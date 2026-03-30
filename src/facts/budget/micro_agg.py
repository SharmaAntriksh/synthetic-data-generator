"""Per-chunk micro-aggregation for budget.

Called inside each worker's _worker_task() after the Arrow table is built.
Collapses ~2M rows to ~1K rows grouped by (country_id, category_id, year, month, channel).

Returns a small dict of numpy arrays (trivial to pickle across IPC).

Performance: ~5-10ms per 2M-row chunk (numpy bincount, single O(n) pass).
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pyarrow as pa


# ----------------------------------------------------------------
# Shared column extraction
# ----------------------------------------------------------------

def _build_dimension_arrays(
    store_keys: np.ndarray,
    product_keys: np.ndarray,
    channel_raw: np.ndarray,
    order_dates: np.ndarray,
    *,
    store_to_country: np.ndarray,
    product_to_cat: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Derive dimension arrays and flat composite key for bincount aggregation.

    Accepts raw numpy arrays (extracted from either an Arrow table or a join
    result) so both sales and returns micro-aggregation can share this logic.

    Returns a dict with: country_id, category_id, year, month, channel_idx,
    channel_uniq, and the per-dimension cardinalities / strides needed for
    the flat composite key.
    """
    # Bounds-safe lookup: keys beyond the lookup array map to 0 (first bucket)
    # rather than raising IndexError or leaving -1 (which breaks np.bincount).
    _sk_safe = np.clip(store_keys, 0, len(store_to_country) - 1)
    _pk_safe = np.clip(product_keys, 0, len(product_to_cat) - 1)
    country_id = store_to_country[_sk_safe]
    category_id = product_to_cat[_pk_safe]

    # Replace any -1 sentinel values (unmapped keys) with 0 so flat_key
    # stays non-negative — np.bincount requires non-negative indices.
    country_id = np.where(country_id < 0, 0, country_id)
    category_id = np.where(category_id < 0, 0, category_id)

    m_int = order_dates.astype("datetime64[M]").astype(np.int64)
    year = (m_int // 12 + 1970).astype(np.int32)
    month = (m_int % 12 + 1).astype(np.int32)

    # Channels: remap to 0-based dense index (with bounds check)
    channel_uniq = np.unique(channel_raw)
    n_ch = int(channel_uniq.size)
    if n_ch == 0:
        raise ValueError("Budget micro-agg: no channel keys found in chunk")
    ch_min = int(channel_uniq.min())
    if ch_min < 0:
        raise ValueError(f"Budget micro-agg: negative channel key {ch_min}")
    ch_max = int(channel_uniq.max()) + 1
    ch_remap = np.zeros(ch_max, dtype=np.int32)
    ch_remap[channel_uniq] = np.arange(n_ch, dtype=np.int32)
    channel_idx = ch_remap[channel_raw.astype(np.int64, copy=False)]

    n_country = int(store_to_country.max()) + 1
    n_cat = int(product_to_cat.max()) + 1
    min_year = int(year.min())
    n_year = int(year.max()) - min_year + 1

    stride_ch = n_ch
    stride_m = 12 * stride_ch
    stride_y = n_year * stride_m
    stride_cat = n_cat * stride_y
    total_cells = n_country * stride_cat

    if total_cells > 2**62:
        raise OverflowError(f"Budget micro-aggregation stride would overflow: {total_cells} cells")

    flat_key = (
        country_id.astype(np.int64) * stride_cat
        + category_id.astype(np.int64) * stride_y
        + (year - min_year).astype(np.int64) * stride_m
        + (month - 1).astype(np.int64) * stride_ch
        + channel_idx.astype(np.int64)
    )

    return {
        "country_id": country_id,
        "category_id": category_id,
        "year": year,
        "month": month,
        "channel_idx": channel_idx,
        "channel_uniq": channel_uniq,
        "min_year": min_year,
        "flat_key": flat_key,
        "total_cells": total_cells,
        "stride_cat": stride_cat,
        "stride_y": stride_y,
        "stride_m": stride_m,
        "stride_ch": stride_ch,
    }


def _extract_columns_from_table(
    table: pa.Table,
    *,
    store_to_country: np.ndarray,
    product_to_cat: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Extract dimension columns from an Arrow table and build flat key."""
    return _build_dimension_arrays(
        store_keys=table.column("StoreKey").to_numpy(zero_copy_only=False).astype(np.int64),
        product_keys=table.column("ProductKey").to_numpy(zero_copy_only=False).astype(np.int64),
        channel_raw=table.column("SalesChannelKey").to_numpy(zero_copy_only=False),
        order_dates=table.column("OrderDate").to_numpy(zero_copy_only=False),
        store_to_country=store_to_country,
        product_to_cat=product_to_cat,
    )


def _decode_flat_key(
    indices: np.ndarray,
    *,
    stride_cat: int,
    stride_y: int,
    stride_m: int,
    stride_ch: int,
    min_year: int,
    channel_uniq: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Decode composite flat keys back into dimension arrays."""
    out_country, rem = np.divmod(indices, stride_cat)
    out_cat, rem = np.divmod(rem, stride_y)
    out_year_idx, rem = np.divmod(rem, stride_m)
    out_month_idx, out_ch_idx = np.divmod(rem, stride_ch)

    return {
        "country_id": out_country.astype(np.int32),
        "category_id": out_cat.astype(np.int32),
        "year": (out_year_idx + min_year).astype(np.int16),
        "month": (out_month_idx + 1).astype(np.int8),
        "channel_key": channel_uniq[out_ch_idx.astype(np.intp)].astype(np.int32),
    }


# ----------------------------------------------------------------
# Shared bincount + decode pipeline
# ----------------------------------------------------------------

def _aggregate_via_bincount(
    dims: Dict[str, np.ndarray],
    weight_pairs: Dict[str, np.ndarray],
) -> Optional[Dict[str, np.ndarray]]:
    """Bincount aggregation over flat composite key, then decode.

    Args:
        dims: output of ``_build_dimension_arrays()``.
        weight_pairs: {output_name: weights_array} — each is bincount'd
            over the same flat_key.

    Returns:
        dict with decoded dimension columns + aggregated weight columns,
        or None if all bins are zero.
    """
    flat_key = dims["flat_key"]
    total_cells = dims["total_cells"]

    sums = {
        name: np.bincount(flat_key, weights=w, minlength=total_cells)
        for name, w in weight_pairs.items()
    }

    # Build non-zero mask directly via functools.reduce (avoids pre-allocating zeros)
    sum_arrays = list(sums.values())
    mask = sum_arrays[0] != 0
    for arr in sum_arrays[1:]:
        mask |= arr != 0
    indices = np.flatnonzero(mask)

    if indices.size == 0:
        return None

    decoded = _decode_flat_key(
        indices,
        stride_cat=dims["stride_cat"],
        stride_y=dims["stride_y"],
        stride_m=dims["stride_m"],
        stride_ch=dims["stride_ch"],
        min_year=dims["min_year"],
        channel_uniq=dims["channel_uniq"],
    )
    for name, arr in sums.items():
        decoded[name] = arr[indices]
    return decoded


# ----------------------------------------------------------------
# Sales micro-aggregation
# ----------------------------------------------------------------

def micro_aggregate_sales(
    table: pa.Table,
    *,
    store_to_country: np.ndarray,   # dense: StoreKey -> country_id (int32)
    product_to_cat: np.ndarray,     # dense: ProductKey -> category_id (int32)
) -> Dict[str, np.ndarray]:
    """
    Collapse a sales chunk to budget-grain aggregates.

    Input:  ~2M row Arrow table with StoreKey, ProductKey, OrderDate,
            Quantity, NetPrice, SalesChannelKey columns.

    Output: dict of aligned arrays (each ~500-2000 rows):
        country_id, category_id, year, month, channel_key,
        sales_amount (Quantity * NetPrice), sales_qty

    Uses np.bincount with a flat composite key — O(n) single pass,
    no sorting, no structured arrays.
    """
    dims = _extract_columns_from_table(
        table,
        store_to_country=store_to_country,
        product_to_cat=product_to_cat,
    )

    qty = table.column("Quantity").to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    net_price = table.column("NetPrice").to_numpy(zero_copy_only=False).astype(np.float64, copy=False)

    result = _aggregate_via_bincount(dims, {
        "sales_amount": qty * net_price,
        "sales_qty": qty,
    })
    # Sales micro-agg always returns a dict (even if empty — caller checks length)
    return result or {
        "country_id": np.empty(0, dtype=np.int32),
        "category_id": np.empty(0, dtype=np.int32),
        "year": np.empty(0, dtype=np.int16),
        "month": np.empty(0, dtype=np.int8),
        "channel_key": np.empty(0, dtype=np.int32),
        "sales_amount": np.empty(0, dtype=np.float64),
        "sales_qty": np.empty(0, dtype=np.float64),
    }


# ----------------------------------------------------------------
# Returns micro-aggregation
# ----------------------------------------------------------------

_RETURNS_REQUIRED_SALES_COLS = {"StoreKey", "ProductKey", "OrderDate", "SalesChannelKey"}


def _join_returns_to_sales(
    returns_table: pa.Table,
    sales_table: pa.Table,
) -> Optional[Dict[str, np.ndarray]]:
    """Join returns to sales via (SalesOrderNumber, SalesOrderLineNumber).

    Returns dict with matched dimension arrays from the sales side plus
    return value arrays, or None if no matches.
    """
    sales_so = sales_table.column("SalesOrderNumber").to_numpy(zero_copy_only=False)
    sales_ln = sales_table.column("SalesOrderLineNumber").to_numpy(zero_copy_only=False)

    ret_so = returns_table.column("SalesOrderNumber").to_numpy(zero_copy_only=False)
    ret_ln = returns_table.column("SalesOrderLineNumber").to_numpy(zero_copy_only=False)

    # Pack (OrderNumber, LineNumber) into a single int64 for fast lookup.
    sales_ln_i64 = sales_ln.astype(np.int64)
    ret_ln_i64 = ret_ln.astype(np.int64)
    max_ln = max(int(sales_ln_i64.max()) if sales_ln_i64.size else 0,
                 int(ret_ln_i64.max()) if ret_ln_i64.size else 0)
    shift_bits = max(16, int(max_ln).bit_length())
    sales_key = (sales_so.astype(np.int64) << shift_bits) | sales_ln_i64
    ret_key = (ret_so.astype(np.int64) << shift_bits) | ret_ln_i64

    # Map each return row to its source detail row via sorted lookup
    sort_idx = np.argsort(sales_key)
    sorted_keys = sales_key[sort_idx]
    positions = np.searchsorted(sorted_keys, ret_key)

    in_bounds = positions < len(sorted_keys)
    positions = np.clip(positions, 0, max(len(sorted_keys) - 1, 0))
    matched = in_bounds & (sorted_keys[positions] == ret_key)
    if not matched.any():
        return None

    detail_row_idx = sort_idx[positions]
    ret_qty = returns_table.column("ReturnQuantity").to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    ret_amt = returns_table.column("ReturnNetPrice").to_numpy(zero_copy_only=False).astype(np.float64, copy=False)

    if not matched.all():
        detail_row_idx = detail_row_idx[matched]
        ret_qty = ret_qty[matched]
        ret_amt = ret_amt[matched]

    return {
        "store_keys": sales_table.column("StoreKey").to_numpy(zero_copy_only=False)[detail_row_idx].astype(np.int64),
        "product_keys": sales_table.column("ProductKey").to_numpy(zero_copy_only=False)[detail_row_idx].astype(np.int64),
        "channel_raw": sales_table.column("SalesChannelKey").to_numpy(zero_copy_only=False)[detail_row_idx],
        "order_dates": sales_table.column("OrderDate").to_numpy(zero_copy_only=False)[detail_row_idx],
        "return_amount": ret_amt,
        "return_qty": ret_qty,
    }


def micro_aggregate_returns(
    returns_table: pa.Table,
    sales_table: pa.Table,
    *,
    store_to_country: np.ndarray,
    product_to_cat: np.ndarray,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Collapse a returns chunk to budget-grain aggregates.

    The returns table only carries return-specific columns (ReturnQuantity,
    ReturnNetPrice, etc.) — it does NOT have StoreKey, ProductKey, OrderDate,
    or SalesChannelKey.  These dimension columns live on the sales (detail)
    table, so we join through (SalesOrderNumber, SalesOrderLineNumber) to
    recover them before aggregating.

    Output grain: (country_id, category_id, year, month, channel_key)
    Values: return_amount (ReturnNetPrice, already prorated by the builder),
            return_qty   (ReturnQuantity)

    Returns None if the returns table is empty or if the sales table lacks
    the dimension columns needed for the join.
    """
    if returns_table is None or returns_table.num_rows == 0:
        return None

    if not _RETURNS_REQUIRED_SALES_COLS.issubset(sales_table.schema.names):
        return None

    joined = _join_returns_to_sales(returns_table, sales_table)
    if joined is None:
        return None

    dims = _build_dimension_arrays(
        store_keys=joined["store_keys"],
        product_keys=joined["product_keys"],
        channel_raw=joined["channel_raw"],
        order_dates=joined["order_dates"],
        store_to_country=store_to_country,
        product_to_cat=product_to_cat,
    )

    return _aggregate_via_bincount(dims, {
        "return_amount": joined["return_amount"],
        "return_qty": joined["return_qty"],
    })
