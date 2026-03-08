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

def _extract_dimension_columns(
    table: pa.Table,
    *,
    store_to_country: np.ndarray,
    product_to_cat: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Extract and derive the dimension arrays shared by sales and returns aggregation.

    Returns a dict with: country_id, category_id, year, month, channel_idx,
    channel_uniq, and the per-dimension cardinalities / strides needed for
    the flat composite key.
    """
    store_keys = table.column("StoreKey").to_numpy(zero_copy_only=False)
    product_keys = table.column("ProductKey").to_numpy(zero_copy_only=False)
    channel_raw = table.column("SalesChannelKey").to_numpy(zero_copy_only=False)

    country_id = store_to_country[store_keys]
    category_id = product_to_cat[product_keys]

    order_dates = table.column("OrderDate").to_numpy(zero_copy_only=False)
    m_int = order_dates.astype("datetime64[M]").astype(np.int64)
    year = (m_int // 12 + 1970).astype(np.int32)
    month = (m_int % 12 + 1).astype(np.int32)

    # Channels: remap to 0-based dense index
    channel_uniq = np.unique(channel_raw)
    n_ch = int(channel_uniq.size)
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
        "channel_key": channel_uniq[out_ch_idx.astype(np.intp)].astype(np.int16),
    }


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
    dims = _extract_dimension_columns(
        table,
        store_to_country=store_to_country,
        product_to_cat=product_to_cat,
    )

    qty = table.column("Quantity").to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    net_price = table.column("NetPrice").to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    amount = qty * net_price

    flat_key = dims["flat_key"]
    total_cells = dims["total_cells"]

    sum_amt = np.bincount(flat_key, weights=amount, minlength=total_cells)
    sum_qty = np.bincount(flat_key, weights=qty, minlength=total_cells)

    mask = (sum_amt != 0) | (sum_qty != 0)
    indices = np.flatnonzero(mask)

    decoded = _decode_flat_key(
        indices,
        stride_cat=dims["stride_cat"],
        stride_y=dims["stride_y"],
        stride_m=dims["stride_m"],
        stride_ch=dims["stride_ch"],
        min_year=dims["min_year"],
        channel_uniq=dims["channel_uniq"],
    )
    decoded["sales_amount"] = sum_amt[indices]
    decoded["sales_qty"] = sum_qty[indices]
    return decoded


# ----------------------------------------------------------------
# Returns micro-aggregation
# ----------------------------------------------------------------

_RETURNS_REQUIRED_SALES_COLS = {"StoreKey", "ProductKey", "OrderDate", "SalesChannelKey"}


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

    # Guard: sales_table must carry the dimension columns we need.
    if not _RETURNS_REQUIRED_SALES_COLS.issubset(sales_table.schema.names):
        return None

    # ---- Build composite join key on sales side ----
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

    # Clamp and validate matches
    positions = np.clip(positions, 0, len(sorted_keys) - 1)
    matched = sorted_keys[positions] == ret_key
    if not matched.any():
        return None

    # Map to original detail row indices; drop unmatched returns
    detail_row_idx = sort_idx[positions]
    ret_qty_raw = returns_table.column("ReturnQuantity").to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    ret_amt_raw = returns_table.column("ReturnNetPrice").to_numpy(zero_copy_only=False).astype(np.float64, copy=False)

    if not matched.all():
        detail_row_idx = detail_row_idx[matched]
        ret_qty_raw = ret_qty_raw[matched]
        ret_amt_raw = ret_amt_raw[matched]

    # ---- Extract dimension arrays from the SALES table at matched rows ----
    store_keys = sales_table.column("StoreKey").to_numpy(zero_copy_only=False)[detail_row_idx]
    product_keys = sales_table.column("ProductKey").to_numpy(zero_copy_only=False)[detail_row_idx]
    channel_raw = sales_table.column("SalesChannelKey").to_numpy(zero_copy_only=False)[detail_row_idx]
    order_dates = sales_table.column("OrderDate").to_numpy(zero_copy_only=False)[detail_row_idx]

    country_id = store_to_country[store_keys]
    category_id = product_to_cat[product_keys]

    m_int = order_dates.astype("datetime64[M]").astype(np.int64)
    year = (m_int // 12 + 1970).astype(np.int32)
    month = (m_int % 12 + 1).astype(np.int32)

    # Channel remap (same logic as _extract_dimension_columns)
    channel_uniq = np.unique(channel_raw)
    n_ch = int(channel_uniq.size)
    ch_max = int(channel_uniq.max()) + 1
    ch_remap = np.zeros(ch_max, dtype=np.int32)
    ch_remap[channel_uniq] = np.arange(n_ch, dtype=np.int32)
    channel_idx = ch_remap[channel_raw.astype(np.int64, copy=False)]

    # ---- Flat composite key + bincount ----
    n_country = int(store_to_country.max()) + 1
    n_cat = int(product_to_cat.max()) + 1
    min_year = int(year.min())
    n_year = int(year.max()) - min_year + 1

    stride_ch = n_ch
    stride_m = 12 * stride_ch
    stride_y = n_year * stride_m
    stride_cat = n_cat * stride_y
    total_cells = n_country * stride_cat

    flat_key = (
        country_id.astype(np.int64) * stride_cat
        + category_id.astype(np.int64) * stride_y
        + (year - min_year).astype(np.int64) * stride_m
        + (month - 1).astype(np.int64) * stride_ch
        + channel_idx.astype(np.int64)
    )

    sum_amt = np.bincount(flat_key, weights=ret_amt_raw, minlength=total_cells)
    sum_qty = np.bincount(flat_key, weights=ret_qty_raw, minlength=total_cells)

    mask = (sum_amt != 0) | (sum_qty != 0)
    indices = np.flatnonzero(mask)

    if indices.size == 0:
        return None

    decoded = _decode_flat_key(
        indices,
        stride_cat=stride_cat,
        stride_y=stride_y,
        stride_m=stride_m,
        stride_ch=stride_ch,
        min_year=min_year,
        channel_uniq=channel_uniq,
    )
    decoded["return_amount"] = sum_amt[indices]
    decoded["return_qty"] = sum_qty[indices]
    return decoded
