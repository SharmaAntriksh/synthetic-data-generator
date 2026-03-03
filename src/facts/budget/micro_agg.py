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
    # ---- Extract columns (zero-copy where possible) ----
    store_keys = table.column("StoreKey").to_numpy(zero_copy_only=False)
    product_keys = table.column("ProductKey").to_numpy(zero_copy_only=False)
    qty = table.column("Quantity").to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    net_price = table.column("NetPrice").to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    channel_raw = table.column("SalesChannelKey").to_numpy(zero_copy_only=False)

    # ---- Vectorized key lookups ----
    country_id = store_to_country[store_keys]    # dense array index, O(n)
    category_id = product_to_cat[product_keys]   # dense array index, O(n)

    # Year/month from OrderDate
    order_dates = table.column("OrderDate").to_numpy(zero_copy_only=False)
    m_int = order_dates.astype("datetime64[M]").astype(np.int64)
    year = (m_int // 12 + 1970).astype(np.int32)
    month = (m_int % 12 + 1).astype(np.int32)

    amount = qty * net_price

    # ---- Compute dimension cardinalities for flat key ----
    n_country = int(store_to_country.max()) + 1
    n_cat = int(product_to_cat.max()) + 1
    min_year = int(year.min())
    n_year = int(year.max()) - min_year + 1
    # Channels: remap to 0-based dense index
    channel_uniq = np.unique(channel_raw)
    n_ch = int(channel_uniq.size)
    # Dense remap: channel_key -> 0..n_ch-1
    ch_max = int(channel_uniq.max()) + 1
    ch_remap = np.full(ch_max, 0, dtype=np.int32)
    ch_remap[channel_uniq] = np.arange(n_ch, dtype=np.int32)
    channel_idx = ch_remap[channel_raw.astype(np.intp, copy=False)]

    # ---- Flat composite key: single int64 ----
    # key = country * (n_cat * n_year * 12 * n_ch)
    #     + category * (n_year * 12 * n_ch)
    #     + year_idx * (12 * n_ch)
    #     + month_idx * n_ch
    #     + channel_idx
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

    # ---- Bincount: O(n), single pass, no sort ----
    sum_amt = np.bincount(flat_key, weights=amount, minlength=total_cells)
    sum_qty = np.bincount(flat_key, weights=qty, minlength=total_cells)

    # ---- Extract non-zero cells ----
    mask = (sum_amt != 0) | (sum_qty != 0)
    indices = np.flatnonzero(mask)

    # Decode composite key back to dimensions
    rem = indices
    out_country = (rem // stride_cat).astype(np.int32)
    rem = rem % stride_cat
    out_cat = (rem // stride_y).astype(np.int32)
    rem = rem % stride_y
    out_year_idx = (rem // stride_m).astype(np.int32)
    rem = rem % stride_m
    out_month_idx = (rem // stride_ch).astype(np.int32)
    out_ch_idx = (rem % stride_ch).astype(np.int32)

    return {
        "country_id": out_country,
        "category_id": out_cat,
        "year": (out_year_idx + min_year).astype(np.int16),
        "month": (out_month_idx + 1).astype(np.int8),
        "channel_key": channel_uniq[out_ch_idx].astype(np.int16),
        "sales_amount": sum_amt[indices],
        "sales_qty": sum_qty[indices],
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

    Output grain: (country_id, category_id, year, channel_key)
    Values: return_amount (ReturnQuantity * ReturnNetPrice)

    Returns None if returns_table is empty.
    """
    if returns_table is None or returns_table.num_rows == 0:
        return None

    # TODO: implement the actual aggregation following same pattern as
    # micro_aggregate_sales but using ReturnQuantity * ReturnNetPrice.
    # Group by (country_id, category_id, year, channel_key) — monthly
    # grain not needed for returns since the budget only uses annual rate.

    return None  # placeholder
