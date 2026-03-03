"""Per-chunk micro-aggregation for budget.

Called inside each worker's _worker_task() after the Arrow table is built.
Collapses ~2M rows to ~1K rows grouped by (country_id, category_id, year, month, channel).

Returns a small dict of numpy arrays (trivial to pickle across IPC).

Performance: ~5-10ms per 2M-row chunk (numpy bincount, two O(n) passes).
"""
from __future__ import annotations

from typing import Dict, Optional

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

    Uses np.bincount with a flat composite key — two O(n) passes, no sorting.
    """
    # ---- Extract columns (zero-copy where possible) ----
    store_keys   = table.column("StoreKey").to_numpy(zero_copy_only=False)
    product_keys = table.column("ProductKey").to_numpy(zero_copy_only=False)
    qty          = table.column("Quantity").to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    net_price    = table.column("NetPrice").to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    channel_raw  = table.column("SalesChannelKey").to_numpy(zero_copy_only=False)

    # ---- Vectorized key lookups ----
    country_id  = store_to_country[store_keys]
    category_id = product_to_cat[product_keys]

    # ---- Year/month from OrderDate ----
    order_dates = table.column("OrderDate").to_numpy(zero_copy_only=False)
    m_int = order_dates.astype("datetime64[M]").astype(np.int64)
    year  = (m_int // 12 + 1970).astype(np.int32)
    month = (m_int % 12 + 1).astype(np.int32)

    amount = qty * net_price

    # ---- Dimension cardinalities ----
    n_country = int(store_to_country.max()) + 1
    n_cat     = int(product_to_cat.max()) + 1
    min_year  = int(year.min())
    n_year    = int(year.max()) - min_year + 1

    # ---- Channel dense remap to 0-based index ----
    channel_uniq = np.unique(channel_raw)
    n_ch  = int(channel_uniq.size)
    ch_max = int(channel_uniq.max()) + 1
    ch_remap = np.zeros(ch_max, dtype=np.int32)
    ch_remap[channel_uniq] = np.arange(n_ch, dtype=np.int32)
    channel_idx = ch_remap[channel_raw.astype(np.intp, copy=False)]

    # ---- Flat composite key: single int64 ----
    # key = country_id  * stride_country
    #     + category_id * stride_cat
    #     + year_idx    * stride_year
    #     + month_idx   * stride_month
    #     + channel_idx
    stride_month   = n_ch
    stride_year    = 12 * stride_month     # = 12 * n_ch
    stride_cat     = n_year * stride_year  # = n_year * 12 * n_ch
    stride_country = n_cat * stride_cat    # = n_cat * n_year * 12 * n_ch
    total_cells    = n_country * stride_country

    flat_key = (
        country_id.astype(np.int64)          * stride_country
        + category_id.astype(np.int64)       * stride_cat
        + (year - min_year).astype(np.int64) * stride_year
        + (month - 1).astype(np.int64)       * stride_month
        + channel_idx.astype(np.int64)
    )

    # ---- Bincount: two O(n) passes, no sort ----
    sum_amt = np.bincount(flat_key, weights=amount, minlength=total_cells)
    sum_qty = np.bincount(flat_key, weights=qty,    minlength=total_cells)

    # ---- Extract non-zero cells ----
    indices = np.flatnonzero((sum_amt != 0) | (sum_qty != 0))

    # ---- Decode composite key back to dimensions ----
    out_country,  rem = np.divmod(indices, stride_country)
    out_cat,      rem = np.divmod(rem,     stride_cat)
    out_year_idx, rem = np.divmod(rem,     stride_year)
    out_month_idx, out_ch_idx = np.divmod(rem, stride_month)

    return {
        "country_id":   out_country.astype(np.int32),
        "category_id":  out_cat.astype(np.int32),
        "year":         (out_year_idx + min_year).astype(np.int16),
        "month":        (out_month_idx + 1).astype(np.int8),
        "channel_key":  channel_uniq[out_ch_idx].astype(np.int16),
        "sales_amount": sum_amt[indices],
        "sales_qty":    sum_qty[indices],
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

    Joins returns to the corresponding sales detail on
    (SalesOrderNumber, SalesOrderLineNumber) to obtain StoreKey,
    ProductKey, and SalesChannelKey, then aggregates by
    (country_id, category_id, year, channel_key).

    Monthly grain is not included — the budget engine uses annual return rates.

    Output: dict of aligned arrays:
        country_id, category_id, year, channel_key, return_amount

    Returns None if returns_table is empty or produces no matched rows.
    """
    if returns_table is None or returns_table.num_rows == 0:
        return None

    # ---- Join returns to sales detail for store/product/channel columns ----
    detail_slice = sales_table.select([
        "SalesOrderNumber", "SalesOrderLineNumber",
        "StoreKey", "ProductKey", "SalesChannelKey",
    ])
    joined = returns_table.join(
        detail_slice,
        keys=["SalesOrderNumber", "SalesOrderLineNumber"],
        join_type="left outer",
    )

    # ---- Filter to matched rows (unmatched produce nulls on right-side columns) ----
    valid_mask = joined.column("StoreKey").is_valid().to_numpy(zero_copy_only=False)
    if not valid_mask.any():
        return None

    def _col(name: str) -> np.ndarray:
        return joined.column(name).to_numpy(zero_copy_only=False)[valid_mask]

    store_keys   = _col("StoreKey")
    product_keys = _col("ProductKey")
    channel_raw  = _col("SalesChannelKey")
    ret_qty      = _col("ReturnQuantity").astype(np.float64, copy=False)
    ret_price    = _col("ReturnNetPrice").astype(np.float64, copy=False)
    ret_dates    = _col("ReturnDate")

    return_amount = ret_qty * ret_price

    # ---- Year from ReturnDate ----
    m_int = ret_dates.astype("datetime64[M]").astype(np.int64)
    year  = (m_int // 12 + 1970).astype(np.int32)

    # ---- Vectorized key lookups ----
    country_id  = store_to_country[store_keys]
    category_id = product_to_cat[product_keys]

    # ---- Channel dense remap to 0-based index ----
    channel_uniq = np.unique(channel_raw)
    n_ch  = int(channel_uniq.size)
    ch_max = int(channel_uniq.max()) + 1
    ch_remap = np.zeros(ch_max, dtype=np.int32)
    ch_remap[channel_uniq] = np.arange(n_ch, dtype=np.int32)
    channel_idx = ch_remap[channel_raw.astype(np.intp, copy=False)]

    # ---- Dimension cardinalities ----
    n_country = int(store_to_country.max()) + 1
    n_cat     = int(product_to_cat.max()) + 1
    min_year  = int(year.min())
    n_year    = int(year.max()) - min_year + 1

    # ---- Flat composite key (annual grain — no month dimension) ----
    # key = country_id  * stride_country
    #     + category_id * stride_cat
    #     + year_idx    * stride_year
    #     + channel_idx
    stride_year    = n_ch
    stride_cat     = n_year * stride_year  # = n_year * n_ch
    stride_country = n_cat * stride_cat    # = n_cat * n_year * n_ch
    total_cells    = n_country * stride_country

    flat_key = (
        country_id.astype(np.int64)          * stride_country
        + category_id.astype(np.int64)       * stride_cat
        + (year - min_year).astype(np.int64) * stride_year
        + channel_idx.astype(np.int64)
    )

    # ---- Bincount: single O(n) pass ----
    sum_amt = np.bincount(flat_key, weights=return_amount, minlength=total_cells)

    # ---- Extract non-zero cells ----
    indices = np.flatnonzero(sum_amt != 0)

    # ---- Decode composite key back to dimensions ----
    out_country,  rem = np.divmod(indices, stride_country)
    out_cat,      rem = np.divmod(rem,     stride_cat)
    out_year_idx, out_ch_idx = np.divmod(rem, stride_year)

    return {
        "country_id":    out_country.astype(np.int32),
        "category_id":   out_cat.astype(np.int32),
        "year":          (out_year_idx + min_year).astype(np.int16),
        "channel_key":   channel_uniq[out_ch_idx].astype(np.int16),
        "return_amount": sum_amt[indices],
    }
