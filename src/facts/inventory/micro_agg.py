"""Per-chunk micro-aggregation for inventory.

Called inside each worker's _worker_task() after the Arrow table is built.
Collapses ~1-2M sales rows to ~(products × stores × months-in-chunk) rows
grouped by (ProductKey, StoreKey, Year, Month).

Returns a small dict of numpy arrays (trivial to pickle across IPC).

Performance: ~2-5ms per 2M-row chunk (numpy bincount, single O(n) pass).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pyarrow as pa


def micro_aggregate_inventory(
    table: pa.Table,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Collapse a sales chunk to inventory-grain demand aggregates.

    Input:  Arrow table with ProductKey, StoreKey, OrderDate, Quantity columns.

    Output: dict of aligned arrays:
        product_key, store_key, year, month, quantity_sold

    Uses np.bincount with a flat composite key — O(n) single pass.
    """
    required = {"ProductKey", "StoreKey", "OrderDate", "Quantity"}
    if not required.issubset(table.schema.names):
        return None

    if table.num_rows == 0:
        return None

    product_keys = table.column("ProductKey").to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
    store_keys = table.column("StoreKey").to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
    order_dates = table.column("OrderDate").to_numpy(zero_copy_only=False)
    qty = table.column("Quantity").to_numpy(zero_copy_only=False).astype(np.float64, copy=False)

    m_int = order_dates.astype("datetime64[M]").astype(np.int64)
    year = (m_int // 12 + 1970).astype(np.int32)
    month = (m_int % 12 + 1).astype(np.int32)

    max_prod = int(product_keys.max())
    max_store = int(store_keys.max())
    min_year = int(year.min())
    n_year = int(year.max()) - min_year + 1

    n_prod = max_prod + 1
    n_store = max_store + 1
    n_month = 12

    stride_month = 1
    stride_year = n_month
    stride_store = n_year * stride_year
    stride_prod = n_store * stride_store
    total_cells = n_prod * stride_prod

    flat_key = (
        product_keys.astype(np.int64) * stride_prod
        + store_keys.astype(np.int64) * stride_store
        + (year - min_year).astype(np.int64) * stride_year
        + (month - 1).astype(np.int64) * stride_month
    )

    sum_qty = np.bincount(flat_key, weights=qty, minlength=total_cells)

    mask = sum_qty != 0
    indices = np.flatnonzero(mask)

    if indices.size == 0:
        return None

    out_prod, rem = np.divmod(indices, stride_prod)
    out_store, rem = np.divmod(rem, stride_store)
    out_year_idx, out_month_idx = np.divmod(rem, stride_year)

    return {
        "product_key": out_prod.astype(np.int32),
        "store_key": out_store.astype(np.int32),
        "year": (out_year_idx + min_year).astype(np.int16),
        "month": (out_month_idx + 1).astype(np.int8),
        "quantity_sold": sum_qty[indices].astype(np.int64),
    }
