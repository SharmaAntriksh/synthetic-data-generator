"""Per-chunk micro-aggregation for inventory.

Called inside each worker's _worker_task() after the Arrow table is built.
Collapses ~1-2M sales rows to ~(products × stores × months-in-chunk) rows
grouped by (ProductKey, StoreKey, Year, Month).

Returns a small dict of numpy arrays (trivial to pickle across IPC).

Uses flat composite int64 key + argsort + np.add.reduceat for O(n log n) groupby.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pyarrow as pa

from src.facts.shared.micro_agg_helpers import (
    validate_required_columns,
    extract_column,
    decompose_dates,
)


def micro_aggregate_inventory(
    table: pa.Table,
) -> Optional[Dict[str, np.ndarray]]:
    """Collapse a sales chunk to inventory-grain demand aggregates.

    Input:  Arrow table with ProductKey, StoreKey, OrderDate, Quantity columns.

    Output: dict of aligned arrays:
        product_key, store_key, year, month, quantity_sold
    """
    if not validate_required_columns(table, {"ProductKey", "StoreKey", "OrderDate", "Quantity"}):
        return None

    product_keys = extract_column(table, "ProductKey", np.int64)
    store_keys = extract_column(table, "StoreKey", np.int64)
    order_dates = table.column("OrderDate").to_numpy(zero_copy_only=False)
    qty = extract_column(table, "Quantity", np.float64)

    year, month = decompose_dates(order_dates)

    n = len(product_keys)
    if n == 0:
        return None

    # Groupby via flat composite int64 key + argsort + reduceat.
    # Much faster than structured-array sort (single contiguous int64 sort
    # vs. multi-field structured dtype sort).
    max_store = int(store_keys.max()) + 1
    max_month = 13  # months 1..12
    year_min = int(year.min())
    n_years = int(year.max()) - year_min + 1
    stride_s = n_years * max_month
    stride_y = max_month

    flat_key = (
        product_keys * (max_store * stride_s)
        + store_keys * stride_s
        + (year - year_min).astype(np.int64) * stride_y
        + month.astype(np.int64)
    )

    order = np.argsort(flat_key)
    s_fk = flat_key[order]
    s_q = qty[order]

    diff = np.empty(n, dtype=bool)
    diff[0] = True
    diff[1:] = s_fk[1:] != s_fk[:-1]
    group_starts = np.flatnonzero(diff)
    group_qty = np.add.reduceat(s_q, group_starts)
    fk_groups = s_fk[group_starts]

    # Decode flat keys back to (product, store, year, month)
    remainder = fk_groups
    g_product = remainder // (max_store * stride_s)
    remainder = remainder % (max_store * stride_s)
    g_store = remainder // stride_s
    remainder = remainder % stride_s
    g_year = remainder // stride_y + year_min
    g_month = remainder % stride_y

    return {
        "product_key": g_product,
        "store_key": g_store,
        "year": g_year.astype(np.int16),
        "month": g_month.astype(np.int8),
        "quantity_sold": group_qty.astype(np.int64),
    }
