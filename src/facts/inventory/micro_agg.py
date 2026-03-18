"""Per-chunk micro-aggregation for inventory.

Called inside each worker's _worker_task() after the Arrow table is built.
Collapses ~1-2M sales rows to ~(products × stores × months-in-chunk) rows
grouped by (ProductKey, StoreKey, Year, Month).

Returns a small dict of numpy arrays (trivial to pickle across IPC).

Uses structured-array sorting + np.add.reduceat for O(n log n) groupby.
Scales safely to millions of distinct products (no dense allocation).
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

    # Build composite sort key: (product, store, year, month)
    # Sort lexicographically then use reduceat for O(n log n) groupby.
    sort_keys = np.empty(n, dtype=[
        ("p", np.int64), ("s", np.int64),
        ("y", np.int16), ("m", np.int8),
    ])
    sort_keys["p"] = product_keys
    sort_keys["s"] = store_keys
    sort_keys["y"] = year
    sort_keys["m"] = month

    order = np.argsort(sort_keys, order=("p", "s", "y", "m"))
    s_p = product_keys[order]
    s_s = store_keys[order]
    s_y = year[order]
    s_m = month[order]
    s_q = qty[order]

    # Find group boundaries (where any key column changes)
    diff = np.empty(n, dtype=bool)
    diff[0] = True
    diff[1:] = (
        (s_p[1:] != s_p[:-1])
        | (s_s[1:] != s_s[:-1])
        | (s_y[1:] != s_y[:-1])
        | (s_m[1:] != s_m[:-1])
    )
    group_starts = np.flatnonzero(diff)

    if group_starts.size == 0:
        return None

    # Sum qty within each group using reduceat
    group_qty = np.add.reduceat(s_q, group_starts)

    # Extract group keys (first element of each group)
    return {
        "product_key": s_p[group_starts],
        "store_key": s_s[group_starts],
        "year": s_y[group_starts].astype(np.int16),
        "month": s_m[group_starts].astype(np.int8),
        "quantity_sold": group_qty.astype(np.int64),
    }
