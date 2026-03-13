"""Micro-aggregate (CustomerKey, ProductKey) pairs from sales detail chunks."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pyarrow as pa


def micro_aggregate_wishlists(detail_table: pa.Table) -> Optional[Dict[str, np.ndarray]]:
    """Extract unique (CustomerKey, ProductKey) pairs from a single sales chunk.

    Returns a compact dict of two int64 arrays, or None if columns are missing.
    """
    if (
        "CustomerKey" not in detail_table.column_names
        or "ProductKey" not in detail_table.column_names
    ):
        return None

    ck = detail_table.column("CustomerKey").to_numpy().astype(np.int64)
    pk = detail_table.column("ProductKey").to_numpy().astype(np.int64)

    # Deduplicate within this chunk
    pairs = np.column_stack([ck, pk])
    unique_pairs = np.unique(pairs, axis=0)

    if len(unique_pairs) == 0:
        return None

    return {
        "customer_key": unique_pairs[:, 0],
        "product_key": unique_pairs[:, 1],
    }
