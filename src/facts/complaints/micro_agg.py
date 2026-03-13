"""Micro-aggregate (CustomerKey, SalesOrderNumber, SalesOrderLineNumber)
from sales detail chunks for the complaints pipeline."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pyarrow as pa


def micro_aggregate_complaints(detail_table: pa.Table) -> Optional[Dict[str, np.ndarray]]:
    """Extract unique (CustomerKey, SalesOrderNumber, SalesOrderLineNumber)
    triples from a single sales chunk.

    Returns a compact dict of three arrays, or None if required columns
    are missing.
    """
    required = {"CustomerKey", "SalesOrderNumber", "SalesOrderLineNumber"}
    if not required.issubset(detail_table.column_names):
        return None

    ck = detail_table.column("CustomerKey").to_numpy().astype(np.int64)
    so = detail_table.column("SalesOrderNumber").to_numpy().astype(np.int64)
    ln = detail_table.column("SalesOrderLineNumber").to_numpy().astype(np.int64)

    # Deduplicate within this chunk
    triples = np.column_stack([ck, so, ln])
    unique = np.unique(triples, axis=0)

    if len(unique) == 0:
        return None

    return {
        "customer_key": unique[:, 0],
        "sales_order_number": unique[:, 1],
        "line_number": unique[:, 2],
    }
