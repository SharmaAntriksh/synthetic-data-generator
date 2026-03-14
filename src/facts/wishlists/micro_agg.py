"""Micro-aggregate (CustomerKey, ProductKey) pairs from sales detail chunks."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pyarrow as pa

from src.facts.shared.micro_agg_helpers import (
    validate_required_columns,
    extract_column,
    deduplicate_tuples,
)


def micro_aggregate_wishlists(detail_table: pa.Table) -> Optional[Dict[str, np.ndarray]]:
    """Extract unique (CustomerKey, ProductKey) pairs from a single sales chunk.

    Returns a compact dict of two int64 arrays, or None if columns are missing.
    """
    if not validate_required_columns(detail_table, {"CustomerKey", "ProductKey"}):
        return None

    ck = extract_column(detail_table, "CustomerKey", np.int64)
    pk = extract_column(detail_table, "ProductKey", np.int64)

    return deduplicate_tuples(
        [ck, pk],
        ["customer_key", "product_key"],
    )
