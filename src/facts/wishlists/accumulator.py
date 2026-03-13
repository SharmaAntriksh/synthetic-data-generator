"""Wishlist accumulator — collects (CustomerKey, ProductKey) purchase pairs
streamed from sales worker chunks."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class WishlistAccumulator:
    """Collects customer-product purchase pairs across all sales chunks."""

    def __init__(self) -> None:
        self._parts: List[Dict[str, np.ndarray]] = []

    def add(self, micro: Optional[Dict[str, np.ndarray]]) -> None:
        if micro is not None and len(micro.get("customer_key", [])) > 0:
            self._parts.append(micro)

    @property
    def has_data(self) -> bool:
        return len(self._parts) > 0

    def finalize(self) -> pd.DataFrame:
        """Return deduplicated (CustomerKey, ProductKey) pairs as a DataFrame."""
        if not self._parts:
            return pd.DataFrame(columns=["CustomerKey", "ProductKey"])

        all_ck = np.concatenate([p["customer_key"] for p in self._parts])
        all_pk = np.concatenate([p["product_key"] for p in self._parts])

        # Global dedup across chunks
        pairs = np.column_stack([all_ck, all_pk])
        unique_pairs = np.unique(pairs, axis=0)

        return pd.DataFrame({
            "CustomerKey": unique_pairs[:, 0],
            "ProductKey": unique_pairs[:, 1],
        })
