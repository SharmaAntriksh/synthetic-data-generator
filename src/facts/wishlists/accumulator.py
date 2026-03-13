"""Wishlist accumulator — collects (CustomerKey, ProductKey) purchase pairs
streamed from sales worker chunks."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.facts.shared.base_accumulator import BaseAccumulator


class WishlistAccumulator(BaseAccumulator):
    """Collects customer-product purchase pairs across all sales chunks."""

    def __init__(self) -> None:
        super().__init__(validator_key="customer_key")

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
