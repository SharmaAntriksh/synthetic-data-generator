"""Inventory accumulator (main-process side).

Collects micro-aggregate dicts as they arrive from workers via IPC,
then produces the final consolidated demand DataFrame for the inventory engine.

Memory: holds (ProductKey × StoreKey × Month) summary rows — typically
~50K-500K rows depending on product/store count.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class InventoryAccumulator:
    """
    Accumulator for per-chunk inventory micro-aggregates.

    Usage:
        acc = InventoryAccumulator()
        acc.add(result.get("_inventory_agg"))
        ...
        demand = acc.finalize()
    """

    def __init__(self) -> None:
        self._parts: List[Dict[str, np.ndarray]] = []

    def add(self, micro: Optional[Dict[str, np.ndarray]]) -> None:
        if micro is not None and len(micro.get("quantity_sold", [])) > 0:
            self._parts.append(micro)

    @property
    def has_data(self) -> bool:
        return len(self._parts) > 0

    def finalize(self) -> pd.DataFrame:
        """
        Merge all micro-aggregates into a single DataFrame.

        Returns DataFrame with columns:
            ProductKey, StoreKey, Year, Month, QuantitySold

        Re-aggregates in case chunk boundaries split a month for the same
        (product, store) pair.
        """
        if not self._parts:
            return pd.DataFrame(columns=[
                "ProductKey", "StoreKey", "Year", "Month", "QuantitySold",
            ])

        df = pd.DataFrame({
            "ProductKey": np.concatenate([p["product_key"] for p in self._parts]),
            "StoreKey": np.concatenate([p["store_key"] for p in self._parts]),
            "Year": np.concatenate([p["year"] for p in self._parts]),
            "Month": np.concatenate([p["month"] for p in self._parts]),
            "QuantitySold": np.concatenate([p["quantity_sold"] for p in self._parts]),
        })

        df = df.groupby(
            ["ProductKey", "StoreKey", "Year", "Month"],
            as_index=False,
        ).agg({"QuantitySold": "sum"})

        df["ProductKey"] = df["ProductKey"].astype(np.int32)
        df["StoreKey"] = df["StoreKey"].astype(np.int32)
        df["Year"] = df["Year"].astype(np.int16)
        df["Month"] = df["Month"].astype(np.int8)
        df["QuantitySold"] = df["QuantitySold"].astype(np.int32)

        return df
