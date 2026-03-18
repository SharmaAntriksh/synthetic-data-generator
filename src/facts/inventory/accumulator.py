"""Inventory accumulator (main-process side).

Collects micro-aggregate dicts as they arrive from workers via IPC,
then produces the final consolidated demand DataFrame for the inventory engine.

Memory: holds (ProductKey × StoreKey × Month) summary rows — typically
~50K-500K rows depending on product/store count.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.facts.shared.base_accumulator import BaseAccumulator


class InventoryAccumulator(BaseAccumulator):
    """
    Accumulator for per-chunk inventory micro-aggregates.

    Usage:
        acc = InventoryAccumulator()
        acc.add(result.get("_inventory_agg"))
        ...
        demand = acc.finalize()
    """

    def __init__(self) -> None:
        super().__init__(validator_key="quantity_sold")

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

        # Concatenate all micro-agg arrays and cast to final dtypes upfront
        _pk = np.concatenate([p["product_key"] for p in self._parts]).astype(np.int32)
        _sk = np.concatenate([p["store_key"] for p in self._parts]).astype(np.int32)
        _yr = np.concatenate([p["year"] for p in self._parts]).astype(np.int16)
        _mo = np.concatenate([p["month"] for p in self._parts]).astype(np.int8)
        _qty = np.concatenate([p["quantity_sold"] for p in self._parts]).astype(np.int32)

        df = pd.DataFrame({
            "ProductKey": _pk, "StoreKey": _sk,
            "Year": _yr, "Month": _mo, "QuantitySold": _qty,
        })

        df = df.groupby(
            ["ProductKey", "StoreKey", "Year", "Month"],
            as_index=False,
        ).agg({"QuantitySold": "sum"})

        return df
