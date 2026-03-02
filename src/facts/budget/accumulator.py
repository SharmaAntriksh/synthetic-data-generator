"""Budget accumulator (main-process side).

Collects micro-aggregate dicts as they arrive from workers via IPC,
then produces the final consolidated actuals DataFrame for the budget engine.

Memory: holds ~50K-200K summary rows in memory (a few MB even at 100M sales).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class BudgetAccumulator:
    """
    Thread-safe (GIL) accumulator for worker micro-aggregates.

    Usage:
        acc = BudgetAccumulator(country_labels, category_labels)

        # In _record_chunk_result:
        acc.add_sales(result["_budget_agg"])
        acc.add_returns(result.get("_returns_agg"))

        # After all chunks:
        actuals = acc.finalize_sales()
        returns = acc.finalize_returns()
    """

    def __init__(
        self,
        country_labels: np.ndarray,
        category_labels: np.ndarray,
    ):
        self._country_labels = country_labels
        self._category_labels = category_labels
        self._sales_parts: List[Dict[str, np.ndarray]] = []
        self._returns_parts: List[Dict[str, np.ndarray]] = []

    def add_sales(self, micro: Optional[Dict[str, np.ndarray]]) -> None:
        """Append a micro-aggregate from one worker chunk."""
        if micro is not None and len(micro.get("sales_amount", [])) > 0:
            self._sales_parts.append(micro)

    def add_returns(self, micro: Optional[Dict[str, np.ndarray]]) -> None:
        """Append a returns micro-aggregate from one worker chunk."""
        if micro is not None and len(micro.get("return_amount", [])) > 0:
            self._returns_parts.append(micro)

    def finalize_sales(self) -> pd.DataFrame:
        """
        Merge all micro-aggregates into a single DataFrame at annual grain.

        Returns DataFrame with columns:
            Country, Category, Year, Month, SalesChannelKey,
            SalesAmount, SalesQuantity

        Then a second-pass groupby to get annual grain:
            Country, Category, Year, SalesChannelKey,
            SalesAmount, SalesQuantity
        """
        if not self._sales_parts:
            return pd.DataFrame(columns=[
                "Country", "Category", "Year", "SalesChannelKey",
                "SalesAmount", "SalesQuantity",
            ])

        # Concat all micro-agg arrays
        df = pd.DataFrame({
            "country_id": np.concatenate([p["country_id"] for p in self._sales_parts]),
            "category_id": np.concatenate([p["category_id"] for p in self._sales_parts]),
            "year": np.concatenate([p["year"] for p in self._sales_parts]),
            "month": np.concatenate([p["month"] for p in self._sales_parts]),
            "channel_key": np.concatenate([p["channel_key"] for p in self._sales_parts]),
            "sales_amount": np.concatenate([p["sales_amount"] for p in self._sales_parts]),
            "sales_qty": np.concatenate([p["sales_qty"] for p in self._sales_parts]),
        })

        # Re-aggregate (workers may produce overlapping month groups in rare edge cases
        # if chunk boundaries split a month — this final groupby is the safety net)
        monthly = df.groupby(
            ["country_id", "category_id", "year", "month", "channel_key"],
            as_index=False,
        ).agg({"sales_amount": "sum", "sales_qty": "sum"})

        # Map ids back to labels
        monthly["Country"] = self._country_labels[monthly["country_id"].to_numpy()]
        monthly["Category"] = self._category_labels[monthly["category_id"].to_numpy()]
        monthly.rename(columns={
            "year": "Year",
            "month": "Month",
            "channel_key": "SalesChannelKey",
            "sales_amount": "SalesAmount",
            "sales_qty": "SalesQuantity",
        }, inplace=True)

        return monthly[["Country", "Category", "Year", "Month",
                         "SalesChannelKey", "SalesAmount", "SalesQuantity"]]

    def finalize_returns(self) -> Optional[pd.DataFrame]:
        """
        Merge return micro-aggregates into annual grain.

        Returns DataFrame with columns:
            Country, Category, Year, SalesChannelKey, ReturnAmount
        """
        if not self._returns_parts:
            return None

        # TODO: same concat + groupby + label mapping as finalize_sales
        return None

    @property
    def has_data(self) -> bool:
        return len(self._sales_parts) > 0
