"""Complaints accumulator — collects (CustomerKey, SalesOrderNumber,
SalesOrderLineNumber) triples streamed from sales worker chunks."""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class ComplaintsAccumulator:
    """Collects customer-order-line triples across all sales chunks."""

    def __init__(self) -> None:
        self._parts: List[Dict[str, np.ndarray]] = []

    def add(self, micro: Optional[Dict[str, np.ndarray]]) -> None:
        if micro is not None and len(micro.get("customer_key", [])) > 0:
            self._parts.append(micro)

    @property
    def has_data(self) -> bool:
        return len(self._parts) > 0

    def finalize(self) -> pd.DataFrame:
        """Return deduplicated (CustomerKey, SalesOrderNumber,
        SalesOrderLineNumber) triples as a DataFrame."""
        if not self._parts:
            return pd.DataFrame(
                columns=["CustomerKey", "SalesOrderNumber", "SalesOrderLineNumber"]
            )

        all_ck = np.concatenate([p["customer_key"] for p in self._parts])
        all_so = np.concatenate([p["sales_order_number"] for p in self._parts])
        all_ln = np.concatenate([p["line_number"] for p in self._parts])

        triples = np.column_stack([all_ck, all_so, all_ln])
        unique = np.unique(triples, axis=0)

        return pd.DataFrame({
            "CustomerKey": unique[:, 0],
            "SalesOrderNumber": unique[:, 1],
            "SalesOrderLineNumber": unique[:, 2],
        })
