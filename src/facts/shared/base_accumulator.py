"""Base accumulator for streaming micro-aggregates from sales workers.

Subclasses only need to implement ``finalize()`` to merge the collected
micro-aggregate dicts into their final DataFrame.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd


class BaseAccumulator:
    """Collects micro-aggregate dicts posted by sales workers.

    Each micro-aggregate is a ``Dict[str, np.ndarray]`` keyed by column
    name.  The *validator_key* is checked in ``add()`` to skip empty or
    missing payloads.

    Subclasses must override ``finalize()`` to produce a DataFrame from
    the collected parts.
    """

    def __init__(self, validator_key: str) -> None:
        self._validator_key = validator_key
        self._parts: List[Dict[str, np.ndarray]] = []

    def add(self, micro: Optional[Dict[str, np.ndarray]]) -> None:
        if micro is not None and len(micro.get(self._validator_key, [])) > 0:
            self._parts.append(micro)

    @property
    def has_data(self) -> bool:
        return len(self._parts) > 0

    def finalize(self) -> pd.DataFrame:
        raise NotImplementedError
