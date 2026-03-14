"""Shared micro-aggregation helpers used by fact micro-agg modules."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa


def validate_required_columns(
    table: pa.Table,
    required: set[str],
) -> bool:
    """Return True if *table* has all *required* columns and is non-empty."""
    return required.issubset(table.schema.names) and table.num_rows > 0


def extract_column(
    table: pa.Table,
    name: str,
    dtype: type = np.int64,
) -> np.ndarray:
    """Extract a column from an Arrow table as a numpy array with type coercion."""
    return table.column(name).to_numpy(zero_copy_only=False).astype(dtype, copy=False)


def decompose_dates(dates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Decompose datetime64 array into (year:int32, month:int32) arrays."""
    m_int = dates.astype("datetime64[M]").astype(np.int64)
    year = (m_int // 12 + 1970).astype(np.int32)
    month = (m_int % 12 + 1).astype(np.int32)
    return year, month


def deduplicate_tuples(
    arrays: List[np.ndarray],
    output_keys: List[str],
) -> Optional[Dict[str, np.ndarray]]:
    """Deduplicate N-ary tuples via hash-based DataFrame.drop_duplicates.

    Returns a dict mapping *output_keys* to the unique columns,
    or None if the result is empty.
    """
    import pandas as pd

    df = pd.DataFrame({key: arr for key, arr in zip(output_keys, arrays)})
    df = df.drop_duplicates(ignore_index=True)
    if len(df) == 0:
        return None
    return {key: df[key].to_numpy() for key in output_keys}
