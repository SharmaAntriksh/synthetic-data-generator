from __future__ import annotations

from typing import Any, Optional

import numpy as np


def build_buckets_from_key(key: Any) -> list[np.ndarray]:
    """
    Build index buckets for a dense-ish non-negative integer key.

    Returns:
        buckets[k] = np.ndarray of row indices i where key[i] == k

    Notes:
      - Keys are coerced to int64.
      - Keys must be non-negative (0..K). Sparse keys are supported but allocate up to max(key)+1.
      - Stable grouping (uses mergesort).
    """
    key_np = np.asarray(key, dtype=np.int64)
    if key_np.size == 0:
        return []

    if key_np.min() < 0:
        raise RuntimeError("Key values must be non-negative integers")

    max_k = int(key_np.max())
    K = max_k + 1

    order = np.argsort(key_np, kind="mergesort")
    k_sorted = key_np[order]

    starts = np.flatnonzero(np.r_[True, k_sorted[1:] != k_sorted[:-1]])
    ends = np.r_[starts[1:], k_sorted.size]

    buckets: list[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(K)]
    for s, e in zip(starts, ends):
        k = int(k_sorted[int(s)])
        buckets[k] = order[int(s) : int(e)].astype(np.int64, copy=False)

    return buckets


def int_or(v: Any, default: int) -> int:
    """Parse int with a safe default for None/empty/invalid."""
    try:
        if v is None or v == "":
            return int(default)
        return int(v)
    except Exception:
        return int(default)


def float_or(v: Any, default: float) -> float:
    """Parse float with a safe default for None/empty/invalid."""
    try:
        if v is None or v == "":
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def str_or(v: Any, default: str) -> str:
    """Parse/normalize string with a safe default for None/empty."""
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def as_int64(x: Any) -> np.ndarray:
    """Coerce array-like to int64 ndarray."""
    return np.asarray(x, dtype=np.int64)


def as_f64(x: Any) -> np.ndarray:
    """Coerce array-like to float64 ndarray."""
    return np.asarray(x, dtype=np.float64)


def dense_map(mapping: Optional[dict]) -> Optional[np.ndarray]:
    """
    Build a dense lookup array: arr[key] -> value, missing -> -1.

    - Keys/values coerced to int64.
    - Returns None for empty/invalid mapping.
    """
    if not mapping:
        return None

    keys = np.fromiter((int(k) for k in mapping.keys()), dtype=np.int64)
    vals = np.fromiter((int(v) for v in mapping.values()), dtype=np.int64)

    if keys.size == 0:
        return None

    max_key = int(keys.max())
    if max_key < 0:
        return None

    arr = np.full(max_key + 1, -1, dtype=np.int64)
    arr[keys] = vals
    return arr


def infer_T_from_date_pool(date_pool: Any) -> int:
    """
    Infer number of unique months in a date pool.

    Returned T is count of unique numpy datetime64[M] buckets.
    """
    dp = np.asarray(date_pool, dtype="datetime64[D]")
    months = dp.astype("datetime64[M]")
    return int(np.unique(months).size)


# ------------------------------------------------------------------
# Backward-compatible aliases (Sales worker init historically used _*)
# ------------------------------------------------------------------
def _build_buckets_from_brand_key(brand_key: Any) -> list[np.ndarray]:
    return build_buckets_from_key(brand_key)


def _int_or(v: Any, default: int) -> int:
    return int_or(v, default)


def _float_or(v: Any, default: float) -> float:
    return float_or(v, default)


def _str_or(v: Any, default: str) -> str:
    return str_or(v, default)


def _as_int64(x: Any) -> np.ndarray:
    return as_int64(x)


def _as_f64(x: Any) -> np.ndarray:
    return as_f64(x)


def _dense_map(mapping: Optional[dict]) -> Optional[np.ndarray]:
    return dense_map(mapping)


def _infer_T_from_date_pool(date_pool: Any) -> int:
    return infer_T_from_date_pool(date_pool)


__all__ = [
    # preferred names
    "build_buckets_from_key",
    "int_or",
    "float_or",
    "str_or",
    "as_int64",
    "as_f64",
    "dense_map",
    "infer_T_from_date_pool",
    # legacy aliases
    "_build_buckets_from_brand_key",
    "_int_or",
    "_float_or",
    "_str_or",
    "_as_int64",
    "_as_f64",
    "_dense_map",
    "_infer_T_from_date_pool",
]
