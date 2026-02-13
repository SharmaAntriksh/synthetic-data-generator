from __future__ import annotations

import os
from typing import Tuple


def arrow():
    """
    Lazy Arrow import (keeps CSV-only runs lighter).

    Returns: (pa, pc, pq)
    """
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.compute as pc  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
        return pa, pc, pq
    except Exception as e:
        raise RuntimeError("pyarrow is required for Parquet merge/Delta writes") from e


def ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


# Backward-compatible aliases (Sales code expects underscore names)
def _arrow():
    return arrow()


def _ensure_dir_for_file(path: str) -> None:
    ensure_dir_for_file(path)


__all__ = ["arrow", "ensure_dir_for_file", "_arrow", "_ensure_dir_for_file"]
