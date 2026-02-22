from __future__ import annotations

import os
from pathlib import Path
from typing import Union

# ---------------------------------------------------------------------
# Logging (soft dependency)
# ---------------------------------------------------------------------
try:
    from src.utils.logging_utils import info, skip, done
except Exception:  # pragma: no cover
    def info(msg: str) -> None:  # type: ignore
        print(msg)

    def skip(msg: str) -> None:  # type: ignore
        print(msg)

    def done(msg: str) -> None:  # type: ignore
        print(msg)


PathLike = Union[str, os.PathLike]


def _arrow():
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


def arrow():
    """Public alias for `_arrow()` (kept for back-compat)."""
    return _arrow()


def _ensure_dir_for_file(path: PathLike) -> None:
    p = Path(os.fspath(path))
    p.parent.mkdir(parents=True, exist_ok=True)


def ensure_dir_for_file(path: str) -> None:
    """Public alias for `_ensure_dir_for_file()` (kept for back-compat)."""
    _ensure_dir_for_file(path)