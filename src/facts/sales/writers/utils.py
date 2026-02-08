from __future__ import annotations

import os


# ----------------------------------------------------------------------
# Internal: lazy Arrow import (keeps CSV-only runs lighter)
# ----------------------------------------------------------------------
def _arrow():
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq

        return pa, pc, pq
    except Exception as e:
        raise RuntimeError("pyarrow is required for Parquet merge/Delta writes") from e


def _ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
