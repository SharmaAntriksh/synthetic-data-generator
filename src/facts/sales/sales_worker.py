"""
Backward-compatible shim.

The implementation was split into src.facts.sales.worker.* modules.
Public call sites that import from src.facts.sales.sales_worker remain valid.
"""

from __future__ import annotations

from .worker.init import (
    init_sales_worker,
    _int_or,
    _str_or,
    _as_int64,
    _as_f64,
    _dense_map,
)
from .worker.task import _worker_task
from .worker.chunk_io import (
    _assert_schema,
    _write_parquet_table,
    _write_csv,
    _pa_csv,
    _pa_compute,
)

__all__ = [
    "init_sales_worker",
    "_worker_task",
    # re-exported internals for max compatibility
    "_int_or",
    "_str_or",
    "_as_int64",
    "_as_f64",
    "_dense_map",
    "_assert_schema",
    "_write_parquet_table",
    "_write_csv",
    "_pa_csv",
    "_pa_compute",
]
