from __future__ import annotations

from .init import (
    init_sales_worker,
    build_buckets_from_key,
    int_or,
    float_or,
    str_or,
    as_int64,
    as_f64,
    dense_map,
    infer_T_from_date_pool,
    _build_buckets_from_brand_key,
)
from .io import (
    ChunkIOConfig,
    add_year_month_from_date,
    normalize_to_schema,
    write_parquet_table,
    write_csv_table,
)
from .pool import PoolRunSpec, iter_imap_unordered
from .returns_builder import ReturnsConfig, build_sales_returns_from_detail
from .schemas import schema_dict_cols
from .task import _worker_task, normalize_tasks, derive_chunk_seed, write_table_by_format

# Legacy back-compat aliases (underscore-prefixed)
_int_or = int_or
_float_or = float_or
_str_or = str_or
_as_int64 = as_int64
_as_f64 = as_f64
_dense_map = dense_map
_infer_T_from_date_pool = infer_T_from_date_pool

__all__ = [
    # Sales worker entrypoints
    "init_sales_worker",
    "_worker_task",

    # Common worker utilities
    "ChunkIOConfig",
    "add_year_month_from_date",
    "normalize_to_schema",
    "write_parquet_table",
    "write_csv_table",

    "PoolRunSpec",
    "iter_imap_unordered",

    "normalize_tasks",
    "derive_chunk_seed",
    "write_table_by_format",

    "ReturnsConfig",
    "build_sales_returns_from_detail",

    "build_buckets_from_key",
    "int_or",
    "float_or",
    "str_or",
    "as_int64",
    "as_f64",
    "dense_map",
    "infer_T_from_date_pool",

    # legacy/back-compat names used across older worker code
    "_build_buckets_from_brand_key",
    "_int_or",
    "_float_or",
    "_str_or",
    "_as_int64",
    "_as_f64",
    "_dense_map",
    "_infer_T_from_date_pool",

    # sales-specific helper
    "schema_dict_cols",
]