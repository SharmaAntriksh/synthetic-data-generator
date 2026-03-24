from __future__ import annotations

# Public policy/constants (stable)
from .encoding import DICT_EXCLUDE, REQUIRED_PRICING_COLS

# Public APIs (stable)
from .parquet_merge import merge_parquet_files, optimize_parquet, DEFAULT_COMPRESSION
from .delta import write_delta_partitioned, DEFAULT_SORT_ROW_LIMIT, DeltaWriteResult

# Convenience re-exports
from .utils import arrow, ensure_dir_for_file
from .projection import project_table_to_schema
from .delta import write_delta_from_parquet_parts


__all__ = [
    # public constants
    "DICT_EXCLUDE",
    "REQUIRED_PRICING_COLS",
    "DEFAULT_COMPRESSION",
    "DEFAULT_SORT_ROW_LIMIT",
    # public APIs
    "merge_parquet_files",
    "optimize_parquet",
    "write_delta_partitioned",
    "write_delta_from_parquet_parts",
    "DeltaWriteResult",
    # re-exported convenience
    "arrow",
    "ensure_dir_for_file",
    "project_table_to_schema",
]