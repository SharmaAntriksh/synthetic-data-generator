from __future__ import annotations

# Public policy/constants (stable)
from .writers.encoding import DICT_EXCLUDE, REQUIRED_PRICING_COLS

# Public APIs (stable)
from .writers.parquet_merge import merge_parquet_files
from .writers.delta import write_delta_partitioned

# Convenience re-exports (kept as in the monolith)
from .writers.utils import arrow, ensure_dir_for_file
from .writers.projection import project_table_to_schema

# Internal/back-compat symbols (underscore names)
from .writers.utils import _arrow, _ensure_dir_for_file
from .writers.encoding import _schema_dict_cols, _validate_required
from .writers.projection import _project_table_to_schema
from .writers.parquet_merge import _read_row_group_projected

# Optional: keep these available even if not in __all__ (was true in the single file)
from .writers.delta import DeltaWriteResult, write_delta_from_parquet_parts  # noqa: F401


__all__ = [
    # public
    "DICT_EXCLUDE",
    "REQUIRED_PRICING_COLS",
    "merge_parquet_files",
    "write_delta_partitioned",
    # re-exported convenience
    "arrow",
    "ensure_dir_for_file",
    "project_table_to_schema",
    # internal (keep for safety / older imports)
    "_arrow",
    "_ensure_dir_for_file",
    "_schema_dict_cols",
    "_validate_required",
    "_project_table_to_schema",
    "_read_row_group_projected",
]