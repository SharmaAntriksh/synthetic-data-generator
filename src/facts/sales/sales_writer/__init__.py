from __future__ import annotations

# Public policy/constants (stable)
from .encoding import DICT_EXCLUDE, REQUIRED_PRICING_COLS

# Public APIs (stable)
from .parquet_merge import merge_parquet_files
from .delta import write_delta_partitioned

# Convenience re-exports
from .utils import arrow, ensure_dir_for_file
from .projection import project_table_to_schema

# Internal/back-compat symbols
from .utils import _arrow, _ensure_dir_for_file
from .encoding import _schema_dict_cols, _validate_required
from .projection import _project_table_to_schema
from .parquet_merge import _read_row_group_projected

# Optional: keep these available even if not in __all__
from .delta import DeltaWriteResult, write_delta_from_parquet_parts  # noqa: F401


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