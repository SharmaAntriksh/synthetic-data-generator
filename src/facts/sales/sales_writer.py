"""
Compatibility shim.

Keep imports stable:
  from src.facts.sales.sales_writer import merge_parquet_files, write_delta_partitioned
"""

from __future__ import annotations

from .writers.constants import DICT_EXCLUDE, REQUIRED_PRICING_COLS
from .writers.encoding import _schema_dict_cols, _validate_required
from .writers.parquet_merge import _read_row_group_projected, merge_parquet_files
from .writers.projection import _project_table_to_schema
from .writers.utils import _arrow, _ensure_dir_for_file
from .writers.sales_delta import write_delta_partitioned

__all__ = [
    # public
    "DICT_EXCLUDE",
    "REQUIRED_PRICING_COLS",
    "merge_parquet_files",
    "write_delta_partitioned",
    # internal (keep for safety / older imports)
    "_arrow",
    "_ensure_dir_for_file",
    "_schema_dict_cols",
    "_validate_required",
    "_project_table_to_schema",
    "_read_row_group_projected",
]
