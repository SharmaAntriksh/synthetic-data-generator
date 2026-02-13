from __future__ import annotations

import os
from typing import Iterable, Optional

from .encoding import _schema_dict_cols, _validate_required, required_pricing_cols_for_table
from .projection import _project_table_to_schema
from .utils import _arrow

from src.facts.common.writers.parquet_merge import (
    merge_parquet_files as _common_merge,
    _build_canonical_schema as _common_build_canonical_schema,          # internal helper, reused
    _restrict_schema_to_expected as _common_restrict_schema_to_expected,  # internal helper, reused
)


def _expected_cols_for_table(table_name: str | None) -> tuple[str, ...] | None:
    if not table_name:
        return None

    norm = table_name.replace("_", "").replace(" ", "").casefold()

    from src.utils.static_schemas import (
        _SALES_ORDER_HEADER_COLS,
        _SALES_ORDER_DETAIL_COLS,
    )

    if norm == "salesorderheader":
        return _SALES_ORDER_HEADER_COLS
    if norm == "salesorderdetail":
        return _SALES_ORDER_DETAIL_COLS
    return None


def _read_row_group_projected(reader, rg_index: int, schema, *, table_name: str | None):
    """
    Kept for backward compatibility: sales_writer.py imports this symbol.

    Sales-specific required pricing columns enforcement + projection to canonical schema.
    """
    required = required_pricing_cols_for_table(table_name)

    available = set(reader.schema_arrow.names)
    missing_required = required - available
    if missing_required:
        raise RuntimeError(
            f"Parquet chunk is missing required columns {sorted(missing_required)}; "
            f"file={getattr(reader, 'path', '')}"
        )

    cols_to_read = [c for c in schema.names if c in available]
    table = reader.read_row_group(rg_index, columns=cols_to_read)
    return _project_table_to_schema(table, schema)


def merge_parquet_files(
    parquet_files: Iterable[str],
    merged_file: str,
    delete_after: bool = False,
    *,
    compression: str = "snappy",
    compression_level: int | None = None,
    write_statistics: bool = True,
    table_name: str | None = None,
    schema_strategy: str = "union",
) -> Optional[str]:
    """
    Sales wrapper around the common Parquet merger.

    Sales keeps:
      - expected column sets for SalesOrderHeader/Detail
      - schema validation via _validate_required
      - dict encoding column selection via _schema_dict_cols
      - required pricing columns policy per table

    Common handles:
      - FD-safe streaming merge
      - row-group projection/casting
      - writer setup and output file creation
    """
    from src.utils.logging_utils import skip

    pa, _, pq = _arrow()

    files = [os.path.abspath(p) for p in parquet_files if p and os.path.exists(p)]
    if not files:
        skip("No parquet chunk files to merge")
        return None

    files.sort()

    # Build canonical schema here so Sales can run its existing schema validation + dict-col policy.
    canonical_schema = _common_build_canonical_schema(files, schema_strategy=schema_strategy, pa=pa, pq=pq)

    expected_cols = _expected_cols_for_table(table_name)
    if expected_cols:
        canonical_schema = _common_restrict_schema_to_expected(
            canonical_schema, expected_cols, pa, table_name=(table_name or "table")
        )

    _validate_required(canonical_schema, table_name=table_name)
    dict_cols = _schema_dict_cols(canonical_schema, table_name=table_name)

    required_cols = required_pricing_cols_for_table(table_name) or None

    return _common_merge(
        files,
        merged_file,
        delete_after=delete_after,
        compression=compression,
        compression_level=compression_level,
        write_statistics=write_statistics,
        canonical_schema=canonical_schema,
        expected_cols=expected_cols,
        strict_expected=bool(expected_cols),
        reject_extra_cols=bool(expected_cols),
        required_cols=required_cols,
        use_dictionary=dict_cols,
        sort_files=True,
    )


__all__ = ["merge_parquet_files", "_read_row_group_projected"]
