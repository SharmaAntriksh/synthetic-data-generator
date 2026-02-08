from __future__ import annotations

import os
from typing import Iterable, Optional

from .constants import REQUIRED_PRICING_COLS
from .encoding import _schema_dict_cols, _validate_required
from .projection import _project_table_to_schema
from .utils import _arrow, _ensure_dir_for_file


def _read_row_group_projected(reader, rg_index: int, schema):
    """
    Read a row group from a ParquetFile and project it to canonical schema
    without failing when some optional columns are missing.
    """
    _, _, pq = _arrow()

    reader_schema = reader.schema_arrow
    available = set(reader_schema.names)

    # Read only the columns that exist in this file (in canonical order)
    cols_to_read = [c for c in schema.names if c in available]
    table = reader.read_row_group(rg_index, columns=cols_to_read)

    # Enforce required columns exist in *this* file too (otherwise data is invalid)
    missing_required = REQUIRED_PRICING_COLS - set(cols_to_read)
    if missing_required:
        raise RuntimeError(
            f"Parquet chunk is missing required columns {sorted(missing_required)}; file={getattr(reader, 'path', '')}"
        )

    return _project_table_to_schema(table, schema)


def merge_parquet_files(
    parquet_files: Iterable[str],
    merged_file: str,
    delete_after: bool = False,
    *,
    compression: str = "snappy",
    write_statistics: bool = True,
) -> Optional[str]:
    """
    Optimized parquet merger:
    - Streams row-groups (bounded memory)
    - Projects mismatched schemas to a canonical schema (first file wins)
    - Dictionary encodes only string-like columns (except exclusions)
    """
    from src.utils.logging_utils import info, skip, done

    _, _, pq = _arrow()

    parquet_files = [os.path.abspath(p) for p in parquet_files if p and os.path.exists(p)]
    if not parquet_files:
        skip("No parquet chunk files to merge")
        return None

    parquet_files = sorted(parquet_files)
    merged_file = os.path.abspath(merged_file)
    _ensure_dir_for_file(merged_file)

    info(f"Merging {len(parquet_files)} chunks: {os.path.basename(merged_file)}")

    readers = [(p, pq.ParquetFile(p)) for p in parquet_files]

    # Canonical schema (first file wins by design)
    canonical_schema = readers[0][1].schema_arrow
    _validate_required(canonical_schema)

    dict_cols = _schema_dict_cols(canonical_schema)

    writer = pq.ParquetWriter(
        merged_file,
        canonical_schema,
        compression=compression,
        use_dictionary=dict_cols,
        write_statistics=bool(write_statistics),
    )

    try:
        for path, reader in readers:
            for i in range(reader.num_row_groups):
                if reader.schema_arrow == canonical_schema:
                    writer.write_table(reader.read_row_group(i))
                else:
                    # Safe projection path
                    table = _read_row_group_projected(reader, i, canonical_schema)
                    writer.write_table(table)
    finally:
        writer.close()

    if delete_after:
        for path, _ in readers:
            try:
                os.remove(path)
            except Exception:
                pass

    done(f"Merged chunks: {os.path.basename(merged_file)}")
    return merged_file
