from __future__ import annotations

import os
from typing import Iterable, Optional

from .encoding import _schema_dict_cols, _validate_required, required_pricing_cols_for_table
from .projection import _project_table_to_schema
from .utils import _arrow, _ensure_dir_for_file


def _open_parquet(path: str, pa, pq):
    """
    Open a parquet file in a way that lets us explicitly close resources.
    This avoids keeping dozens/hundreds of files open during merges.
    """
    try:
        source = pa.memory_map(path, "r")
    except Exception:
        # Fallback for environments where memory_map is problematic
        source = pa.OSFile(path, "rb")
    reader = pq.ParquetFile(source)
    return source, reader


def _schema_equals(a, b, *, check_metadata: bool) -> bool:
    eq = getattr(a, "equals", None)
    if callable(eq):
        return a.equals(b, check_metadata=check_metadata)
    # Fallback (older/other schema types)
    return a == b


def _build_canonical_schema(
    parquet_files: list[str],
    *,
    schema_strategy: str,
    pa,
    pq,
):
    """
    schema_strategy:
      - "first": first file wins (current behavior)
      - "union": keep first-file order, append new fields found later
    """
    src0, r0 = _open_parquet(parquet_files[0], pa, pq)
    try:
        base = r0.schema_arrow
    finally:
        try:
            src0.close()
        except Exception:
            pass

    if schema_strategy == "first":
        return base

    if schema_strategy != "union":
        raise ValueError(f"Unknown schema_strategy={schema_strategy!r} (use 'first' or 'union')")

    fields_by_name = {f.name: f for f in base}
    ordered_names = [f.name for f in base]

    for path in parquet_files[1:]:
        src, r = _open_parquet(path, pa, pq)
        try:
            sch = r.schema_arrow
        finally:
            try:
                src.close()
            except Exception:
                pass

        for f in sch:
            if f.name not in fields_by_name:
                fields_by_name[f.name] = f
                ordered_names.append(f.name)

    # Preserve base metadata; projection will coerce incoming tables to this schema.
    return pa.schema([fields_by_name[n] for n in ordered_names], metadata=getattr(base, "metadata", None))


def _read_row_group_projected(reader, rg_index: int, schema, *, table_name: str | None):
    """
    Read a row group and project it to the canonical schema,
    tolerating missing OPTIONAL columns.

    Required-column enforcement should be based on what the file actually has,
    not what we chose to read.
    """
    required = required_pricing_cols_for_table(table_name)

    available = set(reader.schema_arrow.names)
    missing_required = required - available
    if missing_required:
        raise RuntimeError(
            f"Parquet chunk is missing required columns {sorted(missing_required)}; "
            f"file={getattr(reader, 'path', '')}"
        )

    # Read only the columns that exist in this file (in canonical order)
    cols_to_read = [c for c in schema.names if c in available]
    table = reader.read_row_group(rg_index, columns=cols_to_read)

    return _project_table_to_schema(table, schema)


def merge_parquet_files(
    parquet_files: Iterable[str],
    merged_file: str,
    delete_after: bool = False,
    *,
    compression: str = "snappy",
    compression_level: int | None = None,  # useful for zstd
    write_statistics: bool = True,
    table_name: str | None = None,
    schema_strategy: str = "union",  # "union" avoids dropping new columns
) -> Optional[str]:
    """
    Parquet merger:
    - Streams row-groups (bounded memory)
    - Projects mismatched schemas to a canonical schema
    - Avoids keeping all chunk files open simultaneously (better perf, fewer FD issues)

    table_name makes required pricing-column validation table-aware:
      - Enforced for Sales / SalesOrderDetail
      - Not enforced for SalesOrderHeader
    """
    from src.utils.logging_utils import info, skip, done

    pa, _, pq = _arrow()

    parquet_files = [os.path.abspath(p) for p in parquet_files if p and os.path.exists(p)]
    if not parquet_files:
        skip("No parquet chunk files to merge")
        return None

    parquet_files = sorted(parquet_files)
    merged_file = os.path.abspath(merged_file)
    _ensure_dir_for_file(merged_file)

    info(f"Merging {len(parquet_files)} chunks: {os.path.basename(merged_file)}")

    # Canonical schema
    canonical_schema = _build_canonical_schema(
        parquet_files, schema_strategy=schema_strategy, pa=pa, pq=pq
    )

    # Pricing-col validation: table-aware
    _validate_required(canonical_schema, table_name=table_name)

    dict_cols = _schema_dict_cols(canonical_schema, table_name=table_name)

    writer_kwargs = dict(
        compression=compression,
        use_dictionary=dict_cols,
        write_statistics=bool(write_statistics),
    )
    if compression_level is not None:
        writer_kwargs["compression_level"] = int(compression_level)

    try:
        writer = pq.ParquetWriter(merged_file, canonical_schema, **writer_kwargs)
    except TypeError:
        # Older pyarrow may not accept compression_level; fall back cleanly.
        writer_kwargs.pop("compression_level", None)
        writer = pq.ParquetWriter(merged_file, canonical_schema, **writer_kwargs)

    try:
        for path in parquet_files:
            src, reader = _open_parquet(path, pa, pq)
            try:
                schema_exact = _schema_equals(reader.schema_arrow, canonical_schema, check_metadata=True)
                schema_same_fields = _schema_equals(reader.schema_arrow, canonical_schema, check_metadata=False)

                for i in range(reader.num_row_groups):
                    if schema_exact:
                        writer.write_table(reader.read_row_group(i))
                    elif schema_same_fields:
                        # Same fields, different metadata: cheap normalize to canonical
                        t = reader.read_row_group(i, columns=canonical_schema.names)
                        writer.write_table(_project_table_to_schema(t, canonical_schema))
                    else:
                        t = _read_row_group_projected(reader, i, canonical_schema, table_name=table_name)
                        writer.write_table(t)
            finally:
                try:
                    src.close()
                except Exception:
                    pass
    finally:
        writer.close()

    if delete_after:
        for path in parquet_files:
            try:
                os.remove(path)
            except Exception:
                pass

    done(f"Merged chunks: {os.path.basename(merged_file)}")
    return merged_file
