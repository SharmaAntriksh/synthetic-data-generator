from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, Set, Union

try:
    from src.utils.logging_utils import info, skip, done
except Exception:  # pragma: no cover
    # Fallback for unit tests / isolated usage
    def info(msg: str) -> None:  # type: ignore
        print(msg)

    def skip(msg: str) -> None:  # type: ignore
        print(msg)

    def done(msg: str) -> None:  # type: ignore
        print(msg)


PathLike = Union[str, os.PathLike]


def _arrow():
    """Import pyarrow lazily and return (pa, pc, pq)."""
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.compute as pc  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pyarrow is required for Parquet merge") from e
    return pa, pc, pq


def _ensure_dir_for_file(path: PathLike) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)


def _open_parquet(path: str, pa, pq):
    """
    Open a parquet file with explicit resource ownership so we can close it.
    Avoids keeping dozens/hundreds of files open during merges.
    """
    try:
        source = pa.memory_map(path, "r")
    except Exception:
        source = pa.OSFile(path, "rb")
    reader = pq.ParquetFile(source)
    # Attach path for nicer error messages if callers expect it.
    try:
        setattr(reader, "path", path)
    except Exception:
        pass
    return source, reader


def _schema_equals(schema_a, schema_b, *, check_metadata: bool) -> bool:
    """Conservative schema equality check (field-by-field)."""
    if schema_a is schema_b:
        return True

    if len(schema_a) != len(schema_b):
        return False

    for fa, fb in zip(schema_a, schema_b):
        if fa != fb:
            return False

    if check_metadata:
        return getattr(schema_a, "metadata", None) == getattr(schema_b, "metadata", None)

    return True


def _build_canonical_schema(
    parquet_files: list[str],
    *,
    schema_strategy: str,
    pa,
    pq,
):
    """
    Build a canonical schema used for the merged output.

    schema_strategy:
      - "first": first file wins
      - "union": keep first-file field order; append new fields encountered later
    """
    if not parquet_files:
        raise ValueError("parquet_files cannot be empty")

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

    return pa.schema(
        [fields_by_name[n] for n in ordered_names],
        metadata=getattr(base, "metadata", None),
    )


def _restrict_schema_to_expected(schema, expected_cols: Sequence[str], pa, *, table_name: str = "table"):
    """Restrict and reorder schema to an expected set of columns."""
    fields = {f.name: f for f in schema}
    missing = [c for c in expected_cols if c not in fields]
    if missing:
        raise RuntimeError(
            f"[{table_name}] canonical schema missing expected columns {missing}. "
            "This usually means wrong chunk files were passed to the merge."
        )
    return pa.schema([fields[c] for c in expected_cols], metadata=getattr(schema, "metadata", None))


def _nulls_of_type(pa, pc, n: int, typ):
    """Create a null array of length n with the requested Arrow type."""
    try:
        return pa.nulls(n, type=typ)
    except Exception:
        # Some complex types may not support nulls(type=...)
        return pc.cast(pa.nulls(n), typ, safe=False)


def _project_table_to_schema(table, schema, *, cast_safe: bool, pa, pc):
    """
    Project a table to exactly match the given schema:
      - Missing columns are filled with nulls
      - Existing columns are cast to the target type when needed
      - Extra columns are dropped
    """
    cols = []
    names = set(table.schema.names)
    nrows = table.num_rows

    for field in schema:
        if field.name in names:
            col = table[field.name]  # ChunkedArray
            if not col.type.equals(field.type):
                col = pc.cast(col, field.type, safe=bool(cast_safe))
        else:
            col = _nulls_of_type(pa, pc, nrows, field.type)
        cols.append(col)

    return pa.Table.from_arrays(cols, schema=schema)


def _row_group_num_rows(reader, rg_index: int) -> int:
    try:
        md = reader.metadata
        if md is not None:
            return int(md.row_group(rg_index).num_rows)
    except Exception:
        pass
    return -1


def _read_row_group_projected(
    reader,
    rg_index: int,
    schema,
    *,
    required_cols: Optional[Set[str]] = None,
    cast_safe: bool = True,
    pa=None,
    pc=None,
):
    """
    Read a row group and project it to the canonical schema, tolerating missing optional columns.

    required_cols:
      - If provided, these columns MUST exist in the underlying file schema; otherwise raise.
        This check is based on file schema, not on projected columns.
    """
    available = set(reader.schema_arrow.names)

    if required_cols:
        missing_required = set(required_cols) - available
        if missing_required:
            raise RuntimeError(
                f"Parquet chunk is missing required columns {sorted(missing_required)}; "
                f"file={getattr(reader, 'path', '')}"
            )

    cols_to_read = [c for c in schema.names if c in available]
    if not cols_to_read:
        n = _row_group_num_rows(reader, rg_index)
        if n < 0:
            # Fall back to reading the row group (pyarrow should preserve num_rows even with 0 cols)
            t0 = reader.read_row_group(rg_index, columns=[])
            return _project_table_to_schema(t0, schema, cast_safe=cast_safe, pa=pa, pc=pc)

        # Build projected table directly using row-group size.
        cols = [_nulls_of_type(pa, pc, n, field.type) for field in schema]
        return pa.Table.from_arrays(cols, schema=schema)

    t = reader.read_row_group(rg_index, columns=cols_to_read)
    return _project_table_to_schema(t, schema, cast_safe=cast_safe, pa=pa, pc=pc)


def merge_parquet_files(
    parquet_files: Iterable[PathLike],
    merged_file: PathLike,
    delete_after: bool = False,
    *,
    compression: str = "snappy",
    compression_level: int | None = None,
    write_statistics: bool = True,
    schema_strategy: str = "union",
    canonical_schema=None,
    expected_cols: Optional[Sequence[str]] = None,
    strict_expected: bool = False,
    reject_extra_cols: bool = False,
    required_cols: Optional[Set[str]] = None,
    use_dictionary: Union[bool, Sequence[str]] = True,
    dict_exclude: Optional[Set[str]] = None,
    validate_schema_fn: Optional[Callable[[Any], None]] = None,
    cast_safe: bool = True,
    sort_files: bool = True,
    log_prefix: str = "",
) -> Optional[str]:
    """
    Merge many Parquet chunk files into a single Parquet file in a streaming, FD-safe way.

    This module is intentionally fact-agnostic. Fact-specific rules (like "required columns"
    or exact-schema enforcement) should be expressed via:
      - expected_cols / strict_expected / reject_extra_cols
      - required_cols
      - validate_schema_fn

    Parameters
    ----------
    parquet_files:
        Iterable of chunk file paths.
    merged_file:
        Output parquet file path.
    delete_after:
        If True, delete chunk files after successful merge.
    schema_strategy:
        "union" (default) or "first" when canonical_schema is not provided.
    canonical_schema:
        Optional pyarrow.Schema to enforce for the output. If provided, schema_strategy is ignored.
    expected_cols:
        If provided, output schema is restricted to these column names in this exact order.
        Missing columns in the canonical schema will raise.
    strict_expected:
        If True and expected_cols is provided, each chunk must contain at least the expected columns.
    reject_extra_cols:
        If True and expected_cols is provided, each chunk must contain no columns outside expected_cols.
    required_cols:
        If provided, each chunk file must contain these columns (checked against file schema).
    use_dictionary:
        Passed to pyarrow.parquet.ParquetWriter (bool or list of column names).
    dict_exclude:
        If use_dictionary=True, exclude these columns from dictionary encoding.
    validate_schema_fn:
        Optional callback invoked with the final canonical schema before writing.
        Use this to enforce fact-specific schema rules.
    cast_safe:
        Passed to pyarrow.compute.cast for schema projection.
    sort_files:
        If True, sort chunk file paths for deterministic merges.
    log_prefix:
        Optional string prepended to log lines (e.g., "[Returns] ").
    """
    pa, pc, pq = _arrow()

    files: list[str] = []
    for f in parquet_files:
        if not f:
            continue
        fp = os.path.abspath(os.fspath(f))
        if os.path.exists(fp):
            files.append(fp)

    if not files:
        skip(f"{log_prefix}No parquet chunk files to merge".strip())
        return None

    if sort_files:
        files.sort()

    merged_file_abs = os.path.abspath(os.fspath(merged_file))
    _ensure_dir_for_file(merged_file_abs)

    info(f"{log_prefix}Merging {len(files)} chunks: {os.path.basename(merged_file_abs)}".strip())

    # Canonical schema
    schema = canonical_schema
    if schema is None:
        schema = _build_canonical_schema(files, schema_strategy=schema_strategy, pa=pa, pq=pq)

    if expected_cols:
        schema = _restrict_schema_to_expected(schema, expected_cols, pa, table_name="merge")

    if validate_schema_fn is not None:
        validate_schema_fn(schema)

    # Determine dictionary columns
    use_dict = use_dictionary
    if use_dictionary is True and dict_exclude:
        use_dict = [c for c in schema.names if c not in dict_exclude]

    writer_kwargs = dict(
        compression=compression,
        use_dictionary=use_dict,
        write_statistics=bool(write_statistics),
    )
    if compression_level is not None:
        writer_kwargs["compression_level"] = int(compression_level)

    try:
        writer = pq.ParquetWriter(merged_file_abs, schema, **writer_kwargs)
    except TypeError:
        # Older pyarrow may not accept compression_level; fall back cleanly.
        writer_kwargs.pop("compression_level", None)
        writer = pq.ParquetWriter(merged_file_abs, schema, **writer_kwargs)

    try:
        expected_set = set(expected_cols) if expected_cols else None

        for path in files:
            src, reader = _open_parquet(path, pa, pq)
            try:
                chunk_cols = set(reader.schema_arrow.names)

                if expected_set and strict_expected:
                    missing = sorted(expected_set - chunk_cols)
                    if missing:
                        raise RuntimeError(f"Chunk missing expected columns {missing}: {path}")

                if expected_set and reject_extra_cols:
                    extra = sorted(chunk_cols - expected_set)
                    if extra:
                        raise RuntimeError(f"Chunk has unexpected columns {extra}: {path}")

                schema_exact = _schema_equals(reader.schema_arrow, schema, check_metadata=True)
                schema_same_fields = _schema_equals(reader.schema_arrow, schema, check_metadata=False)

                for rg in range(reader.num_row_groups):
                    # Fast path: exact schema match and no expected-cols restriction.
                    if schema_exact and not expected_cols:
                        writer.write_table(reader.read_row_group(rg))
                        continue

                    # Read only canonical columns and cast if necessary.
                    if schema_same_fields:
                        t = reader.read_row_group(rg, columns=schema.names)
                        writer.write_table(_project_table_to_schema(t, schema, cast_safe=cast_safe, pa=pa, pc=pc))
                        continue

                    # Full projection path (tolerates missing optional columns).
                    t = _read_row_group_projected(
                        reader,
                        rg,
                        schema,
                        required_cols=required_cols,
                        cast_safe=cast_safe,
                        pa=pa,
                        pc=pc,
                    )
                    writer.write_table(t)
            finally:
                try:
                    src.close()
                except Exception:
                    pass
    finally:
        writer.close()

    if delete_after:
        for path in files:
            try:
                os.remove(path)
            except Exception:
                pass

    done(f"{log_prefix}Merged chunks: {os.path.basename(merged_file_abs)}".strip())
    return merged_file_abs


__all__ = ["merge_parquet_files"]
