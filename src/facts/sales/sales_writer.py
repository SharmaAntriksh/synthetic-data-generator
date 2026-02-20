"""
Backward-compatible *single-file* sales writer module.

Goal: make src.facts.sales.sales_writer the single source of truth by inlining
the shared writer utilities previously living under src.facts.common.writers/*.

Keep imports stable:
  from src.facts.sales.sales_writer import merge_parquet_files, write_delta_partitioned
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Sequence, Set, Union

from .output_paths import TABLE_SALES, TABLE_SALES_ORDER_DETAIL

# ---------------------------------------------------------------------
# Logging (soft dependency)
# ---------------------------------------------------------------------
try:
    from src.utils.logging_utils import info, skip, done
except Exception:  # pragma: no cover
    def info(msg: str) -> None:  # type: ignore
        print(msg)

    def skip(msg: str) -> None:  # type: ignore
        print(msg)

    def done(msg: str) -> None:  # type: ignore
        print(msg)


PathLike = Union[str, os.PathLike]

# ===============================================================
# common/writers/utils.py (inlined)
# ===============================================================


def _arrow():
    """
    Lazy Arrow import (keeps CSV-only runs lighter).

    Returns: (pa, pc, pq)
    """
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.compute as pc  # type: ignore
        import pyarrow.parquet as pq  # type: ignore
        return pa, pc, pq
    except Exception as e:
        raise RuntimeError("pyarrow is required for Parquet merge/Delta writes") from e


def arrow():
    """Public alias for `_arrow()` (kept for back-compat)."""
    return _arrow()


def _ensure_dir_for_file(path: PathLike) -> None:
    p = Path(os.fspath(path))
    p.parent.mkdir(parents=True, exist_ok=True)


def ensure_dir_for_file(path: str) -> None:
    """Public alias for `_ensure_dir_for_file()` (kept for back-compat)."""
    _ensure_dir_for_file(path)


# ===============================================================
# constants.py (inlined - Sales policy)
# ===============================================================

# Columns we never dictionary-encode
DICT_EXCLUDE = {"SalesOrderNumber", "CustomerKey"}

# Columns that must always exist in Sales (line-grain)
REQUIRED_PRICING_COLS = {
    "UnitPrice",
    "NetPrice",
    "UnitCost",
    "DiscountAmount",
}

# ===============================================================
# common/writers/encoding.py (inlined)
# ===============================================================


def validate_required_columns(
    schema,
    required: Iterable[str] | Set[str],
    *,
    what: str = "required columns",
) -> None:
    """
    Generic required-column validation.
    Policy (which columns are required) must live in the fact module.
    """
    req = set(required or [])
    if not req:
        return

    names = set(getattr(schema, "names", []) or [])
    missing = req - names
    if missing:
        raise RuntimeError(f"Missing {what}: {sorted(missing)}")


def schema_dict_cols(schema, exclude: Optional[Iterable[str]] = None) -> List[str]:
    """
    Generic dictionary-encoding candidate columns for Parquet writers.

    Mechanics only:
      - dictionary-encode string/binary-like columns
      - skip excluded columns

    Policy stays outside:
      - required-column enforcement
      - table-specific exclusions
    """
    try:
        import pyarrow as pa  # type: ignore
    except Exception as e:
        raise RuntimeError("pyarrow is required for dictionary-encoding decisions") from e

    exclude_set = set(exclude or [])
    out: List[str] = []

    for f in schema:
        if f.name in exclude_set:
            continue

        t = f.type
        if (
            pa.types.is_string(t)
            or pa.types.is_large_string(t)
            or pa.types.is_binary(t)
            or pa.types.is_large_binary(t)
        ):
            out.append(f.name)

    return out


# ===============================================================
# common/writers/projection.py (inlined)
# ===============================================================


def project_table_to_schema(table: Any, schema: Any, *, cast_safe: bool = False) -> Any:
    """
    Project/align an Arrow table to the canonical schema:
      - reorder columns
      - add missing columns as typed nulls
      - cast columns to canonical types when safe/possible

    Notes for reuse across facts:
      - Does NOT validate "required columns" (policy). Callers should validate separately.
      - Drops any extra columns not present in `schema` by construction (policy is external).
    """
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.compute as pc  # type: ignore
    except Exception as e:
        raise RuntimeError("pyarrow is required for schema projection") from e

    if table.schema == schema:
        return table

    n = table.num_rows
    have = set(table.schema.names)

    arrays = []
    for field in schema:
        name = field.name

        if name not in have:
            arrays.append(pa.nulls(n, type=field.type))
            continue

        col = table[name]  # ChunkedArray
        if col.type != field.type:
            try:
                col = pc.cast(col, field.type, safe=bool(cast_safe))
            except Exception as ex:
                raise RuntimeError(
                    f"Failed to cast column '{name}' from {col.type} to {field.type}: {ex}"
                ) from ex

        arrays.append(col)

    return pa.Table.from_arrays(arrays, schema=schema)


def _project_table_to_schema(table: Any, schema: Any) -> Any:
    """Backward-compatible alias (many callers expect underscore name)."""
    return project_table_to_schema(table, schema, cast_safe=False)


# ===============================================================
# Sales-specific encoding policy (inlined)
# ===============================================================


def required_pricing_cols_for_table(table_name: str | None) -> Set[str]:
    """
    Pricing cols are required only for line-grain tables.

    Back-compat:
      - If table_name is None, keep old strict behavior (require pricing cols).
    """
    if table_name is None:
        return set(REQUIRED_PRICING_COLS)
    if table_name in {TABLE_SALES, TABLE_SALES_ORDER_DETAIL}:
        return set(REQUIRED_PRICING_COLS)
    return set()


def _validate_required(schema, *, table_name: str | None = None) -> None:
    """
    Legacy name kept for compatibility with existing imports.
    """
    required = required_pricing_cols_for_table(table_name)
    validate_required_columns(schema, required, what="required pricing columns")


def _schema_dict_cols(
    schema,
    exclude: Optional[Iterable[str]] = None,
    *,
    table_name: str | None = None,
) -> List[str]:
    """
    Legacy name kept for compatibility with existing imports.

    Dictionary-encode only string/binary-like columns (except exclusions).
    """
    _validate_required(schema, table_name=table_name)

    exclude_set = set(DICT_EXCLUDE)
    if exclude:
        exclude_set |= set(exclude)

    return schema_dict_cols(schema, exclude=exclude_set)


# ===============================================================
# common/writers/parquet_merge.py (inlined, prefixed)
# ===============================================================


def _pm_open_parquet(path: str, pa, pq):
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


def _pm_schema_equals(schema_a, schema_b, *, check_metadata: bool) -> bool:
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


def _pm_build_canonical_schema(
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

    src0, r0 = _pm_open_parquet(parquet_files[0], pa, pq)
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
        src, r = _pm_open_parquet(path, pa, pq)
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


def _pm_restrict_schema_to_expected(schema, expected_cols: Sequence[str], pa, *, table_name: str = "table"):
    """Restrict and reorder schema to an expected set of columns."""
    fields = {f.name: f for f in schema}
    missing = [c for c in expected_cols if c not in fields]
    if missing:
        raise RuntimeError(
            f"[{table_name}] canonical schema missing expected columns {missing}. "
            "This usually means wrong chunk files were passed to the merge."
        )
    return pa.schema([fields[c] for c in expected_cols], metadata=getattr(schema, "metadata", None))


def _pm_nulls_of_type(pa, pc, n: int, typ):
    """Create a null array of length n with the requested Arrow type."""
    try:
        return pa.nulls(n, type=typ)
    except Exception:
        # Some complex types may not support nulls(type=...)
        return pc.cast(pa.nulls(n), typ, safe=False)


def _pm_project_table_to_schema(table, schema, *, cast_safe: bool, pa, pc):
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
            col = _pm_nulls_of_type(pa, pc, nrows, field.type)
        cols.append(col)

    return pa.Table.from_arrays(cols, schema=schema)


def _pm_row_group_num_rows(reader, rg_index: int) -> int:
    try:
        md = reader.metadata
        if md is not None:
            return int(md.row_group(rg_index).num_rows)
    except Exception:
        pass
    return -1


def _pm_read_row_group_projected(
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
        n = _pm_row_group_num_rows(reader, rg_index)
        if n < 0:
            # Fall back to reading the row group (pyarrow should preserve num_rows even with 0 cols)
            t0 = reader.read_row_group(rg_index, columns=[])
            return _pm_project_table_to_schema(t0, schema, cast_safe=cast_safe, pa=pa, pc=pc)

        # Build projected table directly using row-group size.
        cols = [_pm_nulls_of_type(pa, pc, n, field.type) for field in schema]
        return pa.Table.from_arrays(cols, schema=schema)

    t = reader.read_row_group(rg_index, columns=cols_to_read)
    return _pm_project_table_to_schema(t, schema, cast_safe=cast_safe, pa=pa, pc=pc)


def _merge_parquet_files_common(
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

    This is fact-agnostic. Fact-specific rules should be expressed via:
      - expected_cols / strict_expected / reject_extra_cols
      - required_cols
      - validate_schema_fn
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

    schema = canonical_schema
    if schema is None:
        schema = _pm_build_canonical_schema(files, schema_strategy=schema_strategy, pa=pa, pq=pq)

    if expected_cols:
        schema = _pm_restrict_schema_to_expected(schema, expected_cols, pa, table_name="merge")

    if validate_schema_fn is not None:
        validate_schema_fn(schema)

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
        writer_kwargs.pop("compression_level", None)
        writer = pq.ParquetWriter(merged_file_abs, schema, **writer_kwargs)

    try:
        expected_set = set(expected_cols) if expected_cols else None

        for path in files:
            src, reader = _pm_open_parquet(path, pa, pq)
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

                schema_exact = _pm_schema_equals(reader.schema_arrow, schema, check_metadata=True)
                schema_same_fields = _pm_schema_equals(reader.schema_arrow, schema, check_metadata=False)

                for rg in range(reader.num_row_groups):
                    if schema_exact and not expected_cols:
                        writer.write_table(reader.read_row_group(rg))
                        continue

                    if schema_same_fields:
                        t = reader.read_row_group(rg, columns=schema.names)
                        writer.write_table(_pm_project_table_to_schema(t, schema, cast_safe=cast_safe, pa=pa, pc=pc))
                        continue

                    t = _pm_read_row_group_projected(
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


# ===============================================================
# Sales-specific Parquet merge wrapper (public API)
# ===============================================================


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
    Kept for backward compatibility: existing Sales code imports this symbol.

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

    Sales policy:
      - expected column sets for SalesOrderHeader/Detail
      - schema validation via _validate_required
      - dict encoding column selection via _schema_dict_cols
      - required pricing columns per table

    Merge mechanics:
      - FD-safe streaming merge
      - row-group projection/casting
      - writer setup and output file creation
    """
    pa, _, pq = _arrow()

    files = [os.path.abspath(p) for p in parquet_files if p and os.path.exists(p)]
    if not files:
        skip("No parquet chunk files to merge")
        return None

    files.sort()

    canonical_schema = _pm_build_canonical_schema(files, schema_strategy=schema_strategy, pa=pa, pq=pq)

    expected_cols = _expected_cols_for_table(table_name)
    if expected_cols:
        canonical_schema = _pm_restrict_schema_to_expected(
            canonical_schema, expected_cols, pa, table_name=(table_name or "table")
        )

    _validate_required(canonical_schema, table_name=table_name)
    dict_cols = _schema_dict_cols(canonical_schema, table_name=table_name)

    required_cols = required_pricing_cols_for_table(table_name) or None

    return _merge_parquet_files_common(
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


# ===============================================================
# common/writers/delta.py (inlined)
# ===============================================================

ValidateSchemaFn = Callable[..., None]


@dataclass(frozen=True)
class DeltaWriteResult:
    part_files: List[str]
    delta_output_folder: str
    partition_cols: List[str]


def _delta_import_pyarrow():
    try:
        import pyarrow.parquet as pq  # type: ignore
        import pyarrow as pa  # type: ignore
        return pa, pq
    except Exception as e:
        raise RuntimeError("pyarrow is required for deltaparquet output") from e


def _delta_import_write_deltalake():
    try:
        from deltalake import write_deltalake  # type: ignore
        return write_deltalake
    except Exception as e1:
        try:
            from deltalake.writer import write_deltalake  # type: ignore
            return write_deltalake
        except Exception as e2:
            raise RuntimeError(
                "deltalake is required for Delta output, but import failed. "
                f"top-level error={e1!r}; fallback error={e2!r}"
            ) from e2


def _delta_resolve_part_files(
    *,
    parts_folder: Optional[str],
    parts: Optional[Iterable[Union[str, dict]]],
) -> List[str]:
    part_files: List[str] = []

    if parts is not None:
        for p in parts:
            if isinstance(p, dict):
                name = p.get("part")
                if not name:
                    continue
                if os.path.isabs(name):
                    pf = name
                else:
                    if not parts_folder:
                        raise ValueError("parts_folder is required when passing relative part names")
                    pf = os.path.join(parts_folder, name)
                if os.path.exists(pf) and pf.endswith(".parquet"):
                    part_files.append(os.path.abspath(pf))
            elif isinstance(p, str):
                pf = p
                if not os.path.isabs(pf) and parts_folder:
                    pf = os.path.join(parts_folder, pf)
                pf = os.path.abspath(pf)
                if os.path.exists(pf) and pf.endswith(".parquet"):
                    part_files.append(pf)
    else:
        if not parts_folder or not os.path.exists(parts_folder):
            raise FileNotFoundError(f"Parts folder not found: {parts_folder}")
        part_files = sorted(
            os.path.join(parts_folder, f)
            for f in os.listdir(parts_folder)
            if f.endswith(".parquet")
        )
        part_files = [os.path.abspath(p) for p in part_files]

    return part_files


def _delta_project_table_to_schema_best_effort(table, canonical_schema):
    """
    Best-effort projection/cast helper (kept intentionally permissive).

    - Keeps only canonical fields in canonical order; cast where possible.
    - Missing fields are filled with typed nulls.
    - Cast failures are tolerated (column kept as-is).
    """
    cols = []
    for field in canonical_schema:
        name = field.name
        if name in table.schema.names:
            col = table[name]
            if col.type != field.type:
                try:
                    col = col.cast(field.type)
                except Exception:
                    pass
            cols.append(col)
        else:
            import pyarrow as pa  # type: ignore
            cols.append(pa.nulls(table.num_rows, type=field.type))
    import pyarrow as pa  # type: ignore
    return pa.table(cols, names=[f.name for f in canonical_schema])


def write_delta_from_parquet_parts(
    *,
    parts_folder: Optional[str],
    delta_output_folder: str,
    partition_cols: Optional[Sequence[str]] = None,
    parts: Optional[Iterable[Union[str, dict]]] = None,
    table_name: Optional[str] = None,
    # Policy knobs
    validate_schema: Optional[ValidateSchemaFn] = None,
    on_missing_partition_cols: str = "drop",  # "drop" | "error"
    # Performance knobs
    sort_small_parts: bool = True,
    sort_row_limit: int = 2_000_000,
    # Housekeeping
    cleanup_parts_folder: bool = False,
    # Delta writer passthrough (best-effort; ignored if unsupported by installed deltalake)
    max_partitions: Optional[int] = None,
) -> DeltaWriteResult:
    """
    Generic delta writer for fact tables.

    - Reads Parquet parts (folder mode or explicit list)
    - Derives canonical schema from the first part (simple + stable)
    - Projects each part to canonical schema when needed
    - Writes as Delta using delta-rs (overwrite first part, append the rest)

    Policy is injected via:
      - validate_schema(canonical_schema, table_name=...)
      - on_missing_partition_cols: error vs drop
    """
    _, pq = _delta_import_pyarrow()
    write_deltalake = _delta_import_write_deltalake()

    parts_folder_abs = os.path.abspath(parts_folder) if parts_folder else None
    delta_output_abs = os.path.abspath(delta_output_folder)

    part_files = _delta_resolve_part_files(parts_folder=parts_folder_abs, parts=parts)
    if not part_files:
        raise RuntimeError("No delta part files found.")

    first_pf = pq.ParquetFile(part_files[0])
    canonical_schema = first_pf.schema_arrow

    if validate_schema is not None:
        validate_schema(canonical_schema, table_name=table_name)

    pcols = list(partition_cols or [])
    missing = [c for c in pcols if c not in canonical_schema.names]
    if missing:
        if on_missing_partition_cols == "error":
            raise RuntimeError(f"Partition columns missing from schema: {missing}")
        pcols = [c for c in pcols if c in canonical_schema.names]
        info(
            f"[DELTA] Partition cols missing for table={table_name or 'unknown'}; "
            f"dropping {missing}. Remaining={pcols}"
        )

    os.makedirs(delta_output_abs, exist_ok=True)

    suffix = f" table={table_name}" if table_name else ""
    info(f"[DELTA] Writing {len(part_files)} parts (Arrow -> Delta){suffix}")

    first = True
    for pf_path in part_files:
        pf = pq.ParquetFile(pf_path)

        num_rows = 0
        try:
            if pf.metadata is not None:
                num_rows = int(pf.metadata.num_rows)
        except Exception:
            num_rows = 0

        table = pq.read_table(pf_path)
        if table.schema != canonical_schema:
            table = _delta_project_table_to_schema_best_effort(table, canonical_schema)

        if pcols and sort_small_parts and num_rows and num_rows <= int(sort_row_limit):
            try:
                sort_keys = [(c, "ascending") for c in pcols]
                table = table.sort_by(sort_keys)
            except Exception:
                pass

        kwargs = dict(
            mode="overwrite" if first else "append",
            partition_by=pcols,
        )
        if max_partitions is not None:
            kwargs["max_partitions"] = int(max_partitions)

        try:
            write_deltalake(delta_output_abs, table, **kwargs)
        except TypeError:
            kwargs.pop("max_partitions", None)
            write_deltalake(delta_output_abs, table, **kwargs)

        first = False

    if cleanup_parts_folder and parts is None and parts_folder_abs:
        try:
            shutil.rmtree(parts_folder_abs, ignore_errors=True)
        except Exception:
            pass

    return DeltaWriteResult(
        part_files=part_files,
        delta_output_folder=delta_output_abs,
        partition_cols=pcols,
    )


# ===============================================================
# Sales-specific delta wrapper (public API)
# ===============================================================


def write_delta_partitioned(
    parts_folder: str,
    delta_output_folder: str,
    partition_cols: Optional[List[str]] = None,
    parts: Optional[Iterable[Union[str, dict]]] = None,
    *,
    sort_small_parts: bool = True,
    sort_row_limit: int = 2_000_000,
    table_name: str | None = None,
):
    # Preserve current behavior:
    # - if table_name is None: partition cols missing => error
    # - else: drop missing partition cols and continue
    on_missing = "error" if table_name is None else "drop"

    write_delta_from_parquet_parts(
        parts_folder=parts_folder,
        delta_output_folder=delta_output_folder,
        partition_cols=partition_cols,
        parts=parts,
        table_name=table_name,
        validate_schema=_validate_required,  # Sales policy
        on_missing_partition_cols=on_missing,  # Sales policy
        sort_small_parts=sort_small_parts,
        sort_row_limit=sort_row_limit,
        cleanup_parts_folder=(parts is None),  # matches current folder-mode cleanup
    )


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
