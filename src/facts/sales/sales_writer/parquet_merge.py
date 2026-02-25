from __future__ import annotations

import os
from typing import Any, Callable, Iterable, Optional, Sequence, Set, Union

from .utils import _arrow, _ensure_dir_for_file, done, info, skip
from .projection import _project_table_to_schema
from .encoding import _schema_dict_cols, _validate_required, required_pricing_cols_for_table

PathLike = Union[str, os.PathLike]


def _pm_open_parquet(path: str, pa, pq):
    try:
        source = pa.memory_map(path, "r")
    except Exception:
        source = pa.OSFile(path, "rb")
    reader = pq.ParquetFile(source)
    try:
        setattr(reader, "path", path)
    except Exception:
        pass
    return source, reader


def _pm_schema_equals(schema_a, schema_b, *, check_metadata: bool) -> bool:
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
    fields = {f.name: f for f in schema}
    missing = [c for c in expected_cols if c not in fields]
    if missing:
        raise RuntimeError(
            f"[{table_name}] canonical schema missing expected columns {missing}. "
            "This usually means wrong chunk files were passed to the merge."
        )
    return pa.schema([fields[c] for c in expected_cols], metadata=getattr(schema, "metadata", None))


def _pm_nulls_of_type(pa, pc, n: int, typ):
    try:
        return pa.nulls(n, type=typ)
    except Exception:
        return pc.cast(pa.nulls(n), typ, safe=False)


def _pm_project_table_to_schema(table, schema, *, cast_safe: bool, pa, pc):
    cols = []
    names = set(table.schema.names)
    nrows = table.num_rows

    for field in schema:
        if field.name in names:
            col = table[field.name]
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
            t0 = reader.read_row_group(rg_index, columns=[])
            return _pm_project_table_to_schema(t0, schema, cast_safe=cast_safe, pa=pa, pc=pc)

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
    log: bool = True,
) -> Optional[str]:
    pa, pc, pq = _arrow()

    files: list[str] = []
    for f in parquet_files:
        if not f:
            continue
        fp = os.path.abspath(os.fspath(f))
        if os.path.exists(fp):
            files.append(fp)

    if not files:
        if log:
            skip(f"{log_prefix}No parquet chunk files to merge".strip())
        return None

    if sort_files:
        files.sort()

    merged_file_abs = os.path.abspath(os.fspath(merged_file))
    _ensure_dir_for_file(merged_file_abs)

    if log:
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

    if log:
        done(f"{log_prefix}Merged chunks: {os.path.basename(merged_file_abs)}".strip())
    return merged_file_abs


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
    Back-compat symbol: sales code may import this.

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
    log: bool = True,
) -> Optional[str]:
    pa, _, pq = _arrow()

    files = [os.path.abspath(p) for p in parquet_files if p and os.path.exists(p)]
    if not files:
        if log:
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
        log=log,
    )