# Sales fact writer (Parquet / Delta)
# Pure I/O layer: does NOT interpret or modify business logic

from __future__ import annotations

import os
import shutil
from typing import Iterable, List, Optional, Union


# Columns we never dictionary-encode
DICT_EXCLUDE = {"SalesOrderNumber", "CustomerKey"}

# Columns that must always exist in Sales
REQUIRED_PRICING_COLS = {
    "UnitPrice",
    "NetPrice",
    "UnitCost",
    "DiscountAmount",
}


# ----------------------------------------------------------------------
# Internal: lazy Arrow import (keeps CSV-only runs lighter)
# ----------------------------------------------------------------------
def _arrow():
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
        return pa, pc, pq
    except Exception as e:
        raise RuntimeError("pyarrow is required for Parquet merge/Delta writes") from e


def _ensure_dir_for_file(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def _schema_dict_cols(schema) -> List[str]:
    """
    Dictionary-encode only string/binary-like columns (except exclusions).
    """
    pa, _, _ = _arrow()
    out: List[str] = []
    for f in schema:
        if f.name in DICT_EXCLUDE:
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


def _validate_required(schema) -> None:
    names = set(schema.names)
    missing = REQUIRED_PRICING_COLS - names
    if missing:
        raise RuntimeError(f"Missing required pricing columns: {sorted(missing)}")


def _project_table_to_schema(table, schema):
    """
    Project/align a table to the canonical schema:
      - reorder columns
      - add missing columns as typed nulls
      - cast columns to canonical types when safe
    """
    pa, pc, _ = _arrow()

    # Fast path
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
                col = pc.cast(col, field.type, safe=False)
            except Exception as ex:
                raise RuntimeError(f"Failed to cast column '{name}' from {col.type} to {field.type}: {ex}") from ex
        arrays.append(col)

    return pa.Table.from_arrays(arrays, schema=schema)


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


# ----------------------------------------------------------------------
# PARQUET MERGER
# ----------------------------------------------------------------------
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


# ----------------------------------------------------------------------
# DELTA-PARQUET PARTITION WRITER
# ----------------------------------------------------------------------
def write_delta_partitioned(
    parts_folder: str,
    delta_output_folder: str,
    partition_cols: Optional[List[str]] = None,
    parts: Optional[Iterable[Union[str, dict]]] = None,
    *,
    sort_small_parts: bool = True,
    sort_row_limit: int = 2_000_000,
):
    """
    Convert worker parquet parts into a partitioned Delta table.

    Modes:
      1) Folder mode (default): read all *.parquet under parts_folder
      2) Explicit parts mode: pass `parts` as iterable of either:
           - dicts like {"part": "delta_part_0001.parquet", "rows": 123}
           - absolute/relative file paths

    Notes:
    - Validates required pricing columns and partition columns
    - Projects schema differences to the first part's schema
    - Optionally sorts *small* parts by partition columns for stable output
      (sorting large parts is skipped for performance).
    """
    from src.utils.logging_utils import info

    pa, _, pq = _arrow()

    parts_folder = os.path.abspath(parts_folder) if parts_folder else None
    delta_output_folder = os.path.abspath(delta_output_folder)

    if partition_cols is None:
        partition_cols = []

    # Resolve part_files
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

    if not part_files:
        raise RuntimeError("No delta part files found.")

    # Validate schema once (first file)
    first_pf = pq.ParquetFile(part_files[0])
    canonical_schema = first_pf.schema_arrow
    _validate_required(canonical_schema)

    missing_part_cols = [c for c in partition_cols if c not in canonical_schema.names]
    if missing_part_cols:
        raise RuntimeError(f"Partition columns missing from schema: {missing_part_cols}")

    os.makedirs(delta_output_folder, exist_ok=True)

    try:
        # Preferred in many delta-rs versions
        from deltalake import write_deltalake  # type: ignore
    except Exception as e1:
        try:
            # Fallback for versions where top-level export differs
            from deltalake.writer import write_deltalake  # type: ignore
        except Exception as e2:
            raise RuntimeError(
                "deltalake is required for Delta output, but import failed. "
                f"top-level error={e1!r}; fallback error={e2!r}"
            ) from e2

    info(f"[DELTA] Writing {len(part_files)} parts (Arrow -> Delta)")

    first = True
    for pf_path in part_files:
        pf = pq.ParquetFile(pf_path)

        # Decide whether to sort (only for smaller parts)
        num_rows = 0
        try:
            if pf.metadata is not None:
                num_rows = int(pf.metadata.num_rows)
        except Exception:
            num_rows = 0

        # Read full table (safe + compatible); project if schema differs
        table = pq.read_table(pf_path)
        if table.schema != canonical_schema:
            table = _project_table_to_schema(table, canonical_schema)

        if partition_cols and sort_small_parts and num_rows and num_rows <= int(sort_row_limit):
            try:
                sort_keys = [(c, "ascending") for c in partition_cols]
                table = table.sort_by(sort_keys)
            except Exception:
                # Sorting is an optimization only; do not fail the write
                pass

        write_deltalake(
            delta_output_folder,
            table,
            mode="overwrite" if first else "append",
            partition_by=partition_cols,
        )
        first = False

    # Cleanup only the parts folder that was used (folder mode)
    if parts is None and parts_folder:
        try:
            shutil.rmtree(parts_folder, ignore_errors=True)
        except Exception:
            pass

    return
