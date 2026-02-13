from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Union

# Policy hook: validate canonical schema (fact/table-specific). Keep optional.
ValidateSchemaFn = Callable[..., None]


@dataclass(frozen=True)
class DeltaWriteResult:
    part_files: List[str]
    delta_output_folder: str
    partition_cols: List[str]


def _import_pyarrow():
    try:
        import pyarrow.parquet as pq  # type: ignore
        import pyarrow as pa  # type: ignore
        return pa, pq
    except Exception as e:
        raise RuntimeError("pyarrow is required for deltaparquet output") from e


def _import_write_deltalake():
    try:
        # Preferred in many delta-rs versions
        from deltalake import write_deltalake  # type: ignore
        return write_deltalake
    except Exception as e1:
        try:
            # Fallback for versions where top-level export differs
            from deltalake.writer import write_deltalake  # type: ignore
            return write_deltalake
        except Exception as e2:
            raise RuntimeError(
                "deltalake is required for Delta output, but import failed. "
                f"top-level error={e1!r}; fallback error={e2!r}"
            ) from e2


def _resolve_part_files(
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


def _project_table_to_schema(table, canonical_schema):
    """
    Minimal projection/cast helper.
    For richer semantics (dict cols, metadata), you can later centralize this into
    common/writers/projection.py and reuse from parquet_merge + delta.
    """
    # Keep only canonical fields in canonical order; cast where needed.
    cols = []
    for field in canonical_schema:
        name = field.name
        if name in table.schema.names:
            col = table[name]
            # Cast only if type differs (safe for most primitives)
            if col.type != field.type:
                try:
                    col = col.cast(field.type)
                except Exception:
                    # Best-effort: if cast fails, keep original (policy can enforce stricter validation)
                    pass
            cols.append(col)
        else:
            # Missing field: add nulls
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
    from src.utils.logging_utils import info

    _, pq = _import_pyarrow()
    write_deltalake = _import_write_deltalake()

    parts_folder_abs = os.path.abspath(parts_folder) if parts_folder else None
    delta_output_abs = os.path.abspath(delta_output_folder)

    part_files = _resolve_part_files(parts_folder=parts_folder_abs, parts=parts)
    if not part_files:
        raise RuntimeError("No delta part files found.")

    # Canonical schema: first file
    first_pf = pq.ParquetFile(part_files[0])
    canonical_schema = first_pf.schema_arrow

    if validate_schema is not None:
        # Let fact/table enforce requirements (pricing cols, strict schema, etc.)
        validate_schema(canonical_schema, table_name=table_name)

    # Partition cols handling
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
            table = _project_table_to_schema(table, canonical_schema)

        if pcols and sort_small_parts and num_rows and num_rows <= int(sort_row_limit):
            try:
                sort_keys = [(c, "ascending") for c in pcols]
                table = table.sort_by(sort_keys)
            except Exception:
                pass

        # Best-effort support for max_partitions (varies by deltalake version)
        kwargs = dict(
            mode="overwrite" if first else "append",
            partition_by=pcols,
        )
        if max_partitions is not None:
            kwargs["max_partitions"] = int(max_partitions)

        try:
            write_deltalake(delta_output_abs, table, **kwargs)
        except TypeError:
            # Installed deltalake doesn't support some kwarg(s) (e.g., max_partitions)
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
