from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Union

from .utils import _arrow, info, warn
from .encoding import _validate_required
from .projection import project_table_to_schema

ValidateSchemaFn = Callable[..., None]


@dataclass(frozen=True)
class DeltaWriteResult:
    part_files: List[str]
    delta_output_folder: str
    partition_cols: List[str]


def _delta_import_write_deltalake():
    try:
        from deltalake import write_deltalake  # type: ignore
        return write_deltalake
    except ImportError as e1:
        try:
            from deltalake.writer import write_deltalake  # type: ignore
            return write_deltalake
        except ImportError as e2:
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


def _schema_fields_equal(schema_a, schema_b) -> bool:
    """
    Compare two Arrow schemas by field names and types only,
    ignoring metadata (e.g. pandas metadata).  This matches
    the comparison semantics used in parquet_merge.py.
    """
    if len(schema_a) != len(schema_b):
        return False
    for fa, fb in zip(schema_a, schema_b):
        if fa.name != fb.name or not fa.type.equals(fb.type):
            return False
    return True


def _delta_write_table(
    write_deltalake,
    delta_output_abs: str,
    table,
    *,
    first: bool,
    pcols: List[str],
    max_partitions: Optional[int],
):
    """Write a single table to delta, handling unsupported kwargs gracefully."""
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
    _, _, pq = _arrow()
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

        # Determine total rows from metadata (single open, fixes double-read)
        num_rows = 0
        try:
            if pf.metadata is not None:
                num_rows = int(pf.metadata.num_rows)
        except (AttributeError, TypeError, ValueError):
            num_rows = 0

        needs_sort = pcols and sort_small_parts and num_rows and num_rows <= int(sort_row_limit)
        needs_projection = not _schema_fields_equal(pf.schema_arrow, canonical_schema)

        if needs_sort:
            # Must read the full table to sort — read from the already-opened ParquetFile
            table = pf.read()
            if needs_projection:
                table = project_table_to_schema(
                    table, canonical_schema, cast_safe=False, on_cast_error="warn"
                )
            try:
                sort_keys = [(c, "ascending") for c in pcols]
                table = table.sort_by(sort_keys)
            except Exception as ex:
                warn(f"[DELTA] Sort failed for {os.path.basename(pf_path)}: {ex}")

            _delta_write_table(
                write_deltalake, delta_output_abs, table,
                first=first, pcols=pcols, max_partitions=max_partitions,
            )
        else:
            # Stream row-group-by-row-group to limit memory usage
            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(rg)
                if needs_projection:
                    table = project_table_to_schema(
                        table, canonical_schema, cast_safe=False, on_cast_error="warn"
                    )
                _delta_write_table(
                    write_deltalake, delta_output_abs, table,
                    first=first, pcols=pcols, max_partitions=max_partitions,
                )
                # Only the very first write across all files is "overwrite"
                first = False

        first = False

    # Verify delta output before cleanup
    if cleanup_parts_folder and parts is None and parts_folder_abs:
        delta_log = os.path.join(delta_output_abs, "_delta_log")
        if not os.path.isdir(delta_log):
            warn(
                f"[DELTA] Delta log not found at {delta_log} after write; "
                "skipping parts folder cleanup to avoid data loss."
            )
        else:
            try:
                shutil.rmtree(parts_folder_abs, ignore_errors=True)
            except OSError as ex:
                warn(f"[DELTA] Failed to clean up parts folder {parts_folder_abs}: {ex}")

    return DeltaWriteResult(
        part_files=part_files,
        delta_output_folder=delta_output_abs,
        partition_cols=pcols,
    )


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
        validate_schema=_validate_required,          # Sales policy
        on_missing_partition_cols=on_missing,        # Sales policy
        sort_small_parts=sort_small_parts,
        sort_row_limit=sort_row_limit,
        cleanup_parts_folder=(parts is None),        # matches current folder-mode cleanup
    )


# ---------------------------------------------------------------------------
# Backward-compat aliases for symbols that were refactored away.
# They were private (underscore-prefixed) and only called internally, but
# external test/plugin code may have imported them.
# ---------------------------------------------------------------------------

def _delta_import_pyarrow():
    """Deprecated: use ``from .utils import _arrow`` instead."""
    pa, _, pq = _arrow()
    return pa, pq


def _delta_project_table_to_schema_best_effort(table, canonical_schema):
    """Deprecated: use ``project_table_to_schema(..., on_cast_error='warn')``."""
    return project_table_to_schema(
        table, canonical_schema, cast_safe=False, on_cast_error="warn"
    )
