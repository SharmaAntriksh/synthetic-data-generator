from __future__ import annotations

import os
import shutil
from typing import Iterable, List, Optional, Union

from .encoding import _validate_required
from .projection import _project_table_to_schema
from .utils import _arrow


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
