from __future__ import annotations

from typing import Iterable, List, Optional, Union

from .encoding import _validate_required
from src.facts.common.writers.delta import write_delta_from_parquet_parts


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
        validate_schema=_validate_required,      # Sales policy
        on_missing_partition_cols=on_missing,    # Sales policy
        sort_small_parts=sort_small_parts,
        sort_row_limit=sort_row_limit,
        cleanup_parts_folder=(parts is None),    # matches current folder-mode cleanup
    )
