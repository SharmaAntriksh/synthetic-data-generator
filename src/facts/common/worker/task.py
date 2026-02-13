from __future__ import annotations

import os
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple, Union

import pyarrow as pa

from .init import int_or


Task = Tuple[int, int, Any]  # (idx, batch_size, seed)
TaskArgs = Union[Task, Sequence[Task]]


def normalize_tasks(args: TaskArgs) -> Tuple[List[Task], bool]:
    """
    Supports:
      - single task: (idx, batch_size, seed)
      - batched tasks: [(idx, batch_size, seed), ...]
    Returns: (tasks, single)
    """
    if isinstance(args, tuple):
        # single task
        if len(args) != 3:
            raise ValueError(f"Task tuple must be (idx,batch_size,seed), got len={len(args)}")
        return [args], True

    tasks = list(args)
    return tasks, False


def derive_chunk_seed(seed: Any, idx: int, *, stride: int = 10_000) -> int:
    """
    Deterministic per-chunk seed derivation.
    """
    base_seed = int_or(seed, 0)
    return int(base_seed) + int(idx) * int(stride)


def write_table_by_format(
    *,
    file_format: str,
    output_paths: Any,
    table_name: str,
    idx: int,
    table: pa.Table,
    write_csv_fn: Callable[[pa.Table, str], None],
    write_parquet_fn: Callable[[pa.Table, str], None],
) -> Union[str, Dict[str, Any]]:
    """
    Generic writer shim used by worker tasks.

    Requires output_paths to provide:
      - delta_part_path(table_name, idx) -> str   (deltaparquet only)
      - chunk_path(table_name, idx, ext) -> str   (csv/parquet)

    Returns:
      - csv/parquet: full path (str)
      - deltaparquet: {"part": basename, "rows": n}
    """
    ff = (file_format or "").strip().lower()
    if ff == "deltaparquet":
        if not hasattr(output_paths, "delta_part_path"):
            raise RuntimeError("output_paths must implement delta_part_path() for deltaparquet")
        path = output_paths.delta_part_path(table_name, int(idx))
        write_parquet_fn(table, path)
        return {"part": os.path.basename(path), "rows": table.num_rows}

    if ff == "csv":
        if not hasattr(output_paths, "chunk_path"):
            raise RuntimeError("output_paths must implement chunk_path() for csv")
        path = output_paths.chunk_path(table_name, int(idx), "csv")
        write_csv_fn(table, path)
        return path

    # parquet (default)
    if not hasattr(output_paths, "chunk_path"):
        raise RuntimeError("output_paths must implement chunk_path() for parquet")
    path = output_paths.chunk_path(table_name, int(idx), "parquet")
    write_parquet_fn(table, path)
    return path


__all__ = ["Task", "TaskArgs", "normalize_tasks", "derive_chunk_seed", "write_table_by_format"]
