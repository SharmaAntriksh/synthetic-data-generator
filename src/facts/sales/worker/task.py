from __future__ import annotations

import os
from typing import Any

import pyarrow as pa

from ..sales_logic import chunk_builder
from ..sales_logic.globals import State
from .chunk_io import _write_csv, _write_parquet_table


def _int_or(v: Any, default: int) -> int:
    try:
        if v is None or v == "":
            return int(default)
        return int(v)
    except Exception:
        return int(default)


def _worker_task(args):
    """
    Supports:
      - single task: (idx, batch_size, seed)
      - batched tasks: [(idx, batch_size, seed), ...]
    """
    if isinstance(args, tuple):
        tasks = [args]
        single = True
    else:
        tasks = list(args)
        single = False

    results = []

    for idx, batch_size, seed in tasks:
        base_seed = _int_or(seed, 0)
        chunk_seed = base_seed + int(idx) * 10_000

        table = chunk_builder.build_chunk_table(
            int(batch_size),
            int(chunk_seed),
            no_discount_key=State.no_discount_key,
        )

        if not isinstance(table, pa.Table):
            raise TypeError("chunk_builder must return pyarrow.Table")

        # DELTA (write tmp parquet parts; merge later in sales_writer.write_delta_partitioned)
        if State.file_format == "deltaparquet":
            name = f"delta_part_{idx:04d}.parquet"
            path = os.path.join(State.delta_output_folder, "_tmp_parts", name)
            _write_parquet_table(table, path)
            rows = table.num_rows
            results.append({"part": name, "rows": rows})
            continue

        # CSV
        if State.file_format == "csv":
            path = os.path.join(State.out_folder, f"sales_chunk{idx:04d}.csv")
            _write_csv(table, path)
            results.append(path)
            continue

        # PARQUET
        path = os.path.join(State.out_folder, f"sales_chunk{idx:04d}.parquet")
        _write_parquet_table(table, path)
        results.append(path)

    return results[0] if single else results
