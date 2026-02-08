from __future__ import annotations

import os
from typing import Any, Dict, Union

import pyarrow as pa

from ..sales_logic import chunk_builder
from ..sales_logic.globals import State
from ..output_paths import (
    TABLE_SALES,
    TABLE_SALES_ORDER_DETAIL,
    TABLE_SALES_ORDER_HEADER,
)
from .chunk_io import _write_csv, _write_parquet_table
from .header_builder import build_header_from_detail


def _int_or(v: Any, default: int) -> int:
    try:
        if v is None or v == "":
            return int(default)
        return int(v)
    except Exception:
        return int(default)


def _drop_order_cols_for_sales(table: pa.Table) -> pa.Table:
    # keep in sync with init.py's order_fields
    drop = {"SalesOrderNumber", "SalesOrderLineNumber"}
    keep = [n for n in table.schema.names if n not in drop]
    return table.select(keep)


def _write_table(table_name: str, idx: int, table: pa.Table) -> Union[str, Dict[str, Any]]:
    """
    Writes one logical table for this chunk.
    Returns:
      - csv/parquet: full path (str)
      - deltaparquet: {"part": basename, "rows": n}
    """
    if State.file_format == "deltaparquet":
        path = State.output_paths.delta_part_path(table_name, int(idx))
        _write_parquet_table(table, path, table_name=table_name)
        return {"part": os.path.basename(path), "rows": table.num_rows}

    if State.file_format == "csv":
        path = State.output_paths.chunk_path(table_name, int(idx), "csv")
        _write_csv(table, path, table_name=table_name)
        return path

    # parquet (default)
    path = State.output_paths.chunk_path(table_name, int(idx), "parquet")
    _write_parquet_table(table, path, table_name=table_name)
    return path


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

        # Build one chunk (init.py ensures order cols exist when needed)
        detail_table = chunk_builder.build_chunk_table(
            int(batch_size),
            int(chunk_seed),
            no_discount_key=State.no_discount_key,
        )

        if not isinstance(detail_table, pa.Table):
            raise TypeError("chunk_builder must return pyarrow.Table")

        mode = getattr(State, "sales_output", "sales")

        # Safeguard: normalized modes require order keys for detail/header
        if mode in {"sales_order", "both"}:
            required = {"SalesOrderNumber", "SalesOrderLineNumber"}
            missing = required.difference(detail_table.schema.names)
            if missing:
                raise RuntimeError(
                    f"sales_output={mode} requires order columns, missing: {sorted(missing)}"
                )

        # Single-table legacy path (keeps return-shape stable)
        if mode == "sales":
            sales_table = detail_table
            if getattr(State, "skip_order_cols_requested", False):
                sales_table = _drop_order_cols_for_sales(sales_table)

            results.append(_write_table(TABLE_SALES, int(idx), sales_table))
            continue

        # Multi-table path: return dict(table -> write_result)
        out: Dict[str, Any] = {}

        if mode == "both":
            sales_table = detail_table
            if getattr(State, "skip_order_cols_requested", False):
                sales_table = _drop_order_cols_for_sales(sales_table)
            out[TABLE_SALES] = _write_table(TABLE_SALES, int(idx), sales_table)

        # Detail always written for normalized modes
        out[TABLE_SALES_ORDER_DETAIL] = _write_table(
            TABLE_SALES_ORDER_DETAIL, int(idx), detail_table
        )

        # Header derived from detail
        header_table = build_header_from_detail(detail_table)
        out[TABLE_SALES_ORDER_HEADER] = _write_table(
            TABLE_SALES_ORDER_HEADER, int(idx), header_table
        )

        results.append(out)

    return results[0] if single else results
