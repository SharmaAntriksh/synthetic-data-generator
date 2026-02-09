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
    """
    Preserve existing 'Sales' behavior:
    if user requested skip_order_cols, drop these from Sales only.
    """
    drop = {"SalesOrderNumber", "SalesOrderLineNumber"}  # keep in sync with init.py order_fields
    keep = [n for n in table.schema.names if n not in drop]
    return table.select(keep)


_DELTA_PART_COLS = {"Year", "Month"}


def _project_for_table(table_name: str, table: pa.Table) -> pa.Table:
    """
    Project an in-memory table to the expected logical columns for that output table.

    Note:
      - For deltaparquet schemas, we do NOT select Year/Month here; chunk_io may add them.
    """
    expected = State.schema_by_table[table_name]
    cols = [n for n in expected.names if n not in _DELTA_PART_COLS]

    got = set(table.schema.names)
    exp = set(cols)

    missing = sorted(exp - got)
    extra = sorted(got - exp)

    if missing:
        raise RuntimeError(
            f"Cannot project {table_name}: missing columns {missing}. "
            f"Available columns: {table.schema.names}"
        )

    # NOTE: extra columns are fine here; projection will drop them intentionally.
    return table.select(cols)


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

        # Build one chunk (full "sales-shaped" table; includes order cols when enabled by init.py)
        detail_table = chunk_builder.build_chunk_table(
            int(batch_size),
            int(chunk_seed),
            no_discount_key=State.no_discount_key,
        )
        if not isinstance(detail_table, pa.Table):
            raise TypeError("chunk_builder must return pyarrow.Table")

        mode = str(getattr(State, "sales_output", "sales") or "sales").strip().lower()

        # If producing SalesOrder* tables, order cols must exist.
        if mode in {"sales_order", "both"}:
            required_order = {"SalesOrderNumber", "SalesOrderLineNumber"}
            missing = sorted(required_order.difference(detail_table.schema.names))
            if missing:
                raise RuntimeError(
                    f"sales_output={mode} requires order columns, missing: {missing}"
                )

            # Header builder needs these to exist on the raw detail table
            header_needs = {"SalesOrderNumber", "CustomerKey", "OrderDate", "IsOrderDelayed"}
            missing_h = sorted(header_needs.difference(detail_table.schema.names))
            if missing_h:
                raise RuntimeError(
                    f"Header build requires columns missing from detail: {missing_h}"
                )

        # Sales-only path (MUST remain unchanged)
        if mode == "sales":
            sales_table = detail_table
            if bool(getattr(State, "skip_order_cols_requested", False)):
                sales_table = _drop_order_cols_for_sales(sales_table)

            results.append(_write_table(TABLE_SALES, int(idx), sales_table))
            continue

        out: Dict[str, Any] = {}

        # Both mode: also write Sales table (unchanged behavior)
        if mode == "both":
            sales_table = detail_table
            if bool(getattr(State, "skip_order_cols_requested", False)):
                sales_table = _drop_order_cols_for_sales(sales_table)
            out[TABLE_SALES] = _write_table(TABLE_SALES, int(idx), sales_table)

        # Build header from the FULL raw detail table
        header_table = build_header_from_detail(detail_table)

        # Project to configured schemas (slim/fat is controlled by init.py schemas)
        detail_out = _project_for_table(TABLE_SALES_ORDER_DETAIL, detail_table)
        header_out = _project_for_table(TABLE_SALES_ORDER_HEADER, header_table)

        out[TABLE_SALES_ORDER_DETAIL] = _write_table(TABLE_SALES_ORDER_DETAIL, int(idx), detail_out)
        out[TABLE_SALES_ORDER_HEADER] = _write_table(TABLE_SALES_ORDER_HEADER, int(idx), header_out)

        results.append(out)

    return results[0] if single else results
