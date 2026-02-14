from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Union

import pyarrow as pa

from src.facts.common.worker.task import (
    derive_chunk_seed,
    normalize_tasks,
    write_table_by_format,
)

from ..sales_logic import chunk_builder
from ..sales_logic.globals import State
from ..output_paths import (
    TABLE_SALES,
    TABLE_SALES_ORDER_DETAIL,
    TABLE_SALES_ORDER_HEADER,
)
from .chunk_io import _write_csv, _write_parquet_table
from .header_builder import build_header_from_detail


# Optional (new) table + builder. We keep imports soft so the file can land before
# you add SalesReturn everywhere else.
try:
    from ..output_paths import TABLE_SALES_RETURN  # type: ignore
except Exception:
    TABLE_SALES_RETURN = None  # type: ignore

try:
    from .returns_builder import ReturnsConfig, build_sales_returns_from_detail  # type: ignore
except Exception:
    ReturnsConfig = None  # type: ignore
    build_sales_returns_from_detail = None  # type: ignore


# Keep in sync with init.py order_fields
_DROP_ORDER_COLS = {"SalesOrderNumber", "SalesOrderLineNumber"}


def _drop_order_cols_for_sales(table: pa.Table) -> pa.Table:
    """
    Preserve existing 'Sales' behavior:
    if user requested skip_order_cols, drop these from Sales only.
    """
    keep = [n for n in table.schema.names if n not in _DROP_ORDER_COLS]
    return table.select(keep)


def _partition_cols() -> set[str]:
    """
    Partition cols are a cross-cutting policy (primarily for deltaparquet).
    Use State.partition_cols when available; otherwise preserve the legacy default.
    """
    cols = getattr(State, "partition_cols", None)
    if isinstance(cols, (list, tuple)) and cols:
        return {str(c) for c in cols}
    return {"Year", "Month"}


def _project_for_table(table_name: str, table: pa.Table) -> pa.Table:
    """
    Project an in-memory table to the expected logical columns for that output table.

    Note:
      - For partitioned/delta schemas, we do NOT select partition cols here; chunk_io may add them.
    """
    expected = State.schema_by_table[table_name]
    part_cols = _partition_cols()
    cols = [n for n in expected.names if n not in part_cols]

    got = set(table.schema.names)
    exp = set(cols)

    missing = sorted(exp - got)
    if missing:
        raise RuntimeError(
            f"Cannot project {table_name}: missing columns {missing}. "
            f"Available columns: {table.schema.names}"
        )

    return table.select(cols)


def _write_table(table_name: str, idx: int, table: pa.Table) -> Union[str, Dict[str, Any]]:
    """
    Sales wrapper over the common format switch.
    Keeps table_name-aware writes (chunk_io handles Parquet options).
    """
    return write_table_by_format(
        file_format=State.file_format,
        output_paths=State.output_paths,
        table_name=table_name,
        idx=int(idx),
        table=table,
        write_csv_fn=lambda t, p: _write_csv(t, p, table_name=table_name),
        write_parquet_fn=lambda t, p: _write_parquet_table(t, p, table_name=table_name),
    )


def _mode() -> str:
    return str(getattr(State, "sales_output", "sales") or "sales").strip().lower()


def _require_cols(table: pa.Table, cols: Sequence[str], *, ctx: str) -> None:
    missing = sorted(set(cols).difference(table.schema.names))
    if missing:
        raise RuntimeError(f"{ctx} missing columns: {missing}")


def _as_list(v: Any, default: Sequence[Any]) -> list[Any]:
    """
    Convert possibly-None / scalar / list-like into a list.
    Avoids list(None) which causes: 'NoneType' object is not iterable.

    Notes:
      - Strings/bytes are treated as scalars (not iterated).
      - numpy/pandas objects with .tolist() are supported.
    """
    if v is None:
        return list(default)
    if isinstance(v, list):
        return v
    if isinstance(v, tuple):
        return list(v)
    if isinstance(v, (str, bytes)):
        return [v]
    tolist = getattr(v, "tolist", None)
    if callable(tolist):
        try:
            out = tolist()
            if isinstance(out, list):
                return out
        except Exception:
            pass
    return [v]


def _maybe_build_returns(source_table: pa.Table, *, chunk_seed: int) -> Optional[pa.Table]:
    """
    Build returns table from a line-grain source_table if enabled.

    Requires:
      - TABLE_SALES_RETURN constant exists
      - returns_builder module exists
      - State.returns_enabled == True
      - source_table includes SalesOrderNumber + SalesOrderLineNumber (and required pricing cols)

    Supported modes:
      - sales: uses the raw chunk table (must still contain order/line identifiers)
      - sales_order / both: uses SalesOrderDetail-shaped table
    """
    if not bool(getattr(State, "returns_enabled", False)):
        return None

    if TABLE_SALES_RETURN is None:
        raise RuntimeError(
            "returns_enabled=True but TABLE_SALES_RETURN is not defined in output_paths.py"
        )

    if build_sales_returns_from_detail is None or ReturnsConfig is None:
        raise RuntimeError(
            "returns_enabled=True but returns_builder.py is not available "
            "(expected .returns_builder ReturnsConfig/build_sales_returns_from_detail)"
        )

    mode = _mode()
    if mode not in {"sales", "sales_order", "both"}:
        return None

    # Minimum columns needed by the returns builder (keep aligned with returns_builder.py).
    _require_cols(
        source_table,
        [
            "SalesOrderNumber",
            "SalesOrderLineNumber",
            "CustomerKey",
            "ProductKey",
            "StoreKey",
            "PromotionKey",
            "CurrencyKey",
            "DeliveryDate",
            "Quantity",
            "UnitPrice",
            "DiscountAmount",
            "NetPrice",
            "UnitCost",
        ],
        ctx="SalesReturn build requires",
    )

    # None-safe knobs (avoid float(None), int(None), list(None))
    rate_raw = getattr(State, "returns_rate", None)
    max_lag_raw = getattr(State, "returns_max_lag_days", None)
    reason_keys_raw = getattr(State, "returns_reason_keys", None)
    reason_probs_raw = getattr(State, "returns_reason_probs", None)

    cfg = ReturnsConfig(
        enabled=True,
        return_rate=float(rate_raw) if rate_raw is not None else 0.0,
        max_lag_days=int(max_lag_raw) if max_lag_raw is not None else 60,
        reason_keys=_as_list(reason_keys_raw, default=[1]),
        reason_probs=_as_list(reason_probs_raw, default=[1.0]),
    )

    # Make returns deterministic but distinct from the main chunk generation.
    returns_seed = int(chunk_seed) ^ 0x5A5A_1234

    return build_sales_returns_from_detail(
        source_table,
        chunk_seed=int(returns_seed),
        cfg=cfg,
    )


def _worker_task(args):
    """
    Supports:
      - single task: (idx, batch_size, seed)
      - batched tasks: [(idx, batch_size, seed), ...]
    """
    tasks, single = normalize_tasks(args)
    results = []

    for idx, batch_size, seed in tasks:
        idx_i = int(idx)
        batch_i = int(batch_size)

        chunk_seed = derive_chunk_seed(seed, idx_i, stride=10_000)

        # Build one chunk (full "sales-shaped" table; includes order cols when enabled by init.py)
        detail_table = chunk_builder.build_chunk_table(
            batch_i,
            int(chunk_seed),
            no_discount_key=State.no_discount_key,
            chunk_idx=idx_i,
            chunk_capacity_orders=int(getattr(State, "chunk_size", batch_i)),
        )
        if not isinstance(detail_table, pa.Table):
            raise TypeError("chunk_builder must return pyarrow.Table")

        mode = _mode()

        # If producing SalesOrder* tables, order cols must exist.
        if mode in {"sales_order", "both"}:
            _require_cols(
                detail_table,
                ["SalesOrderNumber", "SalesOrderLineNumber"],
                ctx=f"sales_output={mode} requires",
            )

            # Header builder needs these to exist on the raw detail table
            _require_cols(
                detail_table,
                ["SalesOrderNumber", "CustomerKey", "OrderDate", "IsOrderDelayed"],
                ctx="Header build requires",
            )

        # Sales-only path: Sales table write MUST remain unchanged.
        # If returns are enabled and identifiers exist, we ALSO emit SalesReturn for this chunk.
        if mode == "sales":
            sales_table = detail_table
            if bool(getattr(State, "skip_order_cols_requested", False)):
                sales_table = _drop_order_cols_for_sales(sales_table)

            returns_table = _maybe_build_returns(detail_table, chunk_seed=int(chunk_seed))
            if returns_table is None:
                results.append(_write_table(TABLE_SALES, idx_i, sales_table))
                continue

            # If we have returns, emit a dict so sales.py can record both outputs.
            assert TABLE_SALES_RETURN is not None
            out: Dict[str, Any] = {}
            out[TABLE_SALES] = _write_table(TABLE_SALES, idx_i, sales_table)

            returns_out = _project_for_table(TABLE_SALES_RETURN, returns_table)
            out[TABLE_SALES_RETURN] = _write_table(TABLE_SALES_RETURN, idx_i, returns_out)

            results.append(out)
            continue

        out: Dict[str, Any] = {}

        # Both mode: also write Sales table (unchanged behavior)
        if mode == "both":
            sales_table = detail_table
            if bool(getattr(State, "skip_order_cols_requested", False)):
                sales_table = _drop_order_cols_for_sales(sales_table)

            # Enforce the canonical Sales schema (drops any unexpected cols, keeps stable order)
            sales_out = _project_for_table(TABLE_SALES, sales_table)
            out[TABLE_SALES] = _write_table(TABLE_SALES, idx_i, sales_out)

        # Build header from the FULL raw detail table
        header_table = build_header_from_detail(detail_table)

        # Project to configured schemas (slim/fat is controlled by init.py schemas)
        detail_out = _project_for_table(TABLE_SALES_ORDER_DETAIL, detail_table)
        header_out = _project_for_table(TABLE_SALES_ORDER_HEADER, header_table)

        out[TABLE_SALES_ORDER_DETAIL] = _write_table(TABLE_SALES_ORDER_DETAIL, idx_i, detail_out)
        out[TABLE_SALES_ORDER_HEADER] = _write_table(TABLE_SALES_ORDER_HEADER, idx_i, header_out)

        # Optional: SalesReturn table (derived from SalesOrderDetail contract)
        returns_table = _maybe_build_returns(detail_out, chunk_seed=int(chunk_seed))
        if returns_table is not None:
            assert TABLE_SALES_RETURN is not None
            returns_out = _project_for_table(TABLE_SALES_RETURN, returns_table)
            out[TABLE_SALES_RETURN] = _write_table(TABLE_SALES_RETURN, idx_i, returns_out)

        results.append(out)

    return results[0] if single else results
