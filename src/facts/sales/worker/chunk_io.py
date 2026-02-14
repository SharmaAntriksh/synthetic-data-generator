from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa

from ..sales_logic.globals import State

from ...common.worker.chunk_io import (
    ChunkIOConfig,
    add_year_month_from_date,
    write_csv_table,
    write_parquet_table,
)


# -----------------------------
# Schema + encoding policies
# -----------------------------
def _expected_schema(table_name: Optional[str]) -> pa.Schema:
    schema_by_table = getattr(State, "schema_by_table", None)
    if table_name and isinstance(schema_by_table, dict) and table_name in schema_by_table:
        return schema_by_table[table_name]
    return State.sales_schema


def _dict_cols(table_name: Optional[str]) -> list[str]:
    m = getattr(State, "parquet_dict_cols_by_table", None)
    if table_name and isinstance(m, dict) and table_name in m:
        return list(m[table_name])
    return list(getattr(State, "parquet_dict_cols", []))


def _schema_needs_year_month(expected: pa.Schema) -> bool:
    names = set(expected.names)
    return ("Year" in names) and ("Month" in names)


# -----------------------------
# Year/Month derivation
# -----------------------------
def _derive_year_month_from_int_order_date(order_date: pa.ChunkedArray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Supports two common integer encodings:
      1) YYYYMMDD (e.g. 20240131)
      2) epoch-days (days since 1970-01-01), typically small (~ 10k-30k for modern years)

    Returns:
      (year:int16 array, month:int16 array)
    """
    x = order_date.combine_chunks().to_numpy(zero_copy_only=False)
    x = np.asarray(x)

    if x.dtype.kind not in {"i", "u"}:
        raise RuntimeError(f"OrderDate integer derivation expected int dtype, got {x.dtype}")

    xi = x.astype(np.int64, copy=False)

    if xi.size == 0:
        return xi.astype(np.int16), xi.astype(np.int16)

    # Reject obvious "null sentinel" values; pipeline shouldn't emit null OrderDate.
    if np.any(xi == np.iinfo(np.int64).min):
        raise RuntimeError("OrderDate contains nulls; cannot derive Year/Month")

    mx = int(np.max(xi))
    mn = int(np.min(xi))

    # Heuristic: YYYYMMDD looks like 19000101..21001231
    if 19_000_000 <= mx <= 210_012_31 and 19_000_000 <= mn <= 210_012_31:
        year = (xi // 10_000).astype(np.int16, copy=False)
        month = ((xi // 100) % 100).astype(np.int16, copy=False)
        return year, month

    # Heuristic: epoch-days are "small" (modern years ~ 18k-25k)
    if -100_000 <= mn <= 200_000 and -100_000 <= mx <= 200_000:
        epoch = np.datetime64("1970-01-01", "D")
        dt = (epoch + xi.astype("timedelta64[D]")).astype("datetime64[D]", copy=False)

        year = (dt.astype("datetime64[Y]").astype(np.int32) + 1970).astype(np.int16)
        months = dt.astype("datetime64[M]").astype(np.int32)
        month = ((months % 12) + 1).astype(np.int16)
        return year, month

    raise RuntimeError(
        f"OrderDate integer format not recognized for Year/Month derivation; min={mn} max={mx}"
    )


def _ensure_year_month_if_needed_for_table(
    table: pa.Table,
    *,
    table_name: str,
    expected_schema: pa.Schema,
) -> pa.Table:
    """
    Derive Year/Month only when expected_schema includes them.

    Uses per-table policy:
      State.date_cols_by_table[table_name] -> list of preferred date columns (in priority order)

    Falls back to:
      ["DeliveryDate", "OrderDate"]

    If no date-like column exists, allows integer OrderDate fallback (YYYYMMDD or epoch-days).
    """
    if ("Year" not in expected_schema.names) or ("Month" not in expected_schema.names):
        return table

    if ("Year" in table.column_names) and ("Month" in table.column_names):
        return table

    policy = getattr(State, "date_cols_by_table", {}) or {}
    candidates = policy.get(table_name) or ["DeliveryDate", "OrderDate"]

    usable: list[str] = []
    for c in candidates:
        if c not in table.column_names:
            continue
        t = table.schema.field(c).type
        if pa.types.is_date32(t) or pa.types.is_date64(t) or pa.types.is_timestamp(t):
            usable.append(c)

    if usable:
        return add_year_month_from_date(table, date_cols=tuple(usable))

    # Fallback: integer OrderDate (older dtype)
    if "OrderDate" in table.column_names and pa.types.is_integer(table.schema.field("OrderDate").type):
        year, month = _derive_year_month_from_int_order_date(table["OrderDate"])
        table = table.append_column("Year", pa.array(year, type=pa.int16()))
        table = table.append_column("Month", pa.array(month, type=pa.int16()))
        return table

    raise RuntimeError(
        f"Cannot derive Year/Month for table={table_name}: no usable date column among {candidates}"
    )


# -----------------------------
# CSV postprocess
# -----------------------------
def _csv_postprocess_sales(table: pa.Table) -> pa.Table:
    """
    Preserve existing CSV behavior: ensure null-safe int8 for IsOrderDelayed.
    """
    try:
        import pyarrow.compute as pc  # type: ignore
    except Exception as e:
        raise RuntimeError("pyarrow.compute is required for CSV postprocess") from e

    if "IsOrderDelayed" in table.column_names:
        idx = table.schema.get_field_index("IsOrderDelayed")
        table = table.set_column(
            idx,
            "IsOrderDelayed",
            pc.cast(pc.fill_null(table["IsOrderDelayed"], 0), pa.int8()),
        )
    return table


# -----------------------------
# Public write wrappers
# -----------------------------
def _write_parquet_table(table: pa.Table, path: str, *, table_name: Optional[str] = None) -> None:
    tn = table_name or "Sales"
    expected = _expected_schema(tn)

    cfg = ChunkIOConfig(
        compression=getattr(State, "compression", "snappy"),
        row_group_size=int(getattr(State, "row_group_size", 1_000_000)),
        write_statistics=bool(getattr(State, "write_statistics", True)),
    )

    need_ym = _schema_needs_year_month(expected)

    # IMPORTANT: common calls ensure_cols_fn(table) without context.
    # So we pass a closure that captures tn + expected.
    ensure_fn = (lambda t: _ensure_year_month_if_needed_for_table(t, table_name=tn, expected_schema=expected)) if need_ym else None

    write_parquet_table(
        table,
        path,
        expected_schema=expected,
        cfg=cfg,
        use_dictionary=_dict_cols(tn),
        table_name=tn,
        ensure_cols=("Year", "Month") if need_ym else (),
        ensure_cols_fn=ensure_fn,
    )


def _write_csv(table: pa.Table, path: str, *, table_name: Optional[str] = None) -> None:
    tn = table_name or "Sales"
    expected = _expected_schema(tn)
    need_ym = _schema_needs_year_month(expected)

    ensure_fn = (lambda t: _ensure_year_month_if_needed_for_table(t, table_name=tn, expected_schema=expected)) if need_ym else None

    write_csv_table(
        table,
        path,
        expected_schema=expected,
        table_name=tn,
        ensure_cols=("Year", "Month") if need_ym else (),
        ensure_cols_fn=ensure_fn,
        postprocess=_csv_postprocess_sales,
    )
