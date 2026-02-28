from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa

from ..sales_logic import State

import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq


@dataclass(frozen=True)
class ChunkIOConfig:
    compression: str = "snappy"
    row_group_size: int = 1_000_000
    write_statistics: bool = True


EnsureColsFn = Callable[[pa.Table], pa.Table]
CsvPostprocessFn = Callable[[pa.Table], pa.Table]


def add_year_month_from_date(
    table: pa.Table,
    *,
    date_cols: Sequence[str] = ("OrderDate", "DeliveryDate"),
    year_col: str = "Year",
    month_col: str = "Month",
    year_type: pa.DataType = pa.int16(),
    month_type: pa.DataType = pa.int16(),
) -> pa.Table:
    date_col = None
    for c in date_cols:
        if c in table.schema.names:
            date_col = c
            break
    if not date_col:
        return table

    year = pc.cast(pc.year(table[date_col]), year_type)
    month = pc.cast(pc.month(table[date_col]), month_type)

    if year_col not in table.schema.names:
        table = table.append_column(year_col, year)
    if month_col not in table.schema.names:
        table = table.append_column(month_col, month)
    return table


def normalize_to_schema(
    table: pa.Table,
    expected: pa.Schema,
    *,
    table_name: Optional[str] = None,
    ensure_cols: Optional[Sequence[str]] = None,
    ensure_cols_fn: Optional[EnsureColsFn] = None,
) -> pa.Table:
    expected = expected.remove_metadata()

    if ensure_cols and ensure_cols_fn:
        wants = all(c in expected.names for c in ensure_cols)
        missing_any = any(c not in table.schema.names for c in ensure_cols)
        if wants and missing_any:
            table = ensure_cols_fn(table)

    got_names = set(table.schema.names)
    exp_names = set(expected.names)
    if got_names != exp_names:
        missing = sorted(exp_names - got_names)
        extra = sorted(got_names - exp_names)
        raise RuntimeError(
            "Schema mismatch in writer.\n"
            f"Table: {table_name or 'unknown'}\n"
            f"Missing: {missing}\n"
            f"Extra: {extra}\n\n"
            f"Expected:\n{expected}\n\nGot:\n{table.schema}"
        )

    arrays = []
    cast_safe = bool(getattr(State, "cast_safe", True))
    arrays = []
    for field in expected:
        col = table[field.name]
        if col.type != field.type:
            try:
                # If projecting to a DECIMAL and source is floating, round to the target scale first
                # to avoid float artifacts prior to cast.
                if pa.types.is_decimal(field.type) and pa.types.is_floating(col.type):
                    col = pc.round(col, ndigits=int(field.type.scale))
                col = pc.cast(col, field.type, safe=cast_safe)
            except Exception as ex:
                raise RuntimeError(
                    f"[{table_name or 'unknown'}] Failed cast '{field.name}': {col.type} -> {field.type} "
                    f"(safe={cast_safe}): {ex}"
                ) from ex
        arrays.append(col)

    return pa.Table.from_arrays(arrays, schema=expected)


def write_parquet_table(
    table: pa.Table,
    path: str,
    *,
    expected_schema: pa.Schema,
    cfg: ChunkIOConfig,
    use_dictionary: Optional[Sequence[str]] = None,
    table_name: Optional[str] = None,
    ensure_cols: Optional[Sequence[str]] = None,
    ensure_cols_fn: Optional[EnsureColsFn] = None,
) -> None:

    table = normalize_to_schema(
        table,
        expected_schema,
        table_name=table_name,
        ensure_cols=ensure_cols,
        ensure_cols_fn=ensure_cols_fn,
    )

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    writer = pq.ParquetWriter(
        path,
        table.schema,
        compression=cfg.compression,
        use_dictionary=list(use_dictionary) if use_dictionary else False,
        write_statistics=bool(cfg.write_statistics),
    )
    try:
        writer.write_table(table, row_group_size=int(max(1, cfg.row_group_size)))
    finally:
        writer.close()


def write_csv_table(
    table: pa.Table,
    path: str,
    *,
    expected_schema: pa.Schema,
    table_name: Optional[str] = None,
    ensure_cols: Optional[Sequence[str]] = None,
    ensure_cols_fn: Optional[EnsureColsFn] = None,
    postprocess: Optional[CsvPostprocessFn] = None,
) -> None:

    table = normalize_to_schema(
        table,
        expected_schema,
        table_name=table_name,
        ensure_cols=ensure_cols,
        ensure_cols_fn=ensure_cols_fn,
    )

    if postprocess is not None:
        table = postprocess(table)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    pacsv.write_csv(
        table,
        path,
        write_options=pacsv.WriteOptions(include_header=True, quoting_style="none"),
    )


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


def _derive_year_month_from_int_order_date(order_date: pa.ChunkedArray) -> Tuple[np.ndarray, np.ndarray]:
    x = order_date.combine_chunks().to_numpy(zero_copy_only=False)
    x = np.asarray(x)

    if x.dtype.kind not in {"i", "u"}:
        raise RuntimeError(f"OrderDate integer derivation expected int dtype, got {x.dtype}")

    xi = x.astype(np.int64, copy=False)
    if xi.size == 0:
        return xi.astype(np.int16), xi.astype(np.int16)

    if np.any(xi == np.iinfo(np.int64).min):
        raise RuntimeError("OrderDate contains nulls; cannot derive Year/Month")

    mx = int(np.max(xi))
    mn = int(np.min(xi))

    if 19_000_000 <= mx <= 210_012_31 and 19_000_000 <= mn <= 210_012_31:
        year = (xi // 10_000).astype(np.int16, copy=False)
        month = ((xi // 100) % 100).astype(np.int16, copy=False)
        return year, month

    if -100_000 <= mn <= 200_000 and -100_000 <= mx <= 200_000:
        epoch = np.datetime64("1970-01-01", "D")
        dt = (epoch + xi.astype("timedelta64[D]")).astype("datetime64[D]", copy=False)

        year = (dt.astype("datetime64[Y]").astype(np.int32) + 1970).astype(np.int16)
        months = dt.astype("datetime64[M]").astype(np.int32)
        month = ((months % 12) + 1).astype(np.int16)
        return year, month

    raise RuntimeError(f"OrderDate integer format not recognized; min={mn} max={mx}")


def _ensure_year_month_if_needed_for_table(
    table: pa.Table,
    *,
    table_name: str,
    expected_schema: pa.Schema,
) -> pa.Table:
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

    if "OrderDate" in table.column_names and pa.types.is_integer(table.schema.field("OrderDate").type):
        year, month = _derive_year_month_from_int_order_date(table["OrderDate"])
        table = table.append_column("Year", pa.array(year, type=pa.int16()))
        table = table.append_column("Month", pa.array(month, type=pa.int16()))
        return table

    raise RuntimeError(f"Cannot derive Year/Month for table={table_name}: {candidates}")


def _csv_postprocess_sales(table: pa.Table) -> pa.Table:
    import pyarrow.compute as pc  # type: ignore

    if "IsOrderDelayed" in table.column_names:
        idx = table.schema.get_field_index("IsOrderDelayed")
        table = table.set_column(
            idx,
            "IsOrderDelayed",
            pc.cast(pc.fill_null(table["IsOrderDelayed"], 0), pa.int8()),
        )
    return table


def _write_parquet_table(table: pa.Table, path: str, *, table_name: Optional[str] = None, is_chunk: Optional[bool] = None) -> None:
    tn = table_name or "Sales"
    expected = _expected_schema(tn)

    if is_chunk is None:
        # Heuristic: worker chunk files are typically written via output_paths.chunk_path()
        # and contain a part identifier.
        p = str(path).lower()
        is_chunk = (".part" in p) or ("chunk" in p) or ("parts" in p)

    write_stats_chunks = bool(getattr(State, "write_statistics_chunks", False))
    write_stats_merged = bool(getattr(State, "write_statistics", True))
    cfg = ChunkIOConfig(
        compression=getattr(State, "compression", "snappy"),
        row_group_size=int(getattr(State, "row_group_size", 1_000_000)),
        write_statistics=write_stats_chunks if bool(is_chunk) else write_stats_merged,
    )

    need_ym = _schema_needs_year_month(expected)
    ensure_fn = (
        (lambda t: _ensure_year_month_if_needed_for_table(t, table_name=tn, expected_schema=expected))
        if need_ym
        else None
    )

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

    ensure_fn = (
        (lambda t: _ensure_year_month_if_needed_for_table(t, table_name=tn, expected_schema=expected))
        if need_ym
        else None
    )

    write_csv_table(
        table,
        path,
        expected_schema=expected,
        table_name=tn,
        ensure_cols=("Year", "Month") if need_ym else (),
        ensure_cols_fn=ensure_fn,
        postprocess=_csv_postprocess_sales,
    )
