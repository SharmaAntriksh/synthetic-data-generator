from __future__ import annotations

from typing import Optional

import pyarrow as pa

from ..sales_logic.globals import State

from src.facts.common.worker.chunk_io import (
    ChunkIOConfig,
    add_year_month_from_date,
    write_csv_table,
    write_parquet_table,
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


def _ensure_year_month_if_needed(table: pa.Table) -> pa.Table:
    """
    Sales default: derive Year/Month from OrderDate else DeliveryDate.
    (keeps existing deltaparquet convenience behavior)
    """
    return add_year_month_from_date(table, date_cols=("OrderDate", "DeliveryDate"))


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


def _write_parquet_table(table: pa.Table, path: str, *, table_name: Optional[str] = None) -> None:
    expected = _expected_schema(table_name)

    cfg = ChunkIOConfig(
        compression=getattr(State, "compression", "snappy"),
        row_group_size=int(getattr(State, "row_group_size", 1_000_000)),
        write_statistics=bool(getattr(State, "write_statistics", True)),
    )

    # Sales convention: if expected schema wants Year/Month, auto-derive them
    write_parquet_table(
        table,
        path,
        expected_schema=expected,
        cfg=cfg,
        use_dictionary=_dict_cols(table_name),
        table_name=table_name or "Sales",
        ensure_cols=("Year", "Month"),
        ensure_cols_fn=_ensure_year_month_if_needed,
    )


def _write_csv(table: pa.Table, path: str, *, table_name: Optional[str] = None) -> None:
    expected = _expected_schema(table_name)

    write_csv_table(
        table,
        path,
        expected_schema=expected,
        table_name=table_name or "Sales",
        ensure_cols=("Year", "Month"),
        ensure_cols_fn=_ensure_year_month_if_needed,
        postprocess=_csv_postprocess_sales,
    )
