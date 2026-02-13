from __future__ import annotations

import os
from typing import Optional

import pyarrow as pa


def _pa_compute():
    import pyarrow.compute as pc
    return pc


def _pa_csv():
    import pyarrow.csv as pacsv
    return pacsv


def _pa_parquet():
    import pyarrow.parquet as pq
    return pq


from ..sales_logic.globals import State


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


def _add_year_month(table: pa.Table) -> pa.Table:
    """
    For deltaparquet outputs: ensure Year/Month exist.
    Prefer OrderDate; fallback to DeliveryDate.
    """
    date_col = None
    for c in ("OrderDate", "DeliveryDate"):
        if c in table.schema.names:
            date_col = c
            break
    if not date_col:
        return table

    pc = _pa_compute()
    year = pc.cast(pc.year(table[date_col]), pa.int16())
    month = pc.cast(pc.month(table[date_col]), pa.int16())

    if "Year" not in table.schema.names:
        table = table.append_column("Year", year)
    if "Month" not in table.schema.names:
        table = table.append_column("Month", month)
    return table


def _normalize_to_schema(table: pa.Table, expected: pa.Schema, *, table_name: Optional[str]) -> pa.Table:
    expected = expected.remove_metadata()

    # If expected wants partition columns, add them automatically (delta convenience)
    wants_year_month = "Year" in expected.names and "Month" in expected.names
    if wants_year_month and ("Year" not in table.schema.names or "Month" not in table.schema.names):
        table = _add_year_month(table)

    got_names = set(table.schema.names)
    exp_names = set(expected.names)
    if got_names != exp_names:
        missing = sorted(exp_names - got_names)
        extra = sorted(got_names - exp_names)
        raise RuntimeError(
            "Schema mismatch in writer.\n"
            f"Table: {table_name or 'Sales'}\n"
            f"Missing: {missing}\n"
            f"Extra: {extra}\n\n"
            f"Expected:\n{expected}\n\nGot:\n{table.schema}"
        )

    # Reorder + cast columns exactly to expected
    pc = _pa_compute()
    arrays = []
    for field in expected:
        col = table[field.name]
        if col.type != field.type:
            col = pc.cast(col, field.type, safe=False)
        arrays.append(col)

    return pa.Table.from_arrays(arrays, schema=expected)


def _write_parquet_table(table: pa.Table, path: str, *, table_name: Optional[str] = None) -> None:
    pq = _pa_parquet()

    expected = _expected_schema(table_name)
    table = _normalize_to_schema(table, expected, table_name=table_name)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    writer = pq.ParquetWriter(
        path,
        table.schema,
        compression=State.compression,
        use_dictionary=_dict_cols(table_name),
        write_statistics=bool(getattr(State, "write_statistics", True)),
    )
    try:
        writer.write_table(table, row_group_size=int(getattr(State, "row_group_size", 1_000_000)))
    finally:
        writer.close()


def _write_csv(table: pa.Table, path: str, *, table_name: Optional[str] = None) -> None:
    pacsv = _pa_csv()
    pc = _pa_compute()

    expected = _expected_schema(table_name)
    table = _normalize_to_schema(table, expected, table_name=table_name)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # Ensure null-safe int8 for CSV
    if "IsOrderDelayed" in table.column_names:
        idx = table.schema.get_field_index("IsOrderDelayed")
        table = table.set_column(
            idx,
            "IsOrderDelayed",
            pc.cast(pc.fill_null(table["IsOrderDelayed"], 0), pa.int8()),
        )

    pacsv.write_csv(
        table,
        path,
        write_options=pacsv.WriteOptions(include_header=True, quoting_style="none"),
    )
