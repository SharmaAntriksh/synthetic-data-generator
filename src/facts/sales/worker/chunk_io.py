from __future__ import annotations

from typing import Optional

import pyarrow as pa
import pyarrow.parquet as pq

from ..sales_logic.globals import State


# Cache CSV/compute modules (avoid importing each chunk)
_PACSV: Optional[object] = None
_PC: Optional[object] = None


def _pa_csv():
    global _PACSV
    if _PACSV is None:
        import pyarrow.csv as pacsv
        _PACSV = pacsv
    return _PACSV


def _pa_compute():
    global _PC
    if _PC is None:
        import pyarrow.compute as pc
        _PC = pc
    return _PC


def _assert_schema(table: pa.Table) -> None:
    schema = State.sales_schema
    if table.schema != schema:
        raise RuntimeError(
            "Schema mismatch in writer.\n"
            f"Expected:\n{schema}\n\nGot:\n{table.schema}"
        )


def _write_parquet_table(table: pa.Table, path: str) -> None:
    """
    Fast path: write whole table with row_group_size; ParquetWriter chunks internally.
    """
    _assert_schema(table)

    writer = pq.ParquetWriter(
        path,
        State.sales_schema,
        compression=State.compression,
        use_dictionary=State.parquet_dict_cols,  # only string columns
        write_statistics=True,
    )
    try:
        writer.write_table(table, row_group_size=State.row_group_size)
    finally:
        writer.close()


def _write_csv(table: pa.Table, path: str) -> None:
    """
    CSV output (mainly for debugging/smaller runs).
    """
    _assert_schema(table)

    pc = _pa_compute()
    pacsv = _pa_csv()

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
