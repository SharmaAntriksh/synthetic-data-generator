from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import pyarrow as pa


# -------------------------
# Lazy imports
# -------------------------
def _pa_compute():
    import pyarrow.compute as pc  # type: ignore
    return pc


def _pa_csv():
    import pyarrow.csv as pacsv  # type: ignore
    return pacsv


def _pa_parquet():
    import pyarrow.parquet as pq  # type: ignore
    return pq


# -------------------------
# Config + hooks
# -------------------------
@dataclass(frozen=True)
class ChunkIOConfig:
    compression: str = "snappy"
    row_group_size: int = 1_000_000
    write_statistics: bool = True


# Hook types (policy injected by fact modules)
EnsureColsFn = Callable[[pa.Table], pa.Table]
CsvPostprocessFn = Callable[[pa.Table], pa.Table]


# -------------------------
# Generic helpers
# -------------------------
def add_year_month_from_date(
    table: pa.Table,
    *,
    date_cols: Sequence[str] = ("OrderDate", "DeliveryDate"),
    year_col: str = "Year",
    month_col: str = "Month",
    year_type: pa.DataType = pa.int16(),
    month_type: pa.DataType = pa.int16(),
) -> pa.Table:
    """
    Generic helper: derive Year/Month from the first available date column.
    Useful for delta partition columns.
    """
    date_col = None
    for c in date_cols:
        if c in table.schema.names:
            date_col = c
            break
    if not date_col:
        return table

    pc = _pa_compute()
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
    """
    Strict normalization:
      - optionally ensure/derive columns (e.g., Year/Month) if expected schema wants them
      - require exact column-name match (set equality)
      - reorder + cast columns exactly to expected

    Policy is external:
      - which columns to ensure
      - how to derive them (ensure_cols_fn)
    """
    expected = expected.remove_metadata()

    # Optional auto-add / derive of partition columns
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

    # Reorder + cast columns exactly
    pc = _pa_compute()
    arrays = []
    for field in expected:
        col = table[field.name]
        if col.type != field.type:
            try:
                col = pc.cast(col, field.type, safe=False)
            except Exception as ex:
                raise RuntimeError(
                    f"[{table_name or 'unknown'}] Failed cast '{field.name}': {col.type} -> {field.type}: {ex}"
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
    pq = _pa_parquet()

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
    pacsv = _pa_csv()

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


__all__ = [
    "ChunkIOConfig",
    "add_year_month_from_date",
    "normalize_to_schema",
    "write_parquet_table",
    "write_csv_table",
]
