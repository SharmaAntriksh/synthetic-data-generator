"""Shared multi-format writer for batch fact tables.

Handles parquet, CSV, and Delta Lake output for fact tables that produce
a complete DataFrame/Table in memory (budget, inventory, complaints).
NOT used by the sales writer, which has its own chunked merge pipeline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.logging_utils import info


def write_fact_table(
    df_or_table: pd.DataFrame | pa.Table,
    out_dir: Path,
    name: str,
    file_format: str,
    *,
    csv_prep_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    csv_float_format: Optional[str] = None,
    compression: str = "snappy",
    row_group_size: int = 500_000,
) -> None:
    """Write a fact table in the requested format.

    Args:
        df_or_table:    DataFrame or PyArrow Table to write.
        out_dir:        Directory for parquet/csv output.
        name:           Base filename (without extension).
        file_format:    "csv" | "parquet" | "deltaparquet".
        csv_prep_fn:    Optional function to prepare DataFrame for CSV
                        (column selection, type casting, rounding).
        csv_float_format: Optional float format string for CSV output.
        compression:    Parquet compression codec.
        row_group_size: Parquet row group size.
    """
    if isinstance(df_or_table, pa.Table):
        table = df_or_table
        n_rows = table.num_rows
    else:
        table = pa.Table.from_pandas(df_or_table, preserve_index=False)
        n_rows = len(df_or_table)

    if file_format == "deltaparquet":
        delta_dir = out_dir.parent / name
        delta_dir.mkdir(parents=True, exist_ok=True)
        try:
            from deltalake import write_deltalake
        except ImportError:
            from deltalake.writer import write_deltalake
        write_deltalake(str(delta_dir), table, mode="overwrite")
        info(f"Wrote {name}/ ({n_rows:,} rows)")
        return

    # Parquet — always written for both parquet and csv formats
    parquet_path = out_dir / f"{name}.parquet"
    pq.write_table(
        table, str(parquet_path),
        compression=compression,
        row_group_size=row_group_size,
        use_dictionary=True,
    )

    if file_format == "csv":
        csv_path = out_dir / f"{name}.csv"
        df = table.to_pandas() if isinstance(df_or_table, pa.Table) else df_or_table
        csv_df = csv_prep_fn(df) if csv_prep_fn else df
        kwargs: dict = {"index": False}
        if csv_float_format:
            kwargs["float_format"] = csv_float_format
        csv_df.to_csv(str(csv_path), **kwargs)
        info(f"Wrote {csv_path.name} ({n_rows:,} rows)")
    else:
        info(f"Wrote {parquet_path.name} ({n_rows:,} rows)")
