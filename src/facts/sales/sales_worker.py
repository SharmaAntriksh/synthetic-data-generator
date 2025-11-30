# sales_worker.py — Optimized for Arrow-first speed, vector worker generation,
# deltaparquet improvements (worker-side writes), and lower memory use.

import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.logging_utils import work
from .sales_logic import bind_globals, _build_chunk_table

# Module-level globals (kept same names as sales_logic expects)
_G_product_np = None
_G_store_keys = None
_G_promo_keys_all = None
_G_promo_pct_all = None
_G_promo_start_all = None
_G_promo_end_all = None
_G_customers = None
_G_store_to_geo = None
_G_geo_to_currency = None
_G_date_pool = None
_G_date_prob = None

_G_out_folder = None
_G_file_format = None
_G_row_group_size = 250_000
_G_compression = "lz4"

_G_no_discount_key = None
_G_delta_output_folder = None
_G_write_delta = False
_G_skip_order_cols = False

_G_partition_enabled = False
_G_partition_cols = None


def init_sales_worker(
    product_np,
    store_keys,
    promo_keys_all,
    promo_pct_all,
    promo_start_all,
    promo_end_all,
    customers,
    store_to_geo,
    geo_to_currency,
    date_pool,
    date_prob,
    out_folder,
    file_format,
    row_group_size,
    compression,
    no_discount_key,
    delta_output_folder,
    write_delta,
    skip_order_cols,
    partition_enabled,
    partition_cols,
):
    """Initialize module globals and bind into sales_logic."""
    global _G_product_np, _G_store_keys, _G_promo_keys_all, _G_promo_pct_all
    global _G_promo_start_all, _G_promo_end_all, _G_customers, _G_store_to_geo
    global _G_geo_to_currency, _G_date_pool, _G_date_prob, _G_out_folder
    global _G_file_format, _G_row_group_size, _G_compression, _G_no_discount_key
    global _G_delta_output_folder, _G_write_delta, _G_skip_order_cols
    global _G_partition_enabled, _G_partition_cols

    _G_product_np = product_np
    _G_store_keys = store_keys
    _G_promo_keys_all = promo_keys_all
    _G_promo_pct_all = promo_pct_all
    _G_promo_start_all = promo_start_all
    _G_promo_end_all = promo_end_all
    _G_customers = customers
    _G_store_to_geo = store_to_geo
    _G_geo_to_currency = geo_to_currency
    _G_date_pool = date_pool
    _G_date_prob = date_prob

    _G_out_folder = out_folder
    _G_file_format = file_format
    _G_row_group_size = row_group_size
    _G_compression = compression

    _G_no_discount_key = no_discount_key
    _G_delta_output_folder = delta_output_folder
    _G_write_delta = write_delta
    _G_skip_order_cols = skip_order_cols

    _G_partition_enabled = partition_enabled
    _G_partition_cols = partition_cols

    # update sales_logic globals in one shot
    bind_globals({
        "_G_product_np": _G_product_np,
        "_G_store_keys": _G_store_keys,
        "_G_promo_keys_all": _G_promo_keys_all,
        "_G_promo_pct_all": _G_promo_pct_all,
        "_G_promo_start_all": _G_promo_start_all,
        "_G_promo_end_all": _G_promo_end_all,
        "_G_customers": _G_customers,
        "_G_store_to_geo": _G_store_to_geo,
        "_G_geo_to_currency": _G_geo_to_currency,
        "_G_date_pool": _G_date_pool,
        "_G_date_prob": _G_date_prob,
        "_G_skip_order_cols": _G_skip_order_cols,
    })


def _stream_write_parquet(table: pa.Table, path: str, compression: str, row_group_size: int):
    """Write a pyarrow.Table to parquet in streaming (row-group) mode to limit memory."""
    # Put typical partition columns at the end for nicer file layout (non-copying select)
    part_cols = [c for c in ("Year", "Month") if c in table.column_names]
    if part_cols:
        normal_cols = [c for c in table.column_names if c not in part_cols]
        table = table.select(normal_cols + part_cols)

    # Enable dictionary encoding for all columns EXCEPT SalesOrderNumber
    dict_cols = [c for c in table.column_names if c != "SalesOrderNumber"]

    writer = pq.ParquetWriter(
        path,
        table.schema,
        compression=compression,
        use_dictionary=dict_cols,   # <-- key change
        write_statistics=False,
    )

    try:
        total = table.num_rows
        # Write in row-group-sized slices to avoid allocating huge memory buffers
        for start in range(0, total, int(row_group_size)):
            length = min(int(row_group_size), total - start)
            writer.write_table(table.slice(start, length))
    finally:
        writer.close()


def _try_write_csv_arrow(table: pa.Table, out_path: str) -> bool:
    """
    Try to write CSV using pyarrow.csv (faster + Arrow-native) when available.
    Returns True if written, False if fallback required.
    """
    try:
        # pyarrow.csv.write_csv exists in pyarrow >= 1.0; prefer that (zero-copy)
        import pyarrow.csv as pacsv  # local import (safe)
        pacsv.write_csv(table, out_path)
        return True
    except Exception:
        return False


def _ensure_arrow_table(table_or_df):
    """Return a pyarrow.Table, minimizing copies when possible."""
    if isinstance(table_or_df, pa.Table):
        return table_or_df
    # Assume pandas DataFrame otherwise; use safe=False to skip copying/validation where possible
    return pa.Table.from_pandas(table_or_df, preserve_index=False, safe=False)


def _extract_partition_cols(table: pa.Table) -> pa.Table:
    """
    Append Year and Month columns from an OrderDate column (if present).
    Uses numpy views on Arrow arrays for high performance.
    """
    if "OrderDate" not in table.column_names:
        return table

    # Use Arrow's to_numpy — returns numpy datetime64[...] (may be zero-copy)
    od = table["OrderDate"].to_numpy()  # datetime64[ns] or similar
    # Convert to year and month integers with vectorized arithmetic
    # Year: convert to datetime64[Y] then to int + 1970
    years = (od.astype("datetime64[Y]").astype(int) + 1970).astype(np.int16)
    months = (od.astype("datetime64[M]").astype(int) % 12 + 1).astype(np.int8)

    # Append both columns (pa.array will copy the small arrays only)
    table = table.append_column("Year", pa.array(years, type=pa.int16()))
    table = table.append_column("Month", pa.array(months, type=pa.int8()))
    return table


def _worker_task(args):
    """
    Worker entrypoint for one chunk.
    args: (idx, batch_size, seed)
    Returns:
      - path string for file outputs ('csv' or 'parquet'), or
      - ("delta", idx, table) when deltaparquet return-to-main is used, or
      - ("delta", idx, path) when deltaparquet + _G_write_delta is enabled (worker wrote file).
    """
    idx, batch_size, seed = args

    # Build the chunk (sales_logic._build_chunk_table should be vectorized when possible)
    table_or_df = _build_chunk_table(batch_size, seed, no_discount_key=_G_no_discount_key)

    # Ensure Arrow table quickly (fast path if already Arrow)
    table = _ensure_arrow_table(table_or_df)

    # Partitioning (vectorized, low-overhead)
    if _G_partition_enabled and "OrderDate" in table.column_names:
        table = _extract_partition_cols(table)

    # ------------------------------------------------------------
    # DELTAPARQUET — Workers write ONLY temp parquet parts
    # ------------------------------------------------------------
    if _G_file_format == "deltaparquet":

        # Workers write into:  <delta_output_folder>/_tmp_parts/
        tmp_dir = os.path.join(_G_delta_output_folder, "_tmp_parts")
        os.makedirs(tmp_dir, exist_ok=True)


        out_path = os.path.join(tmp_dir, f"delta_part_{idx:04d}.parquet")

        _stream_write_parquet(
            table,
            out_path,
            compression=_G_compression,
            row_group_size=int(_G_row_group_size)
        )

        work(f"Delta chunk {idx} → {out_path}")

        # Return path (never Arrow table)
        return {
            "delta_part": out_path,
            "chunk": idx,
            "rows": table.num_rows
        }



    # CSV MODE (prefer Arrow CSV writer)
    if _G_file_format == "csv":
        os.makedirs(_G_out_folder, exist_ok=True)
        out_path = os.path.join(_G_out_folder, f"sales_chunk{idx:04d}.csv")
        if not _try_write_csv_arrow(table, out_path):
            # Fallback: use pandas write (only if Arrow CSV unavailable)
            table.to_pandas(split_blocks=True).to_csv(out_path, index=False)
        work(f"Chunk {idx} → {out_path}")
        return out_path

    # PARQUET MODE (default)
    os.makedirs(_G_out_folder, exist_ok=True)
    out_path = os.path.join(_G_out_folder, f"sales_chunk{idx:04d}.parquet")
    _stream_write_parquet(
        table,
        out_path,
        compression=_G_compression,
        row_group_size=int(_G_row_group_size),
    )
    work(f"Chunk {idx} → {out_path}")
    return out_path
