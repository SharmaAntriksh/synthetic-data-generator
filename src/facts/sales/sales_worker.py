# ========================================================================
# sales_worker.py — FINAL MERGED VERSION
# - Module-level globals (matches sales_logic expectations)
# - Uses bind_globals()
# - Uses _build_chunk_table() for real logic
# - Uses streaming ParquetWriter (fast + low memory)
# - Supports deltaparquet return-to-main behavior
# - Preserves EXACT logging with work()
# ========================================================================

import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.logging_utils import work
from .sales_logic import bind_globals, _build_chunk_table

# =========================================================================
# MODULE GLOBALS — must match what sales_logic expects
# =========================================================================
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


# =========================================================================
# INITIALIZER (executed once per worker)
# =========================================================================
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
    global _G_product_np, _G_store_keys, _G_promo_keys_all, _G_promo_pct_all
    global _G_promo_start_all, _G_promo_end_all, _G_customers, _G_store_to_geo
    global _G_geo_to_currency, _G_date_pool, _G_date_prob, _G_out_folder
    global _G_file_format, _G_row_group_size, _G_compression, _G_no_discount_key
    global _G_delta_output_folder, _G_write_delta, _G_skip_order_cols
    global _G_partition_enabled, _G_partition_cols

    # Store as module-level globals
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

    # Bind into sales_logic (updates sales_logic globals)
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


# =========================================================================
# STREAMING PARQUET WRITER (low-memory)
# =========================================================================
def _stream_write_parquet(table: pa.Table, path: str, compression: str, row_group_size: int):
    # Partition columns last
    part_cols = [c for c in ("Year", "Month") if c in table.column_names]
    normal_cols = [c for c in table.column_names if c not in part_cols]
    table = table.select(normal_cols + part_cols)

    writer = pq.ParquetWriter(
        path,
        table.schema,
        compression=compression,
        use_dictionary=False,
        write_statistics=False,
    )
    try:
        total = table.num_rows
        for start in range(0, total, row_group_size):
            writer.write_table(table.slice(start, min(row_group_size, total - start)))
    finally:
        writer.close()


# =========================================================================
# WORKER TASK (runs once per chunk)
# =========================================================================
def _worker_task(args):
    
    idx, batch_size, seed = args
    table_or_df = _build_chunk_table(batch_size, seed, no_discount_key=_G_no_discount_key)

    if isinstance(table_or_df, pa.Table):
        table = table_or_df
    else:
        table = pa.Table.from_pandas(table_or_df, preserve_index=False)

    # Partitioning logic
    if _G_partition_enabled and "OrderDate" in table.column_names:
        od = table["OrderDate"].to_numpy()
        years = (od.astype("datetime64[Y]").astype(int) + 1970).astype(np.int16)
        months = (od.astype("datetime64[M]").astype(int) % 12 + 1).astype(np.int8)
        table = table.append_column("Year", pa.array(years, type=pa.int16()))
        table = table.append_column("Month", pa.array(months, type=pa.int8()))

    # DELTA mode
    if _G_file_format == "deltaparquet":
        return ("delta", idx, table)

    # CSV MODE
    if _G_file_format == "csv":
        out_path = os.path.join(_G_out_folder, f"sales_chunk{idx:04d}.csv")
        table.to_pandas().to_csv(out_path, index=False)
        work(f"Chunk {idx} → {out_path}")
        return out_path

    # PARQUET MODE (default)
    out_path = os.path.join(_G_out_folder, f"sales_chunk{idx:04d}.parquet")

    _stream_write_parquet(
        table,
        out_path,
        compression=_G_compression,
        row_group_size=int(_G_row_group_size),
    )

    work(f"Chunk {idx} → {out_path}")
    return out_path
