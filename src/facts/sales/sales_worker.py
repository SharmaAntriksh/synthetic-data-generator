# sales_worker.py — cleaned for State-based global binding

import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import time

# from src.utils.logging_utils import work
from .sales_logic import chunk_builder
from .sales_logic.globals import State, bind_globals


# ===============================================================
# Worker initializer
# ===============================================================

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

    """
    Initialize global state using the new 'State' container.
    Called once per worker at pool startup.
    """

    # Build dense numpy mapping arrays for fast vectorized lookup
    store_to_geo_arr = None
    geo_to_currency_arr = None

    try:
        if isinstance(store_to_geo, dict) and store_to_geo:
            max_store = max(store_to_geo.keys())
            arr = np.full(max_store + 1, -1, dtype=np.int64)
            for k, v in store_to_geo.items():
                arr[int(k)] = int(v)
            store_to_geo_arr = arr
    except Exception:
        store_to_geo_arr = None

    try:
        if isinstance(geo_to_currency, dict) and geo_to_currency:
            max_geo = max(geo_to_currency.keys())
            arr = np.full(max_geo + 1, -1, dtype=np.int64)
            for k, v in geo_to_currency.items():
                arr[int(k)] = int(v)
            geo_to_currency_arr = arr
    except Exception:
        geo_to_currency_arr = None

    # Bind everything to the State container
    bind_globals({
        "product_np": product_np,
        "store_keys": store_keys,
        "promo_keys_all": promo_keys_all,
        "promo_pct_all": promo_pct_all,
        "promo_start_all": promo_start_all,
        "promo_end_all": promo_end_all,
        "customers": customers,

        "store_to_geo": store_to_geo,
        "geo_to_currency": geo_to_currency,
        "store_to_geo_arr": store_to_geo_arr,
        "geo_to_currency_arr": geo_to_currency_arr,

        "date_pool": date_pool,
        "date_prob": date_prob,

        "skip_order_cols": skip_order_cols,
        "file_format": file_format,
        "out_folder": out_folder,
        "row_group_size": row_group_size,
        "compression": compression,

        "no_discount_key": no_discount_key,
        "delta_output_folder": os.path.normpath(delta_output_folder),
        "write_delta": write_delta,

        "partition_enabled": partition_enabled,
        "partition_cols": partition_cols,
    })


# ===============================================================
# Utilities
# ===============================================================

def _stream_write_parquet(table: pa.Table, path: str, compression: str, row_group_size: int):
    """Efficient streaming parquet writer for large tables."""
    part_cols = [c for c in ("Year", "Month") if c in table.column_names]
    if part_cols:
        normal_cols = [c for c in table.column_names if c not in part_cols]
        table = table.select(normal_cols + part_cols)

    dict_cols = [c for c in table.column_names if c not in ["SalesOrderNumber", "CustomerKey"]]

    writer = pq.ParquetWriter(
        path,
        table.schema,
        compression=compression,
        use_dictionary=dict_cols,
        write_statistics=True,
    )

    try:
        total = table.num_rows
        for start in range(0, total, int(row_group_size)):
            length = min(int(row_group_size), total - start)
            writer.write_table(table.slice(start, length))
    finally:
        writer.close()


# def _try_write_csv_arrow(table: pa.Table, out_path: str) -> bool:
#     """Try Arrow CSV write; fallback to pandas if needed."""
#     try:
#         import pyarrow.csv as pacsv
#         pacsv.write_csv(table, out_path)
#         return True
#     except Exception:
#         return False


def _ensure_arrow_table(obj):
    if isinstance(obj, pa.Table):
        return obj
    return pa.Table.from_pandas(obj, preserve_index=False, safe=False)


# ===============================================================
# Worker Task (computes + writes one chunk)
# ===============================================================

def _worker_task(args):
    idx, batch_size, seed = args

    # Derive per-chunk seed
    try:
        pid = os.getpid()
    except Exception:
        pid = idx

    base_seed = int(seed) if seed is not None else 0
    seed_for_chunk = base_seed ^ (idx + pid + (int(time.time()) & 0xFFFF))

    # Build the data chunk
    table_or_df = chunk_builder.build_chunk_table(
        batch_size,
        seed_for_chunk,
        no_discount_key=State.no_discount_key,
    )

    table = _ensure_arrow_table(table_or_df)

    # ------------------------------------------------------------
    # DELTAPARQUET MODE
    # ------------------------------------------------------------
    if State.file_format == "deltaparquet":
        tmp_dir = os.path.join(State.delta_output_folder, "_tmp_parts")
        os.makedirs(tmp_dir, exist_ok=True)

        out_path = os.path.join(tmp_dir, f"delta_part_{idx:04d}.parquet")

        _stream_write_parquet(
            table,
            out_path,
            compression=State.compression,
            row_group_size=int(State.row_group_size),
        )

        # work(chunk=idx, outfile=out_path)

        return {
            "delta_part": out_path,
            "chunk": idx,
            "rows": table.num_rows,
        }

    # ------------------------------------------------------------
    # CSV MODE
    # ------------------------------------------------------------
    if State.file_format == "csv":
        os.makedirs(State.out_folder, exist_ok=True)
        out_path = os.path.join(State.out_folder, f"sales_chunk{idx:04d}.csv")

        import pyarrow.compute as pc
        import pyarrow.csv as pacsv

        import pyarrow.compute as pc
        import pyarrow as pa

        if "IsOrderDelayed" in table.column_names:
            col_idx = table.schema.get_field_index("IsOrderDelayed")
            col = table.column(col_idx)

            col = pc.fill_null(col, 0)
            col = pc.cast(col, pa.int8())

            table = table.set_column(col_idx, "IsOrderDelayed", col)

        pacsv.write_csv(
            table,
            out_path,
            write_options=pacsv.WriteOptions(
                include_header=True,
                quoting_style="none"
            )
        )

        # work(msg=f"Chunk {idx} → {os.path.basename(out_path)}")

        return out_path


    # ------------------------------------------------------------
    # PARQUET MODE
    # ------------------------------------------------------------
    os.makedirs(State.out_folder, exist_ok=True)
    out_path = os.path.join(State.out_folder, f"sales_chunk{idx:04d}.parquet")

    _stream_write_parquet(
        table,
        out_path,
        compression=State.compression,
        row_group_size=int(State.row_group_size),
    )

    # work(chunk=idx, outfile=out_path)
    return out_path
