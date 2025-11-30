import os
import numpy as np
import pandas as pd
import pyarrow as pa
import csv
from deltalake import write_deltalake

from src.utils.logging_utils import work
from .sales_logic import _build_chunk_table, bind_globals


# =====================================================================
# GLOBALS SHARED BY EACH WORKER
# =====================================================================
_G_out_folder = None
_G_file_format = None
_G_row_group_size = None
_G_compression = None
_G_no_discount_key = None
_G_delta_output_folder = None
_G_write_delta = None


# =====================================================================
# WORKER INITIALIZER
# =====================================================================
def _init_worker(init_args):
    """
    Each worker receives all read-only global arrays/dicts used by sales_logic.
    This ensures perfect equivalence with the old monolithic design.
    """

    (
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
        skip_flag,
    ) = init_args

    # -----------------------------
    # Bind worker-local globals
    # -----------------------------
    globals().update(dict(
        _G_out_folder=out_folder,
        _G_file_format=file_format,
        _G_row_group_size=row_group_size,
        _G_compression=compression,
        _G_no_discount_key=no_discount_key,
        _G_delta_output_folder=delta_output_folder,
        _G_write_delta=write_delta,
    ))

    # -----------------------------
    # Bind globals into sales_logic
    # -----------------------------
    bind_globals({
        "_G_skip_order_cols":  skip_flag,
        "_G_product_np":       product_np,
        "_G_customers":        customers,
        "_G_date_pool":        date_pool,
        "_G_date_prob":        date_prob,
        "_G_store_keys":       store_keys,
        "_G_promo_keys_all":   promo_keys_all,
        "_G_promo_pct_all":    promo_pct_all,
        "_G_promo_start_all":  promo_start_all,
        "_G_promo_end_all":    promo_end_all,
        "_G_store_to_geo":     store_to_geo,
        "_G_geo_to_currency":  geo_to_currency,
    })

    # -----------------------------
    # Array-optimize store→geo, geo→currency if possible
    # -----------------------------
    try:
        # store → geo
        if store_to_geo:
            max_store = max(store_to_geo.keys())
            arr = np.full(max_store + 1, -1, dtype=np.int64)
            for k, v in store_to_geo.items():
                arr[k] = v
            bind_globals({"_G_store_to_geo_arr": arr})
        else:
            bind_globals({"_G_store_to_geo_arr": None})

        # geo → currency
        if geo_to_currency:
            max_geo = max(geo_to_currency.keys())
            arr = np.full(max_geo + 1, -1, dtype=np.int64)
            for k, v in geo_to_currency.items():
                arr[k] = v
            bind_globals({"_G_geo_to_currency_arr": arr})
        else:
            bind_globals({"_G_geo_to_currency_arr": None})

    except Exception:
        # Fall back to dictionary-only path
        bind_globals({"_G_store_to_geo_arr": None})
        bind_globals({"_G_geo_to_currency_arr": None})


# =====================================================================
# WORKER TASK
# =====================================================================
def _worker_task(args):
    """
    args = (idx, batch_size, total_chunks, seed)
    Builds chunk, writes to the output folder, returns path or delta tuple.
    """
    idx, batch, total_chunks, seed = args

    # -----------------------------
    # Build the chunk
    # -----------------------------
    table_or_df = _build_chunk_table(batch, seed, no_discount_key=_G_no_discount_key)

    # Normalize to Arrow Table
    if isinstance(table_or_df, pa.Table):
        table = table_or_df
    else:
        table = pa.Table.from_pandas(table_or_df, preserve_index=False)

    # Row count (safe for Parquet and CSV)
    nrows = table.num_rows

    # -----------------------------
    # Write output
    # -----------------------------
    if _G_file_format == "csv":
        out = os.path.join(_G_out_folder, f"sales_chunk{idx:04d}.csv")
        df = table.to_pandas()
        df.to_csv(out, index=False, quoting=csv.QUOTE_ALL)

    elif _G_file_format == "deltaparquet":
        # Workers don’t write delta — return to parent.
        return ("delta", idx, table)

    else:
        # Parquet via Arrow (best performance)
        import pyarrow.parquet as pq

        out = os.path.join(_G_out_folder, f"sales_chunk{idx:04d}.parquet")

        pq.write_table(
            table,
            out,
            row_group_size=_G_row_group_size,
            compression=_G_compression,
        )

        # Optional delta append
        if _G_write_delta:
            write_deltalake(_G_delta_output_folder, table, mode="append")

    # -----------------------------
    # Progress indicator (safe)
    # -----------------------------
    pct = int((idx + 1) / total_chunks * 100)

    work(
        chunk=idx + 1,
        total=total_chunks,
        pct=pct,
        rows=nrows,
        outfile=out,
    )

    return out
