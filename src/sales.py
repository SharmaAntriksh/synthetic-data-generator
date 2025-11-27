#!/usr/bin/env python3
"""
Patched sales.py — ultra-optimized generator with optional CuPy acceleration and
DELTAPARQUET (real Delta Lake) support.

Features:
- Multiprocessing with initializer to reduce pickling
- PyArrow table generation (preferred) and fast Parquet write
- Delta Lake writes via 'deltalake.write_deltalake' when enabled
- Backward-compatible API; supports csv | parquet | deltaparquet modes
"""
import os
import glob
import numpy as np
import pandas as pd
import csv
from datetime import datetime
from multiprocessing import Pool, cpu_count
from math import ceil

# Try to import pyarrow; if unavailable we'll fall back to pandas-parquet writer
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except Exception:
    PYARROW_AVAILABLE = False

# Try to import deltalake for real Delta Lake writes
try:
    from deltalake import write_deltalake
    DELTA_AVAILABLE = True
except Exception:
    DELTA_AVAILABLE = False

# Try CuPy (GPU-accelerated numpy). If unavailable, fallback to CPU numpy.
try:
    import cupy as cp
    # Confirm there is at least one CUDA device
    try:
        GPU_AVAILABLE = cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        GPU_AVAILABLE = False
except Exception:
    GPU_AVAILABLE = False

# -------------------------
# Basic helpers
# -------------------------

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_parquet_column(path, col):
    return pd.read_parquet(path, columns=[col])[col].values


def load_parquet_df(path, cols=None):
    return pd.read_parquet(path, columns=cols)

# -------------------------
# Weighted customer selection
# -------------------------

def build_weighted_customers(keys, pct, mult, seed=42):
    rng = np.random.default_rng(seed)
    if len(keys) == 0:
        return keys
    mask = rng.random(len(keys)) < (pct / 100.0)
    heavy = keys[mask]
    normal = keys[~mask]
    parts = []
    if heavy.size:
        parts.append(np.repeat(heavy, mult))
    if normal.size:
        parts.append(normal)
    arr = np.concatenate(parts)
    rng.shuffle(arr)
    return arr

# -------------------------
# Weighted date pool
# -------------------------

def build_weighted_date_pool(start, end, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    years = dates.year.values
    months = dates.month.values
    weekdays = dates.weekday.values
    doy = dates.dayofyear.values
    unique_years = np.unique(years)
    idx_year = {y: i for i, y in enumerate(unique_years)}
    growth = 1.08
    yw = np.array([growth ** idx_year[y] for y in years], float)
    month_weights = {1:0.82,2:0.92,3:1.03,4:0.98,5:1.07,6:1.12,7:1.18,8:1.10,9:0.96,10:1.22,11:1.48,12:1.33}
    mw = np.array([month_weights[m] for m in months], float)
    weekday_weights = {0:0.86,1:0.91,2:1.00,3:1.12,4:1.19,5:1.08,6:0.78}
    wdw = np.array([weekday_weights[d] for d in weekdays], float)
    spike = np.ones(n)
    for s,e,f in [(140,170,1.28),(240,260,1.35),(310,350,1.72)]:
        mask = (doy >= s) & (doy <= e)
        spike[mask] *= f
    ot = np.ones(n)
    for a,b,f in [("2021-06-01","2021-10-31",0.70),("2023-02-01","2023-08-31",1.40)]:
        A = pd.to_datetime(a); B = pd.to_datetime(b)
        mask = (dates >= A) & (dates <= B)
        ot[mask] *= f
    noise = rng.uniform(0.95, 1.05, size=n)
    weights = yw * mw * wdw * spike * ot * noise
    blackout_mask = rng.random(n) < rng.uniform(0.10, 0.18)
    weights[blackout_mask] = 0
    weights /= weights.sum()
    # Return numpy datetime64[D] array for faster ops
    return dates.to_numpy(dtype="datetime64[D]"), weights

# -------------------------
# Worker-shared globals (populated by initializer)
# -------------------------
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
_G_row_group_size = None
_G_compression = None
_G_no_discount_key = 1
_G_delta_output_folder = None
_G_write_delta = False
_G_skip_order_cols = False


def _init_worker(init_args):
    global _G_product_np, _G_store_keys, _G_promo_keys_all, _G_promo_pct_all
    global _G_promo_start_all, _G_promo_end_all, _G_customers
    global _G_store_to_geo, _G_geo_to_currency
    global _G_date_pool, _G_date_prob
    global _G_out_folder, _G_file_format
    global _G_row_group_size, _G_compression
    global _G_no_discount_key
    global _G_delta_output_folder, _G_write_delta
    global _G_skip_order_cols

    # unpack exactly in this order (must match init_args above)
    (
        product_np,         # 1
        store_keys,         # 2
        promo_keys_all,     # 3
        promo_pct_all,      # 4
        promo_start_all,    # 5
        promo_end_all,      # 6
        customers,          # 7
        store_to_geo,       # 8
        geo_to_currency,    # 9
        date_pool,          # 10
        date_prob,          # 11
        out_folder,         # 12
        file_format,        # 13
        row_group_size,     # 14
        compression,        # 15
        no_discount_key,    # 16
        delta_output_folder,# 17
        write_delta,        # 18
        skip_order_flag     # 19
    ) = init_args

    # assign to globals
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
    _G_skip_order_cols = bool(skip_order_flag)

    # DEBUG: print what we actually received (very small, safe)
    try:
        print(f"[_init_worker] received skip_order_flag={skip_order_flag!r} (type={type(skip_order_flag)})", flush=True)
    except Exception:
        pass


# -------------------------
# Helper: write a PyArrow table to parquet (fast)
# -------------------------

def _write_table_or_df(table_or_df, out_path, row_group_size, compression):
    """
    table_or_df: either a pyarrow.Table or a pandas.DataFrame
    Prefer pyarrow for speed; fallback to DataFrame.to_parquet if pyarrow not available
    """
    if PYARROW_AVAILABLE and isinstance(table_or_df, pa.Table):
        pq.write_table(table_or_df, out_path, compression=compression, row_group_size=row_group_size, use_dictionary=False)
        return True
    else:
        # fallback: pandas
        if PYARROW_AVAILABLE and isinstance(table_or_df, pa.Table):
            df = table_or_df.to_pandas()
        else:
            df = table_or_df
        df.to_parquet(out_path, index=False, compression=compression)
        return True

# -------------------------
# Core vectorized chunk -> build pyarrow table directly
# -------------------------

def _build_chunk_table(n, seed, no_discount_key=1):
    """
    Uses worker globals to create arrays and returns a pyarrow.Table (preferred).
    Designed to be fast and allocate minimal intermediate Python objects.
    Supports skip of SalesOrderNumber / SalesOrderLineNumber when _G_skip_order_cols = True.
    """
    rng = np.random.default_rng(seed)
    skip_cols = _G_skip_order_cols  # NEW FLAG

    # shortcuts
    product_np = _G_product_np
    store_keys = _G_store_keys
    promo_keys_all = _G_promo_keys_all
    promo_pct_all = _G_promo_pct_all
    promo_start_all = _G_promo_start_all
    promo_end_all = _G_promo_end_all
    customers = _G_customers
    store_to_geo = _G_store_to_geo
    geo_to_currency = _G_geo_to_currency
    date_pool = _G_date_pool
    date_prob = _G_date_prob

    # ---------------------------------------------------------
    # PRODUCTS
    # ---------------------------------------------------------
    prod_idx = rng.integers(0, len(product_np), size=n)
    prods = product_np[prod_idx]
    product_keys = prods[:,0].astype(np.int64)
    unit_price = prods[:,1].astype(np.float64)
    unit_cost  = prods[:,2].astype(np.float64)

    # ---------------------------------------------------------
    # STORES -> GEO -> CURRENCY mapping
    # ---------------------------------------------------------
    store_key_arr = store_keys[rng.integers(0, len(store_keys), size=n)].astype(np.int64)

    try:
        max_store = int(max(store_to_geo.keys()))
        st2g = np.full(max_store+1, -1, dtype=np.int64)
        for k,v in store_to_geo.items():
            st2g[int(k)] = int(v)
        geo_arr = st2g[store_key_arr]

        max_geo = int(max(geo_to_currency.keys()))
        g2c = np.full(max_geo+1, -1, dtype=np.int64)
        for k,v in geo_to_currency.items():
            g2c[int(k)] = int(v)
        currency_arr = g2c[geo_arr]

    except Exception:
        geo_arr = np.array([ store_to_geo[s] for s in store_key_arr ])
        currency_arr = np.array([ geo_to_currency[g] for g in geo_arr ], dtype=np.int64)

    # ---------------------------------------------------------
    # QUANTITY
    # ---------------------------------------------------------
    qty = np.clip(rng.poisson(3, n) + 1, 1, 10).astype(np.int64)

    # --- ORDER GROUPING (robust, always produces length n arrays) ---
    avg_lines = 2.0
    order_count = max(1, int(n / avg_lines))

    suffix = np.char.zfill(rng.integers(0, 999999, order_count).astype(str), 6)
    od_idx = rng.choice(len(date_pool), size=order_count, p=date_prob)
    order_dates = date_pool[od_idx]

    order_dates_str = np.array([
        str(d.astype("datetime64[D]"))[:10].replace("-", "")
        for d in order_dates
    ])
    order_ids_str = np.char.add(order_dates_str, suffix)
    order_ids_int = order_ids_str.astype(np.int64)

    cust_idx = rng.integers(0, len(customers), order_count)
    order_customers = customers[cust_idx].astype(np.int64)

    lines_per_order = rng.choice([1,2,3,4,5], order_count, p=[0.55,0.25,0.10,0.06,0.04])

    # initial expansions (may overshoot or undershoot)
    sales_order_num = np.repeat(order_ids_str, lines_per_order)
    sales_order_num_int = np.repeat(order_ids_int, lines_per_order)
    line_num = np.concatenate([np.arange(1, c+1) for c in lines_per_order])
    customer_keys = np.repeat(order_customers, lines_per_order)
    order_dates_expanded = np.repeat(order_dates, lines_per_order)

    # pad if undershoot
    current_len = len(sales_order_num)
    if current_len < n:
        extra = n - current_len
        extra_suffix = np.char.zfill(rng.integers(0, 999999, extra).astype(str), 6)
        extra_dates = date_pool[rng.choice(len(date_pool), size=extra, p=date_prob)]

        extra_dates_str = np.array([
            str(d.astype("datetime64[D]"))[:10].replace("-", "")
            for d in extra_dates
        ])
        extra_ids_str = np.char.add(extra_dates_str, extra_suffix)
        extra_ids_int = extra_ids_str.astype(np.int64)

        sales_order_num = np.concatenate([sales_order_num, extra_ids_str])
        sales_order_num_int = np.concatenate([sales_order_num_int, extra_ids_int])
        line_num = np.concatenate([line_num, np.ones(extra, dtype=np.int64)])
        customer_keys = np.concatenate([customer_keys,
                                        customers[rng.integers(0, len(customers), extra)]])
        order_dates_expanded = np.concatenate([order_dates_expanded, extra_dates])

    # final trim to exactly n
    sales_order_num = sales_order_num[:n]
    sales_order_num_int = sales_order_num_int[:n]
    line_num = line_num[:n].astype(np.int64)
    customer_keys = customer_keys[:n].astype(np.int64)
    order_dates_expanded = order_dates_expanded[:n]

    # create date array used later by promotions and delivery
    od_np = order_dates_expanded.astype("datetime64[D]")

    # ---------------------------------------------------------
    # DELIVERY LOGIC (depends on order_id)
    # ---------------------------------------------------------
    hash_vals = sales_order_num_int

    due_offset = (hash_vals % 5).astype(np.int64) + 3
    due_date_np = od_np + due_offset.astype("timedelta64[D]")

    line_seed = (product_keys + (hash_vals % 100)) % 100
    product_seed = (hash_vals + product_keys) % 100
    order_seed   = (hash_vals % 100).astype(np.int64)

    cond_a = order_seed < 60
    cond_b = (order_seed >= 60) & (order_seed < 85) & (product_seed < 60)
    cond_c = (order_seed >= 60) & (order_seed < 85) & (product_seed >= 60)
    cond_d = order_seed >= 85

    base_offset = np.select(
        [cond_a, cond_b, cond_c, cond_d],
        [
            0,
            0,
            (line_seed % 4) + 1,
            (product_seed % 5) + 2
        ],
        default=0
    ).astype(np.int64)

    early_mask = rng.random(n) < 0.10
    early_days = rng.integers(1, 3, n)
    delivery_offset = base_offset.copy()
    delivery_offset[early_mask] = -early_days[early_mask]

    delivery_date_np = due_date_np + delivery_offset.astype("timedelta64[D]")

    delivery_status = np.where(
        delivery_date_np < due_date_np, "Early Delivery",
        np.where(delivery_date_np > due_date_np, "Delayed", "On Time")
    )

    # ---------------------------------------------------------
    # PROMOTIONS
    # ---------------------------------------------------------
    promo_keys = np.full(n, no_discount_key, dtype=np.int64)
    promo_pct  = np.zeros(n, dtype=np.float64)

    if promo_keys_all is not None and promo_keys_all.size > 0:
        od_dates_d = od_np
        active = (promo_start_all[:, None] <= od_dates_d) & (od_dates_d <= promo_end_all[:, None])
        has_active = active.any(axis=0)
        if has_active.any():
            active_idx = np.argmax(active, axis=0)
            promo_keys[has_active] = promo_keys_all[active_idx[has_active]]
            promo_pct [has_active] = promo_pct_all [active_idx[has_active]]

    # ---------------------------------------------------------
    # DISCOUNT LOGIC
    # ---------------------------------------------------------
    promo_disc = unit_price * (promo_pct / 100.0)

    rnd_pct  = rng.choice([0,5,10,15,20], n, p=[0.85,0.06,0.04,0.03,0.02])
    rnd_disc = unit_price * (rnd_pct / 100.0)

    discount_amt = np.maximum(promo_disc, rnd_disc)
    discount_amt *= rng.choice([0.90,0.95,1.00,1.05,1.10], n)
    discount_amt  = np.round(discount_amt * 4) / 4
    discount_amt  = np.minimum(discount_amt, unit_price - 0.01)

    # ---------------------------------------------------------
    # IS ORDER DELAYED
    # ---------------------------------------------------------
    is_delayed_line = (delivery_status == "Delayed").astype(np.int64)

    unique_ids, inverse_idx = np.unique(sales_order_num, return_inverse=True)
    counts = np.bincount(inverse_idx, weights=is_delayed_line, minlength=len(unique_ids))
    delayed_any = (counts > 0).astype(np.int8)
    is_order_delayed = delayed_any[inverse_idx].astype(np.int8)

    # ---------------------------------------------------------
    # FINAL PRICES
    # ---------------------------------------------------------
    factor = rng.uniform(0.43, 0.61)
    final_unit_price   = np.round(unit_price * factor, 2)
    final_unit_cost    = np.round(unit_cost  * factor, 2)
    final_discount_amt = np.round(discount_amt * factor, 2)
    final_net_price    = np.round(final_unit_price - final_discount_amt, 2)
    final_net_price    = np.clip(final_net_price, 0.01, None)

    # ---------------------------------------------------------
    # BUILD PYARROW OR PANDAS
    # ---------------------------------------------------------
    if PYARROW_AVAILABLE:
        pa_cols = {
            "OrderDate":      pa.array(od_np.astype("datetime64[D]")),
            "DueDate":        pa.array(due_date_np.astype("datetime64[D]")),
            "DeliveryDate":   pa.array(delivery_date_np.astype("datetime64[D]")),
            "StoreKey":       pa.array(store_key_arr, type=pa.int64()),
            "ProductKey":     pa.array(product_keys, type=pa.int64()),
            "PromotionKey":   pa.array(promo_keys, type=pa.int64()),
            "CurrencyKey":    pa.array(currency_arr, type=pa.int64()),
            "CustomerKey":    pa.array(customer_keys, type=pa.int64()),
            "Quantity":       pa.array(qty, type=pa.int64()),
            "NetPrice":       pa.array(final_net_price, type=pa.float64()),
            "UnitCost":       pa.array(final_unit_cost, type=pa.float64()),
            "UnitPrice":      pa.array(final_unit_price, type=pa.float64()),
            "DiscountAmount": pa.array(final_discount_amt, type=pa.float64()),
            "DeliveryStatus": pa.array(delivery_status.tolist(), type=pa.string()),
            "IsOrderDelayed": pa.array(is_order_delayed.astype(np.int8), type=pa.int8())
        }

        # include only if NOT skipping
        if not skip_cols:
            pa_cols["SalesOrderNumber"]      = pa.array(sales_order_num.tolist(), type=pa.string())
            pa_cols["SalesOrderLineNumber"]  = pa.array(line_num, type=pa.int64())

        return pa.table(pa_cols)

    else:
        df = {
            "OrderDate":      od_np.astype("datetime64[D]"),
            "DueDate":        due_date_np.astype("datetime64[D]"),
            "DeliveryDate":   delivery_date_np.astype("datetime64[D]"),
            "StoreKey":       store_key_arr,
            "ProductKey":     product_keys,
            "PromotionKey":   promo_keys,
            "CurrencyKey":    currency_arr,
            "CustomerKey":    customer_keys,
            "Quantity":       qty,
            "NetPrice":       final_net_price,
            "UnitCost":       final_unit_cost,
            "UnitPrice":      final_unit_price,
            "DiscountAmount": final_discount_amt,
            "DeliveryStatus": delivery_status,
            "IsOrderDelayed": is_order_delayed
        }

        if not skip_cols:
            df["SalesOrderNumber"]     = sales_order_num.astype(str)
            df["SalesOrderLineNumber"] = line_num

        return pd.DataFrame(df)

# -------------------------
# Worker task wrapper
# -------------------------

def _worker_task(args):
    idx, batch, seed = args
    out_folder = _G_out_folder
    file_format = _G_file_format
    row_group_size = _G_row_group_size
    compression = _G_compression
    no_discount_key = _G_no_discount_key
    delta_output_folder = _G_delta_output_folder
    write_delta = _G_write_delta
    
    print(f"WRITING CHUNK: {idx} FORMAT: {file_format} SKIP: {_G_skip_order_cols}", flush=True)
    print("WRITING CHUNK:", idx, "FORMAT:", file_format, "SKIP:", _G_skip_order_cols)
    print(f"Generating chunk {idx} ({batch:,} rows)...", flush=True)
    table_or_df = _build_chunk_table(batch, seed, no_discount_key=no_discount_key)

    # Ensure we have a pyarrow.Table when possible (for both parquet and delta)
    table_for_arrow = None
    if PYARROW_AVAILABLE and isinstance(table_or_df, pa.Table):
        table_for_arrow = table_or_df
    elif PYARROW_AVAILABLE:
        # convert pandas DF to pa.Table
        table_for_arrow = pa.Table.from_pandas(table_or_df, preserve_index=False)

    # Write CSV mode
    if file_format == "csv":
        out = os.path.join(out_folder, f"sales_chunk{idx:04d}.csv")
        if table_for_arrow is not None:
            df = table_for_arrow.to_pandas()
        else:
            df = table_or_df
        df.to_csv(out, index=False, quoting=csv.QUOTE_ALL)

    # DELTAPARQUET mode (write only delta)
    elif file_format == "deltaparquet":
        # Workers DO NOT write delta (not multiprocessing-safe)
        if not DELTA_AVAILABLE:
            raise RuntimeError("file_format='deltaparquet' but the 'deltalake' library is not installed.")

        # Ensure Arrow table
        if table_for_arrow is None:
            table_for_arrow = pa.Table.from_pandas(table_or_df, preserve_index=False)

        # Return the table to the main process for sequential delta write
        print(f"✓ Finished chunk {idx} (returned to main for delta write)", flush=True)
        return ("delta", idx, table_for_arrow)


    # Parquet mode (and possibly dual-write)
    else:
        out = os.path.join(out_folder, f"sales_chunk{idx:04d}.parquet")
        if table_for_arrow is not None:
            _write_table_or_df(table_for_arrow, out, row_group_size=row_group_size, compression=compression)
        else:
            _write_table_or_df(table_or_df, out, row_group_size=row_group_size, compression=compression)

        # optional dual-write to Delta
        if write_delta:
            if not DELTA_AVAILABLE:
                raise RuntimeError("write_delta=True but the 'deltalake' library is not installed.")
            if table_for_arrow is None:
                table_for_arrow = pa.Table.from_pandas(table_or_df, preserve_index=False)
            write_deltalake(delta_output_folder, table_for_arrow, mode="append")

    print(f"✓ Finished chunk {idx} -> {out}", flush=True)
    return out

# -------------------------
# Merge parquet files (unchanged, but uses pyarrow if available)
# -------------------------

def merge_parquet_files(out_folder, merged_file_name, delete_chunks=True):
    files = sorted(glob.glob(os.path.join(out_folder, "sales_chunk*.parquet")))
    if not files:
        print("No parquet chunk files to merge.")
        return None
    merged_path = os.path.join(out_folder, merged_file_name)
    try:
        if PYARROW_AVAILABLE:
            dataset = pq.ParquetDataset(files) if hasattr(pq, "ParquetDataset") else None
            # use dataset if supported, else read tables and concat
            if dataset is not None:
                table = dataset.read()
                pq.write_table(table, merged_path)
            else:
                dfs = [pd.read_parquet(f) for f in files]
                pd.concat(dfs, ignore_index=True).to_parquet(merged_path, index=False)
        else:
            dfs = [pd.read_parquet(f) for f in files]
            pd.concat(dfs, ignore_index=True).to_parquet(merged_path, index=False)
    except Exception:
        # fallback concat
        dfs = [pd.read_parquet(f) for f in files]
        pd.concat(dfs, ignore_index=True).to_parquet(merged_path, index=False)
    if delete_chunks:
        for f in files:
            try:
                os.remove(f)
            except:
                pass
    return merged_path

# -------------------------
# Auto-tune helper (optional): choose chunk_size so that num_chunks ~= n_workers*2
# -------------------------

def suggest_chunk_size(total_rows, target_workers=None, preferred_chunks_per_worker=2):
    if target_workers is None:
        target_workers = max(1, cpu_count() - 1)
    desired_chunks = int(target_workers * preferred_chunks_per_worker)
    return max(1_000, int(ceil(total_rows / desired_chunks)))

# -------------------------
# Main: generate_sales_fact (public)
# -------------------------

def generate_sales_fact(
    parquet_folder,
    out_folder,
    total_rows,
    chunk_size=2_000_000,
    start_date="2021-01-01",
    end_date="2025-12-31",
    row_group_size=2_000_000,
    compression="snappy",
    merge_parquet=False,
    merged_file="sales.parquet",
    delete_chunks=False,
    heavy_pct=5,
    heavy_mult=5,
    seed=42,
    file_format="parquet",
    workers=None,
    write_pyarrow=True,
    tune_chunk=False,
    write_delta=False,
    delta_output_folder=None,
    skip_order_cols=False
):
    """
    workers: None -> auto choose min(num_chunks, cpu_count()-1)
             int  -> explicit number of worker processes
    write_pyarrow: prefer pyarrow tables if True (requires pyarrow)
    tune_chunk: if True, suggests a chunk size for better parallelism
    write_delta: if True, append each chunk to a Delta Lake at delta_output_folder
    delta_output_folder: path for the delta table
    """

    if file_format == "csv":
        merge_parquet = False
    ensure_dir(out_folder)

    if delta_output_folder is None:
        delta_output_folder = os.path.join(out_folder, "delta")
    ensure_dir(delta_output_folder)

    # load inputs
    cust = load_parquet_column(os.path.join(parquet_folder, "customers.parquet"), "CustomerKey")
    customers = build_weighted_customers(cust, heavy_pct, heavy_mult, seed).astype(np.int64)

    prod_df = load_parquet_df(os.path.join(parquet_folder, "products.parquet"), ["ProductKey","UnitPrice","UnitCost"])
    product_np = prod_df.to_numpy()

    store_keys = load_parquet_column(os.path.join(parquet_folder, "stores.parquet"), "StoreKey").astype(np.int64)

    geo_df = load_parquet_df(os.path.join(parquet_folder, "geography.parquet"), ["GeographyKey", "Country", "ISOCode"])
    currency_df = pd.read_parquet(os.path.join(parquet_folder, "currency.parquet"))
    currency_df = currency_df[["CurrencyKey", "ISOCode"]]

    if "ISOCode" not in geo_df.columns:
        raise ValueError("Geography table must include ISOCode column.")
    
    geo_df = geo_df.merge(currency_df, on="ISOCode", how="left")

    if geo_df["CurrencyKey"].isna().any():
        missing = geo_df[geo_df["CurrencyKey"].isna()][["GeographyKey","Country","ISOCode"]]
        raise ValueError(f"Missing currency for some geographies:\n{missing}")
    
    geo_to_currency = dict(zip(geo_df["GeographyKey"], geo_df["CurrencyKey"]))

    store_df = load_parquet_df(os.path.join(parquet_folder, "stores.parquet"), ["StoreKey", "GeographyKey"])
    store_to_geo = dict(zip(store_df["StoreKey"], store_df["GeographyKey"]))

    promo_df = load_parquet_df(os.path.join(parquet_folder, "promotions.parquet"))
    if not promo_df.empty:
        promo_df["StartDate"] = pd.to_datetime(promo_df["StartDate"]).dt.normalize()
        promo_df["EndDate"]   = pd.to_datetime(promo_df["EndDate"]).dt.normalize()
        promo_keys_all  = promo_df["PromotionKey"].to_numpy(np.int64)
        promo_pct_all   = promo_df["DiscountPct"].to_numpy(np.float64)
        promo_start_all = promo_df["StartDate"].to_numpy("datetime64[D]")
        promo_end_all   = promo_df["EndDate"].to_numpy("datetime64[D]")
    else:
        promo_keys_all = np.array([], dtype=np.int64)
        promo_pct_all = np.array([], dtype=np.float64)
        promo_start_all = np.array([], dtype="datetime64[D]")
        promo_end_all = np.array([], dtype="datetime64[D]")

    date_pool, date_prob = build_weighted_date_pool(start_date, end_date, seed)

    # optional tuning
    if tune_chunk:
        suggested = suggest_chunk_size(total_rows, target_workers=None)
        print(f"Suggested chunk_size for your machine: {suggested:,} rows (use this to better utilize cores)")

    # build tasks
    tasks = []
    remaining = total_rows
    idx = 0
    rng_master = np.random.default_rng(seed + 1)
    while remaining > 0:
        batch = min(chunk_size, remaining)
        tasks.append((idx, batch, int(rng_master.integers(1, 1 << 30))))
        remaining -= batch
        idx += 1

    if workers is None:
        # default: don't spawn more workers than tasks; leave one CPU for OS
        max_cores = max(1, cpu_count() - 1)
        n_workers = min(len(tasks), max_cores)
    else:
        n_workers = max(1, int(workers))

    print(f"=== Generating Sales... ===\nSpawning {n_workers} worker processes...")

    # sanity print in main process (before starting Pool)
    print(f"[MAIN] skip_order_cols={skip_order_cols!r} (type={type(skip_order_cols)})", flush=True)
    # local helper value required by workers
    no_discount_key = 1

    init_args = (
        product_np,          # 1
        store_keys,          # 2
        promo_keys_all,      # 3
        promo_pct_all,       # 4
        promo_start_all,     # 5
        promo_end_all,       # 6
        customers,           # 7
        store_to_geo,        # 8
        geo_to_currency,     # 9
        date_pool,           # 10
        date_prob,           # 11
        out_folder,          # 12
        file_format,         # 13
        row_group_size,      # 14
        compression,         # 15
        no_discount_key,     # 16  (local)
        delta_output_folder, # 17
        write_delta,         # 18
        skip_order_cols      # 19  MUST be last
    )
    created = []

    with Pool(processes=n_workers, initializer=_init_worker, initargs=(init_args,)) as pool:
        for item in pool.imap_unordered(_worker_task, tasks):

            # item can be:
            #   type == "delta"   -> ( "delta", idx, pyarrow.Table )
            #   parquet/csv path  -> string

            if isinstance(item, tuple) and item[0] == "delta":
                _, idx_returned, table = item

                # Safe: Delta write must occur in main process only
                write_deltalake(delta_output_folder, table, mode="append")

            else:
                # Normal CSV or Parquet output path
                created.append(item)

    print("All chunks completed.")

    if file_format == "parquet" and merge_parquet:
        merge_parquet_files(out_folder, merged_file, delete_chunks=delete_chunks)

    return created
