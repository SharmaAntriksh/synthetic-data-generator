#!/usr/bin/env python3

import os
import glob
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
import csv


# ============================================================
# Basic Helpers
# ============================================================

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_parquet_column(path, col):
    return pd.read_parquet(path, columns=[col])[col].values


def load_parquet_df(path, cols=None):
    return pd.read_parquet(path, columns=cols)


# ============================================================
# Weighted customer selection
# ============================================================

def build_weighted_customers(keys, pct, mult, seed=42):
    """Randomly selects pct% heavy buyers, repeats them mult times."""
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


# ============================================================
# Weighted date pool
# ============================================================

def build_weighted_date_pool(start, end, seed=42):
    rng = np.random.default_rng(seed)

    dates = pd.date_range(start, end, freq="D")
    n = len(dates)

    years = dates.year.values
    months = dates.month.values
    weekdays = dates.weekday.values
    doy = dates.dayofyear.values

    # Year-over-year growth
    unique_years = np.unique(years)
    idx_year = {y: i for i, y in enumerate(unique_years)}
    growth = 1.08
    yw = np.array([growth ** idx_year[y] for y in years], float)

    # Monthly seasonality
    month_weights = {
        1:0.82, 2:0.92, 3:1.03, 4:0.98, 5:1.07, 6:1.12,
        7:1.18, 8:1.10, 9:0.96, 10:1.22, 11:1.48, 12:1.33
    }
    mw = np.array([month_weights[m] for m in months], float)

    # Weekday weights
    weekday_weights = {
        0:0.86, 1:0.91, 2:1.00, 3:1.12, 4:1.19, 5:1.08, 6:0.78
    }
    wdw = np.array([weekday_weights[d] for d in weekdays], float)

    # Annual spikes
    spike = np.ones(n)
    for s, e, f in [(140,170,1.28), (240,260,1.35), (310,350,1.72)]:
        mask = (doy >= s) & (doy <= e)
        spike[mask] *= f

    # One-time seasonal windows
    ot = np.ones(n)
    for a, b, f in [
        ("2021-06-01","2021-10-31",0.70),
        ("2023-02-01","2023-08-31",1.40)
    ]:
        A = pd.to_datetime(a); B = pd.to_datetime(b)
        mask = (dates >= A) & (dates <= B)
        ot[mask] *= f

    # Noise
    noise = rng.uniform(0.95, 1.05, size=n)

    # Combine weights
    weights = yw * mw * wdw * spike * ot * noise

    # Blackout days 10–18%
    blackout_mask = rng.random(n) < rng.uniform(0.10, 0.18)
    weights[blackout_mask] = 0

    weights /= weights.sum()

    return np.array(dates.to_pydatetime()), weights


# ============================================================
# Chunk Generator
# ============================================================

def generate_chunk_df(
    n,
    date_pool,
    date_prob,
    product_np,
    store_keys,
    promo_keys_all,
    promo_pct_all,
    promo_start_all,
    promo_end_all,
    customers,
    seed,
    no_discount_key=1
):
    rng = np.random.default_rng(seed)

    # ----------------- Products -----------------
    prod_idx = rng.integers(0, len(product_np), size=n)
    prods = product_np[prod_idx]

    product_keys = prods[:, 0].astype(int)
    unit_price  = prods[:, 1].astype(float)
    unit_cost   = prods[:, 2].astype(float)

    # ----------------- Stores -----------------
    store_key_arr = store_keys[rng.integers(0, len(store_keys), size=n)]

    # ----------------- Quantities -----------------
    qty = np.clip(rng.poisson(lam=3, size=n) + 1, 1, 10)

    # ----------------- Order grouping -----------------
    avg_lines = 2.0
    order_count = max(1, int(n / avg_lines))

    suffix = np.char.zfill(rng.integers(0, 999999, order_count).astype(str), 6)
    od_idx = rng.choice(len(date_pool), size=order_count, p=date_prob)
    order_dates = date_pool[od_idx]

    order_ids = np.char.add(
        np.array([d.strftime("%Y%m%d") for d in order_dates]),
        suffix
    )

    cust_idx = rng.integers(0, len(customers), order_count)
    order_customers = customers[cust_idx]

    lines_per_order = rng.choice(
        [1,2,3,4,5],
        order_count,
        p=[0.55,0.25,0.10,0.06,0.04]
    )

    # Expand to line level
    sales_order_num = np.repeat(order_ids, lines_per_order)[:n]
    line_num = np.concatenate([np.arange(1, c+1) for c in lines_per_order])[:n]
    customer_keys = np.repeat(order_customers, lines_per_order)[:n]
    order_dates_expanded = np.repeat(order_dates, lines_per_order)[:n]

    # If short, patch
    if len(sales_order_num) < n:
        extra = n - len(sales_order_num)
        extra_dates = date_pool[rng.choice(len(date_pool), extra, p=date_prob)]
        extra_ids = (
            np.array([d.strftime("%Y%m%d") for d in extra_dates]) +
            np.char.zfill(rng.integers(0, 999999, extra).astype(str), 6)
        )

        sales_order_num = np.concatenate([sales_order_num, extra_ids])
        customer_keys = np.concatenate([customer_keys,
                customers[rng.integers(0, len(customers), extra)]])
        line_num = np.concatenate([line_num, np.ones(extra, int)])
        order_dates_expanded = np.concatenate([order_dates_expanded, extra_dates])

    # Trim to exact size
    sales_order_num = sales_order_num[:n]
    line_num = line_num[:n]
    customer_keys = customer_keys[:n]
    order_dates_expanded = order_dates_expanded[:n]

    # ============================================================
    # Delivery Logic
    # ============================================================

    hash_vals = np.frompyfunc(hash, 1, 1)(sales_order_num).astype(np.int64)

    due_offset = (hash_vals % 5) + 3
    order_dt64 = pd.Series(pd.to_datetime(order_dates_expanded))
    due_date = pd.Series(order_dt64 + pd.to_timedelta(due_offset, unit="D"))

    order_seed = (hash_vals % 100)
    product_seed = (hash_vals + product_keys) % 100
    line_seed = (line_num + product_keys) % 100

    base_offset = np.select(
        [
            order_seed < 60,
            (order_seed >= 60) & (order_seed < 85) & (product_seed < 60),
            (order_seed >= 60) & (order_seed < 85) & (product_seed >= 60),
            order_seed >= 85
        ],
        [0, 0, (line_seed % 4) + 1, (product_seed % 5) + 2],
        default=0
    )

    # Early deliveries
    early_mask = rng.random(n) < 0.10
    early_days = rng.integers(1, 3, n)
    delivery_offset = base_offset.copy()
    delivery_offset[early_mask] = -early_days[early_mask]

    delivery_date = pd.Series(due_date + pd.to_timedelta(delivery_offset, unit="D"))

    delivery_status = np.where(
        delivery_date < due_date, "Early Delivery",
        np.where(delivery_date > due_date, "Delayed", "On Time")
    )

    # ============================================================
    # Promotions (vectorized)
    # ============================================================

    promo_keys = np.full(n, no_discount_key, int)
    promo_pct = np.zeros(n, float)

    if promo_keys_all.size > 0:
        od_np = pd.to_datetime(order_dates_expanded).values.astype("datetime64[D]")

        active = (
            (promo_start_all[:, None] <= od_np) &
            (od_np <= promo_end_all[:, None])
        )

        has_active = active.any(axis=0)
        active_idx = np.argmax(active, axis=0)

        promo_keys[has_active] = promo_keys_all[active_idx[has_active]]
        promo_pct[has_active] = promo_pct_all[active_idx[has_active]]

    # ============================================================
    # Discount Logic
    # ============================================================

    promo_disc = unit_price * (promo_pct / 100.0)
    rnd_pct = rng.choice([0,5,10,15,20], n, p=[0.85,0.06,0.04,0.03,0.02])
    rnd_disc = unit_price * (rnd_pct / 100.0)

    discount_amt = np.maximum(promo_disc, rnd_disc)
    discount_amt *= rng.choice([0.90,0.95,1.00,1.05,1.10], n)
    discount_amt = np.round(discount_amt * 4) / 4
    discount_amt = np.minimum(discount_amt, unit_price - 0.01)

    net_price = unit_price - discount_amt

    # ============================================================
    # DataFrame
    # ============================================================

    df = pd.DataFrame({
        "SalesOrderNumber": sales_order_num.astype(str),
        "SalesOrderLineNumber": line_num,
        "OrderDate": order_dt64.dt.date,
        "DueDate": due_date.dt.date,
        "DeliveryDate": delivery_date.dt.date,
        "StoreKey": store_key_arr,
        "ProductKey": product_keys,
        "PromotionKey": promo_keys,
        "CurrencyKey": 1,
        "CustomerKey": customer_keys,
        "Quantity": qty,
        "NetPrice": np.round(net_price, 2),
        "UnitCost": np.round(unit_cost, 2),
        "UnitPrice": np.round(unit_price, 2),
        "DiscountAmount": np.round(discount_amt, 2),
        "DeliveryStatus": delivery_status
    })

    df["IsOrderDelayed"] = df.groupby("SalesOrderNumber")["DeliveryStatus"] \
                             .transform(lambda x: int((x == "Delayed").any()))

    # ============================================================
    # Price Reduction  (comment fixed — logic unchanged)
    # ============================================================

    price_factor = np.random.default_rng(seed).uniform(0.43, 0.61)
    df["UnitPrice"] = np.round(df["UnitPrice"] * price_factor, 2)
    df["UnitCost"] = np.round(df["UnitCost"] * price_factor, 2)
    df["DiscountAmount"] = np.round(df["DiscountAmount"] * price_factor, 2)
    df["NetPrice"] = np.round(df["UnitPrice"] - df["DiscountAmount"], 2).clip(0.01)

    return df


# ============================================================
# Parquet Writer
# ============================================================

def write_parquet(df, path, row_group_size=1_000_000, compression="snappy"):
    try:
        import pyarrow as pa, pyarrow.parquet as pq
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(
            table,
            path,
            compression=compression,
            row_group_size=row_group_size
        )
        return True
    except Exception:
        try:
            df.to_parquet(path, index=False, compression=compression)
            return True
        except Exception as e2:
            print("Parquet write failed:", e2)
            return False


# ============================================================
# Merge Parquet Chunks
# ============================================================

def merge_parquet_files(out_folder, merged_file_name, delete_chunks=True):
    files = sorted(glob.glob(os.path.join(out_folder, "sales_chunk*.parquet")))
    if not files:
        print("No parquet chunk files to merge.")
        return None

    merged_path = os.path.join(out_folder, merged_file_name)

    try:
        import pyarrow.dataset as ds, pyarrow.parquet as pq
        dataset = ds.dataset(files, format="parquet")
        pq.write_table(dataset.to_table(), merged_path)
    except Exception:
        dfs = [pd.read_parquet(f) for f in files]
        pd.concat(dfs, ignore_index=True).to_parquet(merged_path, index=False)

    if delete_chunks:
        for f in files:
            try: os.remove(f)
            except: pass

    return merged_path


# ============================================================
# Main Function
# ============================================================

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
    file_format="parquet"
):
    # CSV mode must never merge parquet
    if file_format == "csv":
        merge_parquet = False

    ensure_dir(out_folder)

    # Load customers
    cust = load_parquet_column(
        f"{parquet_folder}/customers.parquet",
        "CustomerKey"
    )
    customers = build_weighted_customers(cust, heavy_pct, heavy_mult, seed)

    # Products
    prod_df = load_parquet_df(
        f"{parquet_folder}/products.parquet",
        ["ProductKey","UnitPrice","UnitCost"]
    )
    product_np = prod_df.to_numpy()

    # Stores
    store_keys = load_parquet_column(
        f"{parquet_folder}/stores.parquet",
        "StoreKey"
    )

    # Promotions
    promo_df = load_parquet_df(f"{parquet_folder}/promotions.parquet")
    promo_df["StartDate"] = promo_df["StartDate"].values.astype("datetime64[D]")
    promo_df["EndDate"]   = promo_df["EndDate"].values.astype("datetime64[D]")

    promo_keys_all  = promo_df["PromotionKey"].to_numpy(int)
    promo_pct_all   = promo_df["DiscountPct"].to_numpy(float)
    promo_start_all = promo_df["StartDate"].to_numpy("datetime64[D]")
    promo_end_all   = promo_df["EndDate"].to_numpy("datetime64[D]")

    # Weighted date pool
    date_pool, date_prob = build_weighted_date_pool(start_date, end_date, seed)

    # Chunk loop
    remaining = total_rows
    idx = 0
    created = []

    while remaining > 0:
        batch = min(chunk_size, remaining)

        df = generate_chunk_df(
            batch,
            date_pool,
            date_prob,
            product_np,
            store_keys,
            promo_keys_all,
            promo_pct_all,
            promo_start_all,
            promo_end_all,
            customers,
            np.random.randint(1, 1 << 30)
        )

        if file_format == "csv":
            out = f"{out_folder}/sales_chunk{idx:04d}.csv"
            df.to_csv(out, index=False, quoting=csv.QUOTE_ALL)
        else:
            out = f"{out_folder}/sales_chunk{idx:04d}.parquet"
            write_parquet(
                df,
                out,
                row_group_size=row_group_size,
                compression=compression
            )

        created.append(out)
        remaining -= batch
        idx += 1

    # Merge parquet chunks
    if file_format == "parquet" and merge_parquet:
        merge_parquet_files(
            out_folder,
            merged_file,
            delete_chunks=delete_chunks
        )

    return created
