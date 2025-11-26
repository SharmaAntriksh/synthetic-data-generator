#!/usr/bin/env python3
"""
updated_generator_v19.py
v19: builds on v18 and enforces promotion timelines when assigning PromotionKey.

Changes vs v18:
- Loads full promotions parquet (StartDate/EndDate).
- Builds promotion start/end arrays and PromotionKey/DiscountPct arrays.
- generate_chunk_df now accepts promo_records and assigns promotions only when order date is inside StartDate..EndDate.
- Falls back to No Discount PromotionKey (1) when no active promotions.
"""

import os
import argparse
import glob
import shutil
import numpy as np
import pandas as pd
from datetime import datetime
import csv


# -------------------- Helpers --------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def load_parquet_column(path, col):
    return pd.read_parquet(path, columns=[col])[col].values

def load_parquet_df(path, cols=None):
    return pd.read_parquet(path, columns=cols)

def build_weighted_customers(keys, pct, mult, seed=42):
    rng = np.random.default_rng(seed)
    if len(keys) == 0:
        return keys
    mask = rng.random(len(keys)) < (pct / 100.0)
    heavy = keys[mask]
    normal = keys[~mask]
    parts = []
    if heavy.size > 0:
        parts.append(np.repeat(heavy, mult))
    if normal.size > 0:
        parts.append(normal)
    arr = np.concatenate(parts)
    rng.shuffle(arr)
    return arr

# -------------------- Weighted date pool --------------------
def build_weighted_date_pool(start, end, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)

    years = dates.year.values
    months = dates.month.values
    weekdays = dates.weekday.values
    doy = dates.dayofyear.values

    # Year growth (original, not copied)
    year_weights_map = {}
    unique_years = sorted(set(years))
    base = 1.0
    growth = 1.08  # ~8% year-over-year growth by default
    for i, y in enumerate(unique_years):
        year_weights_map[y] = base * (growth ** i)
    yw = np.array([year_weights_map[y] for y in years], dtype=float)

    # Month seasonality (retail-like pattern; new values)
    month_weights = {
        1:0.82, 2:0.92, 3:1.03, 4:0.98, 5:1.07, 6:1.12,
        7:1.18, 8:1.10, 9:0.96, 10:1.22, 11:1.48, 12:1.33
    }
    mw = np.array([month_weights[m] for m in months], dtype=float)

    # Weekday multipliers (new values)
    weekday_weights = {0:0.86, 1:0.91, 2:1.00, 3:1.12, 4:1.19, 5:1.08, 6:0.78}
    wdw = np.array([weekday_weights[d] for d in weekdays], dtype=float)

    # Annual spikes (day-of-year ranges) - new values, repeat each year
    spike = np.ones(n, dtype=float)
    annual_spikes = [
        (140, 170, 1.28),  # pre-summer sale
        (240, 260, 1.35),  # back-to-school
        (310, 350, 1.72)   # festival/holiday season
    ]
    for s, e, f in annual_spikes:
        mask = (doy >= s) & (doy <= e)
        spike[mask] *= f

    # One-time spikes (year-specific windows) - new sample events
    ot = np.ones(n, dtype=float)
    one_time_spans = [
        ("2021-06-01", "2021-10-31", 0.70),  # supply dip
        ("2023-02-01", "2023-08-31", 1.40)   # high demand
    ]
    for a, b, f in one_time_spans:
        A = pd.to_datetime(a); B = pd.to_datetime(b)
        mask = (dates >= A) & (dates <= B)
        ot[mask] *= f

    # Noise (small random variation)
    noise = rng.uniform(0.95, 1.05, size=n)

    weights = yw * mw * wdw * spike * ot * noise

    # --- Improved Random 5–10% No-Sales Days ---
    rng = np.random.default_rng(seed)

    final_no_sales = np.zeros(n, dtype=bool)
    df_dates = pd.DataFrame({
        "date": dates,
        "year": years,
        "month": months
    })

    # For each year, draw a fresh random blackout fraction (5%–10%)
    for y, grp_year in df_dates.groupby("year"):

        # year-specific blackout fraction
        frac_y = rng.uniform(0.10, 0.18)

        # total blackout days for this year
        total_days_y = len(grp_year)
        blackout_days_y = max(1, int(total_days_y * frac_y))

        # distribute across months IN THIS YEAR
        for m, grp_month in grp_year.groupby("month"):
            idx = grp_month.index.values
            # proportional blackout count per month
            k = int(len(idx) / total_days_y * blackout_days_y)
            if k > 0:
                chosen = rng.choice(idx, size=k, replace=False)
                final_no_sales[chosen] = True

        # if rounding leaves shortage/excess → adjust inside the year
        current_ys = np.where(df_dates["year"] == y)[0]
        delta = blackout_days_y - final_no_sales[current_ys].sum()
        if delta > 0:  # add missing
            extra = rng.choice(current_ys[~final_no_sales[current_ys]], size=delta, replace=False)
            final_no_sales[extra] = True
        elif delta < 0:  # remove extra
            remove = rng.choice(current_ys[final_no_sales[current_ys]], size=-delta, replace=False)
            final_no_sales[remove] = False

    # zero out weights
    weights[final_no_sales] = 0

    weights = weights / weights.sum()

    return np.array(dates.to_pydatetime()), weights

# -------------------- Chunk generator --------------------
def generate_chunk_df(n, date_pool, date_prob, product_np, store_keys, promo_records, customers, seed, no_discount_key=1):
    """
    promo_records: list of dicts (each with PromotionKey, DiscountPct, StartDate, EndDate, ...)
    no_discount_key: PromotionKey to use when no promotions active (default 1)
    """
    rng = np.random.default_rng(seed)

    # --- Date selection (order-level handled later)
    # will use order-level dates based on weighted date pool via sampling with date_prob

    # --- Products & stores
    prod_idx = rng.integers(0, len(product_np), size=n)
    prods = product_np[prod_idx] if False else product_np[prod_idx]  # placeholder to keep variable name used below
    prods = product_np[prod_idx]
    product_keys = prods[:,0].astype(int)
    unit_price = prods[:,1].astype(float)
    unit_cost = prods[:,2].astype(float)
    store_key_arr = store_keys[rng.integers(0, len(store_keys), size=n)]

    # --- Build promo lookup arrays for fast date-based selection
    # Convert promo_records into numpy arrays
    if promo_records and len(promo_records) > 0:
        promo_df_local = pd.DataFrame(promo_records)
        # ensure StartDate/EndDate are datetimes
        promo_df_local["StartDate"] = pd.to_datetime(promo_df_local["StartDate"])
        promo_df_local["EndDate"] = pd.to_datetime(promo_df_local["EndDate"])
        promo_keys_all = promo_df_local["PromotionKey"].to_numpy(dtype=int)
        promo_dct_all = promo_df_local["DiscountPct"].to_numpy(dtype=float)
        promo_start_all = promo_df_local["StartDate"].to_numpy(dtype="datetime64[D]")
        promo_end_all = promo_df_local["EndDate"].to_numpy(dtype="datetime64[D]")

        # normalize DiscountPct into percentage (0-100) if stored as 0-1
        if promo_dct_all.max() <= 1.0:
            promo_dct_all = promo_dct_all * 100.0
    else:
        promo_keys_all = np.array([], dtype=int)
        promo_dct_all = np.array([], dtype=float)
        promo_start_all = np.array([], dtype="datetime64[D]")
        promo_end_all = np.array([], dtype="datetime64[D]")

    # --- Quantity per line (realistic distribution using Poisson fallback)
    qty = rng.poisson(lam=3, size=n) + 1
    qty = np.clip(qty, 1, 10)   # ensure quantity is between 1 and 10

    # --- Order grouping (consistent CustomerKey per order)
    avg_lines_per_order = 2.0
    order_count = max(1, int(n / avg_lines_per_order))

    suffix = np.char.zfill(rng.integers(0, 999999, size=order_count).astype(str), 6)
    # sample order-level date indices using weighted date probability
    order_date_idx = rng.choice(len(date_pool), size=order_count, p=date_prob)
    order_dates = date_pool[order_date_idx]
    order_date_str = np.array([d.strftime("%Y%m%d") for d in order_dates])
    sales_order_nums = np.char.add(order_date_str, suffix)

    # assign customer per order (consistent)
    order_customer_idx = rng.integers(0, len(customers), size=order_count)
    order_customer_keys = customers[order_customer_idx]

    # lines per order based on a skewed distribution (not uniform)
    lines_per_order = rng.choice([1,2,3,4,5], size=order_count, p=[0.55,0.25,0.10,0.06,0.04])

    # expand order-level arrays to row-level
    sales_order_num = np.repeat(sales_order_nums, lines_per_order)[:n]
    line_num = np.concatenate([np.arange(1, c+1) for c in lines_per_order])[:n]
    customer_keys_per_order = np.repeat(order_customer_keys, lines_per_order)[:n]
    order_dates_expanded = np.repeat(order_dates, lines_per_order)[:n]


    # --- fix: ensure enough rows
    if len(sales_order_num) < n:
        shortage = n - len(sales_order_num)
        extra_suffix = np.char.zfill(rng.integers(0, 999999, size=shortage).astype(str), 6)
        extra_idx = rng.choice(len(date_pool), size=shortage, p=date_prob)
        extra_dates = date_pool[extra_idx]
        extra_date_str = np.array([d.strftime("%Y%m%d") for d in extra_dates])
        extra_orders = np.char.add(extra_date_str, extra_suffix)
        extra_customers = customers[rng.integers(0, len(customers), size=shortage)]
        extra_line_nums = np.ones(shortage, dtype=int)
        sales_order_num = np.concatenate([sales_order_num, extra_orders])
        customer_keys_per_order = np.concatenate([customer_keys_per_order, extra_customers])
        order_dates_expanded = np.concatenate([order_dates_expanded, extra_dates])
        line_num = np.concatenate([line_num, extra_line_nums])
    # ensure arrays trimmed to n
    sales_order_num = sales_order_num[:n]
    line_num = line_num[:n]
    customer_keys = customer_keys_per_order[:n]
    order_level_dates = pd.Series(pd.to_datetime(order_dates_expanded))

    # --- Due / Delivery logic
    vhash = np.vectorize(hash)
    due_offset = (vhash(sales_order_num) % 5) + 3
    due_date = pd.Series(order_level_dates + pd.to_timedelta(due_offset, unit="D"))

    order_seed = vhash(sales_order_num) % 100
    product_seed = vhash(sales_order_num.astype(str) + product_keys.astype(str)) % 100
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

    early_mask = rng.random(n) < 0.10
    early_days = rng.integers(1, 3, size=n)
    delivery_offset = base_offset.copy()
    delivery_offset[early_mask] = -early_days[early_mask]
    delivery_date = pd.Series(due_date + pd.to_timedelta(delivery_offset, unit="D"))

    delivery_status = np.where(
        delivery_date < due_date, "Early Delivery",
        np.where(delivery_date > due_date, "Delayed", "On Time")
    )

    # --- Promotion assignment based on promotion timelines ---
    promo_keys = np.zeros(n, dtype=int)
    promo_pct = np.zeros(n, dtype=float)

    # convert order dates to numpy datetime64[D]
    order_dates_np = pd.to_datetime(order_dates_expanded).values.astype("datetime64[D]")

    if promo_keys_all.size > 0:
        # vectorized-ish per-row selection (still loops across rows but uses numpy arrays for masks)
        for i, od in enumerate(order_dates_np):
            # boolean mask of active promotions for this order date
            mask = (promo_start_all <= od) & (od <= promo_end_all)
            if mask.any():
                # choose randomly among active promos
                choices = np.nonzero(mask)[0]
                chosen_idx = rng.integers(0, len(choices))
                sel = choices[chosen_idx]
                promo_keys[i] = int(promo_keys_all[sel])
                promo_pct[i] = float(promo_dct_all[sel])
            else:
                promo_keys[i] = int(no_discount_key)
                promo_pct[i] = 0.0
    else:
        promo_keys[:] = int(no_discount_key)
        promo_pct[:] = 0.0

    # --- ensure promo_pct is on 0-100 scale (some sources store 0-1)
    if promo_pct.max() <= 1.0:
        promo_pct = promo_pct * 100.0

    # --- Discount logic (quantized + rounded to 0.25)
    promo_disc = unit_price * (promo_pct / 100.0)
    rnd_pct = rng.choice([0,5,10,15,20], size=n, p=[0.85,0.06,0.04,0.03,0.02])
    rnd_disc = unit_price * (rnd_pct / 100.0)
    discount_amt = np.maximum(promo_disc, rnd_disc)
    noise_levels = np.array([0.90, 0.95, 1.00, 1.05, 1.10])
    multipliers = rng.choice(noise_levels, size=n)
    discount_amt = discount_amt * multipliers
    discount_amt = np.round(discount_amt * 4.0) / 4.0
    discount_amt = np.minimum(discount_amt, unit_price - 0.01)
    net_price = unit_price - discount_amt

    # --- Build DataFrame (dates as date only)
    df = pd.DataFrame({
        "SalesOrderNumber": sales_order_num.astype(str),
        "SalesOrderLineNumber": line_num,

        "OrderDate": order_level_dates.dt.date,
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


    df["IsOrderDelayed"] = df.groupby("SalesOrderNumber")["DeliveryStatus"].transform(lambda x: int((x == "Delayed").any()))

    # --- Reduce monetary amounts slightly ---
    rng = np.random.default_rng(seed)
    price_factor = rng.uniform(0.43, 0.61)   # Reduce prices by 3%–7%

    df["UnitPrice"] = np.round(df["UnitPrice"] * price_factor, 2)
    df["UnitCost"] = np.round(df["UnitCost"] * price_factor, 2)
    df["DiscountAmount"] = np.round(df["DiscountAmount"] * price_factor, 2)

    df["NetPrice"] = np.round(df["UnitPrice"] - df["DiscountAmount"], 2)
    df["NetPrice"] = df["NetPrice"].clip(lower=0.01)

    return df

# -------------------- Parquet write (pyarrow) --------------------
def write_parquet(df, path, row_group_size=1000000, compression="snappy"):
    try:
        import pyarrow as pa, pyarrow.parquet as pq
        table = pa.Table.from_pandas(df, preserve_index=False)
        pq.write_table(table, path, compression=compression, row_group_size=row_group_size)
        return True
    except Exception as e:
        # fallback to pandas/fastparquet if pyarrow unavailable
        try:
            df.to_parquet(path, index=False, compression=compression)
            return True
        except Exception as e2:
            print("Parquet write failed:", e, e2)
            return False

# -------------------- Merge helper --------------------
def merge_parquet_files(out_folder, merged_file_name, delete_chunks=True):
    files = sorted(glob.glob(os.path.join(out_folder, "sales_chunk*.parquet")))
    if not files:
        print("No parquet chunk files to merge.")
        return None
    merged_path = os.path.join(out_folder, merged_file_name)
    try:
        import pyarrow.dataset as ds, pyarrow.parquet as pq
        dataset = ds.dataset(files, format="parquet")
        table = dataset.to_table()
        pq.write_table(table, merged_path)
        print(f"Merged {len(files)} files to: {merged_path}")
    except Exception as e:
        print("pyarrow merge failed:", e, "falling back to pandas concat")
        try:
            dfs = [pd.read_parquet(f) for f in files]
            big = pd.concat(dfs, ignore_index=True)
            big.to_parquet(merged_path, index=False)
            print(f"Merged {len(files)} files to: {merged_path} (pandas fallback)")
        except Exception as e2:
            print("Fallback merge failed:", e2)
            return None
    if delete_chunks:
        for f in files:
            try:
                os.remove(f)
            except Exception as e:
                print(f"Warning: failed to delete {f}: {e}")
                
    return merged_path

# -------------------- Main --------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--parquet-folder", required=True)
    p.add_argument("--out-folder", required=True)
    p.add_argument("--total-rows", type=int, required=True)
    p.add_argument("--chunk-size", type=int, default=2000000, help="Rows per chunk (default 2,000,000)")
    p.add_argument("--start-date", required=True)
    p.add_argument("--end-date", required=True)
    p.add_argument("--file-format", choices=["csv","parquet"], default="parquet")
    p.add_argument("--row-group-size", type=int, default=2000000, help="Parquet row group size (default 2,000,000)")
    p.add_argument("--compression", default="snappy")
    p.add_argument("--merge-parquet", choices=["yes","no"], default="no")
    p.add_argument("--merged-file", default="sales.parquet")
    p.add_argument("--delete-chunks-after-merge", action="store_true")
    p.add_argument("--heavy-pct", type=int, default=5)
    p.add_argument("--heavy-mult", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    ensure_dir(args.out_folder)

    # load dims
    cust = load_parquet_column(os.path.join(args.parquet_folder, "customers.parquet"), "CustomerKey")
    customers = build_weighted_customers(cust, args.heavy_pct, args.heavy_mult, args.seed)

    prod_df = load_parquet_df(os.path.join(args.parquet_folder, "products.parquet"), ["ProductKey","UnitPrice","UnitCost"])
    product_np = prod_df.to_numpy()

    store_keys = load_parquet_column(os.path.join(args.parquet_folder, "stores.parquet"), "StoreKey")

    # ---- Load promotions fully (with StartDate/EndDate)
    promo_df = load_parquet_df(os.path.join(args.parquet_folder, "promotions.parquet"))
    # Ensure StartDate/EndDate exist and are parsed
    promo_df["StartDate"] = pd.to_datetime(promo_df["StartDate"])
    promo_df["EndDate"] = pd.to_datetime(promo_df["EndDate"])
    # Convert to records for passing into chunk generator
    promo_records = promo_df.to_dict("records")

    # build weighted date pool using hybrid model
    date_pool, date_prob = build_weighted_date_pool(args.start_date, args.end_date, seed=args.seed)

    remaining = args.total_rows
    idx = 0
    created = []

    print("Starting generation: chunksize=", args.chunk_size, "row_group=", args.row_group_size)
    while remaining > 0:
        batch = min(args.chunk_size, remaining)
        print(f"Generating chunk {idx} ({batch} rows)...")
        df = generate_chunk_df(batch, date_pool, date_prob, product_np, store_keys, promo_records, customers, np.random.randint(1,1<<30), no_discount_key=1)
        if args.file_format == "csv":
            out = os.path.join(args.out_folder, f"sales_chunk{idx:04d}.csv")
            df.to_csv(out, index=False)
        else:
            out = os.path.join(args.out_folder, f"sales_chunk{idx:04d}.parquet")
            ok = write_parquet(df, out, row_group_size=args.row_group_size, compression=args.compression)
            if not ok:
                print("Failed to write parquet for chunk", idx)
                return
        created.append(out)
        print("Wrote", out)
        remaining -= batch
        idx += 1

    if args.file_format == "parquet" and args.merge_parquet == "yes":
        merged = merge_parquet_files(args.out_folder, args.merged_file, delete_chunks=args.delete_chunks_after_merge)
        if merged:
            print("Merged parquet available at:", merged)
        else:
            print("Merging failed or none merged.")

    print("Generation complete. Created chunks:", len(created))


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
    file_format="parquet"  # <-- NEW
):
    ensure_dir(out_folder)

    # Load customers
    cust = load_parquet_column(f"{parquet_folder}/customers.parquet", "CustomerKey")
    customers = build_weighted_customers(cust, heavy_pct, heavy_mult, seed)

    # Load products
    prod_df = load_parquet_df(f"{parquet_folder}/products.parquet",
                              ["ProductKey","UnitPrice","UnitCost"])
    product_np = prod_df.to_numpy()

    # Load stores
    store_keys = load_parquet_column(f"{parquet_folder}/stores.parquet", "StoreKey")

    # Load promotions
    promo_df = load_parquet_df(f"{parquet_folder}/promotions.parquet")
    promo_df["StartDate"] = pd.to_datetime(promo_df["StartDate"])
    promo_df["EndDate"] = pd.to_datetime(promo_df["EndDate"])
    promo_records = promo_df.to_dict("records")

    # Build weighted date pool
    date_pool, date_prob = build_weighted_date_pool(start_date, end_date, seed=seed)

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
            promo_records,
            customers,
            np.random.randint(1, 1 << 30),
            no_discount_key=1
        )

        # --- NEW: CSV or Parquet output ---
        if file_format == "csv":
            out = os.path.join(out_folder, f"sales_chunk{idx:04d}.csv")
            df.to_csv(out, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
        else:
            out = os.path.join(out_folder, f"sales_chunk{idx:04d}.parquet")
            write_parquet(df, out, row_group_size=row_group_size, compression=compression)


        created.append(out)
        remaining -= batch
        idx += 1

    # Only merge parquet files
    if file_format == "parquet" and merge_parquet:
        merge_parquet_files(out_folder, merged_file, delete_chunks=delete_chunks)

        # if delete_chunks:
        #     for f in os.listdir(out_folder):
        #         if f.startswith("sales_chunk") and f.endswith(".parquet"):
        #             os.remove(os.path.join(out_folder, f))

    return created


if __name__ == "__main__":
    main()
