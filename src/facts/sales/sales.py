import os
import glob
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from math import ceil

from src.utils.logging_utils import info, work, skip, done
from .sales_worker import init_sales_worker, _worker_task
from .sales_writer import merge_parquet_files


# =====================================================================
# Helpers
# =====================================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_parquet_column(path, col):
    # Kept as pandas for safety / compatibility
    return pd.read_parquet(path, columns=[col])[col].values


def load_parquet_df(path, cols=None):
    return pd.read_parquet(path, columns=cols)


def build_weighted_customers(keys, pct, mult, seed=42):
    rng = np.random.default_rng(seed)
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


def build_weighted_date_pool(start, end, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, end, freq="D")
    n = len(dates)

    years = dates.year.values
    months = dates.month.values
    weekdays = dates.weekday.values
    doy = dates.dayofyear.values

    unique_years = np.unique(years)
    year_idx = {y: i for i, y in enumerate(unique_years)}
    growth = 1.08
    yw = np.array([growth ** year_idx[y] for y in years])

    month_weights = {
        1:0.82, 2:0.92, 3:1.03, 4:0.98, 5:1.07, 6:1.12,
        7:1.18, 8:1.10, 9:0.96, 10:1.22, 11:1.48, 12:1.33
    }
    mw = np.array([month_weights[m] for m in months])

    weekday_weights = {0:0.86,1:0.91,2:1.00,3:1.12,4:1.19,5:1.08,6:0.78}
    wdw = np.array([weekday_weights[d] for d in weekdays])

    spike = np.ones(n)
    for s, e, f in [(140,170,1.28),(240,260,1.35),(310,350,1.72)]:
        spike[(doy >= s) & (doy <= e)] *= f

    ot = np.ones(n)
    for a,b,f in [("2021-06-01","2021-10-31",0.70),("2023-02-01","2023-08-31",1.40)]:
        mask = (dates >= a) & (dates <= b)
        ot[mask] *= f

    noise = rng.uniform(0.95, 1.05, size=n)

    weights = yw * mw * wdw * spike * ot * noise
    blackout = rng.random(n) < rng.uniform(0.10, 0.18)
    weights[blackout] = 0
    weights /= weights.sum()

    return dates.to_numpy("datetime64[D]"), weights


def suggest_chunk_size(total_rows, target_workers=None, preferred_chunks_per_worker=2):
    if target_workers is None:
        target_workers = max(1, cpu_count() - 1)
    desired_chunks = target_workers * preferred_chunks_per_worker
    return max(1_000, int(ceil(total_rows / desired_chunks)))


# =====================================================================
# Main Fact Generation
# =====================================================================
def generate_sales_fact(
    cfg,
    parquet_folder,
    out_folder,
    total_rows,
    chunk_size=2_000_000,
    start_date=None,
    end_date=None,
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
    tune_chunk=False,
    write_delta=False,   # legacy flag (ignored)
    delta_output_folder=None,
    skip_order_cols=False,
    write_pyarrow=True,
    partition_enabled=False,
    partition_cols=None
):
    # ------------------------------------------------------------
    # Resolve defaults
    # ------------------------------------------------------------
    if start_date is None or end_date is None:
        defaults_section = cfg.get("defaults") or cfg.get("_defaults")
        if not defaults_section or "dates" not in defaults_section:
            raise KeyError("Missing defaults.dates in config")
        defaults = defaults_section["dates"]
        start_date = defaults["start"]
        end_date = defaults["end"]

    # ------------------------------------------------------------
    # Delta folder setup
    # ------------------------------------------------------------
    if file_format == "deltaparquet":
        if delta_output_folder is None:
            delta_output_folder = os.path.join(out_folder, "delta")
        delta_output_folder = os.path.abspath(delta_output_folder)
        ensure_dir(delta_output_folder)
        ensure_dir(os.path.join(delta_output_folder, "_tmp_parts"))

    # ------------------------------------------------------------
    # Load dimensions
    # ------------------------------------------------------------
    customers_raw = load_parquet_column(
        os.path.join(parquet_folder, "customers.parquet"),
        "CustomerKey"
    )
    customers = build_weighted_customers(
        customers_raw, heavy_pct, heavy_mult, seed
    ).astype(np.int64)

    prod_df = load_parquet_df(
        os.path.join(parquet_folder, "products.parquet"),
        ["ProductKey", "UnitPrice", "UnitCost"]
    )
    product_np = prod_df.to_numpy()

    store_keys = load_parquet_column(
        os.path.join(parquet_folder, "stores.parquet"),
        "StoreKey"
    ).astype(np.int64)

    geo_df = load_parquet_df(
        os.path.join(parquet_folder, "geography.parquet"),
        ["GeographyKey", "Country", "ISOCode"]
    )

    currency_df = load_parquet_df(
        os.path.join(parquet_folder, "currency.parquet"),
        ["CurrencyKey", "ISOCode"]
    )

    geo_df = geo_df.merge(currency_df, on="ISOCode", how="left")

    if geo_df["CurrencyKey"].isna().any():
        default_currency = int(currency_df.iloc[0]["CurrencyKey"])
        geo_df["CurrencyKey"] = geo_df["CurrencyKey"].fillna(default_currency)

    geo_to_currency = dict(
        zip(
            geo_df["GeographyKey"].astype(np.int64),
            geo_df["CurrencyKey"].astype(np.int64)
        )
    )

    store_df = load_parquet_df(
        os.path.join(parquet_folder, "stores.parquet"),
        ["StoreKey", "GeographyKey"]
    )

    store_to_geo = dict(
        zip(
            store_df["StoreKey"].astype(np.int64),
            store_df["GeographyKey"].astype(np.int64)
        )
    )

    promo_df = load_parquet_df(
        os.path.join(parquet_folder, "promotions.parquet")
    )

    if promo_df.empty:
        promo_keys_all = np.array([], dtype=np.int64)
        promo_pct_all = np.array([], dtype=np.float64)
        promo_start_all = np.array([], dtype="datetime64[D]")
        promo_end_all = np.array([], dtype="datetime64[D]")
    else:
        promo_df["StartDate"] = pd.to_datetime(promo_df["StartDate"]).dt.normalize()
        promo_df["EndDate"] = pd.to_datetime(promo_df["EndDate"]).dt.normalize()

        promo_keys_all = promo_df["PromotionKey"].to_numpy(np.int64)
        promo_pct_all = promo_df["DiscountPct"].to_numpy(np.float64)
        promo_start_all = promo_df["StartDate"].to_numpy("datetime64[D]")
        promo_end_all = promo_df["EndDate"].to_numpy("datetime64[D]")

    date_pool, date_prob = build_weighted_date_pool(start_date, end_date, seed)

    # ------------------------------------------------------------
    # Chunk scheduling
    # ------------------------------------------------------------
    rng_master = np.random.default_rng(seed + 1)
    total_chunks = ceil(total_rows / chunk_size)
    seeds = rng_master.integers(1, 1 << 30, size=total_chunks)

    tasks = []
    remaining = total_rows
    for idx, s in enumerate(seeds):
        if remaining <= 0:
            break
        batch = min(chunk_size, remaining)
        tasks.append((idx, batch, int(s)))
        remaining -= batch

    # ------------------------------------------------------------
    # Worker count
    # ------------------------------------------------------------
    if workers is None:
        n_workers = min(len(tasks), max(1, cpu_count() - 1))
    else:
        n_workers = int(workers)

    info(f"Spawning {n_workers} worker processes...")

    # ------------------------------------------------------------
    # Worker init args
    # ------------------------------------------------------------
    initargs = (
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
        1,  # no_discount_key
        delta_output_folder,
        file_format == "deltaparquet",
        skip_order_cols,
        file_format == "deltaparquet",
        partition_cols,
    )

    created_files = []

    # ------------------------------------------------------------
    # MULTIPROCESSING
    # ------------------------------------------------------------
    with Pool(
        processes=n_workers,
        initializer=init_sales_worker,
        initargs=initargs
    ) as pool:

        completed = 0
        for result in pool.imap_unordered(_worker_task, tasks):
            completed += 1
            if isinstance(result, str):
                created_files.append(result)
                work(f"[{completed}/{len(tasks)}] → {os.path.basename(result)}")

    done("All chunks completed.")

    # ------------------------------------------------------------
    # FINAL ASSEMBLY
    # ------------------------------------------------------------
    if file_format == "deltaparquet":
        from .sales_writer import write_delta_partitioned
        write_delta_partitioned(
            parts_folder=os.path.join(delta_output_folder, "_tmp_parts"),
            delta_output_folder=delta_output_folder,
            partition_cols=partition_cols
        )
        return created_files

    if file_format == "csv":
        return created_files

    if file_format == "parquet":
        parquet_chunks = sorted(
            f for f in glob.glob(os.path.join(out_folder, "sales_chunk*.parquet"))
            if os.path.isfile(f)
        )
        if parquet_chunks:
            merge_parquet_files(
                parquet_chunks,
                os.path.join(out_folder, merged_file),
                delete_after=True
            )

    return created_files
