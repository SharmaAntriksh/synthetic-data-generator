import os
import glob
import numpy as np
import pandas as pd
from datetime import datetime
from multiprocessing import Pool, cpu_count
from math import ceil

from src.utils.logging_utils import info, work, skip, done, stage
from .sales_worker import init_sales_worker, _worker_task
from .sales_writer import merge_parquet_files
from deltalake import write_deltalake
import pyarrow as pa


# =====================================================================
# Helpers
# =====================================================================
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_parquet_column(path, col):
    return pd.read_parquet(path, columns=[col])[col].values


def load_parquet_df(path, cols=None):
    return pd.read_parquet(path, columns=cols)


def build_weighted_customers(keys, pct, mult, seed=42):
    """Oversample heavy customers (for heavy_pct / heavy_mult)."""
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
    """
    Weighted date distribution:
      - long-term growth
      - seasonal effects
      - weekday differences
      - random spikes
      - occasional blackout days
    """
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
    yw = np.array([growth ** idx_year[y] for y in years])

    # Month weights
    month_weights = {
        1:0.82, 2:0.92, 3:1.03, 4:0.98, 5:1.07, 6:1.12,
        7:1.18, 8:1.10, 9:0.96, 10:1.22, 11:1.48, 12:1.33
    }
    mw = np.array([month_weights[m] for m in months])

    # Weekday weights
    weekday_weights = {0:0.86,1:0.91,2:1.00,3:1.12,4:1.19,5:1.08,6:0.78}
    wdw = np.array([weekday_weights[d] for d in weekdays])

    # Seasonal spikes
    spike = np.ones(n)
    for s, e, f in [(140,170,1.28),(240,260,1.35),(310,350,1.72)]:
        mask = (doy >= s) & (doy <= e)
        spike[mask] *= f

    # One-time events
    ot = np.ones(n)
    for a,b,f in [("2021-06-01","2021-10-31",0.70),("2023-02-01","2023-08-31",1.40)]:
        A = pd.to_datetime(a); B = pd.to_datetime(b)
        mask = (dates >= A) & (dates <= B)
        ot[mask] *= f

    noise = rng.uniform(0.95, 1.05, size=n)

    weights = yw * mw * wdw * spike * ot * noise

    # Blackout days
    blackout_mask = rng.random(n) < rng.uniform(0.10, 0.18)
    weights[blackout_mask] = 0

    weights /= weights.sum()

    return dates.to_numpy(dtype="datetime64[D]"), weights


def suggest_chunk_size(total_rows, target_workers=None, preferred_chunks_per_worker=2):
    if target_workers is None:
        target_workers = max(1, cpu_count() - 1)
    desired_chunks = int(target_workers * preferred_chunks_per_worker)
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
    write_delta=False,            # ← NOT USED ANYMORE
    delta_output_folder=None,
    skip_order_cols=False,
    write_pyarrow=True,
    partition_enabled=False,
    partition_cols=None
):
    """
    Clean patched version — delta uses ONLY partition writer.
    """
    
    # Resolve defaults section robustly (support "defaults" or "_defaults")
    if start_date is None or end_date is None:
        defaults_section = cfg.get("defaults") or cfg.get("_defaults")
        if not defaults_section or "dates" not in defaults_section:
            raise KeyError("Could not find defaults.dates in config. Expected 'defaults' or '_defaults' with a 'dates' mapping.")
        defaults = defaults_section["dates"]
        start_date = defaults["start"]
        end_date = defaults["end"]

    # 1) normalize delta_output_folder immediately
    if file_format == "deltaparquet":
        if delta_output_folder is None:
            delta_output_folder = os.path.join(out_folder, "delta")
        delta_output_folder = os.path.abspath(delta_output_folder)
        ensure_dir(delta_output_folder)
        ensure_dir(os.path.join(delta_output_folder, "_tmp_parts"))

    # ------------------------------------------------------------
    # Load dimensions (unchanged)
    # ------------------------------------------------------------
    # info(f"Loading customers from: {os.path.join(parquet_folder, 'customers.parquet')}")
    customers_raw = load_parquet_column(
        os.path.join(parquet_folder, "customers.parquet"),
        "CustomerKey",
    )
    customers = build_weighted_customers(customers_raw, heavy_pct, heavy_mult, seed).astype(np.int64)

    # info(f"Loading products from: {os.path.join(parquet_folder, 'products.parquet')}")
    prod_df = load_parquet_df(
        os.path.join(parquet_folder, "products.parquet"),
        ["ProductKey", "UnitPrice", "UnitCost"],
    )
    product_np = prod_df.to_numpy()

    # info(f"Loading stores from: {os.path.join(parquet_folder, 'stores.parquet')}")
    store_keys = load_parquet_column(
        os.path.join(parquet_folder, "stores.parquet"),
        "StoreKey"
    ).astype(np.int64)

    geo_path = os.path.join(parquet_folder, "geography.parquet")
    # info(f"Loading geography from: {geo_path}")

    geo_df = load_parquet_df(
        geo_path,
        ["GeographyKey", "Country", "ISOCode"]
    )
    
    # info(f"Loading currency from: {os.path.join(parquet_folder, 'currency.parquet')}")
    currency_df = pd.read_parquet(
        os.path.join(parquet_folder, "currency.parquet")
    )[["CurrencyKey", "ISOCode"]]

    geo_df = geo_df.merge(currency_df, on="ISOCode", how="left")

    if geo_df["CurrencyKey"].isna().any():
        missing_count = geo_df["CurrencyKey"].isna().sum()
        default_row = currency_df[currency_df["ISOCode"] == "USD"].iloc[0] \
            if "USD" in currency_df["ISOCode"].values else currency_df.iloc[0]
        default_currency_key = int(default_row["CurrencyKey"])
        info(f"[sales] {missing_count} geography rows missing CurrencyKey — filling with {default_currency_key}")
        geo_df["CurrencyKey"] = geo_df["CurrencyKey"].fillna(default_currency_key)

    geo_to_currency = dict(
        zip(
            geo_df["GeographyKey"].astype(np.int64),
            geo_df["CurrencyKey"].astype(np.int64),
        )
    )

    store_df = load_parquet_df(
        os.path.join(parquet_folder, "stores.parquet"),
        ["StoreKey", "GeographyKey"]
    )
    store_to_geo = dict(
        zip(
            store_df["StoreKey"].astype(np.int64),
            store_df["GeographyKey"].astype(np.int64),
        )
    )

    referenced_geos = set(map(int, store_df["GeographyKey"].unique()))
    known_geos = set(map(int, geo_df["GeographyKey"].unique()))
    missing_geos = sorted(list(referenced_geos - known_geos))
    if missing_geos:
        default_row = currency_df[currency_df["ISOCode"] == "USD"].iloc[0] \
            if "USD" in currency_df["ISOCode"].values else currency_df.iloc[0]
        default_currency_key = int(default_row["CurrencyKey"])
        info(f"[sales] WARNING: {len(missing_geos)} GeographyKey(s) missing from geography.parquet: {missing_geos}")
        info(f"[sales] Assigning default CurrencyKey={default_currency_key}")
        for mg in missing_geos:
            geo_to_currency[int(mg)] = default_currency_key

    # info(f"Loading promotions from: {os.path.join(parquet_folder, 'promotions.parquet')}")
    promo_df = pd.read_parquet(os.path.join(parquet_folder, "promotions.parquet"))
    if not promo_df.empty:
        promo_df["StartDate"] = pd.to_datetime(promo_df["StartDate"]).dt.normalize()
        promo_df["EndDate"] = pd.to_datetime(promo_df["EndDate"]).dt.normalize()

        promo_keys_all = promo_df["PromotionKey"].to_numpy(np.int64)
        promo_pct_all = promo_df["DiscountPct"].to_numpy(np.float64)
        promo_start_all = promo_df["StartDate"].to_numpy("datetime64[D]")
        promo_end_all = promo_df["EndDate"].to_numpy("datetime64[D]")
    else:
        promo_keys_all = np.array([], dtype=np.int64)
        promo_pct_all = np.array([], dtype=np.float64)
        promo_start_all = np.array([], dtype="datetime64[D]")
        promo_end_all = np.array([], dtype="datetime64[D]")

    # Weighted date pool
    date_pool, date_prob = build_weighted_date_pool(start_date, end_date, seed)

    # Chunk scheduling
    tasks = []
    remaining = total_rows
    rng_master = np.random.default_rng(seed + 1)
    seeds = rng_master.integers(1, 1 << 30, size=ceil(total_rows / chunk_size))

    idx = 0
    for s in seeds:
        batch = min(chunk_size, remaining)
        tasks.append((idx, batch, int(s)))
        remaining -= batch
        idx += 1
        if remaining <= 0:
            break

    total_chunks = len(tasks)
    completed = 0

    # worker count
    if workers is None:
        max_cores = max(1, cpu_count() - 1)
        n_workers = min(total_chunks, max_cores)
    else:
        n_workers = int(workers)

    info(f"Spawning {n_workers} worker processes...")

    no_discount_key = 1

    # Init args for workers
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
        no_discount_key,
        delta_output_folder,
        (file_format == "deltaparquet"),
        skip_order_cols,
        (file_format == "deltaparquet"),   # partition flag in worker
        partition_cols,
    )

    created_files = []
    delta_part_paths = []

    # ------------------------------------------------------------
    # MULTIPROCESSING
    # ------------------------------------------------------------
    with Pool(
        processes=n_workers,
        initializer=init_sales_worker,
        initargs=initargs
    ) as pool:
        for result in pool.imap_unordered(_worker_task, tasks):

            completed += 1

            if isinstance(result, str):
                created_files.append(result)
                work(f"[{completed}/{total_chunks}] → {os.path.basename(result)}")
                continue

            if isinstance(result, dict) and "delta_part" in result:
                delta_part_paths.append(result["delta_part"])
                work(f"[{completed}/{total_chunks}] → {os.path.basename(result['delta_part'])}")
                continue


    done("All chunks completed.")

    # ------------------------------------------------------------
    # **FINAL DELTA-PARQUET ASSEMBLY**
    # ------------------------------------------------------------
    if file_format == "deltaparquet":
        # info("MASTER: assembling partitioned Delta-Parquet dataset...")

        from .sales_writer import write_delta_partitioned

        parts_folder = os.path.join(delta_output_folder, "_tmp_parts")

        write_delta_partitioned(
            parts_folder=parts_folder,
            delta_output_folder=delta_output_folder,
            partition_cols=partition_cols
        )

        info("MASTER: Delta-Parquet partition write DONE.")
        return created_files

    # ------------------------------------------------------------
    # CSV mode — do nothing further
    # ------------------------------------------------------------
    if file_format == "csv":
        return created_files

    # ------------------------------------------------------------
    # STANDARD PARQUET MERGE
    # ------------------------------------------------------------
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

        # cleanup
        for stray in glob.iglob(os.path.join(out_folder, "sales_chunk*.parquet")):
            try:
                os.remove(stray)
            except:
                pass

    return created_files
