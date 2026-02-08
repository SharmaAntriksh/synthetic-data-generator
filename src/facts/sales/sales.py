from __future__ import annotations

import glob
import os
from math import ceil
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.utils.logging_utils import done, info, skip, work
from .sales_logic.globals import State
from .sales_worker import _worker_task, init_sales_worker
from .sales_writer import merge_parquet_files
from .output_paths import OutputPaths, TABLE_SALES, TABLE_SALES_ORDER_DETAIL, TABLE_SALES_ORDER_HEADER


# =====================================================================
# Helpers
# =====================================================================

def _as_np(x, dtype=None) -> np.ndarray:
    """Works for pandas Series/Index AND for already-materialized numpy arrays."""
    return np.asarray(x, dtype=dtype)


def _bool_mask(x) -> np.ndarray:
    """Ensure we always have a numpy bool mask."""
    return np.asarray(x, dtype=bool)


def ensure_dir(path: Union[str, Path]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def load_parquet_column(path: Union[str, Path], col: str) -> np.ndarray:
    """
    Load a single parquet column as a numpy array.
    """
    s = pd.read_parquet(str(path), columns=[col])[col]
    return _as_np(s)


def load_parquet_df(path: Union[str, Path], cols: Optional[Sequence[str]] = None) -> pd.DataFrame:
    return pd.read_parquet(str(path), columns=list(cols) if cols is not None else None)


def _cfg_get(cfg: Any, path: Sequence[str], default: Any = None) -> Any:
    cur = cfg
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _int_or(value: Any, default: int) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _bool_or(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y"}:
            return True
        if v in {"false", "0", "no", "n"}:
            return False
    return bool(default)


def _str_or(value: Any, default: str) -> str:
    if value is None:
        return default
    s = str(value).strip()
    return s if s else default


def _apply_cfg_default(current: Any, default: Any, cfg_value: Any) -> Any:
    """
    Treat cfg as source-of-truth defaults when call-site leaves args at their defaults.
    """
    if cfg_value is None:
        return current
    return cfg_value if current == default else current


def _resolve_date_range(cfg: dict, start_date: Optional[str], end_date: Optional[str]) -> Tuple[str, str]:
    """
    Priority:
      explicit args
      cfg.sales.override.dates.{start,end}
      cfg.defaults.dates.{start,end} (or cfg._defaults.dates)
    """
    if start_date is not None and end_date is not None:
        return str(start_date), str(end_date)

    defaults_section = cfg.get("defaults") or cfg.get("_defaults")
    defaults_dates = (defaults_section or {}).get("dates") if isinstance(defaults_section, dict) else None
    if not isinstance(defaults_dates, dict):
        raise KeyError("Missing defaults.dates in config")

    ov_dates = _cfg_get(cfg, ["sales", "override", "dates"], default={})
    ov_dates = ov_dates if isinstance(ov_dates, dict) else {}

    if start_date is None:
        start_date = ov_dates.get("start") or defaults_dates.get("start")
    if end_date is None:
        end_date = ov_dates.get("end") or defaults_dates.get("end")

    if not start_date or not end_date:
        raise KeyError("Could not resolve start/end dates from config")

    return str(start_date), str(end_date)


def _resolve_seed(cfg: dict, seed: Any, default_seed: int = 42) -> int:
    """
    Priority:
      explicit seed param
      cfg.sales.override.seed
      cfg.sales.seed
      cfg.defaults.seed
      fallback
    """
    if seed is not None:
        return _int_or(seed, default_seed)

    ov = _cfg_get(cfg, ["sales", "override", "seed"], default=None)
    if ov is not None:
        return _int_or(ov, default_seed)

    s = _cfg_get(cfg, ["sales", "seed"], default=None)
    if s is not None:
        return _int_or(s, default_seed)

    d = _cfg_get(cfg, ["defaults", "seed"], default=None)
    if d is None:
        d = _cfg_get(cfg, ["_defaults", "seed"], default=None)

    if d is not None:
        return _int_or(d, default_seed)

    return int(default_seed)


def _normalize_dt_any(x) -> Union[pd.Series, pd.DatetimeIndex]:
    """
    Normalize date-like inputs to midnight.
    Handles Series (has .dt) and DatetimeIndex (has .normalize()).
    """
    dt = pd.to_datetime(x, errors="coerce")
    return dt.dt.normalize() if hasattr(dt, "dt") else dt.normalize()


def build_weighted_date_pool(start: str, end: str, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a weighted daily date pool with realistic seasonality.
    Returns:
      date_pool: datetime64[D] array
      date_prob: normalized probabilities
    """
    rng = np.random.default_rng(_int_or(seed, 42))

    dates = pd.date_range(start, end, freq="D")
    n = len(dates)
    if n <= 0:
        raise ValueError("Date range produced an empty pool")

    years = _as_np(dates.year)
    months = _as_np(dates.month)
    weekdays = _as_np(dates.weekday)
    doy = _as_np(dates.dayofyear)

    # Year growth (vectorized)
    _, inv = np.unique(years, return_inverse=True)
    growth = 1.08
    yw = np.power(growth, inv).astype(np.float64)

    # Month seasonality (index 1..12)
    month_w = np.ones(13, dtype=np.float64)
    month_w[1:] = np.array([0.82, 0.92, 1.03, 0.98, 1.07, 1.12, 1.18, 1.10, 0.96, 1.22, 1.48, 1.33], dtype=np.float64)
    mw = month_w[months]

    # Weekday effect (0=Mon..6=Sun)
    weekday_w = np.array([0.86, 0.91, 1.00, 1.12, 1.19, 1.08, 0.78], dtype=np.float64)
    wdw = weekday_w[weekdays]

    # Promotional spikes (day-of-year windows)
    spike = np.ones(n, dtype=np.float64)
    for s, e, f in ((140, 170, 1.28), (240, 260, 1.35), (310, 350, 1.72)):
        spike[(doy >= s) & (doy <= e)] *= f

    # One-off trends (explicit windows)
    ot = np.ones(n, dtype=np.float64)
    for a, b, f in (("2021-06-01", "2021-10-31", 0.70), ("2023-02-01", "2023-08-31", 1.40)):
        mask = (dates >= a) & (dates <= b)
        if bool(np.any(mask)):
            ot[_bool_mask(mask)] *= f  # <-- FIX: no mask.to_numpy()

    noise = rng.uniform(0.95, 1.05, size=n).astype(np.float64)

    weights = yw * mw * wdw * spike * ot * noise

    # Random blackout days (scalar blackout rate)
    blackout_rate = rng.uniform(0.10, 0.18)
    blackout = rng.random(n) < blackout_rate
    weights[_bool_mask(blackout)] = 0.0

    total = float(weights.sum())
    if total <= 0:
        weights[:] = 1.0 / n
    else:
        weights /= total

    return dates.to_numpy("datetime64[D]"), weights


def suggest_chunk_size(total_rows: int, target_workers: Optional[int] = None, preferred_chunks_per_worker: int = 2) -> int:
    if target_workers is None:
        target_workers = max(1, cpu_count() - 1)
    desired_chunks = max(1, int(target_workers) * int(preferred_chunks_per_worker))
    return max(1_000, int(ceil(int(total_rows) / desired_chunks)))


def batch_tasks(tasks: List[Tuple[int, int, int]], batch_size: int) -> List[List[Tuple[int, int, int]]]:
    if batch_size <= 1:
        return [[t] for t in tasks]
    return [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]


def _normalize_nullable_int_month(arr: Any, n: int) -> np.ndarray:
    """
    Normalize CustomerEndMonth into int64 with -1 meaning "no end".
    """
    if arr is None:
        return np.full(n, -1, dtype=np.int64)

    s = pd.Series(arr)
    v = pd.to_numeric(s, errors="coerce").fillna(-1).astype("int64", copy=False).to_numpy()
    v[v < 0] = -1
    if v.shape[0] != n:
        v = np.resize(v, n)
    return v


def _resolve_partitioning(cfg: dict, partition_enabled: bool, partition_cols: Optional[Sequence[str]]) -> Tuple[bool, Optional[List[str]]]:
    sales_cfg = cfg.get("sales") if isinstance(cfg, dict) else None
    sales_cfg = sales_cfg if isinstance(sales_cfg, dict) else {}

    cfg_enabled = None
    if "partition_enabled" in sales_cfg:
        cfg_enabled = sales_cfg.get("partition_enabled")
    elif "partitioning" in sales_cfg and isinstance(sales_cfg.get("partitioning"), dict):
        cfg_enabled = sales_cfg["partitioning"].get("enabled")

    cfg_cols = None
    if "partition_cols" in sales_cfg:
        cfg_cols = sales_cfg.get("partition_cols")
    elif "partitioning" in sales_cfg and isinstance(sales_cfg.get("partitioning"), dict):
        cfg_cols = sales_cfg["partitioning"].get("columns")

    partition_enabled = _apply_cfg_default(
        partition_enabled,
        False,
        _bool_or(cfg_enabled, False) if cfg_enabled is not None else None
    )

    if partition_cols is None:
        if isinstance(cfg_cols, list) and cfg_cols:
            partition_cols = list(cfg_cols)
        else:
            partition_cols = None
    else:
        partition_cols = list(partition_cols)

    return bool(partition_enabled), partition_cols


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
    heavy_pct=5,           # legacy (kept for API compatibility; ignored if CustomerBaseWeight exists)
    heavy_mult=5,          # legacy (kept for API compatibility; ignored if CustomerBaseWeight exists)
    seed=42,
    file_format="parquet",
    workers=None,
    tune_chunk=False,
    write_delta=False,     # legacy (ignored)
    delta_output_folder=None,
    skip_order_cols=False,
    write_pyarrow=True,
    partition_enabled=False,
    partition_cols=None
):
    # ------------------------------------------------------------
    # Normalize cfg defaults (cfg is source-of-truth when call-site omits)
    # ------------------------------------------------------------
    cfg = cfg if isinstance(cfg, dict) else {}
    sales_cfg = cfg.get("sales") if isinstance(cfg.get("sales"), dict) else {}

    file_format_cfg = sales_cfg.get("file_format")
    if file_format_cfg is not None:
        file_format = _apply_cfg_default(file_format, "parquet", _str_or(file_format_cfg, "parquet").lower())

    merge_parquet = _apply_cfg_default(
        merge_parquet, False,
        _bool_or(sales_cfg.get("merge_parquet"), merge_parquet) if "merge_parquet" in sales_cfg else None
    )
    merged_file = _apply_cfg_default(merged_file, "sales.parquet", sales_cfg.get("merged_file") if "merged_file" in sales_cfg else None)

    delete_chunks = _apply_cfg_default(
        delete_chunks, False,
        _bool_or(sales_cfg.get("delete_chunks"), delete_chunks) if "delete_chunks" in sales_cfg else None
    )

    chunk_size = _apply_cfg_default(chunk_size, 2_000_000, _int_or(sales_cfg.get("chunk_size"), chunk_size) if "chunk_size" in sales_cfg else None)
    row_group_size = _apply_cfg_default(row_group_size, 2_000_000, _int_or(sales_cfg.get("row_group_size"), row_group_size) if "row_group_size" in sales_cfg else None)
    compression = _apply_cfg_default(compression, "snappy", sales_cfg.get("compression") if "compression" in sales_cfg else None)
    workers = _apply_cfg_default(workers, None, sales_cfg.get("workers") if "workers" in sales_cfg else None)
    tune_chunk = _apply_cfg_default(tune_chunk, False, _bool_or(sales_cfg.get("tune_chunk"), tune_chunk) if "tune_chunk" in sales_cfg else None)
    write_pyarrow = _apply_cfg_default(write_pyarrow, True, _bool_or(sales_cfg.get("write_pyarrow"), write_pyarrow) if "write_pyarrow" in sales_cfg else None)

    start_date, end_date = _resolve_date_range(cfg, start_date, end_date)
    seed = _resolve_seed(cfg, seed, default_seed=42)
    partition_enabled, partition_cols = _resolve_partitioning(cfg, partition_enabled, partition_cols)

    # ------------------------------------------------------------
    # Paths / folders
    # ------------------------------------------------------------
    parquet_folder_p = Path(str(parquet_folder))
    out_folder_p = Path(str(out_folder))

    # Resolve delta folder early (so OutputPaths is built with final values)
    if file_format == "deltaparquet":
        if delta_output_folder is None:
            delta_output_folder = str(out_folder_p / "delta")
        delta_output_folder = os.path.abspath(str(delta_output_folder))

    output_paths = OutputPaths(
        file_format=file_format,
        out_folder=str(out_folder_p),
        merged_file=str(merged_file),
        delta_output_folder=(str(delta_output_folder) if file_format == "deltaparquet" else None),
    )

    sales_output = _str_or(sales_cfg.get("sales_output"), "sales").lower()
    if sales_output not in {"sales", "sales_order", "both"}:
        raise ValueError(f"Invalid sales_output: {sales_output}")

    needs_order_cols = sales_output in {"sales_order", "both"}

    # Safeguard: if user generates BOTH and keeps order columns in Sales, output balloons.
    if sales_output == "both" and not bool(skip_order_cols):
        info(
            "Config: sales_output=both and skip_order_cols=false -> Sales will include order columns "
            "AND SalesOrderHeader/Detail will also be written. Expect much larger output. "
            "If you want Sales slimmer, set skip_order_cols=true."
        )

    tables: list[str] = []
    if sales_output in {"sales", "both"}:
        tables.append(TABLE_SALES)
    if sales_output in {"sales_order", "both"}:
        tables += [TABLE_SALES_ORDER_DETAIL, TABLE_SALES_ORDER_HEADER]

    for t in tables:
        output_paths.ensure_dirs(t)

    # Normalize delta_output_folder after OutputPaths decides defaults/abspath (if your class does that)
    delta_output_folder = output_paths.delta_output_folder

    # ------------------------------------------------------------
    # Optional auto chunk sizing
    # ------------------------------------------------------------
    total_rows = _int_or(total_rows, 0)
    if total_rows <= 0:
        skip("No sales rows to generate (total_rows <= 0).")
        return []

    if workers is None:
        n_workers_planned = max(1, cpu_count() - 1)
    else:
        n_workers_planned = max(1, _int_or(workers, cpu_count() - 1))

    if tune_chunk:
        chunk_size = suggest_chunk_size(total_rows, target_workers=n_workers_planned, preferred_chunks_per_worker=2)

    chunk_size = max(1_000, _int_or(chunk_size, 1_000_000))

    # ------------------------------------------------------------
    # Load dimensions
    # ------------------------------------------------------------
    customers_path = parquet_folder_p / "customers.parquet"

    cust_cols = ["CustomerKey", "IsActiveInSales", "CustomerStartMonth", "CustomerEndMonth"]
    cust_df = load_parquet_df(customers_path, cust_cols)

    if cust_df.empty:
        raise RuntimeError("customers.parquet is empty; cannot generate sales")

    customer_keys = _as_np(cust_df["CustomerKey"], np.int64)
    is_active_in_sales = _as_np(cust_df["IsActiveInSales"], np.int64)

    if "CustomerStartMonth" in cust_df.columns:
        customer_start_month = _as_np(cust_df["CustomerStartMonth"], np.int64)
    else:
        customer_start_month = np.zeros(len(customer_keys), dtype=np.int64)

    if "CustomerEndMonth" in cust_df.columns:
        customer_end_month = _normalize_nullable_int_month(_as_np(cust_df["CustomerEndMonth"]), len(customer_keys))
    else:
        customer_end_month = np.full(len(customer_keys), -1, dtype=np.int64)

    # Load customer weight column separately (robust)
    customer_base_weight = None
    for wcol in ("CustomerBaseWeight", "CustomerWeight"):
        try:
            w = pd.read_parquet(str(customers_path), columns=[wcol])[wcol]
            customer_base_weight = _as_np(w, np.float64)
            break
        except Exception:
            pass

    if customer_base_weight is None:
        info("CustomerBaseWeight not found; customer sampling will be uniform unless chunk_builder applies other logic.")
    else:
        info("CustomerBaseWeight loaded; chunk_builder can use it for weighted sampling.")

    # Products: respect runner-bound active_product_np
    if hasattr(State, "active_product_np") and State.active_product_np is not None:
        product_np = State.active_product_np
    else:
        prod_df = load_parquet_df(parquet_folder_p / "products.parquet", ["ProductKey", "UnitPrice", "UnitCost"])
        product_np = prod_df.to_numpy()

    # Stores: read ONCE (keys + geography)
    store_df = load_parquet_df(parquet_folder_p / "stores.parquet", ["StoreKey", "GeographyKey"])
    store_keys = _as_np(store_df["StoreKey"], np.int64)
    store_to_geo = dict(zip(_as_np(store_df["StoreKey"], np.int64), _as_np(store_df["GeographyKey"], np.int64)))

    # Geography + currency mapping
    geo_df = load_parquet_df(parquet_folder_p / "geography.parquet", ["GeographyKey", "ISOCode"])
    currency_df = load_parquet_df(parquet_folder_p / "currency.parquet", ["CurrencyKey", "ToCurrency"])

    geo_df = geo_df.merge(currency_df, left_on="ISOCode", right_on="ToCurrency", how="left")
    if geo_df["CurrencyKey"].isna().any():
        default_currency = int(currency_df.iloc[0]["CurrencyKey"])
        geo_df["CurrencyKey"] = geo_df["CurrencyKey"].fillna(default_currency)

    geo_to_currency = dict(zip(_as_np(geo_df["GeographyKey"], np.int64), _as_np(geo_df["CurrencyKey"], np.int64)))

    # Promotions
    promo_df = load_parquet_df(parquet_folder_p / "promotions.parquet")

    if promo_df.empty:
        promo_keys_all = np.array([], dtype=np.int64)
        promo_pct_all = np.array([], dtype=np.float64)
        promo_start_all = np.array([], dtype="datetime64[D]")
        promo_end_all = np.array([], dtype="datetime64[D]")
    else:
        promo_start = _normalize_dt_any(promo_df["StartDate"])
        promo_end = _normalize_dt_any(promo_df["EndDate"])

        promo_keys_all = _as_np(promo_df["PromotionKey"], np.int64)
        promo_pct_all = _as_np(promo_df["DiscountPct"], np.float64)
        promo_start_all = _as_np(promo_start, "datetime64[D]")
        promo_end_all = _as_np(promo_end, "datetime64[D]")

    # Weighted date pool (deterministic)
    date_pool, date_prob = build_weighted_date_pool(start_date, end_date, seed)

    # ------------------------------------------------------------
    # Chunk scheduling
    # ------------------------------------------------------------
    total_chunks = int(ceil(total_rows / chunk_size))
    rng_master = np.random.default_rng(seed + 1)
    seeds = rng_master.integers(1, 1 << 30, size=total_chunks, dtype=np.int64)

    tasks: List[Tuple[int, int, int]] = []
    remaining = total_rows
    for idx, s in enumerate(seeds):
        if remaining <= 0:
            break
        batch = min(chunk_size, remaining)
        tasks.append((idx, int(batch), int(s)))
        remaining -= batch

    if not tasks:
        skip("No sales rows to generate.")
        return []

    # ------------------------------------------------------------
    # Worker count
    # ------------------------------------------------------------
    if workers is None:
        n_workers = min(len(tasks), max(1, cpu_count() - 1))
    else:
        n_workers = min(len(tasks), max(1, _int_or(workers, cpu_count() - 1)))

    info(f"Spawning {n_workers} worker processes...")

    # ------------------------------------------------------------
    # Worker configuration (keep keys stable for compatibility)
    # ------------------------------------------------------------
    worker_cfg = dict(
        product_np=product_np,
        store_keys=store_keys,
        promo_keys_all=promo_keys_all,
        promo_pct_all=promo_pct_all,
        promo_start_all=promo_start_all,
        promo_end_all=promo_end_all,

        # Backward compat: keep 'customers' as keys array (no duplication)
        customers=customer_keys,

        # New contract for chunk_builder lifecycle eligibility
        customer_keys=customer_keys,
        customer_is_active_in_sales=is_active_in_sales,
        customer_start_month=customer_start_month,
        customer_end_month=customer_end_month,
        customer_base_weight=customer_base_weight,

        store_to_geo=store_to_geo,
        geo_to_currency=geo_to_currency,
        date_pool=date_pool,
        date_prob=date_prob,

        output_paths=output_paths.to_dict() if hasattr(output_paths, "to_dict") else {
            "file_format": output_paths.file_format,
            "out_folder": output_paths.out_folder,
            "merged_file": output_paths.merged_file,
            "delta_output_folder": output_paths.delta_output_folder,
        },
        file_format=file_format,
        out_folder=str(out_folder_p),
        row_group_size=_int_or(row_group_size, 2_000_000),
        compression=_str_or(compression, "snappy"),
        sales_output=sales_output,
        
        # legacy knobs (kept)
        heavy_pct=heavy_pct,
        heavy_mult=heavy_mult,
        no_discount_key=1,

        delta_output_folder=delta_output_folder,
        write_delta=write_delta,
        skip_order_cols=(False if needs_order_cols else bool(skip_order_cols)),
        skip_order_cols_requested=bool(skip_order_cols),

        write_pyarrow=write_pyarrow,

        partition_enabled=partition_enabled,
        partition_cols=partition_cols,

        models_cfg=State.models_cfg,
    )

    # Track outputs per logical table (Sales / SalesOrderDetail / SalesOrderHeader)
    created_by_table: Dict[str, List[Any]] = {t: [] for t in tables}
    created_files: List[str] = []  # flat list of chunk file paths (csv/parquet), kept for backward-compat

    def _record_chunk_result(r: Any, completed_units: int, total_units: int) -> None:
        """
        r can be:
          - str (legacy 'sales' mode): path
          - dict(table_name -> write_result): multi-table modes
              write_result is str for csv/parquet, or {"part":..., "rows":...} for delta
        """
        if isinstance(r, str):
            created_by_table.setdefault(TABLE_SALES, []).append(r)
            created_files.append(r)
            work(f"[{completed_units}/{total_units}] -> {os.path.basename(r)}")
            return

        if isinstance(r, dict):
            for table_name, val in r.items():
                created_by_table.setdefault(table_name, []).append(val)
                if isinstance(val, str):
                    created_files.append(val)
                    work(f"[{completed_units}/{total_units}] -> {os.path.basename(val)}")
                elif isinstance(val, dict) and "part" in val:
                    work(f"[{completed_units}/{total_units}] -> {table_name}:{val['part']}")
            return

        # Unknown / unexpected return type: ignore quietly
        return

    # ------------------------------------------------------------
    # Multiprocessing (batched)
    # ------------------------------------------------------------
    CHUNKS_PER_CALL = 2
    batched_tasks = batch_tasks(tasks, CHUNKS_PER_CALL)

    total_units = len(tasks)
    completed_units = 0

    with Pool(
        processes=n_workers,
        initializer=init_sales_worker,
        initargs=(worker_cfg,),
    ) as pool:
        for result in pool.imap_unordered(_worker_task, batched_tasks):
            if isinstance(result, list):
                for r in result:
                    completed_units += 1
                    _record_chunk_result(r, completed_units, total_units)
            else:
                completed_units += 1
                _record_chunk_result(result, completed_units, total_units)

    done("All chunks completed.")

    # ------------------------------------------------------------
    # Final assembly (TABLE-AWARE)
    # ------------------------------------------------------------
    if file_format == "deltaparquet":
        # IMPORTANT: use the table-aware delta writer (expects parts per table)
        from .writers.sales_delta import write_delta_partitioned

        for t in tables:
            parts_dir = output_paths.delta_parts_dir(t)
            delta_dir = output_paths.delta_table_dir(t)

            # If a table wasn't generated for some reason, skip cleanly
            if not os.path.exists(parts_dir):
                info(f"No delta parts folder for {t}: {parts_dir} (skipping)")
                continue

            write_delta_partitioned(
                parts_folder=parts_dir,
                delta_output_folder=delta_dir,
                partition_cols=partition_cols,
                table_name=t,  # <-- key
            )

        return created_files

    if file_format == "csv":
        return created_files

    if file_format == "parquet":
        if merge_parquet:
            # Prefer globbing per-table rather than filtering created_files
            from .writers.parquet_merge import merge_parquet_files

            for t in tables:
                chunks = sorted(
                    f for f in glob.glob(output_paths.chunk_glob(t, "parquet"))
                    if os.path.isfile(f)
                )

                if not chunks:
                    info(f"No parquet chunks found for {t}; skipping merge")
                    continue

                merge_parquet_files(
                    chunks,
                    output_paths.merged_path(t),
                    delete_after=bool(delete_chunks),
                    table_name=t,  # <-- key
                )

        return created_files

    return created_files

