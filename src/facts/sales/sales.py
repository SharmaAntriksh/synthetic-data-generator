from __future__ import annotations

import glob
import os
from math import ceil
from multiprocessing import cpu_count
from src.facts.sales.sales_worker import PoolRunSpec, iter_imap_unordered

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.utils.logging_utils import done, info, skip, work
from .sales_logic import State
from .sales_worker import _worker_task, init_sales_worker
from .sales_writer import merge_parquet_files
from .output_paths import (
    OutputPaths,
    TABLE_SALES,
    TABLE_SALES_ORDER_DETAIL,
    TABLE_SALES_ORDER_HEADER,
    TABLE_SALES_RETURN,   # NEW
)


from dataclasses import dataclass
from typing import Any, Optional

@dataclass(frozen=True)
class TableOutputs:
    table: str
    file_format: str
    chunks: list[Any]                 # list[str] for csv/parquet; list[{"part":..,"rows":..}] for delta
    merged_path: Optional[str] = None # parquet only
    delta_table_dir: Optional[str] = None  # delta only
    delta_parts_dir: Optional[str] = None  # delta only

@dataclass(frozen=True)
class SalesRunManifest:
    sales_output: str
    file_format: str
    out_folder: str
    tables: dict[str, TableOutputs]


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
    partition_cols=None,
    return_manifest: bool = False,
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
    skip_order_cols = _apply_cfg_default(
        skip_order_cols,
        False,
        _bool_or(sales_cfg.get("skip_order_cols"), skip_order_cols) if "skip_order_cols" in sales_cfg else None,
    )

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
    # ------------------------------------------------------------
    # Returns (optional)
    # ------------------------------------------------------------
    facts_enabled = _cfg_get(cfg, ["facts", "enabled"], default=[])
    facts_enabled = facts_enabled if isinstance(facts_enabled, list) else []

    returns_cfg = cfg.get("returns") if isinstance(cfg.get("returns"), dict) else {}
    returns_enabled = _bool_or(returns_cfg.get("enabled"), False)

    # If facts.enabled is used, treat it as an additional “feature gate”
    if facts_enabled:
        returns_enabled = bool(returns_enabled and ("returns" in {str(x).lower() for x in facts_enabled}))

    returns_rate = float(returns_cfg.get("return_rate", 0.0))
    returns_max_lag_days = int(returns_cfg.get("max_days_after_sale", returns_cfg.get("returns_max_lag_days", 60)))

    # Safeguard: if user generates BOTH and keeps order columns in Sales, output balloons.
    if sales_output == "both" and not bool(skip_order_cols):
        info("Config: both + skip_order_cols=false -> Sales includes order cols.")
        info("Note: output will be large; set skip_order_cols=true for slimmer Sales.")

    # Keep "requested" vs "effective" separate so we can warn+continue.
    returns_enabled_requested = bool(returns_enabled)
    returns_enabled_effective = bool(returns_enabled)

    # Allow returns for ALL modes (sales / sales_order / both),
    # but if sales_output == "sales" and skip_order_cols == True, we cannot derive returns
    # (no order identifiers), so disable returns and warn.
    if returns_enabled_requested and sales_output == "sales" and bool(skip_order_cols):
        info(
            "WARNING: returns.enabled=true with sales_output='sales' and skip_order_cols=true "
            "=> SalesReturn will be skipped (needs SalesOrderNumber/SalesOrderLineNumber). "
            "Sales generation will continue."
        )
        returns_enabled_effective = False

    tables: list[str] = []
    if sales_output in {"sales", "both"}:
        tables.append(TABLE_SALES)
    if sales_output in {"sales_order", "both"}:
        tables += [TABLE_SALES_ORDER_DETAIL, TABLE_SALES_ORDER_HEADER]

    # Returns table is independent of which sales family tables you output
    # (as long as returns_enabled_effective == True).
    if returns_enabled_effective:
        tables.append(TABLE_SALES_RETURN)

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
    product_brand_key = None
    products_path = parquet_folder_p / "products.parquet"

    def _brand_codes_from_series(s: pd.Series) -> np.ndarray:
        # Guarantee no NA => no -1 codes
        s2 = s.fillna("Unknown").astype(str)
        codes, _ = pd.factorize(s2, sort=True)
        return np.asarray(codes, dtype=np.int64)

    if getattr(State, "active_product_np", None) is not None:
        product_np = State.active_product_np

        try:
            brand_df = load_parquet_df(products_path, ["ProductKey", "Brand"])
            brand_df = brand_df.drop_duplicates("ProductKey", keep="first")

            brand_df["ProductKey"] = brand_df["ProductKey"].astype("int64", copy=False)
            brand_df["_BrandKey"] = _brand_codes_from_series(brand_df["Brand"])

            # Map ProductKey -> BrandKey
            brand_map = pd.Series(
                brand_df["_BrandKey"].to_numpy(),
                index=brand_df["ProductKey"].to_numpy(),
            )

            active_keys = np.asarray(product_np[:, 0], dtype=np.int64)
            bk = brand_map.reindex(active_keys).to_numpy(dtype="float64")

            invalid = (~np.isfinite(bk)) | (bk < 0)
            if np.any(invalid):
                info("Brand mapping missing/invalid for some ProductKeys; disabling brand_popularity for this run.")
                product_brand_key = None
            else:
                product_brand_key = bk.astype(np.int64, copy=False)

        except Exception:
            info("Could not load/derive Brand from products.parquet; disabling brand_popularity for this run.")
            product_brand_key = None

    else:
        # Full product path: keep backward compatibility if Brand is absent
        try:
            prod_df = load_parquet_df(products_path, ["ProductKey", "UnitPrice", "UnitCost", "Brand"])
            product_np = prod_df[["ProductKey", "UnitPrice", "UnitCost"]].to_numpy()

            codes = _brand_codes_from_series(prod_df["Brand"])
            product_brand_key = codes if not np.any(codes < 0) else None

        except Exception:
            prod_df = load_parquet_df(products_path, ["ProductKey", "UnitPrice", "UnitCost"])
            product_np = prod_df[["ProductKey", "UnitPrice", "UnitCost"]].to_numpy()
            product_brand_key = None


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
        
    # ------------------------------------------------------------
    # Employees / store assignments -> SalesPersonEmployeeKey
    # ------------------------------------------------------------
    emp_assign_path = parquet_folder_p / "employee_store_assignments.parquet"

    employee_assign_store_key = None
    employee_assign_employee_key = None
    employee_assign_start_date = None
    employee_assign_end_date = None
    employee_assign_fte = None
    employee_assign_is_primary = None
    employee_assign_role = None

    # config-driven allowlist: which RoleAtStore can appear as SalesPersonEmployeeKey
    salesperson_roles = _cfg_get(cfg, ["sales", "salesperson_roles"], default=None)
    if not (isinstance(salesperson_roles, list) and salesperson_roles):
        primary = _cfg_get(cfg, ["employees", "store_assignments", "primary_sales_role"], default="Sales Associate")
        salesperson_roles = [str(primary)]

    if emp_assign_path.exists():
        emp_assign_df = load_parquet_df(
            emp_assign_path,
            cols=[
                "EmployeeKey",
                "StoreKey",
                "StartDate",
                "EndDate",
                "FTE",
                "IsPrimary",
                "RoleAtStore",
            ],
        )

        # Keep only allowed salespeople roles
        if "RoleAtStore" in emp_assign_df.columns:
            emp_assign_df = emp_assign_df[emp_assign_df["RoleAtStore"].isin(salesperson_roles)].copy()

        if not emp_assign_df.empty:
            end_dt = pd.to_datetime(end_date, errors="coerce").normalize()

            start_dt = pd.to_datetime(emp_assign_df["StartDate"], errors="coerce").dt.normalize()
            end_dt_col = pd.to_datetime(emp_assign_df["EndDate"], errors="coerce").dt.normalize()
            end_dt_col = end_dt_col.fillna(end_dt)

            employee_assign_store_key = _as_np(emp_assign_df["StoreKey"], np.int64)
            employee_assign_employee_key = _as_np(emp_assign_df["EmployeeKey"], np.int64)
            employee_assign_start_date = _as_np(start_dt, "datetime64[D]")
            employee_assign_end_date = _as_np(end_dt_col, "datetime64[D]")
            employee_assign_role = _as_np(emp_assign_df["RoleAtStore"].astype(str))

            if "FTE" in emp_assign_df.columns:
                employee_assign_fte = _as_np(emp_assign_df["FTE"], np.float64)
            if "IsPrimary" in emp_assign_df.columns:
                employee_assign_is_primary = _as_np(emp_assign_df["IsPrimary"], bool)

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
        product_brand_key=product_brand_key,
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
        skip_order_cols=bool(skip_order_cols),
        skip_order_cols_requested=bool(skip_order_cols),

        write_pyarrow=write_pyarrow,

        partition_enabled=partition_enabled,
        partition_cols=partition_cols,

        models_cfg=State.models_cfg,
        # Returns (optional)
        returns_enabled=bool(returns_enabled_effective),
        returns_rate=float(returns_rate),
        returns_max_lag_days=int(returns_max_lag_days),

        # deterministic employee assignment lookup
        seed_master= int(seed),
        employee_salesperson_seed= int(seed) + 99173,
        employee_primary_boost= 2.0,

        # employee-store assignment pools
        employee_assign_store_key= employee_assign_store_key,
        employee_assign_employee_key= employee_assign_employee_key,
        employee_assign_start_date= employee_assign_start_date,
        employee_assign_end_date= employee_assign_end_date,
        employee_assign_fte= employee_assign_fte,
        employee_assign_is_primary= employee_assign_is_primary,
        employee_assign_role=employee_assign_role,
        salesperson_roles=salesperson_roles,
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

    pool_spec = PoolRunSpec(
        processes=n_workers,
        chunksize=1,            # keep existing behavior; tune later if needed
        maxtasksperchild=None,  # leave None; can set later for long runs
        label="sales",
    )

    for result in iter_imap_unordered(
        tasks=batched_tasks,
        task_fn=_worker_task,
        spec=pool_spec,
        initializer=init_sales_worker,
        initargs=(worker_cfg,),
    ):
        if isinstance(result, list):
            for r in result:
                completed_units += 1
                _record_chunk_result(r, completed_units, total_units)
        else:
            completed_units += 1
            _record_chunk_result(result, completed_units, total_units)

    done("All chunks completed.")

    # ------------------------------------------------------------
    # Manifest helper (defined BEFORE returns so it actually runs)
    # ------------------------------------------------------------
    def _build_sales_manifest() -> SalesRunManifest:
        per_table: dict[str, TableOutputs] = {}

        for t in tables:
            per_table[t] = TableOutputs(
                table=t,
                file_format=file_format,
                chunks=list(created_by_table.get(t, [])),
                merged_path=(output_paths.merged_path(t) if file_format == "parquet" else None),
                delta_table_dir=(output_paths.delta_table_dir(t) if file_format == "deltaparquet" else None),
                delta_parts_dir=(output_paths.delta_parts_dir(t) if file_format == "deltaparquet" else None),
            )

        return SalesRunManifest(
            sales_output=sales_output,
            file_format=file_format,
            out_folder=str(out_folder_p),
            tables=per_table,
        )

    # ------------------------------------------------------------
    # Final assembly (TABLE-AWARE)
    # ------------------------------------------------------------
    if file_format == "deltaparquet":
        from .sales_worker import write_delta_partitioned

        missing_parts = []
        wrote = 0

        for t in tables:
            parts_dir = output_paths.delta_parts_dir(t)
            delta_dir = output_paths.delta_table_dir(t)

            if not os.path.exists(parts_dir):
                missing_parts.append((t, parts_dir))
                continue

            write_delta_partitioned(
                parts_folder=parts_dir,
                delta_output_folder=delta_dir,
                partition_cols=partition_cols,
                table_name=t,
            )
            wrote += 1

        if wrote == 0:
            msg = " | ".join([f"{t} -> {p}" for t, p in missing_parts]) if missing_parts else "no parts found"
            raise RuntimeError(f"No delta parts found for any table. {msg}")

        manifest = _build_sales_manifest()
        return (created_files, manifest) if return_manifest else created_files

    if file_format == "csv":
        manifest = _build_sales_manifest()
        return (created_files, manifest) if return_manifest else created_files

    if file_format == "parquet":
        if merge_parquet:
            from .sales_writer import merge_parquet_files

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
                    table_name=t,
                )

        manifest = _build_sales_manifest()
        return (created_files, manifest) if return_manifest else created_files

    raise ValueError(f"Unknown file_format: {file_format}")

