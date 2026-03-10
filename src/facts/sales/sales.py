from __future__ import annotations

import glob
import os
import zlib
from dataclasses import dataclass
from math import ceil
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.utils.config_helpers import int_or as _int_or, float_or as _float_or, bool_or as _bool_or, str_or as _str_or
from src.utils.logging_utils import debug, done, info, skip, work
from src.utils.shared_arrays import SharedArrayGroup
from .sales_logic import State
from .sales_worker import PoolRunSpec, iter_imap_unordered, _worker_task, init_sales_worker
from .sales_writer import merge_parquet_files, optimize_parquet
from .output_paths import (
    OutputPaths,
    TABLE_SALES,
    TABLE_SALES_ORDER_DETAIL,
    TABLE_SALES_ORDER_HEADER,
    TABLE_SALES_RETURN,
)

# Budget streaming aggregation (lazy import to avoid hard dependency)
try:
    from src.facts.budget.lookups import build_budget_lookups
    from src.facts.budget.accumulator import BudgetAccumulator
    _BUDGET_AVAILABLE = True
except ImportError:
    _BUDGET_AVAILABLE = False

try:
    from src.facts.inventory.accumulator import InventoryAccumulator
    _INVENTORY_AVAILABLE = True
except ImportError:
    _INVENTORY_AVAILABLE = False

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


def _apply_cfg_default(current: Any, default: Any, cfg_value: Any) -> Any:
    """
    Treat cfg as source-of-truth defaults when call-site leaves args at their defaults.
    """
    if cfg_value is None:
        return current
    return cfg_value if current == default else current


def _resolve_date_range(cfg: dict, start_date: Optional[str], end_date: Optional[str]) -> Tuple[str, str]:
    """
    Resolve the *Sales* fact date window (raw, unbuffered).

    Priority:
      explicit args
      cfg.sales.override.dates.{start,end}
      cfg.dates.override.dates.{start,end}          # global override
      cfg.defaults.dates.{start,end} (or cfg._defaults.dates)
    """
    if start_date is not None and end_date is not None:
        return str(start_date), str(end_date)

    defaults_section = cfg.get("defaults") or cfg.get("_defaults")
    defaults_dates = (defaults_section or {}).get("dates") if isinstance(defaults_section, dict) else None
    if not isinstance(defaults_dates, dict):
        raise KeyError("Missing defaults.dates in config")

    ov_sales_dates = _cfg_get(cfg, ["sales", "override", "dates"], default={})
    ov_sales_dates = ov_sales_dates if isinstance(ov_sales_dates, dict) else {}

    ov_global_dates = _cfg_get(cfg, ["dates", "override", "dates"], default={})
    ov_global_dates = ov_global_dates if isinstance(ov_global_dates, dict) else {}

    if start_date is None:
        start_date = (
            ov_sales_dates.get("start")
            or ov_global_dates.get("start")
            or defaults_dates.get("start")
        )
    if end_date is None:
        end_date = (
            ov_sales_dates.get("end")
            or ov_global_dates.get("end")
            or defaults_dates.get("end")
        )

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
            ot[_bool_mask(mask)] *= f

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
    optimize_after_merge = _bool_or(sales_cfg.get("optimize"), False)

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

    # ------------------------------------------------------------
    # Returns (optional)
    # ------------------------------------------------------------
    facts_enabled = _cfg_get(cfg, ["facts", "enabled"], default=[])
    facts_enabled = facts_enabled if isinstance(facts_enabled, list) else []

    returns_cfg = cfg.get("returns") if isinstance(cfg.get("returns"), dict) else {}
    returns_enabled = _bool_or(returns_cfg.get("enabled"), False)

    # If facts.enabled is used, treat it as an additional "feature gate"
    if facts_enabled:
        returns_enabled = bool(returns_enabled and ("returns" in {str(x).lower() for x in facts_enabled}))

    returns_rate = _float_or(returns_cfg.get("return_rate", 0.0), 0.0)
    if not np.isfinite(returns_rate):
        returns_rate = 0.0
    # keep within [0, 1]
    returns_rate = max(0.0, min(1.0, returns_rate))

    returns_min_lag_days = _int_or(
        returns_cfg.get("min_days_after_sale", returns_cfg.get("returns_min_lag_days", 0)),
        0,
    )
    returns_min_lag_days = max(0, returns_min_lag_days)

    returns_max_lag_days = _int_or(
        returns_cfg.get("max_days_after_sale", returns_cfg.get("returns_max_lag_days", 60)),
        60,
    )
    returns_max_lag_days = max(0, returns_max_lag_days)

    # Ensure min <= max (swap/clamp defensively)
    if returns_min_lag_days > returns_max_lag_days:
        returns_min_lag_days = returns_max_lag_days

    # Safeguard: if user generates BOTH and keeps order columns in Sales, output balloons.

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

    def _empty_manifest() -> SalesRunManifest:
        """Build an empty manifest for early-exit paths."""
        per_table: dict[str, TableOutputs] = {}
        for t in tables:
            per_table[t] = TableOutputs(
                table=t,
                file_format=file_format,
                chunks=[],
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
    # Optional auto chunk sizing
    # ------------------------------------------------------------
    total_rows = _int_or(total_rows, 0)
    if total_rows <= 0:
        skip("No sales rows to generate (total_rows <= 0).")
        if return_manifest:
            return ([], _empty_manifest())
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

    customer_keys = _as_np(cust_df["CustomerKey"], np.int32)
    is_active_in_sales = _as_np(cust_df["IsActiveInSales"], np.int32)

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
        info("CustomerBaseWeight not found; customer sampling will be uniform.")

    # Products: respect runner-bound active_product_np
    product_brand_key = None
    brand_names = None
    product_subcat_key = None
    products_path = parquet_folder_p / "products.parquet"
    assortment_cfg = (cfg.get("stores") or {}).get("assortment") or {}

    def _brand_codes_from_series(s: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        # Guarantee no NA => no -1 codes
        s2 = s.fillna("Unknown").astype(str)
        codes, uniques = pd.factorize(s2, sort=True)
        return np.asarray(codes, dtype=np.int32), np.asarray(uniques, dtype=object)

    # Read schema once (metadata only) to discover available columns
    import pyarrow.parquet as _pq
    _prod_schema = set(_pq.read_schema(str(products_path)).names)
    _has_brand_col = "Brand" in _prod_schema
    _has_subcat_col = "SubcategoryKey" in _prod_schema
    _need_subcat = bool(assortment_cfg.get("enabled")) and _has_subcat_col

    if getattr(State, "active_product_np", None) is not None:
        product_np = State.active_product_np
        active_keys = np.asarray(product_np[:, 0], dtype=np.int32)

        # Single load: Brand + optional SubcategoryKey (one parquet open)
        _prod_cols = ["ProductKey"]
        if _has_brand_col:
            _prod_cols.append("Brand")
        if _need_subcat:
            _prod_cols.append("SubcategoryKey")

        try:
            _prod_df = load_parquet_df(products_path, _prod_cols)
            _prod_df = _prod_df.drop_duplicates("ProductKey", keep="first")
            _prod_keys = _prod_df["ProductKey"].to_numpy(dtype=np.int32)

            # Sorted-key lookup (avoids pandas reindex float64 promotion)
            _sort_idx = np.argsort(_prod_keys)
            _sorted_keys = _prod_keys[_sort_idx]
            _pos = np.clip(np.searchsorted(_sorted_keys, active_keys), 0, max(len(_sorted_keys) - 1, 0))
            _found = _sorted_keys[_pos] == active_keys

            if _has_brand_col:
                codes, brand_names = _brand_codes_from_series(_prod_df["Brand"])
                bk = np.full(len(active_keys), -1, dtype=np.int32)
                bk[_found] = codes[_sort_idx][_pos[_found]]
                if np.any(bk < 0):
                    info("Brand mapping missing/invalid for some ProductKeys; disabling brand_popularity for this run.")
                else:
                    product_brand_key = bk

            if _need_subcat and "SubcategoryKey" in _prod_df.columns:
                subcat_vals = _prod_df["SubcategoryKey"].to_numpy(dtype=np.int32)
                sc = np.zeros(len(active_keys), dtype=np.int32)
                sc[_found] = subcat_vals[_sort_idx][_pos[_found]]
                product_subcat_key = sc

        except Exception as exc:
            info(f"Could not load/derive Brand from products.parquet ({type(exc).__name__}: {exc}); "
                 "disabling brand_popularity for this run.")
            product_brand_key = None

    else:
        # Full product path — single load with all needed columns
        _prod_cols = ["ProductKey", "UnitPrice", "UnitCost"]
        if _has_brand_col:
            _prod_cols.append("Brand")
        if _need_subcat:
            _prod_cols.append("SubcategoryKey")

        prod_df = load_parquet_df(products_path, _prod_cols)
        prod_df["ProductKey"] = pd.to_numeric(prod_df["ProductKey"], errors="coerce")
        prod_df["UnitPrice"] = pd.to_numeric(prod_df["UnitPrice"], errors="coerce")
        prod_df["UnitCost"] = pd.to_numeric(prod_df["UnitCost"], errors="coerce")
        prod_df = prod_df.dropna(subset=["ProductKey", "UnitPrice", "UnitCost"])
        prod_df["ProductKey"] = prod_df["ProductKey"].astype("int32", copy=False)

        product_np = np.column_stack([
            prod_df["ProductKey"].to_numpy(dtype=np.int32, copy=False),
            prod_df["UnitPrice"].to_numpy(dtype=np.float64, copy=False),
            prod_df["UnitCost"].to_numpy(dtype=np.float64, copy=False),
        ])

        if _has_brand_col:
            codes, brand_names = _brand_codes_from_series(prod_df["Brand"])
            product_brand_key = codes if not np.any(codes < 0) else None

        if _need_subcat and "SubcategoryKey" in prod_df.columns:
            product_subcat_key = prod_df["SubcategoryKey"].to_numpy(dtype=np.int32)

    # PopularityScore + SeasonalityProfile from ProductProfile (for weighted sampling)
    product_popularity = None
    product_seasonality = None
    _profile_path = parquet_folder_p / "product_profile.parquet"
    if _profile_path.exists():
        try:
            _pp_df = load_parquet_df(_profile_path, ["ProductKey", "PopularityScore", "SeasonalityProfile"])
            _pp_df = _pp_df.drop_duplicates("ProductKey", keep="first")
            _pp_df["ProductKey"] = _pp_df["ProductKey"].astype("int32")
            active_keys = np.asarray(product_np[:, 0], dtype=np.int32)
            _pp_map_pop = pd.Series(
                _pp_df["PopularityScore"].to_numpy(dtype=np.float64),
                index=_pp_df["ProductKey"].to_numpy(dtype=np.int32),
            )
            product_popularity = _pp_map_pop.reindex(active_keys).fillna(50.0).to_numpy(dtype=np.float64)
            _pp_map_sea = pd.Series(
                _pp_df["SeasonalityProfile"].to_numpy().astype(str),
                index=_pp_df["ProductKey"].to_numpy(dtype=np.int32),
            )
            _sea_str = _pp_map_sea.reindex(active_keys).fillna("None").to_numpy().astype(str)
            # Encode as int8 so it can be shared via shared memory (avoids 8x pickle of object array)
            _SEASON_ENCODE = {"Holiday": 1, "Winter": 2, "Summer": 3, "BackToSchool": 4, "Spring": 5}
            product_seasonality = np.zeros(len(_sea_str), dtype=np.int8)
            for _sname, _scode in _SEASON_ENCODE.items():
                product_seasonality[_sea_str == _sname] = _scode
        except Exception as exc:
            info(f"Could not load product profile ({type(exc).__name__}: {exc}); using uniform product sampling.")
            product_popularity = None
            product_seasonality = None

    # Stores: read ONCE (keys + geography + StoreType for assortment)
    _store_cols = ["StoreKey", "GeographyKey"]
    _store_path = parquet_folder_p / "stores.parquet"
    if _store_path.exists():
        import pyarrow.parquet as _pq
        _store_schema_names = set(_pq.read_schema(str(_store_path)).names)
        if "StoreType" in _store_schema_names:
            _store_cols.append("StoreType")
    store_df = load_parquet_df(_store_path, _store_cols)
    store_keys = _as_np(store_df["StoreKey"], np.int32)
    store_to_geo = dict(zip(_as_np(store_df["StoreKey"], np.int32), _as_np(store_df["GeographyKey"], np.int32)))

    # StoreType map for assortment (StoreKey -> "Supermarket" etc.)
    store_type_map = None
    if "StoreType" in store_df.columns:
        store_type_map = dict(zip(
            store_df["StoreKey"].astype(int).tolist(),
            store_df["StoreType"].astype(str).tolist(),
        ))

    # Geography + currency mapping
    geo_df = load_parquet_df(parquet_folder_p / "geography.parquet", ["GeographyKey", "ISOCode"])
    currency_df = load_parquet_df(parquet_folder_p / "currency.parquet", ["CurrencyKey", "ToCurrency"])

    geo_df = geo_df.merge(currency_df, left_on="ISOCode", right_on="ToCurrency", how="left")
    if geo_df["CurrencyKey"].isna().any():
        n_missing = int(geo_df["CurrencyKey"].isna().sum())
        missing_isos = geo_df.loc[geo_df["CurrencyKey"].isna(), "ISOCode"].unique().tolist()
        default_currency = int(currency_df.iloc[0]["CurrencyKey"])
        geo_df["CurrencyKey"] = geo_df["CurrencyKey"].fillna(default_currency)
        info(f"WARNING: {n_missing} geography row(s) have no currency match (ISOs: {missing_isos}); "
             f"defaulting to CurrencyKey={default_currency}.")

    geo_to_currency = dict(zip(_as_np(geo_df["GeographyKey"], np.int32), _as_np(geo_df["CurrencyKey"], np.int32)))

    # Promotions
    promo_df = load_parquet_df(parquet_folder_p / "promotions.parquet")

    if promo_df.empty:
        promo_keys_all = np.array([], dtype=np.int32)
        promo_start_all = np.array([], dtype="datetime64[D]")
        promo_end_all = np.array([], dtype="datetime64[D]")
        new_customer_promo_keys = np.array([], dtype=np.int32)
    else:
        promo_start = _normalize_dt_any(promo_df["StartDate"])
        promo_end = _normalize_dt_any(promo_df["EndDate"])

        promo_keys_all = _as_np(promo_df["PromotionKey"], np.int32)
        promo_start_all = _as_np(promo_start, "datetime64[D]")
        promo_end_all = _as_np(promo_end, "datetime64[D]")

        if "PromotionType" in promo_df.columns:
            nc_mask = promo_df["PromotionType"].astype(str) == "New Customer"
            new_customer_promo_keys = _as_np(promo_df.loc[nc_mask, "PromotionKey"], np.int32)
        else:
            new_customer_promo_keys = np.array([], dtype=np.int32)

    # ------------------------------------------------------------
    # Employees / store assignments -> SalesPersonEmployeeKey
    # ------------------------------------------------------------
    emp_assign_path = parquet_folder_p / "employee_store_assignments.parquet"

    # Fail fast: the bridge is the single source of truth for SalesPersonEmployeeKey.
    # Without it, every Sales row would get SalesPersonEmployeeKey = -1.
    if not emp_assign_path.exists():
        raise FileNotFoundError(
            f"employee_store_assignments.parquet is required: {emp_assign_path}. "
            f"Run dimension generation first."
        )

    # config-driven allowlist: which RoleAtStore can appear as SalesPersonEmployeeKey
    salesperson_roles = _cfg_get(cfg, ["sales", "salesperson_roles"], default=None)
    if not (isinstance(salesperson_roles, list) and salesperson_roles):
        primary = _cfg_get(cfg, ["employees", "store_assignments", "primary_sales_role"], default="Sales Associate")
        salesperson_roles = [str(primary)]

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

    if emp_assign_df.empty:
        raise RuntimeError(
            f"No employee assignments with role in {salesperson_roles} found in "
            f"{emp_assign_path}. Check employees.store_assignments.primary_sales_role "
            f"and ensure the bridge has been regenerated."
        )

    end_dt = pd.to_datetime(end_date, errors="coerce").normalize()

    start_dt = pd.to_datetime(emp_assign_df["StartDate"], errors="coerce").dt.normalize()
    end_dt_col = pd.to_datetime(emp_assign_df["EndDate"], errors="coerce").dt.normalize()
    end_dt_col = end_dt_col.fillna(end_dt)

    employee_assign_store_key = _as_np(emp_assign_df["StoreKey"], np.int32)
    employee_assign_employee_key = _as_np(emp_assign_df["EmployeeKey"], np.int32)
    employee_assign_start_date = _as_np(start_dt, "datetime64[D]")
    employee_assign_end_date = _as_np(end_dt_col, "datetime64[D]")
    employee_assign_role = _as_np(emp_assign_df["RoleAtStore"].astype(str))

    employee_assign_fte = None
    employee_assign_is_primary = None
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
        if return_manifest:
            return ([], _empty_manifest())
        return []

    # ------------------------------------------------------------
    # Worker count
    # ------------------------------------------------------------
    if workers is None:
        n_workers = min(len(tasks), max(1, cpu_count() - 1))
    else:
        n_workers = min(len(tasks), max(1, _int_or(workers, cpu_count() - 1)))


    info(f"Spawning {n_workers} worker processes...")

    # SalesOrderNumber RunId:
    # - If configured: sales.order_id_run_id (0..999)
    # - Else derive a stable 0..999 id from output folder + seed (unique per run folder)
    order_id_run_id_raw = sales_cfg.get("order_id_run_id", None)
    if order_id_run_id_raw is None:
        key = f"{out_folder_p.resolve()}|{int(seed)}".encode("utf-8")
        order_id_run_id = int(zlib.crc32(key) % 1000)
    else:
        order_id_run_id = int(order_id_run_id_raw) % 1000
    if order_id_run_id < 0:
        order_id_run_id += 1000

    # ------------------------------------------------------------
    # Auto-adjust new_customer_share so all customers can be
    # discovered within the available rows.  Without this, low
    # row-to-customer ratios silently leave thousands of customers
    # without any sales.
    # ------------------------------------------------------------
    _models = State.models_cfg
    if isinstance(_models, dict):
        _cust_mdl = _models.get("customers", {}) or {}
        configured_share = float(_cust_mdl.get("new_customer_share", 0.10))
        configured_max_frac = float(_cust_mdl.get("max_new_fraction_per_month", 0.015))

        total_active = int((is_active_in_sales == 1).sum())
        # Customers with negative start_month are backdated (pre-existing)
        # and will be pre-seeded into seen_customers by chunk_builder.
        # Only customers with start_month >= 0 need discovery.
        warm_start = int(((is_active_in_sales == 1) & (customer_start_month < 0)).sum())
        undiscovered = max(0, total_active - warm_start)

        if undiscovered > 0 and total_rows > 0:
            headroom = 1.5
            # Customer sampling operates at order granularity (avg ~2
            # lines per order), so the effective number of customer
            # slots is roughly total_rows / avg_lines_per_order.
            avg_lines_per_order = 2.0
            effective_slots = total_rows / avg_lines_per_order
            needed_share = (undiscovered * headroom) / effective_slots
            needed_share = float(np.clip(needed_share, 0.0, 0.50))

            # Compute the per-month fraction cap needed to sustain discovery.
            # With T months and avg_eligible customers, each month must discover
            # undiscovered / T customers, i.e. (undiscovered / T) / avg_eligible
            # of the eligible pool.
            months_int = date_pool.astype("datetime64[M]").astype("int64")
            T_est = max(1, int(months_int.max() - months_int.min() + 1))
            avg_eligible = max(1, total_active // 2)
            needed_frac = (undiscovered * headroom) / (avg_eligible * T_est)
            needed_frac = float(np.clip(needed_frac, 0.0, 0.50))

            adjusted = False
            _cust_mdl_copy = dict(_cust_mdl)

            if needed_share > configured_share:
                _cust_mdl_copy["new_customer_share"] = needed_share
                adjusted = True

            if needed_frac > configured_max_frac:
                _cust_mdl_copy["max_new_fraction_per_month"] = needed_frac
                adjusted = True

            if adjusted:
                debug(
                    f"Auto-adjusting customer discovery: new_customer_share "
                    f"{configured_share:.3f} -> {_cust_mdl_copy.get('new_customer_share', configured_share):.3f}, "
                    f"max_new_fraction_per_month "
                    f"{configured_max_frac:.4f} -> {_cust_mdl_copy.get('max_new_fraction_per_month', configured_max_frac):.4f} "
                    f"({undiscovered:,} undiscovered customers across {total_rows:,} rows)."
                )
                _models_copy = dict(_models)
                _models_copy["customers"] = _cust_mdl_copy
                State.models_cfg = _models_copy

    # ------------------------------------------------------------
    # Worker configuration (keep keys stable for compatibility)
    # ------------------------------------------------------------
    worker_cfg = dict(
        product_np=product_np,
        product_brand_key=product_brand_key,
        brand_names=brand_names,
        store_keys=store_keys,
        promo_keys_all=promo_keys_all,
        promo_start_all=promo_start_all,
        promo_end_all=promo_end_all,
        new_customer_promo_keys=new_customer_promo_keys,
        new_customer_window_months=int((cfg.get("promotions") or {}).get("new_customer_window_months", 3)),

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

        # CRITICAL: must be constant across ALL chunks/tasks for SalesOrderNumber uniqueness.
        # worker/task.py must use this constant (never fall back to per-task batch_i).
        chunk_size=int(chunk_size),

        # Optional alias (safe to add): lets us rename later without breaking older workers.
        # In init.py you can do: stride = worker_cfg.get("order_id_stride_orders") or worker_cfg["chunk_size"]
        order_id_stride_orders=int(chunk_size),
        order_id_run_id=int(order_id_run_id),
        max_lines_per_order=int(sales_cfg.get("max_lines_per_order", 5) or 5),

        sales_output=sales_output,

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
        returns_min_lag_days=int(returns_min_lag_days),
        returns_max_lag_days=int(returns_max_lag_days),

        # deterministic employee assignment lookup
        seed_master=int(seed),
        employee_salesperson_seed=int(seed) + 99173,
        employee_primary_boost=2.0,

        # employee-store assignment pools
        employee_assign_store_key=employee_assign_store_key,
        employee_assign_employee_key=employee_assign_employee_key,
        employee_assign_start_date=employee_assign_start_date,
        employee_assign_end_date=employee_assign_end_date,
        employee_assign_fte=employee_assign_fte,
        employee_assign_is_primary=employee_assign_is_primary,
        employee_assign_role=employee_assign_role,
        salesperson_roles=salesperson_roles,

        # Product profile attributes for weighted sampling
        product_popularity=product_popularity,
        product_seasonality=product_seasonality,
    )

    # ------------------------------------------------------------
    # Store-product assortment (optional)
    # ------------------------------------------------------------
    if assortment_cfg.get("enabled") and product_subcat_key is not None and store_type_map is not None:
        worker_cfg["assortment"] = dict(assortment_cfg)
        worker_cfg["product_subcat_key"] = product_subcat_key
        worker_cfg["store_type_map"] = store_type_map
        info("Store-product assortment: enabled")

    # ------------------------------------------------------------
    # Budget streaming aggregation (optional)
    # ------------------------------------------------------------
    budget_cfg = cfg.get("budget") if isinstance(cfg.get("budget"), dict) else {}
    budget_enabled = _BUDGET_AVAILABLE and bool(budget_cfg.get("enabled", False))
    budget_acc = None  # type: Optional[BudgetAccumulator]

    if budget_enabled:
        try:
            budget_lookups = build_budget_lookups(parquet_folder_p)
            # Pass dense arrays to workers via worker_cfg (tiny — few KB, pickle fine)
            worker_cfg["budget_enabled"] = True
            worker_cfg["budget_store_to_country"] = budget_lookups["budget_store_to_country"]
            worker_cfg["budget_product_to_cat"] = budget_lookups["budget_product_to_cat"]
            worker_cfg["parquet_folder"] = str(parquet_folder_p)

            budget_acc = BudgetAccumulator(
                country_labels=budget_lookups["budget_country_labels"],
                category_labels=budget_lookups["budget_category_labels"],
            )
            info("Budget streaming aggregation: enabled")
        except Exception as exc:
            info(f"Budget streaming aggregation: disabled ({type(exc).__name__}: {exc})")
            budget_enabled = False
            budget_acc = None
            worker_cfg["budget_enabled"] = False
    else:
        worker_cfg["budget_enabled"] = False

    # ------------------------------------------------------------
    # Inventory
    # ------------------------------------------------------------
    inv_cfg = cfg.get("inventory") if isinstance(cfg.get("inventory"), dict) else {}
    inventory_enabled = _INVENTORY_AVAILABLE and bool(inv_cfg.get("enabled", False))
    inventory_acc = None

    if inventory_enabled:
        inventory_acc = InventoryAccumulator()
        worker_cfg["inventory_enabled"] = True
        info("Inventory streaming aggregation: enabled")
    else:
        worker_cfg["inventory_enabled"] = False

    # ------------------------------------------------------------
    # Shared memory: place large numpy arrays in OS shared memory
    # so worker processes get zero-copy views instead of pickle copies.
    # ------------------------------------------------------------
    _shm = SharedArrayGroup()
    _shm.publish_dict(worker_cfg, [
        "product_np",
        "product_brand_key",
        "customers",              # alias for customer_keys
        "customer_keys",
        "customer_is_active_in_sales",
        "customer_start_month",
        "customer_end_month",
        "customer_base_weight",
        "product_subcat_key",
        "product_popularity",
        "date_pool",
        "date_prob",
        "promo_keys_all",
        "promo_start_all",
        "promo_end_all",
        "employee_assign_store_key",
        "employee_assign_employee_key",
        "employee_assign_start_date",
        "employee_assign_end_date",
        "employee_assign_fte",
        "employee_assign_is_primary",
        "product_seasonality",
    ])

    # ------------------------------------------------------------
    # Pre-build expensive derived structures once in the main process
    # instead of redundantly in each worker (saves ~2.4 GB RAM and
    # eliminates ~4.5 min worker init delay on Windows).
    # NOTE: local variables (product_np, store_keys, etc.) still
    # reference the original numpy arrays; worker_cfg values are now
    # shared-memory descriptors after publish_dict above.
    # ------------------------------------------------------------
    from .sales_worker.init import (
        _build_store_assortment,
        _build_buckets_from_brand_key,
        _build_brand_prob_by_month_rotate_winner,
        _build_salesperson_effective_by_store,
        _DEFAULT_ASSORTMENT_COVERAGE,
        infer_T_from_date_pool,
        int_or,
        float_or,
    )

    # 1) brand_to_row_idx — list of ~300K arrays, share via jagged shared memory
    #    (pickling 300K individual arrays is extremely slow on Windows spawn)
    _brand_product_counts = None  # reused by brand_prob pre-build
    if product_brand_key is not None:
        _prebuilt_brand = _build_buckets_from_brand_key(product_brand_key)
        worker_cfg["_prebuilt_brand_to_row_idx"] = _shm.publish_jagged(
            "brand_idx", _prebuilt_brand, dtype=np.int32,
        )
        # Extract brand product counts before freeing (reused for brand_prob)
        _brand_product_counts = np.array(
            [len(b) if b is not None else 0 for b in _prebuilt_brand],
            dtype=np.float64,
        )
        del _prebuilt_brand

    # 2) store_to_product_rows (~350 MB — share via jagged shared memory)
    if assortment_cfg.get("enabled") and product_subcat_key is not None and store_type_map is not None:
        store_type_arr = np.array(
            [str(store_type_map.get(int(sk), "Supermarket")) for sk in store_keys],
            dtype=object,
        )
        coverage = assortment_cfg.get("coverage", _DEFAULT_ASSORTMENT_COVERAGE)
        assort_seed = int(assortment_cfg.get("seed", seed))
        _prebuilt_assortment = _build_store_assortment(
            store_keys=store_keys,
            store_type_arr=store_type_arr,
            product_np=product_np,
            product_subcat_key=product_subcat_key,
            coverage_cfg=coverage,
            seed=assort_seed,
        )
        worker_cfg["_prebuilt_store_to_product_rows"] = _shm.publish_jagged(
            "assortment", _prebuilt_assortment, dtype=np.int32,
        )
        del _prebuilt_assortment  # free the original list

    # 3) salesperson_effective_by_store (dict of small arrays — pickle)
    if employee_assign_employee_key is not None and employee_assign_store_key is not None:
        _sp_eff = _build_salesperson_effective_by_store(
            store_keys=store_keys,
            assign_store=employee_assign_store_key,
            assign_emp=employee_assign_employee_key,
            assign_start=employee_assign_start_date,
            assign_end=employee_assign_end_date,
            assign_fte=employee_assign_fte,
            assign_is_primary=employee_assign_is_primary,
            primary_boost=2.0,
        )
        worker_cfg["_prebuilt_salesperson_effective_by_store"] = _sp_eff
        # Pre-compute salesperson_global_pool so workers skip employee_assign_role filtering
        if _sp_eff is not None:
            _all_sp = np.concatenate([v[0] for v in _sp_eff.values()])
            worker_cfg["_prebuilt_salesperson_global_pool"] = np.unique(_all_sp).astype(np.int32)
        del _sp_eff

    # 4) brand_prob_by_month — (T, B) float64 matrix, ~164MB for 72 months × 300K brands.
    #    Deterministic per seed, so build once and share instead of 8× in workers.
    models_cfg = worker_cfg.get("models_cfg")
    if isinstance(models_cfg, dict):
        _models_root = models_cfg.get("models") if isinstance(models_cfg.get("models"), dict) else models_cfg
        _brand_cfg = _models_root.get("brand_popularity") if isinstance(_models_root, dict) else None
        if _brand_cfg and product_brand_key is not None and product_brand_key.size > 0:
            _T = infer_T_from_date_pool(date_pool)
            _B = int(product_brand_key.max()) + 1
            _rng_bp = np.random.default_rng(int(int_or(_brand_cfg.get("seed"), 1234)))

            # Brand product counts (reused from section 1 if available)
            _bp_counts = _brand_product_counts if (_brand_product_counts is not None and len(_brand_product_counts) == _B) else None

            # Explicit brand weights from config
            _bp_explicit = None
            _cfg_weights = _brand_cfg.get("brand_weights")
            if isinstance(_cfg_weights, dict) and _cfg_weights and brand_names is not None and len(brand_names) == _B:
                _bp_explicit = np.zeros(_B, dtype=np.float64)
                _name_to_idx = {str(n): i for i, n in enumerate(brand_names)}
                _has_override = False
                for _bname, _bw in _cfg_weights.items():
                    _idx = _name_to_idx.get(str(_bname))
                    if _idx is not None:
                        _bp_explicit[_idx] = float(_bw)
                        _has_override = True
                if _has_override:
                    _unset = _bp_explicit == 0.0
                    if _unset.any() and _bp_counts is not None:
                        _fallback = np.sqrt(np.maximum(_bp_counts, 1.0))
                        _leftover = max(0.0, 1.0 - _bp_explicit.sum())
                        _fallback_sum = _fallback[_unset].sum()
                        if _fallback_sum > 0 and _leftover > 0:
                            _bp_explicit[_unset] = _fallback[_unset] / _fallback_sum * _leftover
                        elif _leftover > 0:
                            _bp_explicit[_unset] = _leftover / _unset.sum()
                    elif _unset.any():
                        _leftover = max(0.0, 1.0 - _bp_explicit.sum())
                        _bp_explicit[_unset] = _leftover / _unset.sum() if _unset.sum() > 0 else 0.0
                else:
                    _bp_explicit = None

            _brand_prob = _build_brand_prob_by_month_rotate_winner(
                _rng_bp,
                T=_T, B=_B,
                winner_boost=float_or(_brand_cfg.get("winner_boost"), 2.5),
                noise_sd=float_or(_brand_cfg.get("noise_sd"), 0.15),
                min_share=float_or(_brand_cfg.get("min_share"), 0.02),
                year_len_months=int_or(_brand_cfg.get("year_len_months"), 12),
                brand_product_counts=_bp_counts,
                explicit_weights=_bp_explicit,
            )
            worker_cfg["_prebuilt_brand_prob_by_month"] = _shm.publish(
                "brand_prob", _brand_prob,
            )
            del _brand_prob

    # Track outputs per logical table (Sales / SalesOrderDetail / SalesOrderHeader)
    created_by_table: Dict[str, List[Any]] = {t: [] for t in tables}
    created_files: List[str] = []  # flat list of chunk file paths (csv/parquet), kept for backward-compat

    def _record_chunk_result(r: Any, completed_units: int, total_units: int) -> None:
        """
        r can be:
        - str (legacy 'sales' mode): path
        - dict(table_name -> write_result): multi-table modes
            write_result is str for csv/parquet, or {"part":..., "rows":...} for delta
            May also contain _budget_agg / _returns_agg (popped before recording).
        """

        # ---- Extract budget micro-aggregates (if present) ----
        if budget_acc is not None and isinstance(r, dict):
            budget_acc.add_sales(r.pop("_budget_agg", None))
            budget_acc.add_returns(r.pop("_returns_agg", None))

        if inventory_acc is not None and isinstance(r, dict):
            inventory_acc.add(r.pop("_inventory_agg", None))
        def _chunk_tag(path_like: str) -> str:
            b = os.path.basename(path_like)
            i = b.find("chunk")
            if i < 0:
                return b
            j = i + 5
            while j < len(b) and b[j].isdigit():
                j += 1
            return b[i:j]  # e.g. "chunk0004"

        short = {
            TABLE_SALES: "sales",
            TABLE_SALES_ORDER_DETAIL: "detail",
            TABLE_SALES_ORDER_HEADER: "header",
            TABLE_SALES_RETURN: "return",
        }

        if isinstance(r, str):
            created_by_table.setdefault(TABLE_SALES, []).append(r)
            created_files.append(r)
            work(f"[{completed_units}/{total_units}] {_chunk_tag(r)} -> sales")
            return

        if isinstance(r, dict):
            # stable display order: configured tables first, then any extras
            ordered_keys = [t for t in tables if t in r] + [k for k in r.keys() if k not in set(tables)]

            # pick any string path to extract the chunk tag (parquet/csv)
            tag = None
            for k in ordered_keys:
                v = r.get(k)
                if isinstance(v, str):
                    tag = _chunk_tag(v)
                    break

            produced: list[str] = []

            for table_name in ordered_keys:
                val = r.get(table_name)

                # record outputs (preserve manifest + created_files behavior)
                created_by_table.setdefault(table_name, []).append(val)

                if isinstance(val, str):
                    created_files.append(val)
                    produced.append(short.get(table_name, table_name))
                elif isinstance(val, dict) and "part" in val:
                    # delta mode: no file name spam; just note table produced a part
                    produced.append(short.get(table_name, table_name))

            if produced:
                if tag is None:
                    tag = "chunk"
                work(f"[{completed_units}/{total_units}] {tag} -> " + ", ".join(produced))
            return

        # Unknown / unexpected return type: log (helps catch worker bugs)
        info(f"[{completed_units}/{total_units}] Worker returned unsupported result type: {type(r).__name__}")
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

    try:
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
    finally:
        _shm.cleanup()

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
        from .sales_writer import write_delta_partitioned

        missing_parts = []
        wrote = 0

        for t in tables:
            parts_dir = output_paths.delta_parts_dir(t)
            delta_dir = output_paths.delta_table_dir(t)

            # parts_dir may exist (pre-created) even if no parts were written;
            # validate there is at least one parquet file (supports partitioned subfolders).
            part_files = glob.glob(os.path.join(parts_dir, "**", "*.parquet"), recursive=True)
            if not part_files:
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
        return (created_files, manifest, budget_acc, inventory_acc) if return_manifest else created_files

    if file_format == "csv":
        manifest = _build_sales_manifest()
        return (created_files, manifest, budget_acc, inventory_acc) if return_manifest else created_files

    if file_format == "parquet":
        if merge_parquet:
            merge_jobs: list[tuple[str, list[str], str]] = []
            skipped: list[str] = []

            for t in tables:
                chunks = sorted(
                    f for f in glob.glob(output_paths.chunk_glob(t, "parquet"))
                    if os.path.isfile(f)
                )
                if not chunks:
                    skipped.append(t)
                    continue
                merge_jobs.append((t, chunks, output_paths.merged_path(t)))

            if merge_jobs:
                short = {
                    TABLE_SALES: "sales",
                    TABLE_SALES_ORDER_DETAIL: "detail",
                    TABLE_SALES_ORDER_HEADER: "header",
                    TABLE_SALES_RETURN: "return",
                }

                counts = [(short.get(t, t), len(chunks)) for (t, chunks, _out) in merge_jobs]

                if len({c for _, c in counts}) == 1:
                    n = counts[0][1]
                    info(f"Merge parquet: {n} chunks -> " + ", ".join(name for name, _ in counts))
                else:
                    info("Merge parquet: " + ", ".join(f"{name}={n}" for name, n in counts))

                for t, chunks, out in merge_jobs:
                    merge_parquet_files(
                        chunks,
                        out,
                        delete_after=bool(delete_chunks),
                        compression=compression,
                        table_name=t,
                        log=False,
                    )

                if optimize_after_merge:
                    info("Optimize parquet: sorting merged files...")
                    for t, _chunks, out in merge_jobs:
                        result = optimize_parquet(
                            out,
                            table_name=t,
                            compression=compression,
                            row_group_size=row_group_size,
                        )
                        if result:
                            info(f"  Optimized: {os.path.basename(out)}")
            else:
                info("Merge parquet: none")

        manifest = _build_sales_manifest()
        return (created_files, manifest, budget_acc, inventory_acc) if return_manifest else created_files

    raise ValueError(f"Unknown file_format: {file_format}")
