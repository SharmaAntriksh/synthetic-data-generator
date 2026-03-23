from __future__ import annotations

import glob
import os
import zlib
from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.defaults import ONLINE_SALES_REP_ROLE
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

try:
    from src.facts.wishlists.accumulator import WishlistAccumulator
    _WISHLISTS_AVAILABLE = True
except ImportError:
    _WISHLISTS_AVAILABLE = False

try:
    from src.facts.complaints.accumulator import ComplaintsAccumulator
    _COMPLAINTS_AVAILABLE = True
except ImportError:
    _COMPLAINTS_AVAILABLE = False

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
        if not isinstance(cur, Mapping):
            return default
        # Prefer attribute access (Pydantic models) over dict access
        if hasattr(cur, k):
            cur = getattr(cur, k)
        elif isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def _apply_cfg_default(current: Any, default: Any, cfg_value: Any) -> Any:
    """
    Treat cfg as source-of-truth defaults when call-site leaves args at their defaults.
    """
    if cfg_value is None:
        return current
    return cfg_value if current == default else current


def _build_scd2_product_versions(
    products_path: Path,
    pool_product_ids: np.ndarray,
    pool_product_np: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Build per-entity version lookup tables for SCD2 product resolution.

    Returns (starts, data):
      - starts: shape (N_pool, max_ver) int64 — EffectiveStartDate as epoch days,
        sorted ascending per entity, padded with INT64_MAX.
      - data: shape (N_pool, max_ver, 3) float64 — [ProductKey, ListPrice, UnitCost]
        per version slot, padded with IsCurrent=1 values.

    At lookup time: for a sale with epoch-day D on product pool index P,
    ``ver = searchsorted(starts[P], D, side='right') - 1`` gives the correct
    version slot.
    """
    try:
        all_df = pd.read_parquet(
            str(products_path),
            columns=["ProductID", "ProductKey", "ListPrice", "UnitCost",
                     "EffectiveStartDate", "EffectiveEndDate"],
        )
    except (KeyError, ValueError):
        return None

    all_df["eff_start_days"] = pd.to_datetime(all_df["EffectiveStartDate"]).values.astype("datetime64[D]").astype(np.int64)

    N_pool = len(pool_product_ids)

    # Vectorized pool index mapping via dense lookup array
    max_pid = max(int(pool_product_ids.max()), int(all_df["ProductID"].max())) + 1
    pid_lookup = np.full(max_pid, -1, dtype=np.int32)
    pid_lookup[pool_product_ids] = np.arange(N_pool, dtype=np.int32)

    pool_idx = pid_lookup[all_df["ProductID"].to_numpy()]
    mask = pool_idx >= 0
    pool_idx = pool_idx[mask]
    eff_start = all_df["eff_start_days"].to_numpy()[mask]
    pkey_arr = all_df["ProductKey"].to_numpy(dtype=np.float64)[mask]
    lprice_arr = all_df["ListPrice"].to_numpy(dtype=np.float64)[mask]
    ucost_arr = all_df["UnitCost"].to_numpy(dtype=np.float64)[mask]

    # Sort by (pool_idx, eff_start_days) via lexsort (secondary key first)
    order = np.lexsort((eff_start, pool_idx))
    pool_idx = pool_idx[order]
    eff_start = eff_start[order]
    pkey_arr = pkey_arr[order]
    lprice_arr = lprice_arr[order]
    ucost_arr = ucost_arr[order]

    # Compute per-entity version slot indices using group boundaries
    breaks = np.empty(len(pool_idx), dtype=np.int32)
    breaks[0] = 0
    breaks[1:] = np.cumsum(pool_idx[1:] != pool_idx[:-1])
    group_starts = np.concatenate([[0], np.where(pool_idx[1:] != pool_idx[:-1])[0] + 1])
    slot = np.arange(len(pool_idx), dtype=np.int32)
    slot -= np.repeat(group_starts, np.diff(np.append(group_starts, len(pool_idx))))

    max_ver = int(slot.max()) + 1 if len(slot) > 0 else 1

    # Initialize with IsCurrent=1 defaults
    starts = np.full((N_pool, max_ver), np.iinfo(np.int64).max, dtype=np.int64)
    data = np.empty((N_pool, max_ver, 3), dtype=np.float64)
    data[:, :, 0] = pool_product_np[:, 0:1]  # ProductKey broadcast
    data[:, :, 1] = pool_product_np[:, 1:2]  # ListPrice broadcast
    data[:, :, 2] = pool_product_np[:, 2:3]  # UnitCost broadcast

    # Cap slot indices at max_ver - 1
    valid = slot < max_ver
    pi = pool_idx[valid]
    si = slot[valid]

    # Vectorized scatter into output arrays
    starts[pi, si] = eff_start[valid]
    data[pi, si, 0] = pkey_arr[valid]
    data[pi, si, 1] = lprice_arr[valid]
    data[pi, si, 2] = ucost_arr[valid]

    # Clamp first version start to 0 (covers all time before second version)
    starts[pi, 0] = 0

    return starts, data


def _build_scd2_customer_versions(
    customers_path: Path,
    pool_customer_keys: np.ndarray,
    pool_customer_ids: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Build per-entity version lookup tables for SCD2 customer resolution.

    Returns (starts, keys, key_to_pool_idx):
      - starts: shape (N_pool, max_ver) int64 — EffectiveStartDate as epoch days,
        sorted ascending per entity, padded with INT64_MAX.
      - keys: shape (N_pool, max_ver) int32 — CustomerKey per version slot,
        padded with IsCurrent=1 key.
      - key_to_pool_idx: dense int32 array mapping IsCurrent CustomerKey → pool index.

    At lookup time: for a sale with epoch-day D on customer pool index P,
    ``ver = searchsorted(starts[P], D, side='right') - 1`` gives the correct
    version slot.
    """
    try:
        all_df = pd.read_parquet(
            str(customers_path),
            columns=["CustomerID", "CustomerKey",
                     "EffectiveStartDate", "EffectiveEndDate"],
        )
    except (KeyError, ValueError):
        return None

    all_df["eff_start_days"] = pd.to_datetime(all_df["EffectiveStartDate"]).values.astype("datetime64[D]").astype(np.int64)

    N_pool = len(pool_customer_keys)

    # Vectorized reverse lookup: IsCurrent CustomerKey → pool index
    max_key = int(pool_customer_keys.max()) + 1
    key_to_pool_idx = np.full(max_key, -1, dtype=np.int32)
    key_to_pool_idx[pool_customer_keys] = np.arange(N_pool, dtype=np.int32)

    # Vectorized pool index mapping via dense lookup array
    max_cid = max(int(pool_customer_ids.max()), int(all_df["CustomerID"].max())) + 1
    cid_lookup = np.full(max_cid, -1, dtype=np.int32)
    cid_lookup[pool_customer_ids] = np.arange(N_pool, dtype=np.int32)

    all_cids = all_df["CustomerID"].to_numpy()
    pool_idx = cid_lookup[all_cids]
    mask = pool_idx >= 0
    pool_idx = pool_idx[mask]
    eff_start = all_df["eff_start_days"].to_numpy()[mask]
    ckey_arr = all_df["CustomerKey"].to_numpy(dtype=np.int32)[mask]

    # Sort by (pool_idx, eff_start_days) via lexsort (secondary key first)
    order = np.lexsort((eff_start, pool_idx))
    pool_idx = pool_idx[order]
    eff_start = eff_start[order]
    ckey_arr = ckey_arr[order]

    # Compute per-entity version slot indices using group boundaries
    group_starts = np.concatenate([[0], np.where(pool_idx[1:] != pool_idx[:-1])[0] + 1])
    slot = np.arange(len(pool_idx), dtype=np.int32)
    slot -= np.repeat(group_starts, np.diff(np.append(group_starts, len(pool_idx))))

    max_ver = int(slot.max()) + 1 if len(slot) > 0 else 1

    # Initialize: all slots default to IsCurrent=1 key, starts padded with MAX
    starts = np.full((N_pool, max_ver), np.iinfo(np.int64).max, dtype=np.int64)
    # Broadcast fill instead of np.tile (avoids full copy + intermediate array)
    keys = np.empty((N_pool, max_ver), dtype=np.int32)
    keys[:] = pool_customer_keys.astype(np.int32)[:, np.newaxis]

    # Cap slot indices at max_ver - 1
    valid = slot < max_ver
    pi = pool_idx[valid]
    si = slot[valid]

    # Vectorized scatter into output arrays
    starts[pi, si] = eff_start[valid]
    keys[pi, si] = ckey_arr[valid]

    # Clamp first version start to 0 (covers all time before second version)
    starts[pi, 0] = 0

    return starts, keys, key_to_pool_idx


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

    defaults_section = getattr(cfg, "defaults", None) or getattr(cfg, "_defaults", None)
    defaults_dates = getattr(defaults_section, "dates", None) if defaults_section is not None else None
    if not isinstance(defaults_dates, Mapping):
        raise KeyError("Missing defaults.dates in config")

    ov_sales_dates = _cfg_get(cfg, ["sales", "override", "dates"], default={})
    ov_sales_dates = ov_sales_dates if isinstance(ov_sales_dates, Mapping) else {}

    ov_global_dates = _cfg_get(cfg, ["dates", "override", "dates"], default={})
    ov_global_dates = ov_global_dates if isinstance(ov_global_dates, Mapping) else {}

    if start_date is None:
        start_date = (
            ov_sales_dates.get("start")
            or ov_global_dates.get("start")
            or getattr(defaults_dates, "start", None)
        )
    if end_date is None:
        end_date = (
            ov_sales_dates.get("end")
            or ov_global_dates.get("end")
            or getattr(defaults_dates, "end", None)
        )

    if not start_date or not end_date:
        raise KeyError("Could not resolve start/end dates from config")

    return str(start_date), str(end_date)


def _resolve_seed(cfg: Any, seed: Any, default_seed: int = 42) -> int:
    """Resolve sales seed via unified resolver.

    If *seed* is explicitly provided (not None), it takes priority.
    Otherwise delegates to config_precedence.resolve_seed.
    """
    from src.utils.config_precedence import resolve_seed as _resolve
    if seed is not None:
        return _int_or(seed, default_seed)
    sales_cfg = getattr(cfg, "sales", None) if not isinstance(cfg, dict) else cfg.get("sales")
    return _resolve(cfg, sales_cfg, fallback=default_seed)


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

    weekdays = _as_np(dates.weekday)

    # Weekday effect (0=Mon..6=Sun) — within-month date distribution only.
    # Year growth, monthly seasonality, promotional spikes, and one-off trends
    # are controlled by macro_demand settings in models.yaml.
    weekday_w = np.array([0.86, 0.91, 1.00, 1.12, 1.19, 1.08, 0.78], dtype=np.float64)
    wdw = weekday_w[weekdays]

    noise = rng.uniform(0.98, 1.02, size=n).astype(np.float64)

    weights = wdw * noise

    # Random blackout days (scalar blackout rate)
    blackout_rate = rng.uniform(0.10, 0.18)
    blackout = rng.random(n) < blackout_rate
    weights[_bool_mask(blackout)] = 0.0

    total = float(weights.sum())
    if total <= 0:
        weights[:] = 1.0 / n
    else:
        weights /= total
        # Clamp last element to prevent FP rounding from leaving sum != 1.0,
        # which causes searchsorted out-of-bounds (CLAUDE.md gotcha #16).
        # max(0, ...) guards against FP overshoot making it negative.
        weights[-1] = max(0.0, 1.0 - weights[:-1].sum())

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
    v = pd.to_numeric(s, errors="coerce").fillna(-1).astype("int64").to_numpy(copy=True)
    v[v < 0] = -1
    if v.shape[0] != n:
        v = np.resize(v, n)
    return v


def _resolve_partitioning(cfg: dict, partition_enabled: bool, partition_cols: Optional[Sequence[str]]) -> Tuple[bool, Optional[List[str]]]:
    sales_cfg = getattr(cfg, "sales", None) if isinstance(cfg, Mapping) else None
    sales_cfg = sales_cfg if isinstance(sales_cfg, Mapping) else {}

    cfg_enabled = None
    if hasattr(sales_cfg, "partition_enabled"):
        cfg_enabled = getattr(sales_cfg, "partition_enabled", None)
    elif hasattr(sales_cfg, "partitioning"):
        _part = getattr(sales_cfg, "partitioning", None)
        if isinstance(_part, Mapping):
            cfg_enabled = getattr(_part, "enabled", None)

    cfg_cols = None
    if hasattr(sales_cfg, "partition_cols"):
        cfg_cols = getattr(sales_cfg, "partition_cols", None)
    elif hasattr(sales_cfg, "partitioning"):
        _part = getattr(sales_cfg, "partitioning", None)
        if isinstance(_part, Mapping):
            cfg_cols = getattr(_part, "columns", None)

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


def _merge_fact_csv_chunks(
    csv_chunks: list,
    out_dir: Path,
    chunk_prefix: str,
    chunk_size: int,
    delete_chunks: bool,
) -> None:
    """Re-chunk CSV files for a sales fact table to respect chunk_size.

    Output files keep the existing chunk_prefix naming convention
    (e.g. ``sales_chunk0000.csv``, ``sales_return_chunk0000.csv``).

    Writes to a temporary directory first, then moves files into out_dir
    to avoid overwriting source chunks that share the same filename.
    """
    if not csv_chunks:
        return

    with open(csv_chunks[0], "r", encoding="utf-8") as f:
        header = f.readline()

    # Write to a temp directory so we never clobber source chunks.
    import tempfile
    tmp_dir = Path(tempfile.mkdtemp(dir=out_dir, prefix=".merge_"))

    tmp_files: list[Path] = []
    out_f = None
    rows_in_current = 0
    file_idx = 0

    def _open_next():
        nonlocal out_f, rows_in_current, file_idx
        if out_f is not None:
            out_f.close()
        path = tmp_dir / f"{chunk_prefix}{file_idx:04d}.csv"
        tmp_files.append(path)
        out_f = open(path, "w", newline="", encoding="utf-8")
        out_f.write(header)
        rows_in_current = 0
        file_idx += 1

    _open_next()
    for chunk_path in csv_chunks:
        with open(chunk_path, "r", encoding="utf-8") as in_f:
            next(in_f, None)  # skip header
            for line in in_f:
                if rows_in_current >= chunk_size:
                    _open_next()
                out_f.write(line)
                rows_in_current += 1
    if out_f is not None:
        out_f.close()

    # Remove original chunks
    for f in csv_chunks:
        try:
            f.unlink()
        except OSError:
            pass

    # Move merged files from temp into out_dir
    out_files: list[Path] = []
    for tmp_f in tmp_files:
        dest = out_dir / tmp_f.name
        tmp_f.replace(dest)
        out_files.append(dest)

    # Clean up temp directory
    try:
        tmp_dir.rmdir()
    except OSError:
        pass


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
    partition_enabled=False,
    partition_cols=None,
    return_manifest: bool = False,
):
    # ------------------------------------------------------------
    # Normalize cfg defaults (cfg is source-of-truth when call-site omits)
    # ------------------------------------------------------------
    cfg = cfg if isinstance(cfg, Mapping) else {}
    sales_cfg = getattr(cfg, "sales", None)
    sales_cfg = sales_cfg if isinstance(sales_cfg, Mapping) else {}

    file_format_cfg = getattr(sales_cfg, "file_format", None)
    if file_format_cfg is not None:
        file_format = _apply_cfg_default(file_format, "parquet", _str_or(file_format_cfg, "parquet").lower())

    merge_parquet = _apply_cfg_default(
        merge_parquet, False,
        _bool_or(getattr(sales_cfg, "merge_parquet", None), merge_parquet) if hasattr(sales_cfg, "merge_parquet") else None
    )
    merged_file = _apply_cfg_default(merged_file, "sales.parquet", getattr(sales_cfg, "merged_file", None) if hasattr(sales_cfg, "merged_file") else None)

    delete_chunks = _apply_cfg_default(
        delete_chunks, False,
        _bool_or(getattr(sales_cfg, "delete_chunks", None), delete_chunks) if hasattr(sales_cfg, "delete_chunks") else None
    )

    chunk_size = _apply_cfg_default(chunk_size, 2_000_000, _int_or(getattr(sales_cfg, "chunk_size", None), chunk_size) if hasattr(sales_cfg, "chunk_size") else None)
    row_group_size = _apply_cfg_default(row_group_size, 2_000_000, _int_or(sales_cfg.row_group_size, row_group_size))
    compression = _apply_cfg_default(compression, "snappy", sales_cfg.compression)
    workers = _apply_cfg_default(workers, None, sales_cfg.workers)
    tune_chunk = _apply_cfg_default(tune_chunk, False, _bool_or(sales_cfg.tune_chunk, tune_chunk))
    skip_order_cols = _apply_cfg_default(
        skip_order_cols,
        False,
        _bool_or(getattr(sales_cfg, "skip_order_cols", None), skip_order_cols) if hasattr(sales_cfg, "skip_order_cols") else None,
    )

    start_date, end_date = _resolve_date_range(cfg, start_date, end_date)
    optimize_after_merge = _bool_or(getattr(sales_cfg, "optimize", None), False)

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

    sales_output = _str_or(getattr(sales_cfg, "sales_output", None), "sales").lower()
    if sales_output not in {"sales", "sales_order", "both"}:
        raise ValueError(f"Invalid sales_output: {sales_output}")

    # ------------------------------------------------------------
    # Returns (optional)
    # ------------------------------------------------------------
    facts_enabled = _cfg_get(cfg, ["facts", "enabled"], default=[])
    facts_enabled = facts_enabled if isinstance(facts_enabled, list) else []

    _returns_obj = getattr(cfg, "returns", None)
    returns_cfg = _returns_obj if isinstance(_returns_obj, Mapping) else {}
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

    # Single parquet read: discover available columns via schema, then
    # load ALL needed columns in one I/O call instead of 3-5 separate opens.
    import pyarrow.parquet as _pq
    _cust_schema_names = set(_pq.read_schema(str(customers_path)).names)

    _cust_load_cols = ["CustomerKey", "CustomerStartDate", "CustomerEndDate"]
    # Backward compat: also load legacy columns if they still exist
    _cust_legacy = ["IsActiveInSales", "CustomerStartMonth", "CustomerEndMonth"]
    # SCD2 columns (previously a separate parquet open)
    _cust_scd2_cols_wanted = ["CustomerID", "IsCurrent"]
    # Weight columns (previously a separate parquet open)
    _cust_weight_cols = ["CustomerBaseWeight", "CustomerWeight"]
    # Geo column (previously a separate parquet open)
    _cust_geo_cols = ["GeographyKey"]

    _cust_all_cols = list(dict.fromkeys(
        _cust_load_cols
        + [c for c in _cust_legacy if c in _cust_schema_names]
        + [c for c in _cust_scd2_cols_wanted if c in _cust_schema_names]
        + [c for c in _cust_weight_cols if c in _cust_schema_names]
        + [c for c in _cust_geo_cols if c in _cust_schema_names]
    ))

    cust_df_full = load_parquet_df(customers_path, _cust_all_cols)

    if cust_df_full.empty:
        raise RuntimeError("customers.parquet is empty; cannot generate sales")

    # --- SCD2 customer deduplication ---
    _cust_scd2_detected = False
    _cust_pool_ids = None   # CustomerID array (parallel to pool) — set if SCD2 active
    # Save pre-filter geo column before SCD2 filtering (needed later for geo mapping)
    _cust_geo_full = _as_np(cust_df_full["GeographyKey"], np.int32) if "GeographyKey" in cust_df_full.columns else None
    _cust_is_current_full = cust_df_full["IsCurrent"].values if "IsCurrent" in cust_df_full.columns else None

    if ("CustomerID" in cust_df_full.columns and "IsCurrent" in cust_df_full.columns
            and (cust_df_full["IsCurrent"] == 0).any()):
        _cust_scd2_detected = True
        _n_before = len(cust_df_full)
        # Keep only IsCurrent=1 rows for the sampling pool
        _is_current_mask = cust_df_full["IsCurrent"] == 1
        cust_df = cust_df_full[_is_current_mask].reset_index(drop=True)
        cust_df = cust_df.drop(columns=["IsCurrent"], errors="ignore")
        _cust_pool_ids = _as_np(cust_df["CustomerID"], np.int32)
        info(f"Customer SCD2: dedup {_n_before:,} -> {len(cust_df):,} rows (IsCurrent=1 pool)")
    else:
        cust_df = cust_df_full.drop(columns=["IsCurrent"], errors="ignore")

    # Free the full DataFrame — cust_df now holds the filtered pool
    del cust_df_full

    customer_keys = _as_np(cust_df["CustomerKey"], np.int32)

    # --- Derive month indices from dates ---
    config_start = pd.to_datetime(start_date).to_period("M")

    if "CustomerStartMonth" in cust_df.columns:
        # Legacy path: use stored month indices
        customer_start_month = _as_np(cust_df["CustomerStartMonth"], np.int64)
    elif "CustomerStartDate" in cust_df.columns:
        cust_start_ts = pd.to_datetime(cust_df["CustomerStartDate"], errors="coerce")
        cust_start_period = cust_start_ts.dt.to_period("M")
        customer_start_month = (cust_start_period.apply(lambda p: p.ordinal) - config_start.ordinal).to_numpy(dtype=np.int64)
        customer_start_month = np.clip(customer_start_month, 0, None)
    else:
        customer_start_month = np.zeros(len(customer_keys), dtype=np.int64)

    if "CustomerEndMonth" in cust_df.columns:
        customer_end_month = _normalize_nullable_int_month(_as_np(cust_df["CustomerEndMonth"]), len(customer_keys))
    elif "CustomerEndDate" in cust_df.columns:
        cust_end_ts = pd.to_datetime(cust_df["CustomerEndDate"], errors="coerce")
        customer_end_month = np.full(len(customer_keys), -1, dtype=np.int64)
        valid_end = cust_end_ts.notna()
        if valid_end.any():
            end_periods = cust_end_ts[valid_end].dt.to_period("M")
            customer_end_month[valid_end.to_numpy()] = (end_periods.apply(lambda p: p.ordinal) - config_start.ordinal).to_numpy(dtype=np.int64)
    else:
        customer_end_month = np.full(len(customer_keys), -1, dtype=np.int64)

    # --- Derive is_active_in_sales from churn status ---
    if "IsActiveInSales" in cust_df.columns:
        # Legacy path
        is_active_in_sales = _as_np(cust_df["IsActiveInSales"], np.int32)
    else:
        # Active = not churned (no end date, or end date in the future)
        config_end = pd.to_datetime(end_date)
        cust_end_for_active = pd.to_datetime(cust_df.get("CustomerEndDate"), errors="coerce")
        is_active_in_sales = np.where(
            cust_end_for_active.isna() | (cust_end_for_active >= config_end),
            1, 0,
        ).astype(np.int32)

    # Extract customer weight from the already-loaded DataFrame (no extra parquet open)
    customer_base_weight = None
    for wcol in ("CustomerBaseWeight", "CustomerWeight"):
        if wcol in cust_df.columns:
            customer_base_weight = _as_np(cust_df[wcol], np.float64)
            break

    # Products: respect runner-bound active_product_np
    product_brand_key = None
    brand_names = None
    product_subcat_key = None
    products_path = parquet_folder_p / "products.parquet"
    assortment_cfg = (getattr(cfg, "stores", None) or {}).get("assortment") or {}

    def _brand_codes_from_series(s: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        # Guarantee no NA => no -1 codes
        s2 = s.fillna("Unknown").astype(str)
        codes, uniques = pd.factorize(s2, sort=True)
        return np.asarray(codes, dtype=np.int32), np.asarray(uniques, dtype=object)

    # Read schema once (metadata only) to discover available columns
    _prod_schema = set(_pq.read_schema(str(products_path)).names)
    _has_brand_col = "Brand" in _prod_schema
    _has_subcat_col = "SubcategoryKey" in _prod_schema
    _need_subcat = bool(assortment_cfg.get("enabled")) and _has_subcat_col

    # Determine SCD2 capability upfront so we include required columns in
    # the single product load below (avoids extra parquet opens later).
    _prod_has_scd2 = (
        "ProductID" in _prod_schema
        and "IsCurrent" in _prod_schema
        and "EffectiveStartDate" in _prod_schema
    )

    # Cached full product DataFrame — reused for SCD2 version table building
    _prod_df_full = None

    if getattr(State, "active_product_np", None) is not None:
        product_np = State.active_product_np
        active_keys = np.asarray(product_np[:, 0], dtype=np.int32)

        # Single load: Brand + optional SubcategoryKey + SCD2 columns (one parquet open)
        _prod_cols = ["ProductKey"]
        if _has_brand_col:
            _prod_cols.append("Brand")
        if _need_subcat:
            _prod_cols.append("SubcategoryKey")
        # Include SCD2 columns so we can reuse this DataFrame later
        if _prod_has_scd2:
            for _sc in ("ProductID", "IsCurrent"):
                if _sc not in _prod_cols:
                    _prod_cols.append(_sc)

        try:
            _prod_df = load_parquet_df(products_path, _prod_cols)
            # Keep the full DF for SCD2 use before dedup
            if _prod_has_scd2:
                _prod_df_full = _prod_df
            _prod_df_dedup = _prod_df.drop_duplicates("ProductKey", keep="first")
            _prod_keys = _prod_df_dedup["ProductKey"].to_numpy(dtype=np.int32)

            # Sorted-key lookup (avoids pandas reindex float64 promotion)
            _sort_idx = np.argsort(_prod_keys)
            _sorted_keys = _prod_keys[_sort_idx]
            _pos = np.clip(np.searchsorted(_sorted_keys, active_keys), 0, max(len(_sorted_keys) - 1, 0))
            _found = _sorted_keys[_pos] == active_keys

            if _has_brand_col:
                codes, brand_names = _brand_codes_from_series(_prod_df_dedup["Brand"])
                bk = np.full(len(active_keys), -1, dtype=np.int32)
                bk[_found] = codes[_sort_idx][_pos[_found]]
                if np.any(bk < 0):
                    info("Brand mapping missing/invalid for some ProductKeys; disabling brand_popularity for this run.")
                else:
                    product_brand_key = bk

            if _need_subcat and "SubcategoryKey" in _prod_df_dedup.columns:
                subcat_vals = _prod_df_dedup["SubcategoryKey"].to_numpy(dtype=np.int32)
                sc = np.zeros(len(active_keys), dtype=np.int32)
                sc[_found] = subcat_vals[_sort_idx][_pos[_found]]
                product_subcat_key = sc

            del _prod_df_dedup

        except (KeyError, ValueError, TypeError, OSError) as exc:
            info(f"Could not load/derive Brand from products.parquet ({type(exc).__name__}: {exc}); "
                 "disabling brand_popularity for this run.")
            product_brand_key = None

    else:
        # Full product path — single load with all needed columns
        _prod_cols = ["ProductKey", "ListPrice", "UnitCost"]
        if _has_brand_col:
            _prod_cols.append("Brand")
        if _need_subcat:
            _prod_cols.append("SubcategoryKey")
        # SCD2: load IsCurrent + ProductID to reuse for version table building
        if "IsCurrent" in _prod_schema:
            _prod_cols.append("IsCurrent")
        if _prod_has_scd2 and "ProductID" in _prod_schema:
            _prod_cols.append("ProductID")

        prod_df = load_parquet_df(products_path, _prod_cols)
        # Keep the full DF for SCD2 use before filtering
        if _prod_has_scd2:
            _prod_df_full = prod_df
        # SCD2: only use current version rows for the product pool
        if "IsCurrent" in prod_df.columns:
            prod_df = prod_df[prod_df["IsCurrent"] == 1].copy()
            prod_df = prod_df.drop(columns=["IsCurrent"], errors="ignore")
        if "ProductID" in prod_df.columns:
            prod_df = prod_df.drop(columns=["ProductID"], errors="ignore")
        prod_df["ProductKey"] = pd.to_numeric(prod_df["ProductKey"], errors="coerce")
        prod_df["ListPrice"] = pd.to_numeric(prod_df["ListPrice"], errors="coerce")
        prod_df["UnitCost"] = pd.to_numeric(prod_df["UnitCost"], errors="coerce")
        prod_df = prod_df.dropna(subset=["ProductKey", "ListPrice", "UnitCost"])
        prod_df["ProductKey"] = prod_df["ProductKey"].astype("int32", copy=False)

        product_np = np.column_stack([
            prod_df["ProductKey"].to_numpy(dtype=np.int32, copy=False),
            prod_df["ListPrice"].to_numpy(dtype=np.float64, copy=False),
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
        except (KeyError, ValueError, TypeError, OSError) as exc:
            info(f"Could not load product profile ({type(exc).__name__}: {exc}); using uniform product sampling.")
            product_popularity = None
            product_seasonality = None

    # Weighted date pool (deterministic) — needed by SCD2 grid builders below
    date_pool, date_prob = build_weighted_date_pool(start_date, end_date, seed)

    # --- SCD2 version lookup tables (per-entity, per-row resolution) ---
    _product_scd2_active = False
    _product_scd2_starts = None   # (N_pool, max_ver) int64
    _product_scd2_data = None     # (N_pool, max_ver, 3) float64
    _customer_scd2_active = False
    _customer_scd2_starts = None  # (N_pool, max_ver) int64
    _customer_scd2_keys = None    # (N_pool, max_ver) int32
    _cust_key_to_pool_idx = None

    # Detect product SCD2 and build version tables (reusing _prod_df_full
    # from the single product load above — no extra parquet opens).
    if _prod_has_scd2 and _prod_df_full is not None:
        try:
            _has_history = (_prod_df_full["IsCurrent"] == 0).any()
        except (KeyError, ValueError):
            _has_history = False

        if _has_history:
            try:
                _pid_current = _prod_df_full[_prod_df_full["IsCurrent"] == 1].drop_duplicates("ProductKey", keep="first")
                _pid_map = pd.Series(
                    _pid_current["ProductID"].to_numpy(dtype=np.int32),
                    index=_pid_current["ProductKey"].to_numpy(dtype=np.int32),
                )
                _pool_keys = np.asarray(product_np[:, 0], dtype=np.int32)
                _reindexed = _pid_map.reindex(_pool_keys)
                _unmapped = _reindexed.isna()
                if _unmapped.any():
                    info(f"Product SCD2: {int(_unmapped.sum())} pool keys have no "
                         f"IsCurrent=1 ProductID mapping; they will use current-version prices.")
                _pool_product_ids = _reindexed.fillna(-1).to_numpy(dtype=np.int32)
                del _pid_current, _pid_map, _reindexed

                _prod_result = _build_scd2_product_versions(
                    products_path, _pool_product_ids, product_np,
                )
                if _prod_result is not None:
                    _product_scd2_starts, _product_scd2_data = _prod_result
                    _product_scd2_active = True
                    info(f"Product SCD2: {_product_scd2_starts.shape[1]} max versions × "
                         f"{_product_scd2_starts.shape[0]:,} products")
            except (KeyError, ValueError, TypeError, OSError) as exc:
                info(f"Product SCD2 build failed ({type(exc).__name__}: {exc}); "
                     "using current-version prices for all months.")

    # Free the full product DataFrame now that SCD2 is built
    del _prod_df_full

    # Build customer SCD2 version tables (if dedup detected earlier)
    if _cust_scd2_detected and _cust_pool_ids is not None:
        _cust_result = _build_scd2_customer_versions(
            customers_path, customer_keys, _cust_pool_ids,
        )
        if _cust_result is not None:
            _customer_scd2_starts, _customer_scd2_keys, _cust_key_to_pool_idx = _cust_result
            _customer_scd2_active = True
            info(f"Customer SCD2: {_customer_scd2_starts.shape[1]} max versions × "
                 f"{_customer_scd2_starts.shape[0]:,} customers")

    # Stores: read ONCE (keys + geography + StoreType + dates for eligibility)
    _store_cols = ["StoreKey", "GeographyKey"]
    _store_path = parquet_folder_p / "stores.parquet"
    if _store_path.exists():
        _store_schema_names = set(_pq.read_schema(str(_store_path)).names)
        if "StoreType" in _store_schema_names:
            _store_cols.append("StoreType")
        if "OpeningDate" in _store_schema_names:
            _store_cols.append("OpeningDate")
        if "ClosingDate" in _store_schema_names:
            _store_cols.append("ClosingDate")
    store_df = load_parquet_df(_store_path, _store_cols)
    store_keys = _as_np(store_df["StoreKey"], np.int32)
    store_to_geo = dict(zip(_as_np(store_df["StoreKey"], np.int32), _as_np(store_df["GeographyKey"], np.int32)))

    # Store open/close months for eligibility filtering (month-int: months since epoch)
    store_open_month = None
    store_close_month = None
    # Day-level arrays for exact eligibility (indexed by position in store_keys)
    store_open_day = None
    store_close_day = None
    _FAR_PAST_DAY = np.datetime64("1900-01-01", "D")
    _FAR_FUTURE_DAY = np.datetime64("2262-04-11", "D")
    if "OpeningDate" in store_df.columns:
        _open_dt = pd.to_datetime(store_df["OpeningDate"]).values.astype("datetime64[M]")
        store_open_month = _open_dt.astype("int64").astype(np.int64)
        # Day-level: NaT → FAR_PAST (always open)
        _open_dt_d = pd.to_datetime(store_df["OpeningDate"]).values.astype("datetime64[D]")
        _open_nat = np.isnat(_open_dt_d)
        _open_dt_d[_open_nat] = _FAR_PAST_DAY
        store_open_day = _open_dt_d
    if "ClosingDate" in store_df.columns:
        _close_dt = pd.to_datetime(store_df["ClosingDate"]).values
        _close_nat_mask = np.isnat(_close_dt)
        _close_m = _close_dt.astype("datetime64[M]").astype("int64").astype(np.int64)
        # NaT → INT64_MAX means "never closed"
        _close_m[_close_nat_mask] = np.iinfo(np.int64).max
        store_close_month = _close_m
        # Day-level: NaT → FAR_FUTURE (never closed)
        _close_dt_d = _close_dt.astype("datetime64[D]")
        _close_dt_d[_close_nat_mask] = _FAR_FUTURE_DAY
        store_close_day = _close_dt_d

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
    # Employees / store assignments -> EmployeeKey
    # ------------------------------------------------------------
    emp_assign_path = parquet_folder_p / "employee_store_assignments.parquet"

    # Fail fast: the bridge is the single source of truth for EmployeeKey.
    # Without it, every Sales row would get EmployeeKey = -1.
    if not emp_assign_path.exists():
        raise FileNotFoundError(
            f"employee_store_assignments.parquet is required: {emp_assign_path}. "
            f"Run dimension generation first."
        )

    # config-driven allowlist: which RoleAtStore can appear as EmployeeKey
    salesperson_roles = _cfg_get(cfg, ["sales", "salesperson_roles"], default=None)
    if not (isinstance(salesperson_roles, list) and salesperson_roles):
        primary = _cfg_get(cfg, ["employees", "store_assignments", "primary_sales_role"], default="Sales Associate")
        salesperson_roles = [str(primary), ONLINE_SALES_REP_ROLE]

    _esa_base_cols = [
        "EmployeeKey", "StoreKey", "StartDate", "EndDate",
        "FTE", "RoleAtStore",
    ]
    try:
        emp_assign_df = load_parquet_df(
            emp_assign_path, cols=_esa_base_cols + ["IsPrimary"],
        )
    except (KeyError, ValueError):
        # Legacy parquet without IsPrimary column
        emp_assign_df = load_parquet_df(emp_assign_path, cols=_esa_base_cols)

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

    # ------------------------------------------------------------
    # Column correlation lookups (customer-geo, store-channel,
    # product-channel eligibility, promo-channel, delivery days)
    # ------------------------------------------------------------
    # 1) Geography: customer -> country, store -> country, country -> stores
    #    Reuse geo_df (GeographyKey, ISOCode already loaded above) — load Country
    _geo_country_path = parquet_folder_p / "geography.parquet"
    _geo_country_df = load_parquet_df(_geo_country_path, ["GeographyKey", "Country"])
    _unique_countries = _geo_country_df["Country"].fillna("Unknown").unique()
    _country_to_id = {c: i for i, c in enumerate(_unique_countries)}
    _n_countries = len(_unique_countries)

    # geo_to_country_id: dense array GeographyKey -> country_id
    _max_geo = int(_geo_country_df["GeographyKey"].max()) if len(_geo_country_df) else 0
    geo_to_country_id = np.full(_max_geo + 1, 0, dtype=np.int32)
    _geo_keys = _geo_country_df["GeographyKey"].to_numpy(dtype=np.int32)
    _geo_countries = _geo_country_df["Country"].fillna("Unknown").map(_country_to_id).to_numpy(dtype=np.int32)
    geo_to_country_id[_geo_keys] = _geo_countries

    # store_to_country_id: dense StoreKey -> country_id (via store_to_geo)
    _max_sk = int(store_keys.max()) if store_keys.size else 0
    store_to_country_id = np.full(_max_sk + 1, 0, dtype=np.int32)
    _sg_sk = np.fromiter(store_to_geo.keys(), dtype=np.int32, count=len(store_to_geo))
    _sg_gk = np.fromiter(store_to_geo.values(), dtype=np.int32, count=len(store_to_geo))
    _sg_valid = (_sg_sk <= _max_sk) & (_sg_gk <= _max_geo)
    store_to_country_id[_sg_sk[_sg_valid]] = geo_to_country_id[_sg_gk[_sg_valid]]

    # country_to_store_keys: list of arrays
    _sk_country_ids = store_to_country_id[store_keys.astype(np.int32)]
    country_to_store_keys = [
        store_keys[_sk_country_ids == cid].astype(np.int32)
        for cid in range(_n_countries)
    ]

    # customer_geo_key: customer pool index -> GeographyKey
    # Reuse data from the consolidated customer load (no extra parquet open).
    customer_geo_key = None
    if _cust_geo_full is not None:
        if _cust_scd2_detected and _cust_is_current_full is not None:
            # Filter to IsCurrent=1 rows (same filter applied to cust_df)
            customer_geo_key = _cust_geo_full[_cust_is_current_full == 1]
        else:
            customer_geo_key = _cust_geo_full
    elif "GeographyKey" in cust_df.columns:
        customer_geo_key = _as_np(cust_df["GeographyKey"], np.int32)
    # Free the full-length arrays now that they've been consumed
    del _cust_geo_full, _cust_is_current_full

    # 2) Store type -> valid SalesChannelKeys mapping
    # Channel classification:
    #   Physical channels: 1=Store, 10=Kiosk
    #   Digital channels:  2=Online, 6=Web, 7=MobileApp, 8=SocialCommerce
    #   Business channels: 4=B2B, 9=PartnerReseller
    #   Assisted channels: 5=CallCenter
    #   Marketplace:       3=Marketplace
    _PHYSICAL_CHANNELS = np.array([1, 5, 10], dtype=np.int16)          # Store, CallCenter, Kiosk
    _DIGITAL_CHANNELS = np.array([2, 3, 6, 7, 8], dtype=np.int16)     # Online, Marketplace, Web, MobileApp, Social
    _ALL_CHANNELS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.int16)

    # StoreType -> channel affinity. Physical stores skew physical+assisted;
    # Online/fulfillment stores skew digital. All store types can do B2B.
    _STORE_TYPE_CHANNEL_MAP: dict[str, tuple[np.ndarray, np.ndarray]] = {
        # (valid_keys, probability_weights)
        "Online":      (_DIGITAL_CHANNELS, np.array([0.35, 0.10, 0.20, 0.25, 0.10], dtype=np.float64)),
        "Fulfillment": (_DIGITAL_CHANNELS, np.array([0.35, 0.10, 0.20, 0.25, 0.10], dtype=np.float64)),
        # Physical store types: mostly physical channels + some digital
        "Retail":       (np.array([1, 2, 3, 5, 7, 10], dtype=np.int16),
                         np.array([0.50, 0.15, 0.05, 0.10, 0.10, 0.10], dtype=np.float64)),
        "Flagship":     (np.array([1, 2, 5, 7, 10], dtype=np.int16),
                         np.array([0.55, 0.15, 0.10, 0.10, 0.10], dtype=np.float64)),
        "Hypermarket":  (np.array([1, 2, 3, 5, 10], dtype=np.int16),
                         np.array([0.55, 0.15, 0.05, 0.10, 0.15], dtype=np.float64)),
        "Supermarket":  (np.array([1, 2, 5, 10], dtype=np.int16),
                         np.array([0.60, 0.15, 0.10, 0.15], dtype=np.float64)),
        "Convenience":  (np.array([1, 5, 10], dtype=np.int16),
                         np.array([0.70, 0.10, 0.20], dtype=np.float64)),
        "Outlet":       (np.array([1, 2, 5], dtype=np.int16),
                         np.array([0.65, 0.20, 0.15], dtype=np.float64)),
        "Warehouse":    (np.array([1, 2, 4, 9], dtype=np.int16),
                         np.array([0.30, 0.25, 0.25, 0.20], dtype=np.float64)),
    }
    # Default for unknown store types
    _DEFAULT_CHANNEL_MAP = (np.array([1, 2, 3, 5], dtype=np.int16),
                            np.array([0.40, 0.30, 0.15, 0.15], dtype=np.float64))

    # Build per-StoreKey channel arrays (dense, indexed by StoreKey)
    store_channel_keys_list: list = [None] * (_max_sk + 1)
    channel_prob_by_store_list: list = [None] * (_max_sk + 1)
    if store_type_map is not None:
        for sk in store_keys:
            sk_int = int(sk)
            st = store_type_map.get(sk_int, "")
            keys, probs = _STORE_TYPE_CHANNEL_MAP.get(st, _DEFAULT_CHANNEL_MAP)
            store_channel_keys_list[sk_int] = keys
            channel_prob_by_store_list[sk_int] = probs / probs.sum()
    else:
        # No store types available — uniform across all channels
        _uniform_p = np.ones(len(_ALL_CHANNELS), dtype=np.float64) / len(_ALL_CHANNELS)
        for sk in store_keys:
            store_channel_keys_list[int(sk)] = _ALL_CHANNELS
            channel_prob_by_store_list[int(sk)] = _uniform_p

    # 3) Product channel eligibility (from ProductProfile)
    product_channel_eligible = None
    _profile_path = parquet_folder_p / "product_profile.parquet"
    if _profile_path.exists():
        try:
            _elig_cols = ["ProductKey", "EligibleStore", "EligibleOnline", "EligibleMarketplace", "EligibleB2B"]
            _elig_df = pd.read_parquet(str(_profile_path), columns=_elig_cols)
            # Build dense array aligned with product_np rows (ProductKey -> row index)
            # product_np[:, 0] contains ProductKey
            _prod_keys_arr = product_np[:, 0].astype(np.int32)
            _max_pk = int(_prod_keys_arr.max()) if _prod_keys_arr.size else 0
            # Map ProductKey -> product_np row index
            _pk_to_row = np.full(_max_pk + 1, -1, dtype=np.int32)
            _pk_to_row[_prod_keys_arr] = np.arange(len(_prod_keys_arr), dtype=np.int32)
            # 4 groups: store=0, online=1, marketplace=2, b2b=3
            product_channel_eligible = np.ones((len(product_np), 4), dtype=np.int8)
            _elig_pks = _elig_df["ProductKey"].to_numpy(dtype=np.int32)
            _elig_mask = (_elig_pks <= _max_pk)
            _elig_pks_valid = _elig_pks[_elig_mask]
            _elig_rows = _pk_to_row[_elig_pks_valid]
            _mapped = _elig_rows >= 0
            _ri = _elig_rows[_mapped]
            _elig_mask_idx = np.where(_elig_mask)[0][_mapped]
            for col_idx, col_name in enumerate(["EligibleStore", "EligibleOnline",
                                                 "EligibleMarketplace", "EligibleB2B"]):
                product_channel_eligible[_ri, col_idx] = (
                    _elig_df[col_name].to_numpy(dtype=np.int8)[_elig_mask_idx]
                )
        except (KeyError, OSError):
            pass

    # 4) Promotion channel group (from PromotionCategory)
    promo_channel_group = np.zeros(len(promo_keys_all), dtype=np.int8)  # 0=any
    if not promo_df.empty and "PromotionCategory" in promo_df.columns:
        _cat_series = promo_df["PromotionCategory"].astype(str)
        promo_channel_group[_cat_series.isin({"Store", "Physical"}).to_numpy()] = 1
        promo_channel_group[_cat_series.isin({"Online", "Digital"}).to_numpy()] = 2

    # 5) Channel fulfillment days (from SalesChannels dimension)
    channel_fulfillment_days = np.full(11, 3, dtype=np.int16)  # default 3 days
    channel_fulfillment_days[0] = 0   # Unknown
    channel_fulfillment_days[1] = 0   # Store (immediate)
    channel_fulfillment_days[2] = 3   # Online
    channel_fulfillment_days[3] = 5   # Marketplace
    channel_fulfillment_days[4] = 7   # B2B
    channel_fulfillment_days[5] = 2   # CallCenter
    channel_fulfillment_days[6] = 3   # Web
    channel_fulfillment_days[7] = 3   # MobileApp
    channel_fulfillment_days[8] = 5   # SocialCommerce
    channel_fulfillment_days[9] = 7   # PartnerReseller
    channel_fulfillment_days[10] = 0  # Kiosk (immediate)
    _sc_path = parquet_folder_p / "sales_channels.parquet"
    if _sc_path.exists():
        try:
            _sc_df = pd.read_parquet(str(_sc_path))
            if "TypicalFulfillmentDays" in _sc_df.columns and "SalesChannelKey" in _sc_df.columns:
                _sc_keys = _sc_df["SalesChannelKey"].to_numpy(dtype=np.int32)
                _sc_days = _sc_df["TypicalFulfillmentDays"]
                _sc_valid = (_sc_keys >= 0) & (_sc_keys < len(channel_fulfillment_days)) & _sc_days.notna()
                channel_fulfillment_days[_sc_keys[_sc_valid]] = _sc_days.to_numpy(dtype=np.int16)[_sc_valid]
        except (KeyError, OSError):
            pass

    # Channel key -> eligibility group index (for product_channel_eligible)
    # 0=store, 1=online, 2=marketplace, 3=b2b
    _CHANNEL_TO_ELIG_GROUP = np.zeros(11, dtype=np.int8)
    _CHANNEL_TO_ELIG_GROUP[1] = 0   # Store -> EligibleStore
    _CHANNEL_TO_ELIG_GROUP[2] = 1   # Online -> EligibleOnline
    _CHANNEL_TO_ELIG_GROUP[3] = 2   # Marketplace -> EligibleMarketplace
    _CHANNEL_TO_ELIG_GROUP[4] = 3   # B2B -> EligibleB2B
    _CHANNEL_TO_ELIG_GROUP[5] = 1   # CallCenter -> EligibleOnline (same catalog)
    _CHANNEL_TO_ELIG_GROUP[6] = 1   # Web -> EligibleOnline
    _CHANNEL_TO_ELIG_GROUP[7] = 1   # MobileApp -> EligibleOnline
    _CHANNEL_TO_ELIG_GROUP[8] = 2   # SocialCommerce -> EligibleMarketplace
    _CHANNEL_TO_ELIG_GROUP[9] = 3   # PartnerReseller -> EligibleB2B
    _CHANNEL_TO_ELIG_GROUP[10] = 0  # Kiosk -> EligibleStore

    info(f"Column correlations: {_n_countries} countries, "
         f"product eligibility {'loaded' if product_channel_eligible is not None else 'not available'}, "
         f"promo channel groups {int((promo_channel_group > 0).sum())}/{len(promo_channel_group)}")

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

    # Auto-cap workers on Windows to prevent page-file thrashing during
    # spawn-mode init.  Each worker imports numpy/pandas/pyarrow (~350 MB).
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                             ("dwMemoryLoad", ctypes.c_ulong),
                             ("ullTotalPhys", ctypes.c_ulonglong),
                             ("ullAvailPhys", ctypes.c_ulonglong),
                             ("ullTotalPageFile", ctypes.c_ulonglong),
                             ("ullAvailPageFile", ctypes.c_ulonglong),
                             ("ullTotalVirtual", ctypes.c_ulonglong),
                             ("ullAvailVirtual", ctypes.c_ulonglong),
                             ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            stat = MEMORYSTATUSEX(dwLength=ctypes.sizeof(MEMORYSTATUSEX))
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            avail_mb = stat.ullAvailPhys / (1024 * 1024)
            from src.defaults import WORKER_OS_RESERVE_MB, WORKER_ESTIMATE_MB
            usable_mb = max(0, avail_mb - WORKER_OS_RESERVE_MB)
            per_worker_mb = WORKER_ESTIMATE_MB
            mem_cap = max(1, int(usable_mb / per_worker_mb))
            if mem_cap < n_workers:
                info(f"Auto-capping workers {n_workers} -> {mem_cap} (available RAM: {avail_mb:.0f} MB)")
                n_workers = mem_cap
        except (OSError, AttributeError) as exc:
            logging.getLogger(__name__).debug(
                "Could not query available RAM (%s); skipping worker cap", exc
            )

    info(f"Spawning {n_workers} worker processes...")

    # SalesOrderNumber RunId:
    # - If configured: sales.order_id_run_id (0..999)
    # - Else derive a stable 0..999 id from output folder + seed (unique per run folder)
    order_id_run_id_raw = sales_cfg.order_id_run_id
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
    if isinstance(_models, Mapping):
        from src.engine.config.config_schema import CustomersDemandConfig

        _cust_mdl = _models.customers
        if _cust_mdl is None:
            _cust_mdl = CustomersDemandConfig()
        configured_share = float(_cust_mdl.new_customer_share)
        configured_max_frac = float(_cust_mdl.max_new_fraction_per_month)

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

            updates = {}
            if needed_share > configured_share:
                updates["new_customer_share"] = needed_share
            if needed_frac > configured_max_frac:
                updates["max_new_fraction_per_month"] = needed_frac

            if updates:
                new_cust = _cust_mdl.model_copy(update=updates)
                debug(
                    f"Auto-adjusting customer discovery: new_customer_share "
                    f"{configured_share:.3f} -> {new_cust.new_customer_share:.3f}, "
                    f"max_new_fraction_per_month "
                    f"{configured_max_frac:.4f} -> {new_cust.max_new_fraction_per_month:.4f} "
                    f"({undiscovered:,} undiscovered customers across {total_rows:,} rows)."
                )
                _models_copy = _models.model_copy(update={"customers": new_cust})
                State.models_cfg = _models_copy

    # ------------------------------------------------------------
    # Worker configuration (keep keys stable for compatibility)
    # ------------------------------------------------------------
    worker_cfg = dict(
        product_np=product_np,
        product_brand_key=product_brand_key,
        brand_names=brand_names,
        store_keys=store_keys,
        store_open_month=store_open_month,
        store_close_month=store_close_month,
        store_open_day=store_open_day,
        store_close_day=store_close_day,
        promo_keys_all=promo_keys_all,
        promo_start_all=promo_start_all,
        promo_end_all=promo_end_all,
        new_customer_promo_keys=new_customer_promo_keys,
        new_customer_window_months=int((getattr(cfg, "promotions", None) or {}).get("new_customer_window_months", 3)),

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
        max_lines_per_order=int(getattr(sales_cfg, "max_lines_per_order", 5) or 5),

        sales_output=sales_output,

        no_discount_key=1,

        # Disable per-chunk header invariant validation by default (halves
        # group-by work in build_header_from_detail).  Enable via
        # sales.validate_header_invariants: true in config for debugging.
        validate_header_invariants=_bool_or(
            getattr(sales_cfg, "validate_header_invariants", None), False
        ),

        delta_output_folder=delta_output_folder,
        write_delta=write_delta,
        skip_order_cols=bool(skip_order_cols),
        skip_order_cols_requested=bool(skip_order_cols),

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

        # Column correlation data
        customer_geo_key=customer_geo_key,
        geo_to_country_id=geo_to_country_id,
        store_to_country_id=store_to_country_id,
        country_to_store_keys=country_to_store_keys,
        store_channel_keys=store_channel_keys_list,
        channel_prob_by_store=channel_prob_by_store_list,
        product_channel_eligible=product_channel_eligible,
        promo_channel_group=promo_channel_group,
        channel_fulfillment_days=channel_fulfillment_days,
        _channel_to_elig_group=_CHANNEL_TO_ELIG_GROUP,

        # SCD2 version lookup tables
        product_scd2_active=_product_scd2_active,
        product_scd2_starts=_product_scd2_starts,
        product_scd2_data=_product_scd2_data,
        customer_scd2_active=_customer_scd2_active,
        customer_scd2_starts=_customer_scd2_starts,
        customer_scd2_keys=_customer_scd2_keys,
        cust_key_to_pool_idx=_cust_key_to_pool_idx,
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
    _budget_obj = getattr(cfg, "budget", None)
    budget_cfg = _budget_obj if isinstance(_budget_obj, Mapping) else {}
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
        except (KeyError, ValueError, TypeError) as exc:
            info(f"Budget streaming aggregation: disabled ({type(exc).__name__}: {exc})")
            budget_enabled = False
            budget_acc = None
            worker_cfg["budget_enabled"] = False
    else:
        worker_cfg["budget_enabled"] = False

    # ------------------------------------------------------------
    # Inventory
    # ------------------------------------------------------------
    _inv_obj = getattr(cfg, "inventory", None)
    inv_cfg = _inv_obj if isinstance(_inv_obj, Mapping) else {}
    inventory_enabled = _INVENTORY_AVAILABLE and bool(inv_cfg.get("enabled", False))
    inventory_acc = None

    if inventory_enabled:
        inventory_acc = InventoryAccumulator()
        worker_cfg["inventory_enabled"] = True
        info("Inventory streaming aggregation: enabled")
    else:
        worker_cfg["inventory_enabled"] = False

    # ------------------------------------------------------------
    # Wishlists
    # ------------------------------------------------------------
    _wl_obj = getattr(cfg, "wishlists", None)
    wishlists_enabled = _WISHLISTS_AVAILABLE and bool(getattr(_wl_obj, "enabled", False))
    wishlists_acc = None

    if wishlists_enabled:
        wishlists_acc = WishlistAccumulator()
        worker_cfg["wishlists_enabled"] = True
        info("Wishlists streaming aggregation: enabled")
    else:
        worker_cfg["wishlists_enabled"] = False

    # ------------------------------------------------------------
    # Complaints
    # ------------------------------------------------------------
    _cc_obj = getattr(cfg, "complaints", None)
    complaints_enabled = _COMPLAINTS_AVAILABLE and bool(getattr(_cc_obj, "enabled", False))
    complaints_acc = None

    if complaints_enabled:
        complaints_acc = ComplaintsAccumulator()
        worker_cfg["complaints_enabled"] = True
        info("Complaints streaming aggregation: enabled")
    else:
        worker_cfg["complaints_enabled"] = False

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
        "customer_geo_key",
        "geo_to_country_id",
        "store_to_country_id",
        "product_channel_eligible",
        "promo_channel_group",
        "channel_fulfillment_days",
        "_channel_to_elig_group",
        "product_scd2_starts",
        "product_scd2_data",
        "customer_scd2_starts",
        "customer_scd2_keys",
        "cust_key_to_pool_idx",
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
        _build_store_subcat_matrix,
        _build_brand_prob_by_month_rotate_winner,
        _build_salesperson_effective_by_store,
        _DEFAULT_ASSORTMENT_COVERAGE,
        infer_T_from_date_pool,
        int_or,
        float_or,
    )

    # 1) brand_to_row_idx — pre-compute sorted index + offsets (two compact
    #    shared arrays) so workers reconstruct buckets without argsort.
    _brand_product_counts = None  # reused by brand_prob pre-build
    if product_brand_key is not None:
        _bk = np.asarray(product_brand_key, dtype=np.int32)
        _brand_product_counts = np.bincount(_bk).astype(np.float64)
        # Build flat sorted index + offsets for zero-copy brand bucket access
        _brand_order = np.argsort(_bk, kind="mergesort").astype(np.int32)
        _bk_sorted = _bk[_brand_order]
        _brand_starts = np.flatnonzero(np.r_[True, _bk_sorted[1:] != _bk_sorted[:-1]])
        B = int(_bk.max()) + 1
        _brand_offsets = np.zeros(B + 1, dtype=np.int64)
        for s_idx in range(len(_brand_starts)):
            k = int(_bk_sorted[_brand_starts[s_idx]])
            e = int(_brand_starts[s_idx + 1]) if s_idx + 1 < len(_brand_starts) else len(_bk_sorted)
            _brand_offsets[k + 1] = e
        # Forward-fill: offsets must be non-decreasing
        np.maximum.accumulate(_brand_offsets, out=_brand_offsets)
        worker_cfg["_brand_flat_idx"] = _shm.publish("brand_flat_idx", _brand_order)
        worker_cfg["_brand_flat_offsets"] = _shm.publish("brand_flat_off", _brand_offsets)
        del _brand_order, _bk_sorted, _brand_starts, _brand_offsets

    # 2) store-product assortment — compact subcat matrix (stores × subcats bool)
    #    Workers expand to row indices on-the-fly at init, avoiding a 500MB+
    #    shared memory allocation that blocks the main process for minutes.
    if assortment_cfg.get("enabled") and product_subcat_key is not None and store_type_map is not None:
        store_type_arr = np.array(
            [str(store_type_map.get(int(sk), "Supermarket")) for sk in store_keys],
            dtype=object,
        )
        coverage = assortment_cfg.get("coverage", _DEFAULT_ASSORTMENT_COVERAGE)
        assort_seed = int(assortment_cfg.get("seed", seed))
        _unique_subcats, _subcat_matrix = _build_store_subcat_matrix(
            store_keys=store_keys,
            store_type_arr=store_type_arr,
            product_subcat_key=product_subcat_key,
            coverage_cfg=coverage,
            seed=assort_seed,
        )
        # Publish tiny matrix via shared memory (~stores × subcats bytes)
        worker_cfg["_assortment_subcat_matrix"] = _shm.publish(
            "assort_matrix", _subcat_matrix,
        )
        worker_cfg["_assortment_unique_subcats"] = _shm.publish(
            "assort_subcats", _unique_subcats,
        )
        del _subcat_matrix, _unique_subcats

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
    if isinstance(models_cfg, Mapping):
        _brand_cfg = models_cfg.get("brand_popularity") if isinstance(models_cfg, Mapping) else None
        if _brand_cfg and product_brand_key is not None and product_brand_key.size > 0:
            _T = infer_T_from_date_pool(date_pool)
            _B = int(product_brand_key.max()) + 1
            _rng_bp = np.random.default_rng(int(int_or(_brand_cfg.get("seed"), 1234)))

            # Brand product counts (reused from section 1 if available)
            _bp_counts = _brand_product_counts if (_brand_product_counts is not None and len(_brand_product_counts) == _B) else None

            _brand_prob = _build_brand_prob_by_month_rotate_winner(
                _rng_bp,
                T=_T, B=_B,
                winner_boost=float_or(_brand_cfg.get("winner_boost"), 2.5),
                noise_sd=float_or(_brand_cfg.get("noise_sd"), 0.15),
                min_share=float_or(_brand_cfg.get("min_share"), 0.02),
                year_len_months=int_or(_brand_cfg.get("year_len_months"), 12),
                brand_product_counts=_bp_counts,
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
        if budget_acc is not None and isinstance(r, Mapping):
            budget_acc.add_sales(r.pop("_budget_agg", None))
            budget_acc.add_returns(r.pop("_returns_agg", None))

        if inventory_acc is not None and isinstance(r, Mapping):
            inventory_acc.add(r.pop("_inventory_agg", None))

        if wishlists_acc is not None and isinstance(r, Mapping):
            wishlists_acc.add(r.pop("_wishlists_agg", None))

        if complaints_acc is not None and isinstance(r, Mapping):
            complaints_acc.add(r.pop("_complaints_agg", None))
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

        if isinstance(r, Mapping):
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
                elif isinstance(val, Mapping) and "part" in val:
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
        return (created_files, manifest, budget_acc, inventory_acc, wishlists_acc, complaints_acc) if return_manifest else created_files

    if file_format == "csv":
        # Merge small CSV chunk files into chunk_size-respecting files.
        # Sales chunks are already ~chunk_size rows each, but SalesReturn
        # chunks are tiny (return_rate × chunk_size per worker).
        for t in tables:
            csv_dir = Path(output_paths.table_out_dir(t))
            spec = output_paths.spec(t)
            csv_chunks = sorted(csv_dir.glob(f"{spec.chunk_prefix}*.csv"))
            if len(csv_chunks) <= 1:
                continue
            _merge_fact_csv_chunks(csv_chunks, csv_dir, spec.chunk_prefix, chunk_size, delete_chunks)

        manifest = _build_sales_manifest()
        return (created_files, manifest, budget_acc, inventory_acc, wishlists_acc, complaints_acc) if return_manifest else created_files

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
        return (created_files, manifest, budget_acc, inventory_acc, wishlists_acc, complaints_acc) if return_manifest else created_files

    raise ValueError(f"Unknown file_format: {file_format}")
