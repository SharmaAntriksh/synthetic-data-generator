# ---------------------------------------------------------
#  STORES DIMENSION (PIPELINE READY – OPTIMIZED)
#  - Robust config parsing (handles seed: null)
#  - Avoids double-reading geography.parquet
#  - Generates dates at day granularity (date-only)
#  - Writes OpeningDate/ClosingDate as Arrow date32 via write_parquet_with_date32()
#  - Refactored SquareFootage + EmployeeCount to be type/status-aware and configurable
# ---------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.utils.logging_utils import info, skip, stage
from src.utils.output_utils import write_parquet_with_date32
from src.versioning.version_store import should_regenerate, save_version


# ---------------------------------------------------------
# Constants (kept local for clarity)
# ---------------------------------------------------------

_STORE_TYPES = np.array(["Supermarket", "Convenience", "Online", "Hypermarket"], dtype=object)
_STORE_STATUS = np.array(["Open", "Closed", "Renovating"], dtype=object)
_CLOSE_REASONS = np.array(["Low Sales", "Lease Ended", "Renovation", "Moved Location"], dtype=object)

_STORE_TYPES_P = np.array([0.50, 0.30, 0.10, 0.10], dtype=float)
_STORE_STATUS_P = np.array([0.85, 0.10, 0.05], dtype=float)


# ---------------------------------------------------------
# Internals
# ---------------------------------------------------------

def _require_cfg(cfg: Dict) -> Dict:
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dict")
    stores = cfg.get("stores")
    if not isinstance(stores, dict):
        raise KeyError("Missing required config section: 'stores'")
    return stores


def _as_dict(x: Any) -> Dict:
    return x if isinstance(x, dict) else {}


def _int_or(value: Any, default: int) -> int:
    """Safe int parsing: handles None, '', and non-numeric values."""
    try:
        if value is None or value == "":
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _float_or(value: Any, default: float) -> float:
    """Safe float parsing: handles None, '', and non-numeric values."""
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _pick_seed(cfg: Dict, store_cfg: Dict, fallback: int = 42) -> int:
    """
    Seed precedence (robust to nulls):
      stores.override.seed -> stores.seed -> defaults.seed -> fallback
    """
    override = _as_dict(store_cfg.get("override"))
    seed = override.get("seed")
    if seed is None:
        seed = store_cfg.get("seed")
    if seed is None:
        seed = _as_dict(cfg.get("defaults")).get("seed")
    return _int_or(seed, fallback)


def _as_date64d(x: Union[str, pd.Timestamp]) -> np.datetime64:
    """Convert to numpy datetime64[D] (date-only)."""
    ts = pd.to_datetime(x).normalize()
    return np.datetime64(ts.date(), "D")


def _rand_dates_d(
    rng: np.random.Generator,
    start_d: np.datetime64,
    end_d: np.datetime64,
    size: int,
) -> np.ndarray:
    """
    Random date-only array in [start_d, end_d] inclusive.
    Returns numpy datetime64[D].
    """
    if end_d < start_d:
        raise ValueError(f"Invalid date range: {start_d}..{end_d}")

    start_i = start_d.astype("int64")
    end_i = end_d.astype("int64")

    days = rng.integers(start_i, end_i + 1, size=size, dtype=np.int64)
    return days.astype("datetime64[D]")


def _geography_signature(keys: np.ndarray) -> Dict:
    """
    Lightweight signature so changes in geography trigger regeneration.
    Uses row count + min/max GeographyKey.
    """
    if keys.size == 0:
        return {"rows": 0, "min_key": None, "max_key": None}
    return {
        "rows": int(keys.size),
        "min_key": int(keys.min()),
        "max_key": int(keys.max()),
    }


def _range2(x: Any, default_lo: int, default_hi: int) -> Tuple[int, int]:
    if isinstance(x, (list, tuple)) and len(x) == 2:
        return _int_or(x[0], default_lo), _int_or(x[1], default_hi)
    return int(default_lo), int(default_hi)


def _square_footage_from_cfg(
    rng: np.random.Generator,
    store_type: pd.Series,
    sqft_cfg: Dict,
    n: int,
) -> np.ndarray:
    """
    Generates SquareFootage with optional by-type ranges.
    """
    sqft_cfg = _as_dict(sqft_cfg)

    # Defaults tuned to look realistic and to support employee scaling later.
    by_type = _as_dict(sqft_cfg.get("by_type")) or {
        "Convenience": [800, 2500],
        "Supermarket": [3000, 9000],
        "Hypermarket": [10000, 30000],
        "Online": [1500, 6000],
    }

    clamp_min = _int_or(sqft_cfg.get("clamp_min"), 500)
    clamp_max = _int_or(sqft_cfg.get("clamp_max"), 40000)

    st = store_type.astype(str).to_numpy()
    out = np.empty(n, dtype=np.int64)

    # Fill known types
    for t, r in by_type.items():
        lo, hi = _range2(r, 2000, 10000)
        m = st == str(t)
        if m.any():
            out[m] = rng.integers(lo, hi + 1, size=int(m.sum()), dtype=np.int64)

    # Unknown type fallback
    unk = ~np.isin(st, np.array(list(by_type.keys()), dtype=object))
    if unk.any():
        out[unk] = rng.integers(2000, 10000 + 1, size=int(unk.sum()), dtype=np.int64)

    return np.clip(out, clamp_min, clamp_max).astype(np.int64)


def _employee_count_from_cfg(
    rng: np.random.Generator,
    store_type: pd.Series,
    status: pd.Series,
    square_footage: pd.Series,
    emp_cfg: Dict,
) -> np.ndarray:
    """
    EmployeeCount semantics:
      - total headcount for the store (includes store manager)
      - Closed stores: usually 0
      - Renovating stores: small crew (0..3 default)
      - Open stores: type-aware small ranges (demo-friendly)

    Config (optional) under stores.employee_count:
      mode: by_type | scaled_sqft
      open_by_type: { Convenience: [4,10], Supermarket: [8,16], Hypermarket: [12,24], Online: [3,8] }
      renovating_range: [0,3]
      closed_range: [0,0]
      clamp_min, clamp_max
      sqft_per_employee: mapping used by scaled_sqft
      noise_sd: float used by scaled_sqft
    """
    emp_cfg = _as_dict(emp_cfg)

    mode = str(emp_cfg.get("mode") or "by_type").strip().lower()
    open_by_type = _as_dict(emp_cfg.get("open_by_type")) or {
        "Convenience": [4, 10],
        "Supermarket": [8, 16],
        "Hypermarket": [12, 24],
        "Online": [3, 8],
    }

    renovating_lo, renovating_hi = _range2(emp_cfg.get("renovating_range"), 0, 3)
    closed_lo, closed_hi = _range2(emp_cfg.get("closed_range"), 0, 0)
    clamp_min = _int_or(emp_cfg.get("clamp_min"), 0)
    clamp_max = _int_or(emp_cfg.get("clamp_max"), 40)

    n = len(store_type)
    out = np.zeros(n, dtype=np.int64)

    st = store_type.astype(str).to_numpy()
    ss = status.astype(str).to_numpy()

    open_mask = ss == "Open"
    reno_mask = ss == "Renovating"
    closed_mask = ss == "Closed"

    # Closed / Renovating first
    if closed_mask.any():
        out[closed_mask] = rng.integers(closed_lo, closed_hi + 1, size=int(closed_mask.sum()), dtype=np.int64)
    if reno_mask.any():
        out[reno_mask] = rng.integers(renovating_lo, renovating_hi + 1, size=int(reno_mask.sum()), dtype=np.int64)

    # Open stores
    if open_mask.any():
        if mode == "scaled_sqft":
            sqft_per_employee = _as_dict(emp_cfg.get("sqft_per_employee")) or {
                "Convenience": 900,
                "Supermarket": 550,
                "Hypermarket": 380,
                "Online": 1200,
            }
            noise_sd = _float_or(emp_cfg.get("noise_sd"), 1.5)

            sqft = square_footage.astype(np.float64).to_numpy()
            for t, r in open_by_type.items():
                lo, hi = _range2(r, 4, 10)
                m = open_mask & (st == str(t))
                if not m.any():
                    continue
                spe = float(sqft_per_employee.get(t, 700))
                base = np.rint(sqft[m] / max(spe, 50.0) + rng.normal(0.0, noise_sd, size=int(m.sum())))
                base = np.clip(base, lo, hi).astype(np.int64)
                out[m] = base

            # Unknown types fallback
            unk = open_mask & ~np.isin(st, np.array(list(open_by_type.keys()), dtype=object))
            if unk.any():
                out[unk] = rng.integers(4, 10 + 1, size=int(unk.sum()), dtype=np.int64)
        else:
            # by_type (default)
            for t, r in open_by_type.items():
                lo, hi = _range2(r, 4, 10)
                m = open_mask & (st == str(t))
                if not m.any():
                    continue
                out[m] = rng.integers(lo, hi + 1, size=int(m.sum()), dtype=np.int64)

            # Unknown types fallback
            unk = open_mask & ~np.isin(st, np.array(list(open_by_type.keys()), dtype=object))
            if unk.any():
                out[unk] = rng.integers(4, 10 + 1, size=int(unk.sum()), dtype=np.int64)

        # ensure open stores have at least 1 (so a manager conceptually exists)
        out[open_mask] = np.maximum(out[open_mask], 1)

    return np.clip(out, clamp_min, clamp_max).astype(np.int64)


# ---------------------------------------------------------
# Generator
# ---------------------------------------------------------

def generate_store_table(
    *,
    geo_keys: np.ndarray,
    num_stores: int = 200,
    opening_start: str = "2018-01-01",
    opening_end: str = "2023-01-31",
    closing_end: str = "2025-12-31",
    seed: int = 42,
    square_footage_cfg: Optional[Dict] = None,
    employee_count_cfg: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Generate synthetic store dimension table.

    Output columns (unchanged):
      StoreKey, StoreName, StoreType, Status, GeographyKey,
      OpeningDate, ClosingDate, OpenFlag, SquareFootage, EmployeeCount,
      StoreManager, Phone, StoreDescription, CloseReason
    """
    num_stores = _int_or(num_stores, 200)
    if num_stores <= 0:
        raise ValueError(f"num_stores must be > 0, got {num_stores}")

    if not isinstance(geo_keys, np.ndarray) or geo_keys.size == 0:
        raise ValueError("geo_keys must be a non-empty numpy array of GeographyKey values")

    rng = np.random.default_rng(_int_or(seed, 42))

    # Base structure
    store_key = np.arange(1, num_stores + 1, dtype=np.int64)
    df = pd.DataFrame({"StoreKey": store_key})

    # Vectorized names
    key_str4 = pd.Series(store_key).astype(str).str.zfill(4)
    df["StoreName"] = "Store #" + key_str4
    df["StoreManager"] = "Manager " + key_str4

    # Categories
    df["StoreType"] = rng.choice(_STORE_TYPES, size=num_stores, p=_STORE_TYPES_P)
    df["Status"] = rng.choice(_STORE_STATUS, size=num_stores, p=_STORE_STATUS_P)

    # GeographyKey assignment
    df["GeographyKey"] = rng.choice(geo_keys.astype(np.int64), size=num_stores, replace=True)

    # Date generation at DAY granularity
    open_start_d = _as_date64d(opening_start)
    open_end_d = _as_date64d(opening_end)
    close_end_d = _as_date64d(closing_end)

    opening_d = _rand_dates_d(rng, open_start_d, open_end_d, num_stores)
    df["OpeningDate"] = pd.to_datetime(opening_d.astype("datetime64[ns]")).normalize()

    # ClosingDate only for Closed stores, always >= OpeningDate
    df["ClosingDate"] = pd.NaT
    closed_mask = (df["Status"].to_numpy() == "Closed")

    if closed_mask.any():
        open_days = opening_d.astype("int64")[closed_mask]  # days since epoch
        close_end_day = close_end_d.astype("int64")

        # ensure the end bound is >= start bound
        effective_end = np.maximum(open_days, close_end_day)
        close_days = rng.integers(open_days, effective_end + 1, dtype=np.int64)

        close_d = close_days.astype("datetime64[D]")
        df.loc[closed_mask, "ClosingDate"] = pd.to_datetime(close_d.astype("datetime64[ns]")).normalize()

    # Additional attributes
    df["OpenFlag"] = (df["Status"] == "Open").astype(np.int64)

    # SquareFootage (type-aware, configurable)
    df["SquareFootage"] = _square_footage_from_cfg(
        rng=rng,
        store_type=df["StoreType"],
        sqft_cfg=_as_dict(square_footage_cfg),
        n=num_stores,
    )

    # EmployeeCount (type/status-aware, configurable; demo-friendly defaults)
    df["EmployeeCount"] = _employee_count_from_cfg(
        rng=rng,
        store_type=df["StoreType"],
        status=df["Status"],
        square_footage=df["SquareFootage"],
        emp_cfg=_as_dict(employee_count_cfg),
    )

    # Phone (vectorized)
    first = (store_key % 900) + 100  # 100..999
    second = store_key % 10000  # 0..9999
    df["Phone"] = (
        "(555) "
        + pd.Series(first).astype(str).str.zfill(3)
        + "-"
        + pd.Series(second).astype(str).str.zfill(4)
    )

    df["StoreDescription"] = df["StoreType"].astype(str) + " located in GeographyKey " + df["GeographyKey"].astype(str)

    df["CloseReason"] = ""
    if closed_mask.any():
        df.loc[closed_mask, "CloseReason"] = rng.choice(_CLOSE_REASONS, size=int(closed_mask.sum()))

    return df


# ---------------------------------------------------------
# Pipeline entrypoint
# ---------------------------------------------------------

def run_stores(cfg: Dict, parquet_folder: Path) -> None:
    """
    Pipeline wrapper for stores dimension generation.
    - version checks
    - geography dependency signature
    - Parquet write with Arrow date32 for OpeningDate/ClosingDate (Power Query Date)
    """
    store_cfg = _require_cfg(cfg)

    parquet_folder = Path(parquet_folder)
    parquet_folder.mkdir(parents=True, exist_ok=True)

    out_path = parquet_folder / "stores.parquet"
    geo_path = parquet_folder / "geography.parquet"

    if not geo_path.exists():
        raise FileNotFoundError(f"Missing geography parquet: {geo_path}")

    force = bool(store_cfg.get("_force_regenerate", False))

    # Load geography keys once (avoid re-read in generator)
    geo = pd.read_parquet(geo_path, columns=["GeographyKey"])
    if "GeographyKey" not in geo.columns:
        raise ValueError("geography.parquet missing required column: GeographyKey")
    geo_keys = geo["GeographyKey"].astype(np.int64).to_numpy()

    # Pull optional nested configs
    sqft_cfg = _as_dict(store_cfg.get("square_footage"))
    emp_cfg = _as_dict(store_cfg.get("employee_count"))

    # Versioning config:
    # - keep backwards compatibility with existing configs
    # - bump schema_version to force regen once because EmployeeCount semantics changed
    version_cfg = dict(store_cfg)
    version_cfg.pop("_force_regenerate", None)
    version_cfg["schema_version"] = 2
    version_cfg["_geography_sig"] = _geography_signature(geo_keys)

    if not force and not should_regenerate("stores", version_cfg, out_path):
        skip("Stores up-to-date; skipping.")
        return

    # Optional parquet settings
    compression = store_cfg.get("parquet_compression", "snappy")
    compression_level = store_cfg.get("parquet_compression_level", None)
    force_date32 = bool(store_cfg.get("force_date32", True))

    opening_cfg = _as_dict(store_cfg.get("opening"))

    with stage("Generating Stores"):
        df = generate_store_table(
            geo_keys=geo_keys,
            num_stores=_int_or(store_cfg.get("num_stores"), 200),
            opening_start=opening_cfg.get("start") or "1995-01-01",
            opening_end=opening_cfg.get("end") or "2020-12-31",
            closing_end=store_cfg.get("closing_end") or "2025-12-31",
            seed=_pick_seed(cfg, store_cfg, fallback=42),
            square_footage_cfg=sqft_cfg,
            employee_count_cfg=emp_cfg,
        )

        # Cast only the intended date fields to Arrow date32
        write_parquet_with_date32(
            df,
            out_path,
            date_cols=["OpeningDate", "ClosingDate"],
            cast_all_datetime=False,
            compression=str(compression),
            compression_level=(int(compression_level) if compression_level is not None else None),
            force_date32=force_date32,
        )

    save_version("stores", version_cfg, out_path)
    info(f"Stores dimension written: {out_path}")
