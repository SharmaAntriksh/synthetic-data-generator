# ---------------------------------------------------------
#  STORES DIMENSION (PIPELINE READY – OPTIMIZED)
#  - Robust config parsing (handles seed: null)
#  - Avoids double-reading geography.parquet
#  - Generates dates at day granularity (date-only)
#  - Writes OpeningDate/ClosingDate as Arrow date32 via write_parquet_with_date32()
# ---------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Union

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


def _as_dict(x) -> Dict:
    return x if isinstance(x, dict) else {}


def _int_or(value, default: int) -> int:
    """Safe int parsing: handles None, '', and non-numeric values."""
    try:
        if value is None or value == "":
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


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
        effective_end = np.maximum(open_days, close_end_day)

        close_days = rng.integers(open_days, effective_end + 1, dtype=np.int64)
        close_d = close_days.astype("datetime64[D]")
        df.loc[closed_mask, "ClosingDate"] = pd.to_datetime(close_d.astype("datetime64[ns]")).normalize()

    # Additional attributes
    df["OpenFlag"] = (df["Status"] == "Open").astype(np.int64)
    df["SquareFootage"] = rng.integers(2000, 10000, size=num_stores, dtype=np.int64)
    df["EmployeeCount"] = rng.integers(10, 120, size=num_stores, dtype=np.int64)

    # Phone (vectorized)
    first = (store_key % 900) + 100          # 100..999
    second = store_key % 10000               # 0..9999
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

    version_cfg = {
        **store_cfg,
        "_geography_sig": _geography_signature(geo_keys),
    }

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
