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

# ----------------------------
# Naming (demo-friendly)
# ----------------------------

_BRANDS = np.array(
    [
        "Northwind Market", "Contoso Mart", "Fabrikam Foods", "Woodgrove Grocers",
        "Adventure Works Retail", "Tailspin Superstores", "Wingtip Fresh", "Proseware Market",
        "CitySquare Grocers", "Harborview Market", "Summit Retail", "BlueSky Foods",
    ],
    dtype=object,
)

_AREAS = np.array(
    [
        "Downtown", "Uptown", "Midtown", "Riverside", "Lakeside", "Hillcrest",
        "Old Town", "West End", "Eastside", "Northgate", "Southpark", "Harbor",
        "Airport", "Market District", "Central", "University",
    ],
    dtype=object,
)

_MANAGER_FIRST = np.array(
    [
        "James","John","Robert","Michael","William","David","Richard","Joseph","Thomas","Charles",
        "Christopher","Daniel","Matthew","Anthony","Mark","Steven","Paul","Andrew","Joshua","Ryan",
        "Mary","Patricia","Jennifer","Linda","Elizabeth","Barbara","Susan","Jessica","Sarah","Karen",
        "Nancy","Lisa","Margaret","Sandra","Ashley","Kimberly","Emily","Donna","Michelle","Laura",
        "Alex","Jordan","Taylor","Casey","Morgan","Riley","Jamie","Avery","Cameron","Quinn",
    ],
    dtype=object,
)

_MANAGER_LAST = np.array(
    [
        "Smith","Johnson","Williams","Brown","Jones","Miller","Davis","Wilson","Anderson","Thomas",
        "Taylor","Moore","Jackson","Martin","Lee","Perez","Thompson","White","Harris","Clark",
        "Lewis","Robinson","Walker","Young","Allen","King","Wright","Scott","Green","Baker",
        "Adams","Nelson","Hill","Campbell","Mitchell","Carter","Roberts","Turner","Phillips","Parker",
    ],
    dtype=object,
)

_ONLINE_SUFFIX = np.array(
    ["Online", "Digital", "E-Commerce", "Web Store", "Direct"],
    dtype=object,
)


# ---------------------------------------------------------
# Internals
# ---------------------------------------------------------
def _safe_read_geography(geo_path: Path) -> pd.DataFrame:
    """
    Reads geography.parquet with best-effort columns.
    Falls back to reading full file if column projection fails.
    """
    cols = [
        "GeographyKey",
        "City",
        "StateProvinceName",
        "CountryRegionName",
        "Country",
        "RegionCountryName",
        "ISOCode",
    ]
    try:
        return pd.read_parquet(geo_path, columns=cols)
    except Exception:
        return pd.read_parquet(geo_path)


def _build_location_maps(geo: pd.DataFrame) -> tuple[dict[int, str], dict[int, str]]:
    """
    Returns (loc_short_map, loc_full_map) keyed by GeographyKey.
    Uses whatever columns exist; always falls back to 'Geo <key>'.
    """
    geo = geo.copy()
    if "GeographyKey" not in geo.columns:
        raise ValueError("geography.parquet missing required column: GeographyKey")

    def col(name: str) -> pd.Series | None:
        return geo[name].astype(str) if name in geo.columns else None

    city = col("City")
    state = col("StateProvinceName")

    country = col("CountryRegionName")
    if country is None:
        country = col("Country")
    if country is None:
        country = col("RegionCountryName")

    gk = geo["GeographyKey"].astype("int64")

    # Short = City else State else Country else Geo <key>
    if city is not None:
        loc_short = city
    elif state is not None:
        loc_short = state
    elif country is not None:
        loc_short = country
    else:
        loc_short = pd.Series([""] * len(geo))

    # Full = "City, State, Country" from available parts
    parts: list[pd.Series] = []
    if city is not None:
        parts.append(city)
    if state is not None:
        parts.append(state)
    if country is not None:
        parts.append(country)

    if parts:
        loc_full = parts[0]
        for p in parts[1:]:
            loc_full = loc_full + ", " + p
    else:
        loc_full = pd.Series([""] * len(geo))

    short_map = {
        int(k): (str(s).strip() if str(s).strip() else f"Geo {int(k)}")
        for k, s in zip(gk, loc_short)
    }
    full_map = {
        int(k): (str(s).strip() if str(s).strip() else f"Geo {int(k)}")
        for k, s in zip(gk, loc_full)
    }
    return short_map, full_map


def _require_cfg(cfg: Dict) -> Dict:
    stores_cfg = cfg.get("stores")
    if not isinstance(stores_cfg, dict):
        raise ValueError("config missing required block: stores")
    return stores_cfg


def _as_dict(v: Any) -> Dict:
    return v if isinstance(v, dict) else {}


def _int_or(v: Any, default: int) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except Exception:
        return default


def _float_or(v: Any, default: float) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except Exception:
        return default


def _pick_seed(cfg: Dict, local_cfg: Dict, fallback: int = 42) -> int:
    # Prefer local override, else global seed, else fallback
    if "seed" in local_cfg and local_cfg["seed"] is not None:
        return _int_or(local_cfg["seed"], fallback)
    if "seed" in cfg and cfg["seed"] is not None:
        return _int_or(cfg["seed"], fallback)
    return fallback


def _as_date64d(s: Union[str, np.datetime64]) -> np.datetime64:
    if isinstance(s, np.datetime64):
        return s.astype("datetime64[D]")
    # pandas handles many formats reliably
    return np.datetime64(pd.to_datetime(str(s)).date(), "D")


def _rand_dates_d(rng: np.random.Generator, start_d: np.datetime64, end_d: np.datetime64, n: int) -> np.ndarray:
    """
    Returns np.datetime64[D] array in [start_d, end_d], inclusive.
    """
    start_i = start_d.astype("int64")
    end_i = end_d.astype("int64")
    if end_i < start_i:
        start_i, end_i = end_i, start_i
    days = rng.integers(start_i, end_i + 1, size=n, dtype=np.int64)
    return days.astype("datetime64[D]")


def _geography_signature(geo_keys: np.ndarray) -> str:
    # Stable, cheap signature to detect upstream geography changes
    arr = np.asarray(geo_keys, dtype=np.int64)
    if arr.size == 0:
        return "empty"
    return f"{int(arr.size)}:{int(arr.min())}:{int(arr.max())}:{int(arr[: min(5, arr.size)].sum())}"


def _range2(v: Any, default_lo: float, default_hi: float) -> tuple[float, float]:
    """
    Parse a 2-tuple/list like [lo, hi]. Returns (lo, hi) where hi >= lo.
    """
    lo = default_lo
    hi = default_hi
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        lo = _float_or(v[0], default_lo)
        hi = _float_or(v[1], default_hi)
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


def _square_footage_from_cfg(
    *,
    rng: np.random.Generator,
    store_type: pd.Series,
    sqft_cfg: Dict,
    n: int,
) -> np.ndarray:
    """
    Config shape (optional):
      stores:
        square_footage:
          default: [5000, 60000]
          Supermarket: [15000, 80000]
          Convenience: [1200, 8000]
          Hypermarket: [50000, 200000]
          Online: [200, 2000]
    """
    default_lo, default_hi = _range2(sqft_cfg.get("default"), 5000.0, 60000.0)

    out = np.empty(n, dtype=np.int64)
    st = store_type.astype(str).to_numpy()

    for t in np.unique(st):
        lo, hi = _range2(sqft_cfg.get(t), default_lo, default_hi)
        mask = st == t
        if mask.any():
            vals = rng.uniform(lo, hi, size=int(mask.sum()))
            out[mask] = np.maximum(1, np.round(vals)).astype(np.int64)

    # Any gaps -> default
    missing = out == 0
    if missing.any():
        vals = rng.uniform(default_lo, default_hi, size=int(missing.sum()))
        out[missing] = np.maximum(1, np.round(vals)).astype(np.int64)

    return out


def _employee_count_from_cfg(
    *,
    rng: np.random.Generator,
    store_type: pd.Series,
    status: pd.Series,
    square_footage: pd.Series,
    emp_cfg: Dict,
) -> np.ndarray:
    """
    Config shape (optional):
      stores:
        employee_count:
          base_per_1000_sqft: 0.35
          online_base: [5, 60]
          closed_multiplier: 0.15
          renovating_multiplier: 0.6
          min: 3
          max: 800
    """
    base_rate = _float_or(emp_cfg.get("base_per_1000_sqft"), 0.35)
    online_lo, online_hi = _range2(emp_cfg.get("online_base"), 5.0, 60.0)

    closed_mult = _float_or(emp_cfg.get("closed_multiplier"), 0.15)
    reno_mult = _float_or(emp_cfg.get("renovating_multiplier"), 0.60)

    emp_min = _int_or(emp_cfg.get("min"), 3)
    emp_max = _int_or(emp_cfg.get("max"), 800)

    st = store_type.astype(str).to_numpy()
    ss = status.astype(str).to_numpy()
    sqft = square_footage.to_numpy(dtype=np.float64)

    # baseline by sqft
    baseline = np.maximum(1.0, (sqft / 1000.0) * base_rate)

    # online override (small ops)
    online_mask = st == "Online"
    if online_mask.any():
        baseline[online_mask] = rng.uniform(online_lo, online_hi, size=int(online_mask.sum()))

    # status scaling
    closed_mask = ss == "Closed"
    if closed_mask.any():
        baseline[closed_mask] = baseline[closed_mask] * closed_mult

    reno_mask = ss == "Renovating"
    if reno_mask.any():
        baseline[reno_mask] = baseline[reno_mask] * reno_mult

    # jitter + clamp
    jitter = rng.normal(loc=1.0, scale=0.10, size=baseline.size)
    baseline = baseline * np.clip(jitter, 0.7, 1.4)

    out = np.round(baseline).astype(np.int64)
    out = np.clip(out, emp_min, emp_max)
    return out


# ---------------------------------------------------------
# Generator
# ---------------------------------------------------------


def _sample_geography_keys(
    *,
    rng: np.random.Generator,
    geo_keys: np.ndarray,
    n: int,
    iso_by_geo: Optional[dict[int, str]] = None,
    ensure_iso_coverage: bool = False,
) -> np.ndarray:
    """
    Sample GeographyKey values for Stores.

    Default behavior: uniform random choice with replacement (backwards compatible).

    If ensure_iso_coverage is True and iso_by_geo is provided, attempt to cover as many distinct
    ISOCode groups as possible by ensuring at least one store per group (up to n groups).
    """
    keys = np.asarray(geo_keys, dtype=np.int64)
    if keys.size == 0:
        raise ValueError("geo_keys empty")

    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")

    # Backwards compatible default.
    if not ensure_iso_coverage or not iso_by_geo:
        return rng.choice(keys, size=n, replace=True)

    iso_arr = np.array([iso_by_geo.get(int(k), "") for k in keys], dtype=object)
    valid = iso_arr != ""
    if not np.any(valid):
        return rng.choice(keys, size=n, replace=True)

    uniq_iso = sorted(set(iso_arr[valid].tolist()))
    if not uniq_iso:
        return rng.choice(keys, size=n, replace=True)

    # Choose which ISO groups to cover (if n is small, cover n random groups).
    if n < len(uniq_iso):
        chosen_iso = rng.choice(np.array(uniq_iso, dtype=object), size=n, replace=False).tolist()
    else:
        chosen_iso = uniq_iso

    first: list[int] = []
    for code in chosen_iso:
        pool = keys[iso_arr == code]
        if pool.size:
            first.append(int(rng.choice(pool, size=1)[0]))

    first_arr = np.asarray(first, dtype=np.int64)
    remaining = n - int(first_arr.size)
    rest = rng.choice(keys, size=remaining, replace=True) if remaining > 0 else np.empty(0, dtype=np.int64)

    out = np.concatenate([first_arr, rest])
    rng.shuffle(out)
    return out


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
    geo_loc_short: Optional[Dict[int, str]] = None,
    geo_loc_full: Optional[Dict[int, str]] = None,
    iso_by_geo: Optional[dict[int, str]] = None,
    ensure_currency_coverage: bool = False,
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
    sk = store_key.astype(np.int64)

    # Categories (MUST come before name logic that depends on StoreType/Status)
    df["StoreType"] = rng.choice(_STORE_TYPES, size=num_stores, p=_STORE_TYPES_P)
    df["Status"] = rng.choice(_STORE_STATUS, size=num_stores, p=_STORE_STATUS_P)

    # GeographyKey assignment (MUST come before location-based naming)
    df["GeographyKey"] = _sample_geography_keys(
        rng=rng,
        geo_keys=geo_keys.astype(np.int64),
        n=num_stores,
        iso_by_geo=iso_by_geo,
        ensure_iso_coverage=bool(ensure_currency_coverage),
    )

    # Location strings (best-effort)
    if geo_loc_short is None:
        loc_short = df["GeographyKey"].astype(np.int64).map(lambda k: f"Geo {int(k)}")
    else:
        loc_short = df["GeographyKey"].astype(np.int64).map(lambda k: geo_loc_short.get(int(k), f"Geo {int(k)}"))

    if geo_loc_full is None:
        loc_full = loc_short
    else:
        loc_full = df["GeographyKey"].astype(np.int64).map(lambda k: geo_loc_full.get(int(k), f"Geo {int(k)}"))

    # Manager names (deterministic per StoreKey)
    mf = _MANAGER_FIRST[(sk * 5 + int(seed)) % len(_MANAGER_FIRST)]
    ml = _MANAGER_LAST[(sk * 11 + int(seed) * 3) % len(_MANAGER_LAST)]
    df["StoreManager"] = pd.Series(mf, dtype="object").astype(str) + " " + pd.Series(ml, dtype="object").astype(str)

    # StoreName pattern (depends on StoreType)
    brand = _BRANDS[(sk + int(seed)) % len(_BRANDS)]
    area = _AREAS[(sk * 7 + int(seed) * 13) % len(_AREAS)]
    stype = df["StoreType"].astype(str)
    is_online = (stype.to_numpy() == "Online")

    online_suffix = _ONLINE_SUFFIX[(sk * 3 + int(seed) * 17) % len(_ONLINE_SUFFIX)]

    store_name = (
        pd.Series(brand, dtype="object").astype(str)
        + " "
        + loc_short.astype(str)
        + " "
        + pd.Series(area, dtype="object").astype(str)
    )
    store_name_online = (
        pd.Series(brand, dtype="object").astype(str)
        + " "
        + pd.Series(online_suffix, dtype="object").astype(str)
    )

    df["StoreName"] = store_name
    df.loc[is_online, "StoreName"] = store_name_online.loc[is_online].to_numpy()

    # Date generation at DAY granularity
    open_start_d = _as_date64d(opening_start)
    open_end_d = _as_date64d(opening_end)
    close_end_d = _as_date64d(closing_end)

    opening_d = _rand_dates_d(rng, open_start_d, open_end_d, num_stores)
    df["OpeningDate"] = pd.to_datetime(opening_d.astype("datetime64[ns]")).normalize()

    # ClosingDate only for Closed stores, always >= OpeningDate
    df["ClosingDate"] = pd.NaT
    status = df["Status"].astype(str)
    closed_mask = (status.to_numpy() == "Closed")

    if closed_mask.any():
        open_days = opening_d.astype("int64")[closed_mask]  # days since epoch
        close_end_day = close_end_d.astype("int64")

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
    second = store_key % 10000       # 0..9999
    df["Phone"] = (
        "(555) "
        + pd.Series(first).astype(str).str.zfill(3)
        + "-"
        + pd.Series(second).astype(str).str.zfill(4)
    )

    # CloseReason must exist BEFORE StoreDescription references it
    df["CloseReason"] = ""
    if closed_mask.any():
        df.loc[closed_mask, "CloseReason"] = rng.choice(_CLOSE_REASONS, size=int(closed_mask.sum()))

    # StoreDescription
    opened = df["OpeningDate"].dt.strftime("%Y-%m-%d")
    closed = df["ClosingDate"].dt.strftime("%Y-%m-%d").fillna("")
    stype = df["StoreType"].astype(str)

    base = (
        df["StoreName"].astype(str)
        + " is a "
        + stype.str.lower()
        + " ("
        + status.str.lower()
        + ") location in "
        + loc_full.astype(str)
        + ". Opened "
        + opened
        + ". Size "
        + df["SquareFootage"].astype(str)
        + " sqft; headcount "
        + df["EmployeeCount"].astype(str)
        + ". Manager: "
        + df["StoreManager"].astype(str)
        + "."
    )

    # Online phrasing
    online_mask = (stype == "Online")
    if online_mask.any():
        base.loc[online_mask] = (
            df.loc[online_mask, "StoreName"].astype(str)
            + " operates as an online channel serving "
            + loc_full.loc[online_mask].astype(str)
            + ". Launched "
            + opened.loc[online_mask]
            + ". Estimated headcount "
            + df.loc[online_mask, "EmployeeCount"].astype(str)
            + "."
        )

    # Closed addendum
    if closed_mask.any():
        base.loc[closed_mask] = (
            base.loc[closed_mask]
            + " Closed "
            + closed.loc[closed_mask]
            + " ("
            + df.loc[closed_mask, "CloseReason"].astype(str)
            + ")."
        )

    # Renovating addendum
    reno_mask = (status == "Renovating")
    if reno_mask.any():
        base.loc[reno_mask] = base.loc[reno_mask] + " Currently renovating; limited operations."

    df["StoreDescription"] = base

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
    geo = _safe_read_geography(geo_path)
    geo_keys = geo["GeographyKey"].astype(np.int64).to_numpy()
    loc_short_map, loc_full_map = _build_location_maps(geo)

    # Optional: ensure currency diversity even with small num_stores.
    # This affects GeographyKey sampling only; output schema/columns remain unchanged.
    ensure_currency_coverage = bool(store_cfg.get("ensure_currency_coverage", False))

    iso_by_geo: Optional[dict[int, str]] = None
    if ensure_currency_coverage and "ISOCode" in geo.columns:
        g = geo[["GeographyKey", "ISOCode"]].dropna()
        iso_by_geo = dict(
            zip(
                g["GeographyKey"].astype(np.int64).to_numpy(),
                g["ISOCode"].astype(str).to_numpy(),
            )
        )

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
            geo_loc_short=loc_short_map,
            geo_loc_full=loc_full_map,
            iso_by_geo=iso_by_geo,
            ensure_currency_coverage=ensure_currency_coverage,
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