from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.utils.logging_utils import info, skip, stage, warn
from src.utils.output_utils import write_parquet_with_date32
from src.versioning.version_store import should_regenerate, save_version
from src.utils.name_pools import (
    assign_person_names,
    hash_u64,
    load_people_pools,
    resolve_people_folder,
)
from src.utils.config_helpers import (
    as_dict,
    int_or,
    float_or,
    pick_seed_flat,
    range2,
    region_from_iso_code,
)

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------

_STORE_TYPES = np.array(["Supermarket", "Convenience", "Online", "Hypermarket"], dtype=object)
_STORE_STATUS = np.array(["Open", "Closed", "Renovating"], dtype=object)
_CLOSE_REASONS = np.array(["Low Sales", "Lease Ended", "Renovation", "Moved Location"], dtype=object)

_STORE_TYPES_P = np.array([0.50, 0.30, 0.10, 0.10], dtype=float)
_STORE_STATUS_P = np.array([0.85, 0.10, 0.05], dtype=float)

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

# StoreFormat — choices and probabilities keyed by StoreType
_STORE_FORMATS: dict[str, tuple[list[str], list[float]]] = {
    "Online":      (["Digital"],                                   [1.00]),
    "Hypermarket": (["Flagship", "Standard"],                      [0.30, 0.70]),
    "Supermarket": (["Flagship", "Standard", "Express"],           [0.10, 0.60, 0.30]),
    "Convenience": (["Standard", "Express", "Drive-Thru"],         [0.10, 0.50, 0.40]),
}
_DEFAULT_FORMATS: tuple[list[str], list[float]] = (["Standard", "Express"], [0.50, 0.50])

# OwnershipType — choices and probabilities keyed by StoreType
_OWNERSHIP_TYPES: dict[str, tuple[list[str], list[float]]] = {
    "Online":      (["Corporate", "Licensed"],              [0.70, 0.30]),
    "Hypermarket": (["Corporate", "Franchise", "Licensed"], [0.80, 0.15, 0.05]),
    "Supermarket": (["Corporate", "Franchise", "Licensed"], [0.50, 0.35, 0.15]),
    "Convenience": (["Corporate", "Franchise", "Licensed"], [0.30, 0.50, 0.20]),
}
_DEFAULT_OWNERSHIP: tuple[list[str], list[float]] = (
    ["Corporate", "Franchise", "Licensed"], [0.50, 0.35, 0.15],
)

_REVENUE_CLASSES   = np.array(["A", "B", "C"], dtype=object)
_REVENUE_CLASSES_P = np.array([0.20, 0.60, 0.20], dtype=float)

# StoreZone derived from ISO/currency code
_ISO_TO_ZONE: dict[str, str] = {
    "USD": "Americas",    "CAD": "Americas",    "MXN": "Americas",    "BRL": "Americas",
    "ARS": "Americas",    "CLP": "Americas",    "COP": "Americas",    "PEN": "Americas",
    "GBP": "Europe",      "EUR": "Europe",      "CHF": "Europe",      "SEK": "Europe",
    "NOK": "Europe",      "DKK": "Europe",      "PLN": "Europe",      "CZK": "Europe",
    "HUF": "Europe",      "RON": "Europe",
    "INR": "South Asia",
    "AUD": "Asia Pacific", "NZD": "Asia Pacific", "CNY": "Asia Pacific", "JPY": "Asia Pacific",
    "HKD": "Asia Pacific", "SGD": "Asia Pacific", "KRW": "Asia Pacific", "TWD": "Asia Pacific",
    "THB": "Asia Pacific", "IDR": "Asia Pacific", "PHP": "Asia Pacific", "MYR": "Asia Pacific",
}

# Brand → email domain
_BRAND_DOMAINS: dict[str, str] = {
    "Northwind Market":       "northwindmarket.com",
    "Contoso Mart":           "contosomart.com",
    "Fabrikam Foods":         "fabrikamfoods.com",
    "Woodgrove Grocers":      "woodgrovegrocers.com",
    "Adventure Works Retail": "adventureworks.com",
    "Tailspin Superstores":   "tailspinstores.com",
    "Wingtip Fresh":          "wingtipfresh.com",
    "Proseware Market":       "prosewaremarket.com",
    "CitySquare Grocers":     "citysquaregrocers.com",
    "Harborview Market":      "harborviewmarket.com",
    "Summit Retail":          "summitretail.com",
    "BlueSky Foods":          "blueskyfoods.com",
}

# EU country code rotation for phone generation
_EU_COUNTRY_CODES = [33, 34, 39, 49, 31, 32, 41, 46, 47, 45]


# ---------------------------------------------------------
# Internals
# ---------------------------------------------------------

def _safe_read_geography(geo_path: Path) -> pd.DataFrame:
    """Read geography.parquet with best-effort column projection."""
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
    Returns ``(loc_short_map, loc_full_map)`` keyed by GeographyKey.
    Uses whatever columns exist; always falls back to ``'Geo <key>'``.
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

    if city is not None:
        loc_short = city
    elif state is not None:
        loc_short = state
    elif country is not None:
        loc_short = country
    else:
        loc_short = pd.Series([""] * len(geo))

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


def _as_date64d(s: str | np.datetime64) -> np.datetime64:
    if isinstance(s, np.datetime64):
        return s.astype("datetime64[D]")
    return np.datetime64(pd.to_datetime(str(s)).date(), "D")


def _rand_dates_d(
    rng: np.random.Generator,
    start_d: np.datetime64,
    end_d: np.datetime64,
    n: int,
) -> np.ndarray:
    """Returns ``np.datetime64[D]`` array in ``[start_d, end_d]``."""
    start_i = start_d.astype("int64")
    end_i = end_d.astype("int64")
    if end_i < start_i:
        start_i, end_i = end_i, start_i
    days = rng.integers(start_i, end_i + 1, size=n, dtype=np.int64)
    return days.astype("datetime64[D]")


def _geography_signature(geo_keys: np.ndarray) -> str:
    arr = np.asarray(geo_keys, dtype=np.int64)
    if arr.size == 0:
        return "empty"
    return f"{int(arr.size)}:{int(arr.min())}:{int(arr.max())}:{int(arr[: min(5, arr.size)].sum())}"


def _square_footage_from_cfg(
    *,
    rng: np.random.Generator,
    store_type: pd.Series,
    sqft_cfg: Dict,
    n: int,
) -> np.ndarray:
    """
    Config shape (optional)::

      stores:
        square_footage:
          default: [5000, 60000]
          Supermarket: [15000, 80000]
          ...
    """
    default_lo, default_hi = range2(sqft_cfg.get("default"), 5000.0, 60000.0)

    # Use zeros so the missing-mask fallback (`out == 0`) is reliable
    out = np.zeros(n, dtype=np.int64)
    st = store_type.astype(str).to_numpy()

    for t in np.unique(st):
        lo, hi = range2(sqft_cfg.get(t), default_lo, default_hi)
        mask = st == t
        if mask.any():
            vals = rng.uniform(lo, hi, size=int(mask.sum()))
            out[mask] = np.maximum(1, np.round(vals)).astype(np.int64)

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
    Config shape (optional)::

      stores:
        employee_count:
          base_per_1000_sqft: 0.35
          online_base: [5, 60]
          ...
    """
    base_rate = float_or(emp_cfg.get("base_per_1000_sqft"), 0.35)
    online_lo, online_hi = range2(emp_cfg.get("online_base"), 5.0, 60.0)

    closed_mult = float_or(emp_cfg.get("closed_multiplier"), 0.15)
    reno_mult = float_or(emp_cfg.get("renovating_multiplier"), 0.60)

    emp_min = int_or(emp_cfg.get("min"), 3)
    emp_max = int_or(emp_cfg.get("max"), 800)

    st = store_type.astype(str).to_numpy()
    ss = status.astype(str).to_numpy()
    sqft = square_footage.to_numpy(dtype=np.float64)

    baseline = np.maximum(1.0, (sqft / 1000.0) * base_rate)

    online_mask = st == "Online"
    if online_mask.any():
        baseline[online_mask] = rng.uniform(online_lo, online_hi, size=int(online_mask.sum()))

    closed_mask = ss == "Closed"
    if closed_mask.any():
        baseline[closed_mask] = baseline[closed_mask] * closed_mult

    reno_mask = ss == "Renovating"
    if reno_mask.any():
        baseline[reno_mask] = baseline[reno_mask] * reno_mult

    jitter = rng.normal(loc=1.0, scale=0.10, size=baseline.size)
    baseline = baseline * np.clip(jitter, 0.7, 1.4)

    out = np.round(baseline).astype(np.int64)
    out = np.clip(out, emp_min, emp_max)
    return out


# ---------------------------------------------------------
# Geography sampling
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

    If *ensure_iso_coverage* is True, attempt to cover as many distinct
    ISOCode groups as possible by seeding at least one store per group.
    """
    keys = np.asarray(geo_keys, dtype=np.int64)
    if keys.size == 0:
        raise ValueError("geo_keys empty")
    if n <= 0:
        raise ValueError(f"n must be > 0, got {n}")

    if not ensure_iso_coverage or not iso_by_geo:
        return rng.choice(keys, size=n, replace=True)

    iso_arr = np.array([iso_by_geo.get(int(k), "") for k in keys], dtype=object)
    valid = iso_arr != ""
    if not np.any(valid):
        return rng.choice(keys, size=n, replace=True)

    uniq_iso = sorted(set(iso_arr[valid].tolist()))
    if not uniq_iso:
        return rng.choice(keys, size=n, replace=True)

    if n < len(uniq_iso):
        chosen_iso = rng.choice(
            np.array(uniq_iso, dtype=object), size=n, replace=False,
        ).tolist()
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


# ---------------------------------------------------------
# New column helpers
# ---------------------------------------------------------

def _iso_to_zone(iso: str) -> str:
    """Map an ISO/currency code to a geographic zone label."""
    return _ISO_TO_ZONE.get(str(iso).strip().upper(), "International")


def _build_phone(key: int, iso: str) -> str:
    """Return a country-formatted synthetic phone number for a single store."""
    iso = str(iso).strip().upper()
    a = (key * 7 + 131) % 900 + 100       # 3-digit part  (100-999)
    b = (key * 13 + 271) % 9000 + 1000    # 4-digit part (1000-9999)
    c = key % 10000                         # 4-digit suffix

    if iso in ("USD", "CAD", "MXN"):
        return f"+1 (555) {a:03d}-{c:04d}"
    if iso == "GBP":
        return f"+44 {a:03d} {b // 10:03d} {c:04d}"
    if iso == "EUR":
        cc = _EU_COUNTRY_CODES[key % len(_EU_COUNTRY_CODES)]
        return f"+{cc} {a // 10:02d} {b:04d} {c:04d}"
    if iso == "INR":
        return f"+91 {a * 10 + b % 10:05d} {c:05d}"
    if iso == "AUD":
        return f"+61 {key % 9 + 2} {a:04d} {c:04d}"
    if iso in ("CNY", "HKD"):
        return f"+86 {a:03d} {b:04d} {c:04d}"
    if iso == "JPY":
        return f"+81 {a % 90 + 10:02d} {b:04d} {c:04d}"
    if iso == "SGD":
        return f"+65 {b:04d} {c:04d}"
    # Generic international fallback
    cc = 10 + key % 89
    return f"+{cc} {a:03d} {b // 10:03d} {c:04d}"


def _build_store_format(rng: np.random.Generator, store_type: pd.Series) -> np.ndarray:
    """Assign StoreFormat correlated with StoreType."""
    st = store_type.astype(str).to_numpy()
    out = np.full(len(st), "Standard", dtype=object)
    for t in np.unique(st):
        mask = st == t
        choices, probs = _STORE_FORMATS.get(t, _DEFAULT_FORMATS)
        out[mask] = rng.choice(np.array(choices, dtype=object), size=int(mask.sum()), p=probs)
    return out


def _build_ownership_type(rng: np.random.Generator, store_type: pd.Series) -> np.ndarray:
    """Assign OwnershipType correlated with StoreType."""
    st = store_type.astype(str).to_numpy()
    out = np.full(len(st), "Corporate", dtype=object)
    for t in np.unique(st):
        mask = st == t
        choices, probs = _OWNERSHIP_TYPES.get(t, _DEFAULT_OWNERSHIP)
        out[mask] = rng.choice(np.array(choices, dtype=object), size=int(mask.sum()), p=probs)
    return out


def _build_hierarchy(
    geo_keys: np.ndarray,
    iso_by_geo: Optional[dict[int, str]],
    country_by_geo: Optional[dict[int, str]] = None,
    district_size: int = 10,
    districts_per_region: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Derive (StoreZone, StoreDistrict, StoreRegion) from GeographyKey + ISOCode.

    Districts are assigned per *country within zone* so that stores in
    different countries (e.g. China vs Australia) never share a district,
    even when they belong to the same zone ("Asia Pacific").  Region IDs
    roll up districts using ``districts_per_region``.
    """
    n = len(geo_keys)
    zones = np.empty(n, dtype=object)

    if iso_by_geo:
        for i, gk in enumerate(geo_keys):
            iso = iso_by_geo.get(int(gk), "")
            zones[i] = _iso_to_zone(iso)
    else:
        zones[:] = "International"

    # Country label per store (used for sub-grouping within a zone)
    countries = np.empty(n, dtype=object)
    if country_by_geo:
        for i, gk in enumerate(geo_keys):
            countries[i] = country_by_geo.get(int(gk), "Unknown")
    else:
        countries[:] = "Unknown"

    # Assign districts: iterate zone → country within zone → sequential IDs
    district_id = np.zeros(n, dtype=np.int16)
    next_did = 1
    for z in sorted(np.unique(zones)):
        z_mask = zones == z
        z_countries = countries[z_mask]
        z_idx = np.where(z_mask)[0]

        for c in sorted(np.unique(z_countries)):
            c_local = z_countries == c
            idx = z_idx[c_local]
            local_did = np.arange(len(idx)) // district_size
            district_id[idx] = (local_did + next_did).astype(np.int16)
            next_did += int(local_did.max()) + 1 if len(idx) > 0 else 1

    region_id = ((district_id - 1) // districts_per_region + 1).astype(np.int16)

    store_districts = np.array([f"District {d}" for d in district_id], dtype=object)
    store_regions   = np.array([f"Region {r}"   for r in region_id],   dtype=object)

    return zones, store_districts, store_regions


def _build_emails(
    brand_arr: np.ndarray,
    area_arr: np.ndarray,
    store_number_arr: np.ndarray,
    is_online: np.ndarray,
) -> np.ndarray:
    """Generate ``StoreEmail`` addresses keyed to brand domain."""
    n = len(brand_arr)
    out = np.empty(n, dtype=object)
    for i in range(n):
        domain = _BRAND_DOMAINS.get(str(brand_arr[i]), "retailstore.com")
        sn = str(store_number_arr[i]).lower().replace("-", ".")
        if is_online[i]:
            out[i] = f"online.{sn}@{domain}"
        else:
            area_slug = str(area_arr[i]).lower().replace(" ", ".").replace("-", ".")
            out[i] = f"{area_slug}.{sn}@{domain}"
    return out


def _build_analytical(
    rng: np.random.Generator,
    store_type: pd.Series,
    revenue_class: np.ndarray,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate five analytical columns:
      AvgTransactionValue, CustomerSatisfactionScore,
      InventoryTurnoverTarget, LastAuditScore, ShrinkageRatePct

    All values are correlated with StoreType and/or RevenueClass.
    """
    st = store_type.astype(str).to_numpy()
    rc = np.asarray(revenue_class, dtype=object)

    # AvgTransactionValue — base ranges by type, then adjusted for revenue class
    atv_base = np.zeros(n, dtype=np.float64)
    for t in np.unique(st):
        mask = st == t
        cnt = int(mask.sum())
        if t == "Hypermarket":
            atv_base[mask] = rng.uniform(80.0, 250.0, cnt)
        elif t == "Supermarket":
            atv_base[mask] = rng.uniform(40.0, 120.0, cnt)
        elif t == "Convenience":
            atv_base[mask] = rng.uniform(10.0, 30.0, cnt)
        elif t == "Online":
            atv_base[mask] = rng.uniform(50.0, 200.0, cnt)
        else:
            warn(f"Unknown StoreType {t!r} in AvgTransactionValue; using Supermarket range.")
            atv_base[mask] = rng.uniform(40.0, 120.0, cnt)

    rc_mult = np.where(rc == "A", 1.30, np.where(rc == "C", 0.80, 1.00))
    avg_transaction = np.round(atv_base * rc_mult, 2)

    # CustomerSatisfactionScore — 1.0-10.0, positive skew via beta distribution
    raw_csat = rng.beta(a=6, b=2, size=n)
    csat = np.round(1.0 + raw_csat * 9.0, 1)

    # InventoryTurnoverTarget — turns/year, varies by store type
    inv_turn = np.zeros(n, dtype=np.float64)
    for t in np.unique(st):
        mask = st == t
        cnt = int(mask.sum())
        if t == "Hypermarket":
            inv_turn[mask] = rng.uniform(15.0, 25.0, cnt)
        elif t == "Supermarket":
            inv_turn[mask] = rng.uniform(18.0, 30.0, cnt)
        elif t == "Convenience":
            inv_turn[mask] = rng.uniform(30.0, 50.0, cnt)
        elif t == "Online":
            inv_turn[mask] = rng.uniform(20.0, 40.0, cnt)
        else:
            inv_turn[mask] = rng.uniform(18.0, 30.0, cnt)
    inv_turn = np.round(inv_turn, 1)

    # LastAuditScore — integer 50-100
    audit = rng.integers(50, 101, size=n, dtype=np.int64)

    # ShrinkageRatePct — Convenience is higher risk; Online is lower
    shrink = np.zeros(n, dtype=np.float64)
    for t in np.unique(st):
        mask = st == t
        cnt = int(mask.sum())
        if t == "Convenience":
            shrink[mask] = rng.uniform(1.5, 4.0, cnt)
        elif t == "Online":
            shrink[mask] = rng.uniform(0.2, 1.0, cnt)
        else:  # Supermarket, Hypermarket, or any future type
            shrink[mask] = rng.uniform(0.5, 2.5, cnt)
    shrink = np.round(shrink, 2)

    return avg_transaction, csat, inv_turn, audit, shrink


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
    geo_loc_short: Optional[Dict[int, str]] = None,
    geo_loc_full: Optional[Dict[int, str]] = None,
    iso_by_geo: Optional[dict[int, str]] = None,
    country_by_geo: Optional[dict[int, str]] = None,
    ensure_iso_coverage: bool = False,
    people_pools=None,
) -> pd.DataFrame:
    """
    Generate synthetic store dimension table.

    Output columns:
      StoreKey, StoreNumber, StoreName, StoreType, StoreFormat, OwnershipType,
      RevenueClass, Status, GeographyKey, StoreZone, StoreDistrict, StoreRegion,
      OpeningDate, ClosingDate, OpenFlag, SquareFootage, EmployeeCount,
      StoreManager, Phone, StoreEmail, StoreDescription, CloseReason,
      AvgTransactionValue, CustomerSatisfactionScore, InventoryTurnoverTarget,
      LastAuditScore, ShrinkageRatePct
    """
    num_stores = int_or(num_stores, 200)
    if num_stores <= 0:
        raise ValueError(f"num_stores must be > 0, got {num_stores}")

    if not isinstance(geo_keys, np.ndarray) or geo_keys.size == 0:
        raise ValueError("geo_keys must be a non-empty numpy array of GeographyKey values")

    rng = np.random.default_rng(int_or(seed, 42))

    store_key = np.arange(1, num_stores + 1, dtype=np.int64)
    df = pd.DataFrame({"StoreKey": store_key})
    sk = store_key.astype(np.int64)

    # StoreNumber — human-readable alphanumeric key (STR-0001 … STR-9999)
    df["StoreNumber"] = pd.array([f"STR-{k:04d}" for k in store_key], dtype="object")

    df["StoreType"] = rng.choice(_STORE_TYPES, size=num_stores, p=_STORE_TYPES_P)
    df["StoreFormat"] = _build_store_format(rng, df["StoreType"])
    df["OwnershipType"] = _build_ownership_type(rng, df["StoreType"])
    df["RevenueClass"] = rng.choice(_REVENUE_CLASSES, size=num_stores, p=_REVENUE_CLASSES_P)
    df["Status"] = rng.choice(_STORE_STATUS, size=num_stores, p=_STORE_STATUS_P)

    df["GeographyKey"] = _sample_geography_keys(
        rng=rng,
        geo_keys=geo_keys.astype(np.int64),
        n=num_stores,
        iso_by_geo=iso_by_geo,
        ensure_iso_coverage=bool(ensure_iso_coverage),
    )

    # Location strings
    if geo_loc_short is None:
        loc_short = df["GeographyKey"].astype(np.int64).map(lambda k: f"Geo {int(k)}")
    else:
        loc_short = df["GeographyKey"].astype(np.int64).map(
            lambda k: geo_loc_short.get(int(k), f"Geo {int(k)}")
        )

    if geo_loc_full is None:
        loc_full = loc_short
    else:
        loc_full = df["GeographyKey"].astype(np.int64).map(
            lambda k: geo_loc_full.get(int(k), f"Geo {int(k)}")
        )

    # Hierarchy: StoreZone, StoreDistrict, StoreRegion
    zones, districts, regions = _build_hierarchy(
        geo_keys=df["GeographyKey"].to_numpy(dtype=np.int64),
        iso_by_geo=iso_by_geo,
        country_by_geo=country_by_geo,
    )
    df["StoreZone"]     = zones
    df["StoreDistrict"] = districts
    df["StoreRegion"]   = regions

    # Manager names
    if people_pools is not None:
        if iso_by_geo:
            gk_arr = df["GeographyKey"].to_numpy(dtype=np.int64)
            iso_arr = np.array(
                [iso_by_geo.get(int(k), "") for k in gk_arr], dtype=object,
            )
            region = np.array(
                [region_from_iso_code(x, default_region="EU") for x in iso_arr],
                dtype=object,
            )
        else:
            region = np.full(num_stores, "US", dtype=object)

        h = hash_u64(sk.astype(np.uint64), int(seed), 7001)
        gender = np.where(
            (h & np.uint64(1)) == 0, "Male", "Female",
        ).astype(object)

        first, last, _ = assign_person_names(
            keys=sk,
            region=region,
            gender=gender,
            is_org=np.zeros(num_stores, dtype=bool),
            pools=people_pools,
            seed=int(seed),
            include_middle=False,
            default_region="US",
        )
        df["StoreManager"] = (
            pd.Series(first, dtype="object").astype(str)
            + " "
            + pd.Series(last, dtype="object").astype(str)
        )
    else:
        mf = _MANAGER_FIRST[(sk * 5 + int(seed)) % len(_MANAGER_FIRST)]
        ml = _MANAGER_LAST[(sk * 11 + int(seed) * 3) % len(_MANAGER_LAST)]
        df["StoreManager"] = (
            pd.Series(mf, dtype="object").astype(str)
            + " "
            + pd.Series(ml, dtype="object").astype(str)
        )

    # StoreName
    brand = _BRANDS[(sk + int(seed)) % len(_BRANDS)]
    area  = _AREAS[(sk * 7 + int(seed) * 13) % len(_AREAS)]
    stype = df["StoreType"].astype(str)
    is_online = stype.to_numpy() == "Online"

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

    # Dates at DAY granularity
    open_start_d = _as_date64d(opening_start)
    open_end_d   = _as_date64d(opening_end)
    close_end_d  = _as_date64d(closing_end)

    opening_d = _rand_dates_d(rng, open_start_d, open_end_d, num_stores)
    df["OpeningDate"] = pd.to_datetime(opening_d.astype("datetime64[ns]")).normalize()

    df["ClosingDate"] = pd.NaT
    status = df["Status"].astype(str)
    closed_mask = status.to_numpy() == "Closed"

    if closed_mask.any():
        open_days     = opening_d.astype("int64")[closed_mask]
        close_end_day = close_end_d.astype("int64")

        late_openers = int((open_days > close_end_day).sum())
        if late_openers:
            warn(
                f"{late_openers} closed store(s) have OpeningDate after "
                f"closing_end={closing_end!r}; ClosingDate will equal OpeningDate for those stores."
            )

        effective_end = np.maximum(open_days, close_end_day)
        close_days = rng.integers(open_days, effective_end + 1, dtype=np.int64)
        close_d = close_days.astype("datetime64[D]")
        df.loc[closed_mask, "ClosingDate"] = pd.to_datetime(
            close_d.astype("datetime64[ns]")
        ).normalize()

    df["OpenFlag"] = (df["Status"] == "Open").astype(np.int64)

    df["SquareFootage"] = _square_footage_from_cfg(
        rng=rng,
        store_type=df["StoreType"],
        sqft_cfg=as_dict(square_footage_cfg),
        n=num_stores,
    )

    df["EmployeeCount"] = _employee_count_from_cfg(
        rng=rng,
        store_type=df["StoreType"],
        status=df["Status"],
        square_footage=df["SquareFootage"],
        emp_cfg=as_dict(employee_count_cfg),
    )

    # Phone — format varies by country via ISO/currency code
    gk_arr  = df["GeographyKey"].to_numpy(dtype=np.int64)
    iso_arr = np.array(
        [iso_by_geo.get(int(k), "") if iso_by_geo else "" for k in gk_arr],
        dtype=object,
    )
    df["Phone"] = pd.array(
        [_build_phone(int(k), str(iso)) for k, iso in zip(store_key, iso_arr)],
        dtype="object",
    )

    # StoreEmail
    df["StoreEmail"] = _build_emails(
        brand_arr=brand,
        area_arr=area,
        store_number_arr=df["StoreNumber"].to_numpy(),
        is_online=is_online,
    )

    # CloseReason (must exist before StoreDescription references it)
    df["CloseReason"] = ""
    if closed_mask.any():
        df.loc[closed_mask, "CloseReason"] = rng.choice(
            _CLOSE_REASONS, size=int(closed_mask.sum()),
        )

    # StoreDescription
    opened = df["OpeningDate"].dt.strftime("%Y-%m-%d")
    closed = df["ClosingDate"].dt.strftime("%Y-%m-%d").fillna("")

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

    online_mask = stype == "Online"
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

    if closed_mask.any():
        base.loc[closed_mask] = (
            base.loc[closed_mask]
            + " Closed "
            + closed.loc[closed_mask]
            + " ("
            + df.loc[closed_mask, "CloseReason"].astype(str)
            + ")."
        )

    reno_mask = status == "Renovating"
    if reno_mask.any():
        base.loc[reno_mask] = base.loc[reno_mask] + " Currently renovating; limited operations."

    df["StoreDescription"] = base

    # Analytical columns
    avg_txn, csat, inv_turn, audit, shrink = _build_analytical(
        rng=rng,
        store_type=df["StoreType"],
        revenue_class=df["RevenueClass"].to_numpy(),
        n=num_stores,
    )
    df["AvgTransactionValue"]      = avg_txn
    df["CustomerSatisfactionScore"] = csat
    df["InventoryTurnoverTarget"]   = inv_turn
    df["LastAuditScore"]            = audit
    df["ShrinkageRatePct"]          = shrink

    return df


# ---------------------------------------------------------
# Pipeline entrypoint
# ---------------------------------------------------------

def run_stores(cfg: Dict, parquet_folder: Path) -> None:
    """
    Pipeline wrapper for stores dimension generation.
    """
    store_cfg = _require_cfg(cfg)

    parquet_folder = Path(parquet_folder)
    parquet_folder.mkdir(parents=True, exist_ok=True)

    out_path = parquet_folder / "stores.parquet"
    geo_path = parquet_folder / "geography.parquet"

    if not geo_path.exists():
        raise FileNotFoundError(f"Missing geography parquet: {geo_path}")

    force = bool(store_cfg.get("_force_regenerate", False))

    geo = _safe_read_geography(geo_path)
    geo_keys = geo["GeographyKey"].astype(np.int64).to_numpy()
    loc_short_map, loc_full_map = _build_location_maps(geo)

    ensure_iso_coverage = bool(store_cfg.get("ensure_iso_coverage", False))

    iso_by_geo: Optional[dict[int, str]] = None
    if "ISOCode" in geo.columns:
        g = geo[["GeographyKey", "ISOCode"]].dropna()
        iso_by_geo = dict(
            zip(
                g["GeographyKey"].astype(np.int64).to_numpy(),
                g["ISOCode"].astype(str).to_numpy(),
            )
        )

    # Country lookup — used for district sub-grouping within zones
    country_col = next(
        (c for c in ("Country", "CountryRegionName", "RegionCountryName") if c in geo.columns),
        None,
    )
    country_by_geo: Optional[dict[int, str]] = None
    if country_col is not None:
        gc = geo[["GeographyKey", country_col]].dropna()
        country_by_geo = dict(
            zip(
                gc["GeographyKey"].astype(np.int64).to_numpy(),
                gc[country_col].astype(str).to_numpy(),
            )
        )

    sqft_cfg = as_dict(store_cfg.get("square_footage"))
    emp_cfg  = as_dict(store_cfg.get("employee_count"))

    version_cfg = dict(store_cfg)
    version_cfg.pop("_force_regenerate", None)
    version_cfg["schema_version"] = 3
    version_cfg["_geography_sig"] = _geography_signature(geo_keys)

    if not force and not should_regenerate("stores", version_cfg, out_path):
        skip("Stores up-to-date; skipping.")
        return

    compression       = store_cfg.get("parquet_compression", "snappy")
    compression_level = store_cfg.get("parquet_compression_level", None)
    force_date32      = bool(store_cfg.get("force_date32", True))

    opening_cfg = as_dict(store_cfg.get("opening"))

    use_name_pools = bool(store_cfg.get("use_name_pools", True))
    people_pools = None
    if use_name_pools:
        people_folder = resolve_people_folder(cfg)
        pf = Path(people_folder)
        enable_asia = (
            (pf / "asia_male_first.csv").exists()
            and (pf / "asia_female_first.csv").exists()
            and (pf / "asia_last.csv").exists()
        )
        people_pools = load_people_pools(
            people_folder, enable_asia=enable_asia, legacy_support=True,
        )

    with stage("Generating Stores"):
        df = generate_store_table(
            geo_keys=geo_keys,
            num_stores=int_or(store_cfg.get("num_stores"), 200),
            opening_start=opening_cfg.get("start") or "1995-01-01",
            opening_end=opening_cfg.get("end") or "2020-12-31",
            closing_end=store_cfg.get("closing_end") or "2025-12-31",
            seed=pick_seed_flat(cfg, store_cfg, fallback=42),
            square_footage_cfg=sqft_cfg,
            employee_count_cfg=emp_cfg,
            geo_loc_short=loc_short_map,
            geo_loc_full=loc_full_map,
            iso_by_geo=iso_by_geo,
            country_by_geo=country_by_geo,
            ensure_iso_coverage=ensure_iso_coverage,
            people_pools=people_pools,
        )

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
