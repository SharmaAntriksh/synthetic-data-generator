from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.utils import info, skip, stage
from src.versioning import should_regenerate, save_version
from src.engine.dimension_loader import load_dimension
from src.utils.name_pools import (
    PeopleNamePools,
    resolve_people_folder,
    load_people_pools,
    assign_person_names,
    resolve_org_names_file,
    load_org_names,
    assign_org_names,
    slugify_domain_label,
)

# ---------------------------------------------------------
# Configurable defaults (imported from src.defaults)
# ---------------------------------------------------------
from src.defaults import (
    CUSTOMER_PERSONAL_EMAIL_DOMAINS as PERSONAL_EMAIL_DOMAINS,
    CUSTOMER_MARITAL_STATUS_LABELS as MARITAL_STATUS_LABELS,
    CUSTOMER_MARITAL_STATUS_PROBS as MARITAL_STATUS_PROBS,
    CUSTOMER_EDUCATION_LABELS as EDUCATION_LABELS,
    CUSTOMER_EDUCATION_PROBS as EDUCATION_PROBS,
    CUSTOMER_OCCUPATION_LABELS as OCCUPATION_LABELS,
    CUSTOMER_OCCUPATION_PROBS as OCCUPATION_PROBS,
    CUSTOMER_AGE_MIN_DAYS as AGE_MIN_DAYS,
    CUSTOMER_AGE_MAX_DAYS as AGE_MAX_DAYS,
    CUSTOMER_INCOME_MIN as INCOME_MIN,
    CUSTOMER_INCOME_MAX as INCOME_MAX,
    CUSTOMER_MAX_CHILDREN as MAX_CHILDREN,
    CUSTOMER_LOYALTY_W_WEIGHT as LOYALTY_W_WEIGHT,
    CUSTOMER_LOYALTY_W_TEMP as LOYALTY_W_TEMP,
    CUSTOMER_LOYALTY_W_INCOME as LOYALTY_W_INCOME,
    CUSTOMER_EDUCATION_INCOME_PARAMS as EDUCATION_INCOME_PARAMS,
    CUSTOMER_OCCUPATION_INCOME_MULT as OCCUPATION_INCOME_MULT,
    CUSTOMER_INCOME_ROUND_TO as INCOME_ROUND_TO,
    CUSTOMER_AGE_GROUP_EDGES as AGE_GROUP_EDGES,
    CUSTOMER_AGE_GROUP_LABELS as AGE_GROUP_LABELS,
    CUSTOMER_INCOME_GROUP_EDGES as INCOME_GROUP_EDGES,
    CUSTOMER_INCOME_GROUP_LABELS as INCOME_GROUP_LABELS,
    CUSTOMER_HOME_OWNERSHIP_LABELS as HOME_OWNERSHIP_LABELS,
    CUSTOMER_HOME_OWNERSHIP_PROBS_BY_INCOME as HOME_OWNERSHIP_PROBS_BY_INCOME,
    CUSTOMER_CONTACT_METHOD_LABELS as CONTACT_METHOD_LABELS,
    CUSTOMER_CONTACT_METHOD_PROBS as CONTACT_METHOD_PROBS,
    CUSTOMER_MARITAL_PROBS_BY_AGE as MARITAL_PROBS_BY_AGE,
    CUSTOMER_EDUCATION_PROBS_BY_AGE as EDUCATION_PROBS_BY_AGE,
    CUSTOMER_OCCUPATION_PROBS_BY_EDUCATION as OCCUPATION_PROBS_BY_EDUCATION,
    CUSTOMER_CHILDREN_LAMBDA_BY_MARITAL_AGE as CHILDREN_LAMBDA_BY_MARITAL_AGE,
    CUSTOMER_HOME_OWNERSHIP_AGE_SHIFT as HOME_OWNERSHIP_AGE_SHIFT,
    CUSTOMER_CAR_LAMBDA_BY_AGE as CAR_LAMBDA_BY_AGE,
    CUSTOMER_ORG_EMAIL_PREFIXES as ORG_EMAIL_PREFIXES,
    CUSTOMER_PHONE_COUNTRY_CODES as _PHONE_COUNTRY_CODES,
    CUSTOMER_STREET_NAMES as _STREET_NAMES,
    CUSTOMER_STREET_TYPES as _STREET_TYPES,
    CUSTOMER_REGION_LAT_LON_CENTER as _REGION_LAT_LON_CENTER,
    CUSTOMER_LAT_LON_JITTER as _LAT_LON_JITTER,
    CUSTOMER_REGION_TIMEZONE as _REGION_TIMEZONE,
    CUSTOMER_URBAN_RURAL_LABELS as _URBAN_RURAL_LABELS,
    CUSTOMER_URBAN_RURAL_PROBS as _URBAN_RURAL_PROBS,
    CUSTOMER_POSTCODE_FMT as _POSTCODE_FMT,
    CUSTOMER_LANGUAGE_BY_REGION as _LANGUAGE_BY_REGION,
)

_PAYMENT_METHOD_LABELS = np.array([
    "Credit Card", "Debit Card", "Cash", "Digital Wallet", "Bank Transfer",
])
_PAYMENT_METHOD_PROBS = np.array([0.35, 0.25, 0.10, 0.20, 0.10])

_DEVICE_PREFS = np.array(["Mobile", "Desktop", "Tablet"])
_NEWSLETTER_FREQ = np.array(["Weekly", "Monthly", "None"])

_DISTANCE_BY_AREA = {"Urban": (0.5, 5.0), "Suburban": (3.0, 15.0), "Rural": (10.0, 50.0)}

# ---------------------------------------------------------
# OrganizationProfile constants
# ---------------------------------------------------------
_ORG_INDUSTRY_LABELS = np.array([
    "Technology", "Healthcare", "Manufacturing", "Retail", "Finance",
    "Education", "Logistics", "Construction", "Energy", "Media",
])
_ORG_INDUSTRY_PROBS = np.array([
    0.18, 0.14, 0.13, 0.12, 0.11, 0.09, 0.08, 0.06, 0.05, 0.04,
])

_ORG_SIZE_LABELS = np.array(["Startup", "Small", "Medium", "Large", "Enterprise"])
_ORG_SIZE_EMPLOYEE_RANGES = {
    "Startup":    (5, 50),
    "Small":      (50, 250),
    "Medium":     (250, 1000),
    "Large":      (1000, 10000),
    "Enterprise": (10000, 50000),
}

_ORG_SIZE_PROBS = np.array([0.20, 0.30, 0.25, 0.15, 0.10])

_ORG_REVENUE_PARAMS = {
    "Startup":    (12.0, 0.8),    # median ~$160K
    "Small":      (14.0, 0.6),    # median ~$1.2M
    "Medium":     (16.0, 0.5),    # median ~$8.9M
    "Large":      (18.0, 0.4),    # median ~$65M
    "Enterprise": (20.0, 0.35),   # median ~$485M
}

_ORG_CONTACT_ROLES = np.array([
    "Procurement Manager", "CFO", "Operations Director", "Owner",
    "VP of Purchasing", "Supply Chain Manager", "General Manager",
])

_ORG_PROCUREMENT_CYCLE = np.array(["Monthly", "Quarterly", "Annual"])
_ORG_CONTRACT_TYPE = np.array(["Spot", "Annual", "Multi-Year"])
_ORG_CREDIT_RATINGS = np.array(["AAA", "AA", "A", "BBB", "BB", "B"])
_ORG_PAYMENT_TERMS = np.array(["Net30", "Net60", "Net90", "Prepaid"])
_ORG_SHIPPING_METHODS = np.array(["Standard", "Express", "Freight", "White Glove"])
_ORG_ESG_RATINGS = np.array(["A", "B", "C", "D", "Unrated"])
_ORG_SATISFACTION_TIERS = np.array(["Platinum", "Gold", "Silver", "Bronze"])

_REGION_HQ_COUNTRY = {
    "US": "United States", "IN": "India",
    "EU": np.array(["United Kingdom", "Germany", "France"]),
    "AS": np.array(["Japan", "China", "Australia"]),
}


# ---------------------------------------------------------
# Helper: timeline month index space
# ---------------------------------------------------------
def _parse_cfg_dates(cfg: Dict) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Resolve timeline dates from (priority order):
      1) cfg['customers']['global_dates']  (runner injected)
      2) cfg['defaults']['dates']
      3) cfg['_defaults']['dates']         (backward compatibility)

    Returns normalized pandas Timestamps (midnight).
    """
    cust = cfg.get("customers") or {}
    if isinstance(cust, dict):
        gd = cust.get("global_dates")
        if isinstance(gd, dict) and gd.get("start") and gd.get("end"):
            start = pd.to_datetime(gd["start"]).normalize()
            end = pd.to_datetime(gd["end"]).normalize()
            if end < start:
                raise ValueError("defaults.dates.end must be >= defaults.dates.start")
            return start, end

    try:
        defaults = cfg.get("defaults") or cfg.get("_defaults")
        dcfg = defaults["dates"]
        start = pd.to_datetime(dcfg["start"]).normalize()
        end = pd.to_datetime(dcfg["end"]).normalize()
    except Exception as e:
        raise ValueError("Missing or invalid defaults.dates.start/end in config.yaml") from e

    if end < start:
        raise ValueError("defaults.dates.end must be >= defaults.dates.start")

    return start, end


def _month_index_space(start_date: pd.Timestamp, end_date: pd.Timestamp):
    """
    Build a month index space [0..T-1] over the inclusive month range.

    Assumes start_date / end_date are already normalized Timestamps
    (as returned by _parse_cfg_dates).

    Returns:
      start_month0 : Timestamp at month start for start_date's month
      end_month0   : Timestamp at month start for end_date's month
      T            : int month count inclusive
    """
    sp = start_date.to_period("M")
    ep = end_date.to_period("M")

    if ep < sp:
        raise ValueError("defaults.dates.end must be >= defaults.dates.start")

    T = int(ep.ordinal - sp.ordinal) + 1
    start_month0 = sp.to_timestamp(how="start")
    end_month0 = ep.to_timestamp(how="start")
    return start_month0, end_month0, T


def _month_idx_to_date(month0, month_idx):
    """
    Convert a 0-based month index array into timestamps (month start).

    month0: pandas.Timestamp OR pandas.Period("M")
    month_idx: int or ndarray[int]
    Returns: numpy array of datetime64[ns] at month start.
    """
    if isinstance(month0, pd.Period):
        base_period = month0
    else:
        base_period = pd.to_datetime(month0).to_period("M")

    idx = np.asarray(month_idx, dtype=np.int64)
    ords = base_period.ordinal + idx
    pi = pd.PeriodIndex.from_ordinals(ords, freq="M")
    return pi.to_timestamp(how="start").to_numpy()


# ---------------------------------------------------------
# Helper: Acquisition curve (StartMonth distribution)
# ---------------------------------------------------------
def _acquisition_weights(T: int, curve: str, params: Dict) -> np.ndarray:
    """
    Produces weights over months 0..T-1 for sampling CustomerStartMonth.

    curve:
      - "uniform": flat
      - "linear_ramp": w ~ (m+1)^shape
      - "logistic": S-curve
    """
    if T <= 0:
        raise ValueError("T must be > 0")

    m = np.arange(T, dtype="float64")
    curve = (curve or "linear_ramp").lower()

    if curve == "uniform":
        w = np.ones(T, dtype="float64")

    elif curve == "linear_ramp":
        shape = float(params.get("shape", 2.0))
        if shape <= 0:
            raise ValueError("acquisition_params.shape must be > 0")
        w = (m + 1.0) ** shape

    elif curve == "logistic":
        midpoint = float(params.get("midpoint", 0.55))
        steep = float(params.get("steepness", 10.0))
        midpoint = float(np.clip(midpoint, 0.0, 1.0))
        steep = max(steep, 1e-6)

        x0 = midpoint * (T - 1)
        cdf = 1.0 / (1.0 + np.exp(-steep * ((m - x0) / max(T - 1, 1))))
        w = np.diff(np.r_[0.0, cdf])
        w = np.clip(w, 1e-9, None)

    else:
        raise ValueError(f"Unknown acquisition curve: {curve}")

    wsum = float(w.sum())
    if not np.isfinite(wsum) or wsum <= 0:
        raise ValueError("Invalid acquisition weights; check acquisition parameters")
    return w / wsum


def _simulate_end_month(
    rng: np.random.Generator,
    start_month: np.ndarray,
    churn_bias: np.ndarray,
    T: int,
    enable: bool,
    base_monthly_churn: float,
    min_tenure_months: int,
) -> np.ndarray:
    """
    For each customer, possibly sample an end month (churn month).
    If churn is disabled, returns pd.NA for all entries.

    Vectorized implementation: uses geometric sampling to determine
    tenure length, avoiding per-customer Python loops.
    """
    N = len(start_month)

    if not enable:
        return np.full(N, pd.NA, dtype="object")

    if base_monthly_churn < 0:
        raise ValueError("base_monthly_churn must be >= 0")

    end_month = np.full(N, pd.NA, dtype="object")
    mt = max(int(min_tenure_months), 0)

    # Clamp start months
    s = np.clip(start_month.astype("int64"), 0, T)

    # Per-customer hazard rate
    hazard = np.clip(base_monthly_churn * churn_bias.astype("float64"), 0.0, 0.95)

    # Customers that can potentially churn: start < T and hazard > 0
    eligible = (s < T) & (hazard > 0.0)

    if not eligible.any():
        return end_month

    # For eligible customers, sample a geometric tenure from the earliest
    # eligible month (start + min_tenure).  geometric(p) gives the number
    # of Bernoulli trials until first success (1-based), so the churn
    # happens at month = start + min_tenure + (sample - 1).
    eligible_idx = np.where(eligible)[0]
    h = hazard[eligible_idx]
    tenure_samples = rng.geometric(p=h)  # shape: (eligible_count,)

    # Churn month = start + min_tenure + (sample - 1)
    churn_month = s[eligible_idx] + mt + (tenure_samples - 1)

    # Only apply churn if it falls within the timeline
    within = churn_month < T
    apply_idx = eligible_idx[within]
    apply_vals = churn_month[within]

    end_month[apply_idx] = apply_vals.astype(int)

    return end_month


def _validate_percentages(
    pct_india: float, pct_us: float, pct_eu: float, pct_asia: float = 0.0
) -> Tuple[float, float, float, float]:
    p = np.array([pct_india, pct_us, pct_eu, pct_asia], dtype="float64")
    if np.any(~np.isfinite(p)) or np.any(p < 0):
        raise ValueError("pct_india/pct_us/pct_eu/pct_asia must be finite and >= 0")
    s = float(p.sum())
    if s <= 0:
        raise ValueError("pct_india/pct_us/pct_eu/pct_asia must sum to > 0")
    p = p / s
    return float(p[0]), float(p[1]), float(p[2]), float(p[3])


def _read_parquet_dim(parquet_dims_folder: Path, dim_name: str) -> pd.DataFrame:
    path = parquet_dims_folder / f"{dim_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing dimension parquet: {path}. "
            f"Run dimensions generation first (dim_name={dim_name})."
        )
    return pd.read_parquet(path)


def _first_existing_col(df: pd.DataFrame, candidates) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"None of the expected columns found. candidates={candidates}, cols={list(df.columns)}")


def _normalize_probs(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype="float64")
    if np.any(~np.isfinite(p)) or np.any(p < 0):
        raise ValueError("Probabilities must be finite and >= 0")
    s = float(p.sum())
    if s <= 0:
        raise ValueError("Probabilities must sum to > 0")
    return p / s


def _default_tier_probs(k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be > 0")
    if k == 1:
        return np.array([1.0])
    if k == 2:
        return np.array([0.75, 0.25])
    if k == 3:
        return np.array([0.60, 0.28, 0.12])
    if k == 4:
        return np.array([0.55, 0.25, 0.15, 0.05])
    x = np.linspace(1.0, 0.35, k)
    return _normalize_probs(x)


def _assign_tier_by_score(
    *,
    score: np.ndarray,
    tier_keys_sorted_low_to_high: np.ndarray,
    tier_probs_low_to_high: np.ndarray,
) -> np.ndarray:
    k = int(len(tier_keys_sorted_low_to_high))
    if k == 0:
        raise ValueError("No tiers available")
    if k == 1:
        return np.full(score.shape[0], tier_keys_sorted_low_to_high[0], dtype=tier_keys_sorted_low_to_high.dtype)

    p = _normalize_probs(tier_probs_low_to_high)
    cut = np.cumsum(p)[:-1]
    cuts = np.quantile(score, cut)

    idx = np.searchsorted(cuts, score, side="right")
    return tier_keys_sorted_low_to_high[idx]


def _acquisition_weights_from_names(names: np.ndarray, *, org: bool) -> np.ndarray:
    lower = np.char.lower(names.astype(str))
    w = np.ones(lower.shape[0], dtype="float64")

    def boost(substrs, mult):
        mask = np.zeros_like(w, dtype=bool)
        for s in substrs:
            mask |= np.char.find(lower, s) >= 0
        w[mask] *= mult

    if org:
        boost(["partner", "b2b", "reseller", "outbound", "sales"], 2.5)
        boost(["event", "conference"], 1.6)
        boost(["referral"], 1.3)
        boost(["social", "influenc"], 0.7)
    else:
        boost(["organic", "direct"], 1.6)
        boost(["referral"], 1.5)
        boost(["search", "seo", "sem"], 1.3)
        boost(["social", "influenc"], 1.2)
        boost(["email"], 1.1)
        boost(["partner", "b2b", "reseller", "outbound"], 0.7)

    return _normalize_probs(w)


# ---------------------------------------------------------
# Helper: correlated income generation
# ---------------------------------------------------------
def _generate_correlated_income(
    rng: np.random.Generator,
    education: np.ndarray,
    occupation: np.ndarray,
    person_mask: np.ndarray,
    N: int,
) -> np.ndarray:
    """
    Generate YearlyIncome using lognormal distributions parameterized
    by education level, with an occupation-based multiplier.

    Returns int64 array (0 for org rows) rounded to INCOME_ROUND_TO.
    """
    income = np.zeros(N, dtype="float64")
    n_p = int(person_mask.sum())
    if n_p == 0:
        return income.astype("int64")

    edu_mu = np.full(n_p, 10.65, dtype="float64")
    edu_sigma = np.full(n_p, 0.38, dtype="float64")
    edu_vals = education[person_mask]
    for label, (mu, sigma) in EDUCATION_INCOME_PARAMS.items():
        m = edu_vals == label
        edu_mu[m] = mu
        edu_sigma[m] = sigma

    occ_mult = np.ones(n_p, dtype="float64")
    occ_vals = occupation[person_mask]
    for label, mult in OCCUPATION_INCOME_MULT.items():
        occ_mult[occ_vals == label] = mult

    z = rng.normal(0.0, 1.0, size=n_p)
    raw = np.exp(edu_mu + edu_sigma * z) * occ_mult
    raw = np.round(raw / INCOME_ROUND_TO) * INCOME_ROUND_TO
    income[person_mask] = np.clip(raw, INCOME_MIN, INCOME_MAX)
    return income.astype("int64")


# ---------------------------------------------------------
# Helper: phone number generation
# ---------------------------------------------------------
def _generate_phone_numbers(
    rng: np.random.Generator,
    region: np.ndarray,
    N: int,
) -> np.ndarray:
    """Regional-format synthetic phone numbers."""
    phones = np.empty(N, dtype=object)
    raw = rng.integers(1_000_000_000, 9_999_999_999, size=N, dtype="int64")

    def _batch(mask, fmt_fn):
        if mask.any():
            phones[mask] = np.array(
                [fmt_fn(v) for v in raw[mask]], dtype=object
            )

    _batch(
        region == "US",
        lambda v: f"+1 ({(v // 10_000_000) % 800 + 200}) "
                  f"{(v // 10_000) % 1000:03d}-{v % 10_000:04d}",
    )
    _batch(
        region == "IN",
        lambda v: f"+91 {(v // 100_000) % 100_000:05d} {v % 100_000:05d}",
    )
    _batch(
        region == "EU",
        lambda v: f"+44 {(v // 1_000_000) % 10_000:04d} {v % 1_000_000:06d}",
    )
    _batch(
        region == "AS",
        lambda v: f"+81 {(v // 10_000_000) % 100:02d}-"
                  f"{(v // 10_000) % 1000:03d}-{v % 10_000:04d}",
    )
    remaining = pd.isna(phones) | (phones == None)  # noqa: E711
    _batch(
        remaining,
        lambda v: f"+1 ({(v // 10_000_000) % 800 + 200}) "
                  f"{(v // 10_000) % 1000:03d}-{v % 10_000:04d}",
    )
    return phones


# ---------------------------------------------------------
# Helper: credit score generation
# ---------------------------------------------------------
def _generate_credit_scores(
    rng: np.random.Generator,
    income_norm: np.ndarray,
    education: np.ndarray,
    person_mask: np.ndarray,
    N: int,
) -> np.ndarray:
    """
    CreditScore 300-850, correlated with income (normalized 0-1) and education.
    """
    scores = np.full(N, pd.NA, dtype=object)
    n_p = int(person_mask.sum())
    if n_p == 0:
        return scores

    base = rng.normal(loc=680, scale=60, size=n_p)

    edu_boost = np.zeros(n_p, dtype="float64")
    edu_p = education[person_mask]
    for label, delta in [("High School", -40), ("Bachelors", 0),
                         ("Masters", 30), ("PhD", 50)]:
        edu_boost[edu_p == label] = delta

    income_boost = income_norm[person_mask] * 60

    raw = base + edu_boost + income_boost
    raw = np.clip(np.round(raw), 300, 850).astype("int64")
    scores[person_mask] = raw
    return scores


# ---------------------------------------------------------
# Helper: synthetic address generation
# ---------------------------------------------------------
def _generate_addresses(
    rng: np.random.Generator,
    region: np.ndarray,
    customer_key: np.ndarray,
    geo_city: np.ndarray,
    geo_state: np.ndarray,
    N: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate unique HomeAddress and WorkAddress strings."""
    street_num = rng.integers(1, 9999, size=N).astype(str)
    street_name = rng.choice(_STREET_NAMES, size=N)
    street_type = rng.choice(_STREET_TYPES, size=N)

    _UNIT_LABELS = np.array(["Apt", "Suite", "Unit", "Fl", "#"])
    unit_label = rng.choice(_UNIT_LABELS, size=N)
    unit_num = customer_key.astype(str)

    home_street = (
        street_num.astype(object) + " " + street_name + " " + street_type
        + ", " + unit_label + " " + unit_num
    )
    HomeAddress = (
        home_street + ", " + geo_city.astype(object)
        + ", " + geo_state.astype(object)
    )

    work_street_num = rng.integers(1, 9999, size=N).astype(str)
    work_street_name = rng.choice(_STREET_NAMES, size=N)
    work_street_type = rng.choice(_STREET_TYPES, size=N)
    work_unit_label = rng.choice(_UNIT_LABELS, size=N)

    same_city = rng.random(N) < 0.70
    work_city = np.where(same_city, geo_city, rng.permutation(geo_city))
    work_state = np.where(same_city, geo_state, rng.permutation(geo_state))

    work_street = (
        work_street_num.astype(object) + " " + work_street_name + " " + work_street_type
        + ", " + work_unit_label + " " + unit_num
    )
    WorkAddress = (
        work_street + ", " + work_city.astype(object)
        + ", " + work_state.astype(object)
    )
    return HomeAddress, WorkAddress


def _generate_lat_lon(
    rng: np.random.Generator,
    region: np.ndarray,
    N: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Regional center + jitter, rounded to 4 decimal places."""
    lat = np.zeros(N, dtype="float64")
    lon = np.zeros(N, dtype="float64")
    for rc, (clat, clon) in _REGION_LAT_LON_CENTER.items():
        mask = region == rc
        n = int(mask.sum())
        if n:
            jlat, jlon = _LAT_LON_JITTER.get(rc, (5.0, 5.0))
            lat[mask] = np.round(clat + rng.uniform(-jlat, jlat, size=n), 4)
            lon[mask] = np.round(clon + rng.uniform(-jlon, jlon, size=n), 4)
    remaining = (lat == 0) & (lon == 0)
    n_rem = int(remaining.sum())
    if n_rem:
        lat[remaining] = np.round(39.8 + rng.uniform(-8, 8, size=n_rem), 4)
        lon[remaining] = np.round(-98.5 + rng.uniform(-15, 15, size=n_rem), 4)
    return lat, lon


def _generate_postal_codes(
    rng: np.random.Generator,
    region: np.ndarray,
    N: int,
) -> np.ndarray:
    """Region-appropriate synthetic postal codes."""
    codes = np.empty(N, dtype=object)
    for rc, fmt in _POSTCODE_FMT.items():
        mask = region == rc
        n = int(mask.sum())
        if not n:
            continue
        if fmt == "5digit":
            codes[mask] = np.array(
                [f"{v:05d}" for v in rng.integers(10001, 99999, size=n)],
                dtype=object,
            )
        elif fmt == "6digit":
            codes[mask] = np.array(
                [f"{v:06d}" for v in rng.integers(100001, 999999, size=n)],
                dtype=object,
            )
        elif fmt == "uk":
            prefix = rng.choice(
                np.array(["SW", "EC", "W", "SE", "N", "NW", "E", "WC"]),
                size=n,
            )
            num = rng.integers(1, 19, size=n).astype(str)
            suffix = rng.integers(1, 9, size=n).astype(str)
            letter = rng.choice(np.array(list("ABCDEFGHJKLMNPRSTUWXYZ")), size=n)
            letter2 = rng.choice(np.array(list("ABCDEFGHJKLMNPRSTUWXYZ")), size=n)
            codes[mask] = (
                prefix.astype(object) + num.astype(object) + " "
                + suffix.astype(object) + letter + letter2
            )
        elif fmt == "jp":
            codes[mask] = np.array(
                [f"{v // 10000:03d}-{v % 10000:04d}" for v in rng.integers(1000000, 9999999, size=n)],
                dtype=object,
            )
    remaining = pd.isna(codes) | (codes == None)  # noqa: E711
    n_rem = int(remaining.sum())
    if n_rem:
        codes[remaining] = np.array(
            [f"{v:05d}" for v in rng.integers(10001, 99999, size=n_rem)],
            dtype=object,
        )
    return codes


# ---------------------------------------------------------
# Helper: OrganizationProfile generation
# ---------------------------------------------------------
def _generate_org_profile(
    rng: np.random.Generator,
    customer_key: np.ndarray,
    is_org: np.ndarray,
    org_name: np.ndarray,
    region: np.ndarray,
    customer_start_date: np.ndarray,
    churn_bias: np.ndarray,
    customer_weight: np.ndarray,
    people_pools: PeopleNamePools,
    end_date: pd.Timestamp,
    seed: int,
) -> pd.DataFrame:
    """Generate OrganizationProfile for org-type customers only."""
    org_idx = np.where(is_org)[0]
    M = len(org_idx)
    if M == 0:
        return pd.DataFrame()

    org_keys = customer_key[org_idx]
    org_region = region[org_idx]
    org_names = org_name[org_idx]
    org_start = customer_start_date[org_idx]
    org_churn = churn_bias[org_idx]
    org_weight = customer_weight[org_idx]

    Industry = rng.choice(_ORG_INDUSTRY_LABELS, size=M, p=_ORG_INDUSTRY_PROBS)
    CompanySize = rng.choice(_ORG_SIZE_LABELS, size=M, p=_ORG_SIZE_PROBS)

    NumberOfEmployees = np.zeros(M, dtype="int64")
    for label, (lo, hi) in _ORG_SIZE_EMPLOYEE_RANGES.items():
        mask = CompanySize == label
        n = int(mask.sum())
        if n:
            NumberOfEmployees[mask] = rng.integers(lo, hi, size=n)

    AnnualRevenue = np.zeros(M, dtype="float64")
    for label, (mu, sigma) in _ORG_REVENUE_PARAMS.items():
        mask = CompanySize == label
        n = int(mask.sum())
        if n:
            raw = np.exp(rng.normal(mu, sigma, size=n))
            AnnualRevenue[mask] = np.round(raw / 10_000) * 10_000

    size_order = {"Startup": 0, "Small": 1, "Medium": 2, "Large": 3, "Enterprise": 4}
    size_num = np.array([size_order.get(s, 1) for s in CompanySize], dtype="float64")

    current_year = end_date.year
    founded_base = current_year - (size_num * 8 + rng.integers(1, 15, size=M)).astype(int)
    FoundedYear = np.clip(founded_base, 1960, current_year - 1)

    is_large = (CompanySize == "Large") | (CompanySize == "Enterprise")
    IsPubliclyTraded = np.where(
        is_large & (rng.random(M) < 0.25), "Yes", "No"
    )

    HeadquarterCountry = np.empty(M, dtype=object)
    for rc, country in _REGION_HQ_COUNTRY.items():
        mask = org_region == rc
        n = int(mask.sum())
        if n:
            if isinstance(country, np.ndarray):
                HeadquarterCountry[mask] = rng.choice(country, size=n)
            else:
                HeadquarterCountry[mask] = country
    remaining = pd.isna(HeadquarterCountry) | (HeadquarterCountry == None)  # noqa: E711
    if remaining.any():
        HeadquarterCountry[remaining] = "United States"

    Website = np.array(
        ["www." + slugify_domain_label(str(n)) + ".com" for n in org_names],
        dtype=object,
    )

    OrgDomain = np.array(
        [slugify_domain_label(str(n)) + ".com" for n in org_names],
        dtype=object,
    )

    org_street_num = rng.integers(1, 9999, size=M).astype(str)
    org_street_name = rng.choice(_STREET_NAMES, size=M)
    org_street_type = rng.choice(_STREET_TYPES, size=M)
    _SUITE_LABELS = np.array(["Suite", "Floor", "Bldg"])
    org_suite = rng.choice(_SUITE_LABELS, size=M)
    org_suite_num = org_keys.astype(str)
    geo_lookup_hq = {
        "United States": ("New York,Chicago,Houston,Phoenix,Dallas", "NY,IL,TX,AZ,TX"),
        "India": ("Mumbai,Delhi,Bangalore,Chennai,Hyderabad", "MH,DL,KA,TN,TG"),
        "United Kingdom": ("London,Manchester,Birmingham,Leeds,Glasgow", "England,England,England,England,Scotland"),
        "Germany": ("Berlin,Munich,Hamburg,Frankfurt,Cologne", "BE,BY,HH,HE,NW"),
        "France": ("Paris,Lyon,Marseille,Toulouse,Nice", "IDF,ARA,PAC,OCC,PAC"),
        "Japan": ("Tokyo,Osaka,Yokohama,Nagoya,Sapporo", "Tokyo,Osaka,Kanagawa,Aichi,Hokkaido"),
        "China": ("Shanghai,Beijing,Shenzhen,Guangzhou,Chengdu", "SH,BJ,GD,GD,SC"),
        "Australia": ("Sydney,Melbourne,Brisbane,Perth,Adelaide", "NSW,VIC,QLD,WA,SA"),
    }
    hq_city = np.empty(M, dtype=object)
    hq_state = np.empty(M, dtype=object)
    for i in range(M):
        country = str(HeadquarterCountry[i])
        if country in geo_lookup_hq:
            cities_str, states_str = geo_lookup_hq[country]
            cities = cities_str.split(",")
            states = states_str.split(",")
            idx = rng.integers(0, len(cities))
            hq_city[i] = cities[idx]
            hq_state[i] = states[idx]
        else:
            hq_city[i] = "New York"
            hq_state[i] = "NY"

    OrgAddress = (
        org_street_num.astype(object) + " " + org_street_name + " " + org_street_type
        + ", " + org_suite + " " + org_suite_num
        + ", " + hq_city + ", " + hq_state
    )

    us_pool = people_pools.region("US")
    contact_first = rng.choice(us_pool.any_first(), size=M)
    contact_last = rng.choice(us_pool.any_last(), size=M)
    PrimaryContactName = (contact_first.astype(str) + " " + contact_last.astype(str)).astype(object)
    PrimaryContactRole = rng.choice(_ORG_CONTACT_ROLES, size=M)

    NumberOfLocations = np.ones(M, dtype="int64")
    for label, mult in [("Small", 2), ("Medium", 5), ("Large", 20), ("Enterprise", 100)]:
        mask = CompanySize == label
        n = int(mask.sum())
        if n:
            NumberOfLocations[mask] = np.clip(rng.integers(1, mult + 1, size=n), 1, 500)

    proc_probs_by_size = {
        "Startup":    np.array([0.50, 0.35, 0.15]),
        "Small":      np.array([0.40, 0.40, 0.20]),
        "Medium":     np.array([0.25, 0.45, 0.30]),
        "Large":      np.array([0.15, 0.40, 0.45]),
        "Enterprise": np.array([0.10, 0.35, 0.55]),
    }
    ProcurementCycle = np.empty(M, dtype=object)
    for label, probs in proc_probs_by_size.items():
        mask = CompanySize == label
        n = int(mask.sum())
        if n:
            ProcurementCycle[mask] = rng.choice(_ORG_PROCUREMENT_CYCLE, size=n, p=probs)

    contract_probs_by_size = {
        "Startup":    np.array([0.60, 0.30, 0.10]),
        "Small":      np.array([0.40, 0.40, 0.20]),
        "Medium":     np.array([0.25, 0.40, 0.35]),
        "Large":      np.array([0.15, 0.35, 0.50]),
        "Enterprise": np.array([0.10, 0.25, 0.65]),
    }
    ContractType = np.empty(M, dtype=object)
    for label, probs in contract_probs_by_size.items():
        mask = CompanySize == label
        n = int(mask.sum())
        if n:
            ContractType[mask] = rng.choice(_ORG_CONTRACT_TYPE, size=n, p=probs)

    revenue_norm = np.clip(np.log1p(AnnualRevenue) / 25.0, 0, 1)
    age_norm = np.clip((current_year - FoundedYear) / 60.0, 0, 1)
    credit_score = 0.5 * revenue_norm + 0.3 * age_norm + 0.2 * rng.random(M)
    credit_idx = np.clip(
        (credit_score * len(_ORG_CREDIT_RATINGS)).astype(int),
        0, len(_ORG_CREDIT_RATINGS) - 1,
    )
    CreditRating = _ORG_CREDIT_RATINGS[credit_idx]

    pay_probs_by_size = {
        "Startup":    np.array([0.50, 0.20, 0.05, 0.25]),
        "Small":      np.array([0.40, 0.30, 0.15, 0.15]),
        "Medium":     np.array([0.30, 0.35, 0.25, 0.10]),
        "Large":      np.array([0.20, 0.30, 0.40, 0.10]),
        "Enterprise": np.array([0.15, 0.25, 0.50, 0.10]),
    }
    PaymentTerms = np.empty(M, dtype=object)
    for label, probs in pay_probs_by_size.items():
        mask = CompanySize == label
        n = int(mask.sum())
        if n:
            PaymentTerms[mask] = rng.choice(_ORG_PAYMENT_TERMS, size=n, p=probs)

    ship_probs_by_size = {
        "Startup":    np.array([0.50, 0.30, 0.15, 0.05]),
        "Small":      np.array([0.35, 0.30, 0.25, 0.10]),
        "Medium":     np.array([0.20, 0.25, 0.35, 0.20]),
        "Large":      np.array([0.15, 0.20, 0.40, 0.25]),
        "Enterprise": np.array([0.10, 0.15, 0.40, 0.35]),
    }
    PreferredShippingMethod = np.empty(M, dtype=object)
    for label, probs in ship_probs_by_size.items():
        mask = CompanySize == label
        n = int(mask.sum())
        if n:
            PreferredShippingMethod[mask] = rng.choice(_ORG_SHIPPING_METHODS, size=n, p=probs)

    HasDedicatedAccountTeam = np.where(
        is_large & (rng.random(M) < 0.45), "Yes",
        np.where(~is_large & (rng.random(M) < 0.08), "Yes", "No"),
    )

    AvgOrderValueUSD = np.round(
        AnnualRevenue * 0.001 * rng.uniform(0.5, 2.0, size=M)
        + rng.exponential(500, size=M),
        2,
    )
    AvgOrderValueUSD = np.clip(AvgOrderValueUSD, 100, 500_000)

    revenue_rank = np.argsort(np.argsort(-AnnualRevenue))
    top_10_pct = int(max(M * 0.10, 1))
    IsStrategicAccount = np.where(revenue_rank < top_10_pct, "Yes", "No")

    start_ts = pd.to_datetime(org_start)
    delta = end_date - start_ts
    YearsAsCustomer = np.clip(
        (delta.days / 365.25).astype("float64"),
        0, 100,
    ).round(1)

    churn_norm_org = np.clip(org_churn / (org_churn.max() + 1e-9), 0, 1)
    sat_score = 1.0 - churn_norm_org + rng.normal(0, 0.15, size=M)
    sat_idx = np.clip(
        (np.clip(sat_score, 0, 1) * len(_ORG_SATISFACTION_TIERS)).astype(int),
        0, len(_ORG_SATISFACTION_TIERS) - 1,
    )
    SatisfactionTier = _ORG_SATISFACTION_TIERS[sat_idx]

    HasExclusiveDeal = np.where(
        (IsStrategicAccount == "Yes") & (rng.random(M) < 0.35), "Yes",
        np.where(rng.random(M) < 0.03, "Yes", "No"),
    )

    esg_probs_by_size = {
        "Startup":    np.array([0.05, 0.10, 0.15, 0.10, 0.60]),
        "Small":      np.array([0.05, 0.15, 0.20, 0.15, 0.45]),
        "Medium":     np.array([0.10, 0.20, 0.25, 0.15, 0.30]),
        "Large":      np.array([0.15, 0.25, 0.25, 0.15, 0.20]),
        "Enterprise": np.array([0.20, 0.30, 0.20, 0.15, 0.15]),
    }
    ESGRating = np.empty(M, dtype=object)
    for label, probs in esg_probs_by_size.items():
        mask = CompanySize == label
        n = int(mask.sum())
        if n:
            ESGRating[mask] = rng.choice(_ORG_ESG_RATINGS, size=n, p=probs)

    return pd.DataFrame({
        "CustomerKey": org_keys,
        "Industry": Industry,
        "CompanySize": CompanySize,
        "AnnualRevenue": AnnualRevenue,
        "FoundedYear": pd.array(FoundedYear, dtype="Int64"),
        "IsPubliclyTraded": IsPubliclyTraded,
        "HeadquarterCountry": HeadquarterCountry,
        "Website": Website,
        "OrgDomain": OrgDomain,
        "OrgAddress": OrgAddress,
        "PrimaryContactName": PrimaryContactName,
        "PrimaryContactRole": PrimaryContactRole,
        "NumberOfEmployees": pd.array(NumberOfEmployees, dtype="Int64"),
        "NumberOfLocations": pd.array(NumberOfLocations, dtype="Int64"),
        "ProcurementCycle": ProcurementCycle,
        "ContractType": ContractType,
        "CreditRating": CreditRating,
        "PaymentTerms": PaymentTerms,
        "PreferredShippingMethod": PreferredShippingMethod,
        "HasDedicatedAccountTeam": HasDedicatedAccountTeam,
        "AvgOrderValueUSD": AvgOrderValueUSD,
        "IsStrategicAccount": IsStrategicAccount,
        "YearsAsCustomer": YearsAsCustomer,
        "SatisfactionTier": SatisfactionTier,
        "HasExclusiveDeal": HasExclusiveDeal,
        "ESGRating": ESGRating,
    })


# ---------------------------------------------------------
# Main generator
# ---------------------------------------------------------
def generate_synthetic_customers(cfg: Dict, parquet_dims_folder: Path):
    cust_cfg = cfg["customers"]
    total_customers = int(cust_cfg["total_customers"])
    if total_customers <= 0:
        raise ValueError("customers.total_customers must be > 0")

    default_seed = cfg.get("defaults", {}).get("seed", 42)
    override_seed = (cust_cfg.get("override") or {}).get("seed")
    seed = override_seed if override_seed is not None else default_seed
    rng = np.random.default_rng(int(seed))

    start_date, end_date = _parse_cfg_dates(cfg)
    start_month0, _end_month0, T = _month_index_space(start_date, end_date)

    active_ratio = cust_cfg.get("active_ratio", 1.0)
    if not isinstance(active_ratio, (int, float)) or not (0 < float(active_ratio) <= 1):
        raise ValueError("customers.active_ratio must be a number in the range (0, 1]")

    pct_india = float(cust_cfg["pct_india"])
    pct_us = float(cust_cfg["pct_us"])
    pct_eu = float(cust_cfg["pct_eu"])
    pct_asia = float(cust_cfg.get("pct_asia", 0.0))  # optional; defaults to 0
    pct_org = float(cust_cfg["pct_org"])

    if not np.isfinite(pct_org) or pct_org < 0 or pct_org > 100:
        raise ValueError("customers.pct_org must be a finite number in [0, 100]")

    p_in, p_us, p_eu, p_as = _validate_percentages(pct_india, pct_us, pct_eu, pct_asia)

    # --- shared name pools ---
    names_folder = resolve_people_folder(cfg)
    enable_asia = p_as > 0.0
    people_pools = load_people_pools(names_folder, enable_asia=enable_asia, legacy_support=True)

    geography, _ = load_dimension("geography", parquet_dims_folder, cfg["geography"])
    geo_keys = geography["GeographyKey"].to_numpy()

    geo_lookup = geography.set_index("GeographyKey")[["City", "State", "Country"]]

    N = int(total_customers)
    CustomerKey = np.arange(1, N + 1, dtype="int64")

    active_count = int(np.floor(N * float(active_ratio)))
    if active_count <= 0:
        raise ValueError(
            "customers.active_ratio results in zero active customers; "
            "increase active_ratio or total_customers"
        )

    if active_count < N:
        active_customer_keys = rng.choice(CustomerKey, size=active_count, replace=False)
    else:
        active_customer_keys = CustomerKey

    is_active = np.zeros(N, dtype="int64")
    is_active[(active_customer_keys - 1).astype("int64")] = 1

    region_labels = ["IN", "US", "EU"] + (["AS"] if enable_asia else [])
    region_probs = [p_in, p_us, p_eu] + ([p_as] if enable_asia else [])
    Region = rng.choice(region_labels, size=N, p=region_probs)

    IsOrg = rng.random(N) < (pct_org / 100.0)

    Gender = np.empty(N, dtype=object)
    Gender[~IsOrg] = rng.choice(["Male", "Female"], size=(~IsOrg).sum())
    Gender[IsOrg] = "Org"

    GeographyKey = rng.choice(geo_keys, size=N, replace=True)

    # --- names via name_pools (no output schema change) ---
    FirstName, LastName, _ = assign_person_names(
        keys=CustomerKey,
        region=Region,
        gender=Gender,
        is_org=IsOrg,
        pools=people_pools,
        seed=int(seed),
        include_middle=False,
        default_region="US",
    )
    safe_first = np.where(pd.isna(FirstName), "", FirstName.astype(object))
    safe_last = np.where(pd.isna(LastName), "", LastName.astype(object))

    # -----------------------------------------------------
    # Organization handling (meaningful org names from pool)
    # -----------------------------------------------------
    org_file = resolve_org_names_file(cfg)
    org_pool = load_org_names(org_file)
    OrgName = assign_org_names(
        keys=CustomerKey,
        is_org=IsOrg,
        org_pool=org_pool,
        seed=int(seed),
    )

    # -----------------------------------------------------
    # Email (with deduplication via CustomerKey suffix)
    # -----------------------------------------------------
    Email = np.empty(N, dtype=object)

    personal_mask = ~IsOrg
    n_personal = int(personal_mask.sum())
    if n_personal:
        domain = rng.choice(PERSONAL_EMAIL_DOMAINS, size=n_personal, replace=True)
        user = (
            safe_first[personal_mask].astype(str)
            + "."
            + safe_last[personal_mask].astype(str)
        ).astype(str)
        user = np.char.lower(np.char.replace(user, " ", ""))
        suffix = CustomerKey[personal_mask].astype(str)
        Email[personal_mask] = user + suffix + "@" + domain

    n_org = int(IsOrg.sum())
    if n_org:
        org_slugs = np.array(
            [slugify_domain_label(x) for x in OrgName[IsOrg].astype(str)],
            dtype=object,
        )
        org_key_suffix = CustomerKey[IsOrg].astype(str)
        OrgDomain = org_slugs + org_key_suffix + np.full(n_org, ".com", dtype=object)
        org_prefix = rng.choice(ORG_EMAIL_PREFIXES, size=n_org, replace=True)
        Email[IsOrg] = org_prefix.astype(str) + "@" + OrgDomain

    # -----------------------------------------------------
    # CustomerName
    # -----------------------------------------------------
    CustomerName = np.where(
        IsOrg,
        OrgName.astype(str),
        safe_first.astype(str) + " " + safe_last.astype(str),
    )

    # -----------------------------------------------------
    # Demographics
    # -----------------------------------------------------
    BirthDate = np.full(N, np.datetime64("NaT"), dtype="datetime64[ns]")
    ages_days = np.zeros(N, dtype="int64")
    person_mask = ~IsOrg
    n_person = int(person_mask.sum())
    if n_person:
        ages_days[person_mask] = rng.integers(AGE_MIN_DAYS, AGE_MAX_DAYS, size=n_person)
        anchor = end_date.normalize()
        dates = anchor - pd.to_timedelta(ages_days[person_mask], unit="D")
        BirthDate[person_mask] = pd.to_datetime(dates).to_numpy("datetime64[ns]")

    # Age bracket for person rows (0=18-24 .. 5=65+), used to condition demographics
    ages_years = ages_days / 365.25
    person_age_bracket = (
        np.searchsorted(AGE_GROUP_EDGES, ages_years[person_mask])
        if n_person else np.array([], dtype="int64")
    )
    person_idx = np.where(person_mask)[0]

    MaritalStatus = np.empty(N, dtype=object)
    MaritalStatus[IsOrg] = None
    if n_person:
        for bracket in range(len(AGE_GROUP_LABELS)):
            bmask = person_age_bracket == bracket
            n_b = int(bmask.sum())
            if n_b:
                MaritalStatus[person_idx[bmask]] = rng.choice(
                    MARITAL_STATUS_LABELS, size=n_b,
                    p=MARITAL_PROBS_BY_AGE[bracket],
                )

    Education = np.empty(N, dtype=object)
    Education[:] = None
    if n_person:
        for bracket in range(len(AGE_GROUP_LABELS)):
            bmask = person_age_bracket == bracket
            n_b = int(bmask.sum())
            if n_b:
                Education[person_idx[bmask]] = rng.choice(
                    EDUCATION_LABELS, size=n_b,
                    p=EDUCATION_PROBS_BY_AGE[bracket],
                )

    Occupation = np.empty(N, dtype=object)
    Occupation[:] = None
    if n_person:
        for edu_label in EDUCATION_LABELS:
            emask = Education[person_mask] == edu_label
            n_e = int(emask.sum())
            if n_e:
                Occupation[person_idx[emask]] = rng.choice(
                    OCCUPATION_LABELS, size=n_e,
                    p=OCCUPATION_PROBS_BY_EDUCATION[edu_label],
                )

    income_raw = _generate_correlated_income(rng, Education, Occupation, person_mask, N)
    YearlyIncome = pd.array(
        np.where(IsOrg, pd.NA, income_raw), dtype="Int64"
    )

    children_raw = np.zeros(N, dtype="int64")
    if n_person:
        person_marital = MaritalStatus[person_mask]
        for ms in MARITAL_STATUS_LABELS:
            for bracket in range(len(AGE_GROUP_LABELS)):
                combo = (person_marital == ms) & (person_age_bracket == bracket)
                n_c = int(combo.sum())
                if n_c:
                    lam = CHILDREN_LAMBDA_BY_MARITAL_AGE.get((ms, bracket), 1.0)
                    children_raw[person_idx[combo]] = np.clip(
                        rng.poisson(lam=lam, size=n_c), 0, MAX_CHILDREN - 1,
                    )
    TotalChildren = pd.array(
        np.where(IsOrg, pd.NA, children_raw), dtype="Int64",
    )

    # -----------------------------------------------------
    # Derived demographic columns
    # -----------------------------------------------------
    ages_years = ages_days / 365.25

    AgeGroup = np.empty(N, dtype=object)
    AgeGroup[:] = None
    if n_person:
        idx = np.searchsorted(AGE_GROUP_EDGES, ages_years[person_mask])
        AgeGroup[person_mask] = AGE_GROUP_LABELS[idx]

    income_for_grouping = income_raw.astype("float64")
    IncomeGroup = np.empty(N, dtype=object)
    IncomeGroup[:] = None
    if n_person:
        idx = np.searchsorted(INCOME_GROUP_EDGES, income_for_grouping[person_mask])
        IncomeGroup[person_mask] = INCOME_GROUP_LABELS[idx]

    income_norm = np.zeros(N, dtype="float64")
    if n_person:
        income_norm[person_mask] = (
            (income_for_grouping[person_mask] - INCOME_MIN)
            / max(INCOME_MAX - INCOME_MIN, 1)
        )
    income_norm = np.clip(income_norm, 0.0, 1.0)

    CreditScore = _generate_credit_scores(rng, income_norm, Education, person_mask, N)

    HomeOwnership = np.empty(N, dtype=object)
    HomeOwnership[:] = None
    if n_person:
        ig = IncomeGroup[person_mask]
        for grp, base_probs in HOME_OWNERSHIP_PROBS_BY_INCOME.items():
            grp_mask_local = ig == grp
            if not grp_mask_local.any():
                continue
            for bracket in range(len(AGE_GROUP_LABELS)):
                combo = grp_mask_local & (person_age_bracket == bracket)
                n_c = int(combo.sum())
                if n_c:
                    adjusted = np.clip(base_probs + HOME_OWNERSHIP_AGE_SHIFT[bracket], 0.01, None)
                    adjusted = adjusted / adjusted.sum()
                    HomeOwnership[person_idx[combo]] = rng.choice(
                        HOME_OWNERSHIP_LABELS, size=n_c, p=adjusted,
                    )

    NumberOfCars = np.full(N, pd.NA, dtype=object)
    if n_person:
        car_lambda = CAR_LAMBDA_BY_AGE[person_age_bracket]
        base_cars = rng.poisson(lam=car_lambda)
        income_boost_cars = (income_norm[person_mask] > 0.5).astype(int)
        us_boost = (Region[person_mask] == "US").astype(int)
        raw_cars = np.clip(base_cars + income_boost_cars + us_boost, 0, 4)
        NumberOfCars[person_mask] = raw_cars

    PhoneNumber = _generate_phone_numbers(rng, Region, N)

    REFERRAL_SOURCE_LABELS = np.array(["None", "Friend", "Family", "Colleague"])
    REFERRAL_SOURCE_PROBS = np.array([0.50, 0.25, 0.13, 0.12])
    ReferralSource = rng.choice(REFERRAL_SOURCE_LABELS, size=N, p=REFERRAL_SOURCE_PROBS)

    PreferredContactMethod = rng.choice(
        CONTACT_METHOD_LABELS, size=N, p=CONTACT_METHOD_PROBS
    )

    # -----------------------------------------------------
    # Lifecycle + behavioral knobs
    # -----------------------------------------------------
    lifecycle_cfg = cust_cfg.get("lifecycle", {}) or {}

    initial_active_raw = lifecycle_cfg.get("initial_active_customers", 0.45) or 0
    initial_active_raw = float(initial_active_raw)
    if 0.0 < initial_active_raw <= 1.0:
        initial_active_customers = int(round(initial_active_raw * N))
    else:
        initial_active_customers = int(initial_active_raw)
    initial_spread_months = int(lifecycle_cfg.get("initial_spread_months", 3) or 0)

    even_start_months = bool(lifecycle_cfg.get("even_start_months", False))

    acquisition_curve = lifecycle_cfg.get("acquisition_curve")
    if acquisition_curve is None:
        acquisition_curve = "logistic"

    acquisition_params = lifecycle_cfg.get("acquisition_params", {}) or {}
    if acquisition_curve == "logistic" and not acquisition_params:
        acquisition_params = {"midpoint": 0.30, "steepness": 8.0}
    weights = _acquisition_weights(T, acquisition_curve, acquisition_params)

    CustomerStartMonth = rng.choice(np.arange(T), size=N, p=weights).astype("int64")

    if initial_active_customers > 0:
        k = min(initial_active_customers, N)

        active_idx = np.where(is_active == 1)[0]
        if active_idx.size > 0:
            k2 = min(k, int(active_idx.size))
            warm_idx = rng.choice(active_idx, size=k2, replace=False)
            if k2 < k:
                rest = np.setdiff1d(np.arange(N), warm_idx, assume_unique=False)
                extra = rng.choice(rest, size=(k - k2), replace=False)
                warm_idx = np.concatenate([warm_idx, extra])
        else:
            warm_idx = rng.choice(np.arange(N), size=k, replace=False)

        spread_hi = int(min(max(initial_spread_months, 0), max(T - 1, 0)))
        if spread_hi > 0:
            CustomerStartMonth[warm_idx] = rng.integers(0, spread_hi + 1, size=warm_idx.size, dtype=np.int64)
        else:
            CustomerStartMonth[warm_idx] = 0

    enable_churn = bool(lifecycle_cfg.get("enable_churn", True))
    base_monthly_churn = float(lifecycle_cfg.get("base_monthly_churn", 0.04))
    min_tenure_months = int(lifecycle_cfg.get("min_tenure_months", 3))

    CustomerWeight = rng.lognormal(mean=0.0, sigma=0.6, size=N).astype("float64")
    CustomerTemperature = np.clip(rng.normal(loc=0.6, scale=0.25, size=N), 0.05, 1.0).astype("float64")

    segment_cfg = lifecycle_cfg.get("segments", {}) or {}
    seg_names = np.array(segment_cfg.get("names", ["Budget", "Mainstream", "Premium"]), dtype=object)
    seg_probs = np.array(segment_cfg.get("probs", [0.35, 0.5, 0.15]), dtype="float64")
    seg_probs = seg_probs / seg_probs.sum()
    CustomerSegment = rng.choice(seg_names, size=N, p=seg_probs)

    churn_bias_cfg = lifecycle_cfg.get("churn_bias", {}) or {}
    bias_sigma = float(churn_bias_cfg.get("sigma", 0.5))
    CustomerChurnBias = rng.lognormal(mean=0.0, sigma=bias_sigma, size=N).astype("float64")

    # -----------------------------------------------------
    # Loyalty tier + acquisition channel
    # -----------------------------------------------------
    enrich_cfg = (cust_cfg.get("enrichment") or {}) if isinstance(cust_cfg, dict) else {}
    loyalty_cfg = (enrich_cfg.get("loyalty_tier") or {}) if isinstance(enrich_cfg, dict) else {}
    acq_cfg = (enrich_cfg.get("acquisition_channel") or {}) if isinstance(enrich_cfg, dict) else {}

    loyalty_dim = _read_parquet_dim(parquet_dims_folder, "loyalty_tiers")
    acq_dim = _read_parquet_dim(parquet_dims_folder, "customer_acquisition_channels")

    loyalty_key_col = _first_existing_col(loyalty_dim, ["LoyaltyTierKey", "TierKey", "Key"])
    loyalty_name_col = _first_existing_col(loyalty_dim, ["LoyaltyTier", "TierName", "Name"])

    acq_key_col = _first_existing_col(acq_dim, ["CustomerAcquisitionChannelKey", "AcquisitionChannelKey", "ChannelKey", "Key"])
    acq_name_col = _first_existing_col(acq_dim, ["CustomerAcquisitionChannel", "AcquisitionChannel", "ChannelName", "Name"])

    loyalty_dim = loyalty_dim[[loyalty_key_col, loyalty_name_col]].dropna(subset=[loyalty_key_col]).copy()
    loyalty_dim[loyalty_key_col] = pd.to_numeric(loyalty_dim[loyalty_key_col], errors="coerce")
    loyalty_dim = loyalty_dim.dropna(subset=[loyalty_key_col]).sort_values(loyalty_key_col)

    tier_keys = loyalty_dim[loyalty_key_col].astype("int64").to_numpy()

    score = (
        LOYALTY_W_WEIGHT * np.log1p(CustomerWeight)
        + LOYALTY_W_TEMP * CustomerTemperature
        + LOYALTY_W_INCOME * income_norm
    )

    probs = loyalty_cfg.get("probs_low_to_high")
    if probs is None:
        tier_probs = _default_tier_probs(len(tier_keys))
    else:
        tier_probs = _normalize_probs(np.array(probs, dtype="float64"))
        if len(tier_probs) != len(tier_keys):
            raise ValueError(
                f"customers.enrichment.loyalty_tier.probs_low_to_high length must match tiers "
                f"({len(tier_keys)}), got {len(tier_probs)}"
            )

    LoyaltyTierKey = _assign_tier_by_score(
        score=score,
        tier_keys_sorted_low_to_high=tier_keys,
        tier_probs_low_to_high=tier_probs,
    )

    if IsOrg.any():
        org_mode = str(loyalty_cfg.get("org_mode", "top2")).lower()
        if org_mode == "top1":
            LoyaltyTierKey[IsOrg] = tier_keys[-1]
        elif org_mode == "top2" and len(tier_keys) >= 2:
            top2 = tier_keys[-2:]
            LoyaltyTierKey[IsOrg] = rng.choice(top2, size=int(IsOrg.sum()), p=[0.35, 0.65])

    acq_dim = acq_dim[[acq_key_col, acq_name_col]].dropna(subset=[acq_key_col]).copy()
    acq_dim[acq_key_col] = pd.to_numeric(acq_dim[acq_key_col], errors="coerce")
    acq_dim = acq_dim.dropna(subset=[acq_key_col]).sort_values(acq_key_col)

    acq_keys = acq_dim[acq_key_col].astype("int64").to_numpy()
    acq_names = acq_dim[acq_name_col].astype(str).to_numpy()

    w_ind = _acquisition_weights_from_names(acq_names, org=False)
    w_org = _acquisition_weights_from_names(acq_names, org=True)

    CustomerAcquisitionChannelKey = rng.choice(acq_keys, size=N, replace=True, p=w_ind)
    if IsOrg.any():
        CustomerAcquisitionChannelKey[IsOrg] = rng.choice(acq_keys, size=int(IsOrg.sum()), replace=True, p=w_org)

    CustomerEndMonth = _simulate_end_month(
        rng=rng,
        start_month=CustomerStartMonth,
        churn_bias=CustomerChurnBias,
        T=T,
        enable=enable_churn,
        base_monthly_churn=base_monthly_churn,
        min_tenure_months=min_tenure_months,
    )

    CustomerStartDate = _month_idx_to_date(start_month0, CustomerStartMonth)

    if enable_churn:
        end_idx = np.array([int(x) if not pd.isna(x) else -1 for x in CustomerEndMonth], dtype="int64")
        CustomerEndDate = np.where(
            end_idx >= 0,
            _month_idx_to_date(start_month0, end_idx),
            pd.NaT,
        )
    else:
        CustomerEndDate = pd.Series(pd.NaT, index=np.arange(N), dtype="datetime64[ns]").to_numpy()

    CustomerType = np.where(IsOrg, "Organization", "Individual")
    CompanyName = np.where(IsOrg, OrgName, None)

    # -----------------------------------------------------
    # Customer satisfaction (correlated with churn bias)
    # -----------------------------------------------------
    churn_norm = np.clip(CustomerChurnBias / (CustomerChurnBias.max() + 1e-9), 0, 1)
    csat_raw = 5.0 - 3.0 * churn_norm + rng.normal(0, 0.5, size=N)
    CustomerSatisfactionScore = np.clip(np.round(csat_raw).astype(int), 1, 5)

    # =====================================================
    # CustomerProfile columns
    # =====================================================

    # --- Geography-derived: City, State, Country per customer ---
    geo_mapped = geo_lookup.reindex(GeographyKey)
    geo_city = geo_mapped["City"].to_numpy().astype(object)
    geo_state = geo_mapped["State"].to_numpy().astype(object)
    geo_country = geo_mapped["Country"].to_numpy().astype(object)

    CurrentCity = geo_city.copy()
    BirthCity = geo_city.copy()
    relocated = rng.random(N) < 0.25
    if relocated.any():
        BirthCity[relocated] = rng.permutation(geo_city)[: int(relocated.sum())]

    # --- Address columns ---
    HomeAddress, WorkAddress = _generate_addresses(
        rng, Region, CustomerKey, geo_city, geo_state, N,
    )
    PostalCode = _generate_postal_codes(rng, Region, N)
    Latitude, Longitude = _generate_lat_lon(rng, Region, N)

    # --- Urban/Rural, TimeZone ---
    UrbanRural = np.empty(N, dtype=object)
    for rc, probs in _URBAN_RURAL_PROBS.items():
        mask = Region == rc
        n_rc = int(mask.sum())
        if n_rc:
            UrbanRural[mask] = rng.choice(_URBAN_RURAL_LABELS, size=n_rc, p=probs)
    remaining_ur = pd.isna(UrbanRural) | (UrbanRural == None)  # noqa: E711
    if remaining_ur.any():
        UrbanRural[remaining_ur] = rng.choice(
            _URBAN_RURAL_LABELS, size=int(remaining_ur.sum()),
            p=_URBAN_RURAL_PROBS["US"],
        )

    TimeZone = np.empty(N, dtype=object)
    for rc, tz_pool in _REGION_TIMEZONE.items():
        mask = Region == rc
        n_rc = int(mask.sum())
        if n_rc:
            TimeZone[mask] = rng.choice(tz_pool, size=n_rc)
    remaining_tz = pd.isna(TimeZone) | (TimeZone == None)  # noqa: E711
    if remaining_tz.any():
        TimeZone[remaining_tz] = "America/New_York"

    # --- Distance to nearest store (correlated with UrbanRural) ---
    DistanceToNearestStoreKm = np.zeros(N, dtype="float64")
    for area, (lo, hi) in _DISTANCE_BY_AREA.items():
        mask = UrbanRural == area
        n_a = int(mask.sum())
        if n_a:
            DistanceToNearestStoreKm[mask] = np.round(
                rng.uniform(lo, hi, size=n_a), 1
            )

    # --- Digital & Engagement ---
    PreferredLanguage = np.empty(N, dtype=object)
    for rc, lang_pool in _LANGUAGE_BY_REGION.items():
        mask = Region == rc
        n_rc = int(mask.sum())
        if n_rc:
            PreferredLanguage[mask] = rng.choice(lang_pool, size=n_rc)
    remaining_lang = pd.isna(PreferredLanguage) | (PreferredLanguage == None)  # noqa: E711
    if remaining_lang.any():
        PreferredLanguage[remaining_lang] = "English"

    age_young = np.zeros(N, dtype="float64")
    if n_person:
        age_young[person_mask] = np.clip(1.0 - (ages_years[person_mask] - 18) / 52, 0.1, 0.95)
    org_digital = np.full(N, 0.75)
    young_factor = np.where(IsOrg, org_digital, age_young)

    HasOnlineAccount = np.where(rng.random(N) < (0.55 + 0.30 * young_factor), "Yes", "No")
    OptInMarketing = np.where(rng.random(N) < 0.65, "Yes", "No")
    SocialMediaFollower = np.where(
        rng.random(N) < (0.20 + 0.40 * young_factor), "Yes", "No"
    )
    AppInstalled = np.where(
        (HasOnlineAccount == "Yes") & (rng.random(N) < (0.30 + 0.35 * young_factor)),
        "Yes", "No",
    )

    NewsletterFrequency = np.where(
        OptInMarketing == "No",
        "None",
        rng.choice(_NEWSLETTER_FREQ, size=N, p=np.array([0.30, 0.45, 0.25])),
    )

    device_probs_young = np.array([0.60, 0.25, 0.15])
    device_probs_old = np.array([0.25, 0.55, 0.20])
    DevicePreference = np.empty(N, dtype=object)
    young_mask = young_factor > 0.5
    n_young = int(young_mask.sum())
    n_old = N - n_young
    if n_young:
        DevicePreference[young_mask] = rng.choice(_DEVICE_PREFS, size=n_young, p=device_probs_young)
    if n_old:
        DevicePreference[~young_mask] = rng.choice(_DEVICE_PREFS, size=n_old, p=device_probs_old)

    days_back = rng.integers(0, 180, size=N)
    LastWebVisitDate = pd.to_datetime(end_date) - pd.to_timedelta(days_back, unit="D")
    LastWebVisitDate = LastWebVisitDate.to_numpy("datetime64[ns]")

    # --- Financial & Behavioral ---
    PreferredPaymentMethod = rng.choice(
        _PAYMENT_METHOD_LABELS, size=N, p=_PAYMENT_METHOD_PROBS,
    )

    start_dates_ts = pd.to_datetime(CustomerStartDate)
    days_after_start = rng.integers(0, 90, size=N)
    MemberSinceDate = (start_dates_ts + pd.to_timedelta(days_after_start, unit="D")).to_numpy("datetime64[ns]")

    IsEmployee = np.where(rng.random(N) < 0.02, "Yes", "No")
    IsEmployee[IsOrg] = "No"

    spend_score = np.clip(
        0.4 * CustomerWeight / (CustomerWeight.max() + 1e-9)
        + 0.4 * income_norm
        + 0.2 * rng.random(N),
        0, 1,
    )
    AnnualSpendBucket = np.where(
        spend_score < 0.30, "Low",
        np.where(spend_score < 0.60, "Medium",
        np.where(spend_score < 0.85, "High", "VIP")),
    )

    HasGiftCardBalance = np.where(rng.random(N) < 0.15, "Yes", "No")

    tier_norm = LoyaltyTierKey.astype("float64") / max(tier_keys.max(), 1)
    RewardPointsBalance = np.clip(
        (tier_norm * 30000 + rng.exponential(5000, size=N)).astype(int),
        0, 50000,
    )

    AvgOrderFrequencyDays = np.clip(
        (90.0 / (CustomerWeight + 0.1) + rng.normal(0, 10, size=N)).astype(int),
        7, 365,
    )

    # --- CX analytics ---
    nps_base = (CustomerSatisfactionScore - 1) * 2.5
    NPS = np.clip(
        (nps_base + rng.normal(0, 1.0, size=N)).astype(int),
        0, 10,
    )

    tenure_months = np.clip(T - CustomerStartMonth, 1, T).astype("float64")
    clv_raw = (
        income_norm * 2000
        + CustomerWeight * 500
        + tenure_months * 20
        + rng.exponential(300, size=N)
    )
    CustomerLifetimeValue = np.round(np.clip(clv_raw, 50, 100_000), 2)

    churn_risk_score = (
        0.45 * churn_norm
        + 0.30 * (1.0 - CustomerSatisfactionScore / 5.0)
        + 0.25 * (1.0 - np.clip(tenure_months / T, 0, 1))
    )
    ChurnRisk = np.where(
        churn_risk_score < 0.33, "Low",
        np.where(churn_risk_score < 0.66, "Medium", "High"),
    )

    # =====================================================
    # Build Customers dataframe (identity + engine)
    # =====================================================
    customers_df = pd.DataFrame(
        {
            "CustomerKey": CustomerKey,
            "CustomerName": CustomerName,
            "DOB": BirthDate,
            "Gender": Gender,
            "EmailAddress": Email,
            "PhoneNumber": PhoneNumber,
            "HomeAddress": HomeAddress,
            "WorkAddress": WorkAddress,
            "Latitude": Latitude,
            "Longitude": Longitude,
            "PostalCode": PostalCode,
            "CustomerType": CustomerType,
            "CompanyName": CompanyName,
            "GeographyKey": GeographyKey,
            "LoyaltyTierKey": pd.Series(LoyaltyTierKey, dtype="Int64"),
            "CustomerAcquisitionChannelKey": pd.Series(CustomerAcquisitionChannelKey, dtype="Int64"),
            "IsActiveInSales": is_active,
            "CustomerStartMonth": CustomerStartMonth.astype("int64"),
            "CustomerEndMonth": pd.Series(CustomerEndMonth, dtype="Int64"),
            "CustomerStartDate": pd.to_datetime(CustomerStartDate),
            "CustomerEndDate": pd.to_datetime(CustomerEndDate),
            "CustomerWeight": CustomerWeight,
            "CustomerTemperature": CustomerTemperature,
            "CustomerSegment": CustomerSegment,
            "CustomerChurnBias": CustomerChurnBias,
        }
    )

    # =====================================================
    # Build CustomerProfile dataframe (analytical slicers)
    # =====================================================
    profile_df = pd.DataFrame(
        {
            "CustomerKey": CustomerKey,
            "AgeGroup": AgeGroup,
            "YearlyIncome": YearlyIncome,
            "IncomeGroup": IncomeGroup,
            "MaritalStatus": MaritalStatus,
            "Education": Education,
            "Occupation": Occupation,
            "TotalChildren": TotalChildren,
            "HomeOwnership": HomeOwnership,
            "NumberOfCars": pd.Series(NumberOfCars, dtype="Int64"),
            "CreditScore": pd.Series(CreditScore, dtype="Int64"),
            "UrbanRural": UrbanRural,
            "TimeZone": TimeZone,
            "BirthCity": BirthCity,
            "CurrentCity": CurrentCity,
            "DistanceToNearestStoreKm": DistanceToNearestStoreKm,
            "PreferredLanguage": PreferredLanguage,
            "HasOnlineAccount": HasOnlineAccount,
            "OptInMarketing": OptInMarketing,
            "SocialMediaFollower": SocialMediaFollower,
            "AppInstalled": AppInstalled,
            "NewsletterFrequency": NewsletterFrequency,
            "DevicePreference": DevicePreference,
            "LastWebVisitDate": pd.to_datetime(LastWebVisitDate),
            "PreferredPaymentMethod": PreferredPaymentMethod,
            "PreferredContactMethod": PreferredContactMethod,
            "ReferralSource": ReferralSource,
            "MemberSinceDate": pd.to_datetime(MemberSinceDate),
            "IsEmployee": IsEmployee,
            "AnnualSpendBucket": AnnualSpendBucket,
            "HasGiftCardBalance": HasGiftCardBalance,
            "RewardPointsBalance": pd.array(RewardPointsBalance, dtype="Int64"),
            "AvgOrderFrequencyDays": pd.array(AvgOrderFrequencyDays, dtype="Int64"),
            "CustomerSatisfactionScore": pd.array(CustomerSatisfactionScore, dtype="Int64"),
            "NPS": pd.array(NPS, dtype="Int64"),
            "CustomerLifetimeValue": CustomerLifetimeValue,
            "ChurnRisk": ChurnRisk,
        }
    )

    # =====================================================
    # Build OrganizationProfile dataframe (org-only)
    # =====================================================
    org_profile_df = _generate_org_profile(
        rng=rng,
        customer_key=CustomerKey,
        is_org=IsOrg,
        org_name=OrgName,
        region=Region,
        customer_start_date=CustomerStartDate,
        churn_bias=CustomerChurnBias,
        customer_weight=CustomerWeight,
        people_pools=people_pools,
        end_date=end_date,
        seed=int(seed),
    )

    active_customer_set = set(active_customer_keys.tolist())
    return customers_df, profile_df, org_profile_df, active_customer_set


# ---------------------------------------------------------
# Pipeline entry
# ---------------------------------------------------------
def run_customers(cfg: Dict, parquet_folder: Path):
    out_path = parquet_folder / "customers.parquet"
    profile_out_path = parquet_folder / "customer_profile.parquet"
    org_profile_out_path = parquet_folder / "organization_profile.parquet"

    cust_cfg = cfg["customers"]

    version_cfg = dict(cust_cfg)
    version_cfg["_schema_version"] = 5
    version_cfg["_has_loyalty_tier"] = True
    version_cfg["_has_acquisition_channel"] = True
    version_cfg["_has_customer_profile"] = True
    version_cfg["_has_org_profile"] = True

    if not should_regenerate("customers", version_cfg, out_path):
        skip("Customers up-to-date")
        return

    with stage("Generating Customers"):
        customers_df, profile_df, org_profile_df, _active = generate_synthetic_customers(cfg, parquet_folder)
        customers_df.to_parquet(out_path, index=False)
        profile_df.to_parquet(profile_out_path, index=False)

        n_ind = int((customers_df["CustomerType"] == "Individual").sum())
        n_org = int((customers_df["CustomerType"] == "Organization").sum())
        info(f"Customers: {len(customers_df):,} rows ({n_ind:,} individual, {n_org:,} org)")
        info(f"Customer Profile: {len(profile_df):,} rows")

        if not org_profile_df.empty:
            org_profile_df.to_parquet(org_profile_out_path, index=False)
            info(f"Organization Profile: {len(org_profile_df):,} rows")

    save_version("customers", version_cfg, out_path)
