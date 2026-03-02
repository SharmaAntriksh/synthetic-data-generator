from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.utils import info, skip, stage
from src.versioning import should_regenerate, save_version
from src.engine.dimension_loader import load_dimension
from src.utils.name_pools import (
    resolve_people_folder,
    load_people_pools,
    assign_person_names,
    resolve_org_names_file,
    load_org_names,
    assign_org_names,
    slugify_domain_label,
)

# ---------------------------------------------------------
# Configurable defaults (extracted from inline magic numbers)
# ---------------------------------------------------------
PERSONAL_EMAIL_DOMAINS = np.array(
    ["gmail.com", "yahoo.com", "outlook.com", "hotmail.com"]
)

MARITAL_STATUS_LABELS = np.array(["Married", "Single"])
MARITAL_STATUS_PROBS = np.array([0.55, 0.45])

EDUCATION_LABELS = np.array(["High School", "Bachelors", "Masters", "PhD"])
EDUCATION_PROBS = np.array([0.20, 0.50, 0.25, 0.05])

OCCUPATION_LABELS = np.array(
    ["Professional", "Clerical", "Skilled", "Service", "Executive"]
)
OCCUPATION_PROBS = np.array([0.50, 0.20, 0.15, 0.10, 0.05])

AGE_MIN_DAYS = 18 * 365
AGE_MAX_DAYS = 70 * 365
INCOME_MIN = 20_000
INCOME_MAX = 200_000
MAX_CHILDREN = 5  # exclusive upper bound for rng.integers

# Loyalty score component weights
LOYALTY_W_WEIGHT = 0.55
LOYALTY_W_TEMP = 0.30
LOYALTY_W_INCOME = 0.15


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

    for i, v in zip(apply_idx, apply_vals):
        end_month[i] = int(v)

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

    # FIX: validate pct_org bounds (was unchecked — values >100 made everyone an org)
    if not np.isfinite(pct_org) or pct_org < 0 or pct_org > 100:
        raise ValueError("customers.pct_org must be a finite number in [0, 100]")

    p_in, p_us, p_eu, p_as = _validate_percentages(pct_india, pct_us, pct_eu, pct_asia)

    # --- shared name pools ---
    names_folder = resolve_people_folder(cfg)
    enable_asia = p_as > 0.0
    people_pools = load_people_pools(names_folder, enable_asia=enable_asia, legacy_support=True)

    geography, _ = load_dimension("geography", parquet_dims_folder, cfg["geography"])
    geo_keys = geography["GeographyKey"].to_numpy()

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
    # FIX: use pd.isna for null checks instead of fragile `== None`
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
        # FIX: append CustomerKey to prevent email collisions (e.g. two "John Smith")
        suffix = CustomerKey[personal_mask].astype(str)
        Email[personal_mask] = user + suffix + "@" + domain

    OrgDomain = np.empty(N, dtype=object)
    OrgDomain[IsOrg] = (
        np.array(
            [slugify_domain_label(x) for x in OrgName[IsOrg].astype(str)], dtype=object
        )
        + ".com"
    )
    OrgDomain[~IsOrg] = None
    Email[IsOrg] = "info@" + OrgDomain[IsOrg]

    # -----------------------------------------------------
    # CustomerName
    # -----------------------------------------------------
    CustomerName = np.where(
        IsOrg,
        OrgName.astype(str) + " (" + CustomerKey.astype(str) + ")",
        safe_first.astype(str) + " " + safe_last.astype(str),
    )

    # -----------------------------------------------------
    # Demographics
    # -----------------------------------------------------
    # FIX: store BirthDate as datetime64[ns] instead of Python date objects
    # (faster downstream, consistent dtype).  Final parquet write converts as needed.
    BirthDate = np.full(N, np.datetime64("NaT"), dtype="datetime64[ns]")
    person_mask = ~IsOrg
    n_person = int(person_mask.sum())
    if n_person:
        ages = rng.integers(AGE_MIN_DAYS, AGE_MAX_DAYS, size=n_person)
        anchor = end_date.normalize()
        dates = anchor - pd.to_timedelta(ages, unit="D")
        BirthDate[person_mask] = pd.to_datetime(dates).to_numpy("datetime64[ns]")

    MaritalStatus = np.empty(N, dtype=object)
    MaritalStatus[~IsOrg] = rng.choice(
        MARITAL_STATUS_LABELS,
        size=int((~IsOrg).sum()),
        p=MARITAL_STATUS_PROBS,
    )
    MaritalStatus[IsOrg] = None

    # FIX: produce proper numeric array + use pd.array with Int64 for nullable ints
    income_raw = rng.integers(INCOME_MIN, INCOME_MAX, size=N).astype("int64")
    YearlyIncome = pd.array(
        np.where(IsOrg, pd.NA, income_raw), dtype="Int64"
    )

    TotalChildren = pd.array(
        np.where(IsOrg, pd.NA, rng.integers(0, MAX_CHILDREN, size=N)),
        dtype="Int64",
    )

    Education = np.where(
        IsOrg,
        None,
        rng.choice(EDUCATION_LABELS, size=N, p=EDUCATION_PROBS),
    )

    Occupation = np.where(
        IsOrg,
        None,
        rng.choice(OCCUPATION_LABELS, size=N, p=OCCUPATION_PROBS),
    )

    # -----------------------------------------------------
    # Lifecycle + behavioral knobs
    # -----------------------------------------------------
    lifecycle_cfg = cust_cfg.get("lifecycle", {}) or {}

    initial_active_customers = int(lifecycle_cfg.get("initial_active_customers", 0) or 0)
    initial_spread_months = int(lifecycle_cfg.get("initial_spread_months", 0) or 0)

    even_start_months = bool(lifecycle_cfg.get("even_start_months", False))

    acquisition_curve = lifecycle_cfg.get("acquisition_curve")
    if acquisition_curve is None:
        acquisition_curve = "uniform" if (even_start_months or initial_spread_months > 0) else "linear_ramp"

    acquisition_params = lifecycle_cfg.get("acquisition_params", {}) or {}
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

    enable_churn = bool(lifecycle_cfg.get("enable_churn", False))
    base_monthly_churn = float(lifecycle_cfg.get("base_monthly_churn", 0.01))
    min_tenure_months = int(lifecycle_cfg.get("min_tenure_months", 2))

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

    # FIX: use nullable Int64 YearlyIncome — extract numeric values safely
    income = np.zeros(N, dtype="float64")
    if (~IsOrg).any():
        income_series = pd.array(YearlyIncome, dtype="Int64")
        income_ind = (
            pd.to_numeric(pd.Series(income_series[~IsOrg]), errors="coerce")
            .fillna(0)
            .to_numpy(dtype="float64")
        )
        income[~IsOrg] = (income_ind - float(INCOME_MIN)) / float(INCOME_MAX - INCOME_MIN)
    income = np.clip(income, 0.0, 1.0)

    score = (
        LOYALTY_W_WEIGHT * np.log1p(CustomerWeight)
        + LOYALTY_W_TEMP * CustomerTemperature
        + LOYALTY_W_INCOME * income
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
    # Build dataframe (DO NOT change schema)
    # -----------------------------------------------------
    df = pd.DataFrame(
        {
            "CustomerKey": CustomerKey,
            "CustomerName": CustomerName,
            "DOB": BirthDate,
            "MaritalStatus": MaritalStatus,
            "Gender": Gender,
            "EmailAddress": Email,
            "YearlyIncome": YearlyIncome,
            "TotalChildren": TotalChildren,
            "Education": Education,
            "Occupation": Occupation,
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

    active_customer_set = set(active_customer_keys.tolist())
    return df, active_customer_set


# ---------------------------------------------------------
# Pipeline entry
# ---------------------------------------------------------
def run_customers(cfg: Dict, parquet_folder: Path):
    out_path = parquet_folder / "customers.parquet"

    cust_cfg = cfg["customers"]
    force = cust_cfg.get("_force_regenerate", False)

    version_cfg = dict(cust_cfg)
    version_cfg["_schema_version"] = 3  # name_pools integration
    version_cfg["_has_loyalty_tier"] = True
    version_cfg["_has_acquisition_channel"] = True

    if not force and not should_regenerate("customers", version_cfg, out_path):
        skip("Customers up-to-date; skipping.")
        return

    with stage("Generating Customers"):
        df, _active_customer_keys = generate_synthetic_customers(cfg, parquet_folder)
        df.to_parquet(out_path, index=False)

    save_version("customers", version_cfg, out_path)
    info(f"Customers dimension written: {out_path}")
