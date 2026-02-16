import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.utils import info, skip, stage
from src.versioning import should_regenerate, save_version
from src.engine.dimension_loader import load_dimension


# ---------------------------------------------------------
# Helper: Load CSV lists
# ---------------------------------------------------------
def load_list(path: str) -> np.ndarray:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")

    s = pd.read_csv(path, header=None, dtype=str)[0].str.strip()
    # Keep only simple name tokens; drop junk rows
    s = s[s.str.match(r"^[A-Za-z\-\'. ]+$")]
    return s.str.title().unique()


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

    Returns:
      start_month0 : Timestamp at month start for start_date's month
      end_month0   : Timestamp at month start for end_date's month
      T            : int month count inclusive
    """
    start_ts = pd.to_datetime(start_date).normalize()
    end_ts = pd.to_datetime(end_date).normalize()

    sp = start_ts.to_period("M")
    ep = end_ts.to_period("M")

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

    Note: This function supports negative month_idx values, but the wider pipeline
    may not expect negative CustomerStartMonth. This file keeps defaults non-negative.
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
      - "linear_ramp": w ~ (m+1)^shape   (shape>1 back-loads; shape<1 front-loads)
      - "logistic": S-curve

    params vary by curve; all optional.
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

    IMPORTANT: This assumes start_month is within [0..T-1] (the default behavior).
    """
    if not enable:
        return np.full(len(start_month), pd.NA, dtype="object")

    if base_monthly_churn < 0:
        raise ValueError("base_monthly_churn must be >= 0")

    end_month = np.full(len(start_month), pd.NA, dtype="object")
    mt = max(int(min_tenure_months), 0)

    for i in range(len(start_month)):
        s = int(start_month[i])
        # safety clamp, in case upstream config is edited incorrectly
        if s < 0:
            s = 0
        if s >= T:
            continue

        hazard = min(max(base_monthly_churn * float(churn_bias[i]), 0.0), 0.95)
        m = s + mt
        while m < T:
            if rng.random() < hazard:
                end_month[i] = int(m)
                break
            m += 1

    return end_month


def _validate_percentages(pct_india: float, pct_us: float, pct_eu: float) -> Tuple[float, float, float]:
    p = np.array([pct_india, pct_us, pct_eu], dtype="float64")
    if np.any(~np.isfinite(p)) or np.any(p < 0):
        raise ValueError("pct_india/pct_us/pct_eu must be finite and >= 0")
    s = float(p.sum())
    if s <= 0:
        raise ValueError("pct_india/pct_us/pct_eu must sum to > 0")
    p = p / s
    return float(p[0]), float(p[1]), float(p[2])

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
    # More tiers: skew toward lower tiers
    x = np.linspace(1.0, 0.35, k)
    return _normalize_probs(x)


def _assign_tier_by_score(
    *,
    score: np.ndarray,
    tier_keys_sorted_low_to_high: np.ndarray,
    tier_probs_low_to_high: np.ndarray,
) -> np.ndarray:
    """
    Assigns a tier key deterministically by mapping score quantiles to tier buckets.

    tier_probs are proportions for [lowest..highest].
    """
    k = int(len(tier_keys_sorted_low_to_high))
    if k == 0:
        raise ValueError("No tiers available")
    if k == 1:
        return np.full(score.shape[0], tier_keys_sorted_low_to_high[0], dtype=tier_keys_sorted_low_to_high.dtype)

    p = _normalize_probs(tier_probs_low_to_high)
    cut = np.cumsum(p)[:-1]  # k-1 cuts
    cuts = np.quantile(score, cut)

    idx = np.searchsorted(cuts, score, side="right")  # 0..k-1
    return tier_keys_sorted_low_to_high[idx]


def _acquisition_weights_from_names(names: np.ndarray, *, org: bool) -> np.ndarray:
    """
    Heuristic weights based on channel names.
    Works with arbitrary channel vocab as long as names contain some common substrings.
    """
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

    # Global seed: defaults.seed (fallback 42), customers.override.seed takes precedence
    default_seed = cfg.get("defaults", {}).get("seed", 42)
    override_seed = (cust_cfg.get("override") or {}).get("seed")
    seed = override_seed if override_seed is not None else default_seed
    rng = np.random.default_rng(int(seed))

    # Timeline (month index space)
    start_date, end_date = _parse_cfg_dates(cfg)
    start_month0, _end_month0, T = _month_index_space(start_date, end_date)

    # Active gate (must preserve semantics)
    active_ratio = cust_cfg.get("active_ratio", 1.0)
    if not isinstance(active_ratio, (int, float)) or not (0 < float(active_ratio) <= 1):
        raise ValueError("customers.active_ratio must be a number in the range (0, 1]")

    # Region / org mix (existing behavior)
    pct_india = float(cust_cfg["pct_india"])
    pct_us = float(cust_cfg["pct_us"])
    pct_eu = float(cust_cfg["pct_eu"])
    pct_org = float(cust_cfg["pct_org"])
    p_in, p_us, p_eu = _validate_percentages(pct_india, pct_us, pct_eu)

    # -------------------------------------------------
    # Email domain pools
    # -------------------------------------------------
    PERSONAL_EMAIL_DOMAINS = np.array(["gmail.com", "yahoo.com", "outlook.com", "hotmail.com"])

    names_folder = cust_cfg["names_folder"]

    # -----------------------------------------------------
    # Load names
    # -----------------------------------------------------
    paths = {
        "us_male": os.path.join(names_folder, "us_male_first.csv"),
        "us_female": os.path.join(names_folder, "us_female_first.csv"),
        "us_last": os.path.join(names_folder, "us_surnames.csv"),
        "in_first": os.path.join(names_folder, "india_first.csv"),
        "in_last": os.path.join(names_folder, "india_last.csv"),
        "eu_first": os.path.join(names_folder, "eu_first.csv"),
        "eu_last": os.path.join(names_folder, "eu_last.csv"),
    }

    us_male = load_list(paths["us_male"])
    us_female = load_list(paths["us_female"])
    us_last = load_list(paths["us_last"])
    in_first = load_list(paths["in_first"])
    in_last = load_list(paths["in_last"])
    eu_first = load_list(paths["eu_first"])
    eu_last = load_list(paths["eu_last"])

    # -----------------------------------------------------
    # Load Geography
    # -----------------------------------------------------
    geography, _ = load_dimension("geography", parquet_dims_folder, cfg["geography"])
    geo_keys = geography["GeographyKey"].to_numpy()

    # -----------------------------------------------------
    # Allocate arrays
    # -----------------------------------------------------
    N = int(total_customers)
    CustomerKey = np.arange(1, N + 1, dtype="int64")

    # --- Preserve global sales eligibility ---
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

    Region = rng.choice(["IN", "US", "EU"], size=N, p=[p_in, p_us, p_eu])
    IsOrg = rng.random(N) < (pct_org / 100.0)

    Gender = np.empty(N, dtype=object)
    Gender[~IsOrg] = rng.choice(["Male", "Female"], size=(~IsOrg).sum())
    Gender[IsOrg] = "Org"

    GeographyKey = rng.choice(geo_keys, size=N, replace=True)

    # -----------------------------------------------------
    # Names
    # -----------------------------------------------------
    FirstName = np.empty(N, dtype=object)
    LastName = np.empty(N, dtype=object)

    mask = (Region == "IN") & (~IsOrg)
    LastName[mask] = rng.choice(in_last, size=mask.sum())

    mask = (Region == "US") & (~IsOrg)
    LastName[mask] = rng.choice(us_last, size=mask.sum())

    mask = (Region == "EU") & (~IsOrg)
    LastName[mask] = rng.choice(eu_last, size=mask.sum())

    LastName[IsOrg] = None

    mask = (Region == "IN") & (~IsOrg)
    FirstName[mask] = rng.choice(in_first, size=mask.sum())

    mask = (Region == "US") & (~IsOrg) & (Gender == "Male")
    FirstName[mask] = rng.choice(us_male, size=mask.sum())

    mask = (Region == "US") & (~IsOrg) & (Gender == "Female")
    FirstName[mask] = rng.choice(us_female, size=mask.sum())

    mask = (Region == "EU") & (~IsOrg)
    FirstName[mask] = rng.choice(eu_first, size=mask.sum())

    FirstName[IsOrg] = None

    safe_first = np.where(FirstName is None, "", FirstName)
    safe_first = np.where(FirstName == None, "", FirstName.astype(object))
    safe_last = np.where(LastName == None, "", LastName.astype(object))

    # -----------------------------------------------------
    # Organization handling
    # -----------------------------------------------------
    company_pool = np.array(
        [
            "TechNova",
            "BrightWave",
            "ZenithSystems",
            "PrimeSource",
            "ApexCorp",
            "BluePeak",
            "NimbusWorks",
            "Evergreen Holdings",
            "Sunrise Global",
            "MetroLink",
            "Ironclad Industries",
            "Pioneer Labs",
            "CoreAxis",
            "Vertex Partners",
            "OmniTrade",
            "Silverline Group",
            "Summit Ridge",
            "Northstar Logistics",
            "QuantumBridge",
            "Cascade Ventures",
        ],
        dtype=object,
    )

    OrgName = np.empty(N, dtype=object)
    OrgName[IsOrg] = rng.choice(company_pool, size=IsOrg.sum(), replace=True)
    OrgName[~IsOrg] = None

    # -----------------------------------------------------
    # Email
    # -----------------------------------------------------
    Email = np.empty(N, dtype=object)

    personal_mask = ~IsOrg
    if personal_mask.sum():
        domain = rng.choice(PERSONAL_EMAIL_DOMAINS, size=personal_mask.sum(), replace=True)
        user = (safe_first[personal_mask].astype(str) + "." + safe_last[personal_mask].astype(str)).astype(str)
        user = np.char.lower(np.char.replace(user, " ", ""))
        Email[personal_mask] = user + "@" + domain

    OrgDomain = np.empty(N, dtype=object)
    OrgDomain[IsOrg] = np.char.lower(np.char.replace(OrgName[IsOrg].astype(str), " ", "")) + ".com"
    OrgDomain[~IsOrg] = None
    Email[IsOrg] = "info@" + OrgDomain[IsOrg]

    # -----------------------------------------------------
    # CustomerName
    # -----------------------------------------------------
    CustomerName = np.where(IsOrg, "Organization " + CustomerKey.astype(str), safe_first.astype(str) + " " + safe_last.astype(str))

    # -----------------------------------------------------
    # Demographics
    # -----------------------------------------------------
    BirthDate = np.empty(N, dtype=object)
    person_mask = ~IsOrg
    if person_mask.sum():
        ages = rng.integers(18 * 365, 70 * 365, size=person_mask.sum())
        anchor = end_date.normalize()
        dates = anchor - pd.to_timedelta(ages, unit="D")
        BirthDate[person_mask] = pd.to_datetime(dates).date
    BirthDate[IsOrg] = None

    MaritalStatus = np.empty(N, dtype=object)
    MaritalStatus[~IsOrg] = rng.choice(["Married", "Single"], size=(~IsOrg).sum(), p=[0.55, 0.45])
    MaritalStatus[IsOrg] = None

    YearlyIncome = np.where(IsOrg, None, rng.integers(20000, 200000, size=N))
    TotalChildren = pd.Series(np.where(IsOrg, pd.NA, rng.integers(0, 5, size=N)), dtype="Int64")

    Education = np.where(
        IsOrg,
        None,
        rng.choice(["High School", "Bachelors", "Masters", "PhD"], size=N, p=[0.2, 0.5, 0.25, 0.05]),
    )

    Occupation = np.where(
        IsOrg,
        None,
        rng.choice(["Professional", "Clerical", "Skilled", "Service", "Executive"], size=N, p=[0.5, 0.2, 0.15, 0.1, 0.05]),
    )

    # -----------------------------------------------------
    # Lifecycle + behavioral knobs
    # -----------------------------------------------------
    lifecycle_cfg = cust_cfg.get("lifecycle", {}) or {}

    initial_active_customers = int(lifecycle_cfg.get("initial_active_customers", 0) or 0)
    initial_spread_months = int(lifecycle_cfg.get("initial_spread_months", 0) or 0)

    # New knob (optional): force even distribution of start months
    even_start_months = bool(lifecycle_cfg.get("even_start_months", False))

    # Acquisition curve selection:
    # - if explicitly configured, honor it
    # - else if even_start_months OR initial_spread_months>0, default to uniform (even split)
    # - else keep original default linear_ramp
    acquisition_curve = lifecycle_cfg.get("acquisition_curve")
    if acquisition_curve is None:
        acquisition_curve = "uniform" if (even_start_months or initial_spread_months > 0) else "linear_ramp"

    acquisition_params = lifecycle_cfg.get("acquisition_params", {}) or {}
    weights = _acquisition_weights(T, acquisition_curve, acquisition_params)

    CustomerStartMonth = rng.choice(np.arange(T), size=N, p=weights).astype("int64")

    # -----------------------------------------------------
    # Warm-start cohort: EXISTING base, but do NOT force all to month 0
    # -----------------------------------------------------
    if initial_active_customers > 0:
        k = min(initial_active_customers, N)

        # Prefer customers that are active for sales (IsActiveInSales=1)
        active_idx = np.where(is_active == 1)[0]
        if active_idx.size > 0:
            k2 = min(k, active_idx.size)
            warm_idx = rng.choice(active_idx, size=k2, replace=False)
            if k2 < k:
                rest = np.setdiff1d(np.arange(N), warm_idx, assume_unique=False)
                extra = rng.choice(rest, size=(k - k2), replace=False)
                warm_idx = np.concatenate([warm_idx, extra])
        else:
            warm_idx = rng.choice(np.arange(N), size=k, replace=False)

        # Warm-start placement mode:
        # - If initial_spread_months > 0: spread across [0..min(initial_spread_months, T-1)]
        # - Else: month 0 (backward compatible)
        spread_hi = int(min(max(initial_spread_months, 0), max(T - 1, 0)))
        if spread_hi > 0:
            CustomerStartMonth[warm_idx] = rng.integers(0, spread_hi + 1, size=warm_idx.size, dtype=np.int64)
        else:
            CustomerStartMonth[warm_idx] = 0

    # Churn settings
    enable_churn = bool(lifecycle_cfg.get("enable_churn", False))
    base_monthly_churn = float(lifecycle_cfg.get("base_monthly_churn", 0.01))
    min_tenure_months = int(lifecycle_cfg.get("min_tenure_months", 2))

    # Behavior knobs
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

    # Load dims from parquet (generated by dimensions_runner)
    loyalty_dim = _read_parquet_dim(parquet_dims_folder, "loyalty_tiers")
    acq_dim = _read_parquet_dim(parquet_dims_folder, "customer_acquisition_channels")

    loyalty_key_col = _first_existing_col(loyalty_dim, ["LoyaltyTierKey", "TierKey", "Key"])
    loyalty_name_col = _first_existing_col(loyalty_dim, ["LoyaltyTier", "TierName", "Name"])

    acq_key_col = _first_existing_col(acq_dim, ["CustomerAcquisitionChannelKey", "AcquisitionChannelKey", "ChannelKey", "Key"])
    acq_name_col = _first_existing_col(acq_dim, ["CustomerAcquisitionChannel", "AcquisitionChannel", "ChannelName", "Name"])

    # Ensure stable ordering: lowest->highest for tiers (by key)
    loyalty_dim = loyalty_dim[[loyalty_key_col, loyalty_name_col]].dropna(subset=[loyalty_key_col]).copy()
    loyalty_dim[loyalty_key_col] = pd.to_numeric(loyalty_dim[loyalty_key_col], errors="coerce")
    loyalty_dim = loyalty_dim.dropna(subset=[loyalty_key_col]).sort_values(loyalty_key_col)

    tier_keys = loyalty_dim[loyalty_key_col].astype("int64").to_numpy()
    tier_names = loyalty_dim[loyalty_name_col].astype(str).to_numpy()

    # Loyalty score (only meaningful for individuals; org handled separately)
    income = np.zeros(N, dtype="float64")
    if (~IsOrg).any():
        # YearlyIncome is object because org has None :contentReference[oaicite:2]{index=2}
        income_ind = pd.to_numeric(pd.Series(YearlyIncome[~IsOrg]), errors="coerce").fillna(0).to_numpy(dtype="float64")
        income[~IsOrg] = (income_ind - 20000.0) / (200000.0 - 20000.0)
    income = np.clip(income, 0.0, 1.0)

    score = (0.55 * np.log1p(CustomerWeight)) + (0.30 * CustomerTemperature) + (0.15 * income)

    # Tier distribution (optional override)
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

    # Organizations skew higher (configurable)
    if IsOrg.any():
        org_mode = str(loyalty_cfg.get("org_mode", "top2")).lower()
        if org_mode == "top1":
            LoyaltyTierKey[IsOrg] = tier_keys[-1]
        elif org_mode == "top2" and len(tier_keys) >= 2:
            top2 = tier_keys[-2:]
            LoyaltyTierKey[IsOrg] = rng.choice(top2, size=int(IsOrg.sum()), p=[0.35, 0.65])
        else:
            # fallback: no special handling
            pass

    # Acquisition channel sampling (by name heuristics)
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
    # Build dataframe (preserve schema)
    # -----------------------------------------------------
    df = pd.DataFrame(
        {
            # --- legacy schema (stable) ---
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

            # --- keep for now; strip at packaging ---
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

    # Return active_customer_keys as a set for backward compatibility
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
    version_cfg["_schema_version"] = 2
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
