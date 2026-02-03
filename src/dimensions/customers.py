# ---------------------------------------------------------
#  CUSTOMERS DIMENSION (REALISTIC CONTOSO VERSION)
#  - Adds timeline-aware lifecycle fields for realistic acquisition over time
#  - Preserves customers.active_ratio as the global "eligible for sales" gate
# ---------------------------------------------------------

import os
from pathlib import Path
from typing import Dict, Tuple, Optional

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
    s = s[s.str.match(r"^[A-Za-z\-\'. ]+$")]
    return s.str.title().unique()


# ---------------------------------------------------------
# Helper: timeline month index space
# ---------------------------------------------------------
def _parse_cfg_dates(cfg: Dict) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Reads cfg['defaults']['dates']['start'/'end'].
    Accepts strings parseable by pandas.
    """
    try:
        dcfg = cfg["defaults"]["dates"]
        start = pd.to_datetime(dcfg["start"])
        end = pd.to_datetime(dcfg["end"])
    except Exception as e:
        raise ValueError(
            "Missing or invalid defaults.dates.start/end in config.yaml"
        ) from e

    if end < start:
        raise ValueError("defaults.dates.end must be >= defaults.dates.start")

    return start, end


def _month_index_space(start: pd.Timestamp, end: pd.Timestamp) -> Tuple[pd.Period, pd.Period, int]:
    """
    Converts any start/end date into month buckets inclusive.
    """
    start_m = start.to_period("M")
    end_m = end.to_period("M")
    months = int((end_m - start_m) + 1)
    if months <= 0:
        raise ValueError("Computed non-positive number of months from defaults.dates")
    return start_m, end_m, months


def _month_idx_to_date(month0: pd.Period, month_idx: np.ndarray) -> np.ndarray:
    """
    Maps month indices (0..T-1) to a date (first day of month).
    Returns numpy array of python 'date' objects.
    """
    # month0.to_timestamp() gives first day of month at 00:00
    base = month0.to_timestamp(how="start")
    ts = base + pd.to_timedelta(month_idx.astype("int64") * 30, unit="D")  # coarse
    # Replace coarse 30D with exact Period arithmetic
    # Build PeriodIndex via add of months
    p = (month0 + month_idx.astype("int64")).to_timestamp(how="start")
    return pd.to_datetime(p).date


# ---------------------------------------------------------
# Helper: Acquisition curve (StartMonth distribution)
# ---------------------------------------------------------
def _acquisition_weights(T: int, curve: str, params: Dict) -> np.ndarray:
    """
    Produces weights over months 0..T-1 for sampling CustomerStartMonth.

    curve:
      - "linear_ramp": w ~ (m+1)^shape
      - "logistic": S-curve, slower start then plateau
      - "uniform": flat

    params vary by curve; all optional.
    """
    m = np.arange(T, dtype="float64")

    curve = (curve or "linear_ramp").lower()

    if curve == "uniform":
        w = np.ones(T, dtype="float64")

    elif curve == "linear_ramp":
        shape = float(params.get("shape", 2.0))  # higher = steeper ramp toward later months
        w = (m + 1.0) ** shape

    elif curve == "logistic":
        # midpoint and steepness control
        midpoint = float(params.get("midpoint", 0.55))  # fraction of T
        steep = float(params.get("steepness", 10.0))
        x0 = midpoint * (T - 1)
        # logistic increasing, then we convert to per-month mass by differencing cumulative
        cdf = 1.0 / (1.0 + np.exp(-steep * ((m - x0) / max(T - 1, 1))))
        # convert to pmf-like weights (differences); ensure non-zero
        w = np.diff(np.r_[0.0, cdf])
        w = np.clip(w, 1e-9, None)

    else:
        raise ValueError(f"Unknown customers.lifecycle.acquisition_curve: {curve}")

    w = np.clip(w, 1e-12, None)
    return w / w.sum()


# ---------------------------------------------------------
# Helper: Optional churn end-month simulation
# ---------------------------------------------------------
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
    Returns CustomerEndMonth as Int64 array with pd.NA for "never churns inside window".
    If enable=False: returns all pd.NA.
    """
    if not enable:
        return np.full(len(start_month), pd.NA, dtype="object")

    if base_monthly_churn < 0 or base_monthly_churn > 0.5:
        raise ValueError("customers.lifecycle.base_monthly_churn must be in [0, 0.5]")

    min_tenure_months = int(max(min_tenure_months, 0))

    end_month = np.full(len(start_month), pd.NA, dtype="object")

    # Vectorizing churn with varying hazards is non-trivial; do a tight loop.
    # N is usually manageable (tens of thousands).
    for i in range(len(start_month)):
        s = int(start_month[i])
        hazard = float(base_monthly_churn) * float(churn_bias[i])
        hazard = min(max(hazard, 0.0), 0.95)

        # enforce minimum tenure
        m = s + min_tenure_months

        while m < T:
            if rng.random() < hazard:
                end_month[i] = int(m)
                break
            m += 1

    return end_month


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
    override_seed = cust_cfg.get("override", {}).get("seed")
    seed = override_seed if override_seed is not None else default_seed
    rng = np.random.default_rng(int(seed))

    # Timeline (month index space)
    start_date, end_date = _parse_cfg_dates(cfg)
    start_month0, end_month0, T = _month_index_space(start_date, end_date)

    # Active gate (must preserve semantics)
    active_ratio = cust_cfg.get("active_ratio", 1.0)
    if not isinstance(active_ratio, (int, float)) or not (0 < active_ratio <= 1):
        raise ValueError("customers.active_ratio must be a number in the range (0, 1]")

    # Region / org mix (existing behavior)
    pct_india = float(cust_cfg["pct_india"])
    pct_us = float(cust_cfg["pct_us"])
    pct_eu = float(cust_cfg["pct_eu"])
    pct_org = float(cust_cfg["pct_org"])

    # -------------------------------------------------
    # Email domain pools (simple, realistic)
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
    N = total_customers
    CustomerKey = np.arange(1, N + 1, dtype="int64")

    # --- Preserve global sales eligibility ---
    active_count = int(np.floor(N * float(active_ratio)))
    if active_count == 0:
        raise ValueError(
            "customers.active_ratio results in zero active customers; "
            "increase active_ratio or total_customers"
        )

    if active_count < N:
        active_customer_keys = rng.choice(CustomerKey, size=active_count, replace=False)
    else:
        active_customer_keys = CustomerKey

    # More efficient than converting to Python set+list for np.isin
    # Build mask by key->index
    is_active = np.zeros(N, dtype="int64")
    is_active[(active_customer_keys - 1).astype("int64")] = 1

    Region = rng.choice(
        ["IN", "US", "EU"],
        size=N,
        p=[pct_india / 100.0, pct_us / 100.0, pct_eu / 100.0],
    )

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

    safe_first = np.where(FirstName == None, "", FirstName.astype(str))
    safe_last = np.where(LastName == None, "", LastName.astype(str))

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
            "GlobalWorks",
            "VertexInnovations",
            "OmniSoft",
            "NimbusSolutions",
            "SilverlineTech",
        ]
    )

    CompanyName = np.empty(N, dtype=object)
    CompanyName[IsOrg] = company_pool[rng.integers(0, len(company_pool), size=IsOrg.sum())]
    CompanyName[~IsOrg] = None

    safe_company = np.where(CompanyName == None, "", CompanyName.astype(str))
    OrgDomain = np.where(IsOrg, np.char.lower(safe_company) + ".com", None)

    # -----------------------------------------------------
    # Emails
    # -----------------------------------------------------
    Email = np.empty(N, dtype=object)
    person_mask = ~IsOrg

    email_domain = rng.choice(PERSONAL_EMAIL_DOMAINS, size=person_mask.sum())
    Email[person_mask] = (
        np.char.lower(safe_first[person_mask])
        + "."
        + np.char.lower(safe_last[person_mask])
        + rng.integers(10, 99999, size=person_mask.sum()).astype(str)
        + "@"
        + email_domain
    )
    Email[IsOrg] = "info@" + OrgDomain[IsOrg]

    # -----------------------------------------------------
    # CustomerName
    # -----------------------------------------------------
    CustomerName = np.where(
        IsOrg,
        "Organization " + CustomerKey.astype(str),
        safe_first + " " + safe_last,
    )

    # -----------------------------------------------------
    # Demographics
    # -----------------------------------------------------
    BirthDate = np.empty(N, dtype=object)
    if person_mask.sum():
        # Use a stable "as of" date so reruns are deterministic: pick cfg end_date
        # (Previously used pd.Timestamp("today"), which changes every day.)
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
    # Lifecycle + behavioral knobs (NEW)
    # -----------------------------------------------------
    lifecycle_cfg = cust_cfg.get("lifecycle", {}) or {}

    acquisition_curve = lifecycle_cfg.get("acquisition_curve", "linear_ramp")
    acquisition_params = lifecycle_cfg.get("acquisition_params", {}) or {}

    # If you want to ensure some customers exist at month 0, allow a "launch_cohort" fraction.
    launch_frac = float(lifecycle_cfg.get("launch_cohort_fraction", 0.10))
    launch_frac = min(max(launch_frac, 0.0), 1.0)
    launch_n = int(np.floor(N * launch_frac))

    weights = _acquisition_weights(T=T, curve=acquisition_curve, params=acquisition_params)

    CustomerStartMonth = np.empty(N, dtype="int64")
    if launch_n > 0:
        # Force an initial cohort at month 0, then sample the rest from the curve
        idx = np.arange(N)
        rng.shuffle(idx)
        launch_idx = idx[:launch_n]
        rest_idx = idx[launch_n:]
        CustomerStartMonth[launch_idx] = 0
        CustomerStartMonth[rest_idx] = rng.choice(np.arange(T), size=len(rest_idx), p=weights)
    else:
        CustomerStartMonth[:] = rng.choice(np.arange(T), size=N, p=weights)

    # Behavioral columns (defaults)
    # Segments: useful for later multipliers; keep small and stable
    seg_cfg = lifecycle_cfg.get("segments", None)
    if isinstance(seg_cfg, dict) and "names" in seg_cfg and "p" in seg_cfg:
        seg_names = seg_cfg["names"]
        seg_p = seg_cfg["p"]
    else:
        seg_names = ["Value", "Core", "Budget"]
        seg_p = [0.15, 0.65, 0.20]

    CustomerSegment = rng.choice(seg_names, size=N, p=np.array(seg_p, dtype="float64") / np.sum(seg_p))

    # Base weight: long-tail with some whales (lognormal works well)
    base_mu = float(lifecycle_cfg.get("base_weight_mu", -0.1))
    base_sigma = float(lifecycle_cfg.get("base_weight_sigma", 0.9))
    CustomerBaseWeight = rng.lognormal(mean=base_mu, sigma=base_sigma, size=N).astype("float64")

    # Segment multipliers (small, plausible)
    seg_mult = {seg_names[0]: 1.8, seg_names[1]: 1.0, seg_names[2]: 0.7} if len(seg_names) >= 3 else {}
    if seg_mult:
        CustomerBaseWeight *= np.vectorize(lambda s: seg_mult.get(s, 1.0))(CustomerSegment)

    # Temperature: burstiness/volatility; keep in [0.2, 2.0] range
    CustomerTemperature = rng.lognormal(mean=-0.2, sigma=0.6, size=N).astype("float64")
    CustomerTemperature = np.clip(CustomerTemperature, 0.2, 2.5)

    # ChurnBias: hazard multiplier; most near 1.0, some higher
    CustomerChurnBias = rng.lognormal(mean=0.0, sigma=0.5, size=N).astype("float64")
    CustomerChurnBias = np.clip(CustomerChurnBias, 0.3, 4.0)

    # Optional churn end month (nullable)
    enable_churn = bool(lifecycle_cfg.get("enable_churn", False))
    base_monthly_churn = float(lifecycle_cfg.get("base_monthly_churn", 0.01))
    min_tenure_months = int(lifecycle_cfg.get("min_tenure_months", 2))

    CustomerEndMonth = _simulate_end_month(
        rng=rng,
        start_month=CustomerStartMonth,
        churn_bias=CustomerChurnBias,
        T=T,
        enable=enable_churn,
        base_monthly_churn=base_monthly_churn,
        min_tenure_months=min_tenure_months,
    )

    # Human-readable dates (useful for debugging/Power BI/SQL)
    CustomerStartDate = _month_idx_to_date(start_month0, CustomerStartMonth)

    # CustomerEndDate: nullable
    if enable_churn:
        end_idx = np.array([int(x) if x is not pd.NA else -1 for x in CustomerEndMonth], dtype="int64")
        CustomerEndDate = np.where(
            end_idx >= 0,
            _month_idx_to_date(start_month0, end_idx),
            pd.NaT,
        )
    else:
        CustomerEndDate = np.full(N, pd.NaT, dtype="datetime64[ns]")

    # -----------------------------------------------------
    # Final DataFrame
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
            "CustomerType": np.where(IsOrg, "Organization", "Person"),
            "CompanyName": CompanyName,
            "GeographyKey": GeographyKey,
            # Preserve the existing column name & semantics (global eligibility)
            "IsActiveInSales": is_active.astype("int64"),
            # NEW: lifecycle & behavior
            "CustomerStartMonth": CustomerStartMonth.astype("int64"),
            "CustomerEndMonth": pd.Series(CustomerEndMonth, dtype="Int64"),
            "CustomerStartDate": CustomerStartDate,
            "CustomerEndDate": pd.to_datetime(CustomerEndDate),
            "CustomerSegment": CustomerSegment,
            "CustomerBaseWeight": CustomerBaseWeight,
            "CustomerTemperature": CustomerTemperature,
            "CustomerChurnBias": CustomerChurnBias,
        }
    )

    # Return active_customer_keys as a set for backward compatibility with your pipeline
    # (some callers expect this)
    active_customer_set = set(active_customer_keys.tolist())
    return df, active_customer_set


# ---------------------------------------------------------
# Pipeline entry
# ---------------------------------------------------------
def run_customers(cfg: Dict, parquet_folder: Path):
    out_path = parquet_folder / "customers.parquet"

    cust_cfg = cfg["customers"]
    force = cust_cfg.get("_force_regenerate", False)

    if not force and not should_regenerate("customers", cust_cfg, out_path):
        skip("Customers up-to-date; skipping.")
        return

    with stage("Generating Customers"):
        df, _active_customer_keys = generate_synthetic_customers(cfg, parquet_folder)
        df.to_parquet(out_path, index=False)

    save_version("customers", cust_cfg, out_path)
    info(f"Customers dimension written: {out_path}")
