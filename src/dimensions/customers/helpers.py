"""Customer dimension helper/utility functions.

Includes: date parsing, month-index arithmetic, acquisition curves,
churn simulation, demographic generation (income, phone, credit,
address, lat/lon, postal codes), and loyalty-tier helpers.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.exceptions import DimensionError
from src.defaults import (
    CUSTOMER_EDUCATION_INCOME_PARAMS as EDUCATION_INCOME_PARAMS,
    CUSTOMER_OCCUPATION_INCOME_MULT as OCCUPATION_INCOME_MULT,
    CUSTOMER_INCOME_ROUND_TO as INCOME_ROUND_TO,
    CUSTOMER_INCOME_MIN as INCOME_MIN,
    CUSTOMER_INCOME_MAX as INCOME_MAX,
    CUSTOMER_INCOME_AGE_MULT as INCOME_AGE_MULT,
    CUSTOMER_INCOME_REGION_MULT as INCOME_REGION_MULT,
    CUSTOMER_INCOME_REGION_MULT_DEFAULT as INCOME_REGION_MULT_DEFAULT,
    CUSTOMER_STREET_NAMES as _STREET_NAMES,
    CUSTOMER_STREET_TYPES as _STREET_TYPES,
    CUSTOMER_POSTCODE_FMT_BY_COUNTRY as _POSTCODE_FMT_BY_COUNTRY,
    CUSTOMER_POSTCODE_FMT_DEFAULT as _POSTCODE_FMT_DEFAULT,
)


# ---------------------------------------------------------
# Helper: timeline month index space
# ---------------------------------------------------------
def parse_cfg_dates(cfg: Dict) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Resolve timeline dates from (priority order):
      1) cfg['customers']['global_dates']  (runner injected)
      2) cfg['defaults']['dates']
      3) cfg['_defaults']['dates']         (backward compatibility)

    Returns normalized pandas Timestamps (midnight).
    """
    cust = cfg.customers if hasattr(cfg, "customers") else {}
    if isinstance(cust, Mapping):
        gd = cust.global_dates if hasattr(cust, "global_dates") else None
        if isinstance(gd, Mapping) and getattr(gd, "start", None) and getattr(gd, "end", None):
            start = pd.to_datetime(gd.start).normalize()
            end = pd.to_datetime(gd.end).normalize()
            if end < start:
                raise DimensionError("defaults.dates.end must be >= defaults.dates.start")
            return start, end

    try:
        defaults = cfg.defaults if hasattr(cfg, "defaults") else getattr(cfg, "_defaults", None)
        if not isinstance(defaults, Mapping):
            raise KeyError("defaults")
        dcfg = defaults.dates
        start = pd.to_datetime(dcfg.start).normalize()
        end = pd.to_datetime(dcfg.end).normalize()
    except (KeyError, TypeError, ValueError) as e:
        raise DimensionError("Missing or invalid defaults.dates.start/end in config.yaml") from e

    if end < start:
        raise DimensionError("defaults.dates.end must be >= defaults.dates.start")

    return start, end


def month_index_space(start_date: pd.Timestamp, end_date: pd.Timestamp):
    """
    Build a month index space [0..T-1] over the inclusive month range.

    Returns:
      start_month0 : Timestamp at month start for start_date's month
      end_month0   : Timestamp at month start for end_date's month
      T            : int month count inclusive
    """
    sp = start_date.to_period("M")
    ep = end_date.to_period("M")

    if ep < sp:
        raise DimensionError("defaults.dates.end must be >= defaults.dates.start")

    T = int(ep.ordinal - sp.ordinal) + 1
    start_month0 = sp.to_timestamp(how="start")
    end_month0 = ep.to_timestamp(how="start")
    return start_month0, end_month0, T


def month_idx_to_date(month0, month_idx):
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
def acquisition_weights(T: int, curve: str, params: Dict) -> np.ndarray:
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
    w = w / wsum
    # Clamp last element so cumsum-based CDFs end at exactly 1.0
    # (guards against floating-point rounding; see CLAUDE.md gotcha #16)
    w[-1] = max(w[-1], 1.0 - w[:-1].sum())
    return w


def simulate_end_month(
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
    """
    N = len(start_month)

    if not enable:
        return np.full(N, pd.NA, dtype="object")

    if base_monthly_churn < 0:
        raise ValueError("base_monthly_churn must be >= 0")

    end_month = np.full(N, pd.NA, dtype="object")
    mt = max(int(min_tenure_months), 0)

    s = np.clip(start_month.astype("int64"), 0, T)
    hazard = np.clip(base_monthly_churn * churn_bias.astype("float64"), 0.0, 0.95)

    eligible = (s < T) & (hazard > 0.0)
    if not eligible.any():
        return end_month

    eligible_idx = np.where(eligible)[0]
    h = hazard[eligible_idx]
    tenure_samples = rng.geometric(p=h)

    churn_month = s[eligible_idx] + mt + (tenure_samples - 1)

    within = churn_month < T
    apply_idx = eligible_idx[within]
    apply_vals = churn_month[within]

    end_month[apply_idx] = apply_vals.astype(int)

    return end_month


def validate_percentages(
    pct_india: float, pct_us: float, pct_eu: float, pct_asia: float = 0.0
) -> Tuple[float, float, float, float]:
    p = np.array([pct_india, pct_us, pct_eu, pct_asia], dtype="float64")
    if np.any(~np.isfinite(p)) or np.any(p < 0):
        raise DimensionError("pct_india/pct_us/pct_eu/pct_asia must be finite and >= 0")
    s = float(p.sum())
    if s <= 0:
        raise DimensionError("pct_india/pct_us/pct_eu/pct_asia must sum to > 0")
    p = p / s
    return float(p[0]), float(p[1]), float(p[2]), float(p[3])


def read_parquet_dim(parquet_dims_folder, dim_name: str) -> pd.DataFrame:
    path = parquet_dims_folder / f"{dim_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing dimension parquet: {path}. "
            f"Run dimensions generation first (dim_name={dim_name})."
        )
    return pd.read_parquet(path)


def first_existing_col(df: pd.DataFrame, candidates) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise DimensionError(f"None of the expected columns found. candidates={candidates}, cols={list(df.columns)}")


def normalize_probs(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype="float64")
    if np.any(~np.isfinite(p)) or np.any(p < 0):
        raise ValueError("Probabilities must be finite and >= 0")
    s = float(p.sum())
    if s <= 0:
        raise ValueError("Probabilities must sum to > 0")
    return p / s


def default_tier_probs(k: int) -> np.ndarray:
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
    return normalize_probs(x)


def assign_tier_by_score(
    *,
    score: np.ndarray,
    tier_keys_sorted_low_to_high: np.ndarray,
    tier_probs_low_to_high: np.ndarray,
) -> np.ndarray:
    k = int(len(tier_keys_sorted_low_to_high))
    if k == 0:
        raise DimensionError("No tiers available")
    if k == 1:
        return np.full(score.shape[0], tier_keys_sorted_low_to_high[0], dtype=tier_keys_sorted_low_to_high.dtype)

    p = normalize_probs(tier_probs_low_to_high)
    cut = np.cumsum(p)[:-1]
    cut = np.minimum(cut, 1.0 - 1e-12)
    cuts = np.quantile(score, cut)

    idx = np.searchsorted(cuts, score, side="right")
    idx = np.clip(idx, 0, k - 1)
    return tier_keys_sorted_low_to_high[idx]


def acquisition_weights_from_names(names: np.ndarray, *, org: bool) -> np.ndarray:
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

    return normalize_probs(w)


# ---------------------------------------------------------
# Helper: correlated income generation
# ---------------------------------------------------------
def generate_correlated_income(
    rng: np.random.Generator,
    education: np.ndarray,
    occupation: np.ndarray,
    person_mask: np.ndarray,
    N: int,
    *,
    age_bracket: np.ndarray | None = None,
    region: np.ndarray | None = None,
) -> np.ndarray:
    """
    Generate YearlyIncome using lognormal distributions parameterized
    by education level, with an occupation-based multiplier.

    When provided, ``age_bracket`` (person-indexed, length = person_mask.sum())
    applies an experience/age premium and ``region`` (full length N) applies a
    regional purchasing-power factor, so income varies across age and region
    rather than depending on education+occupation alone.

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

    if age_bracket is not None:
        raw *= INCOME_AGE_MULT[age_bracket]

    if region is not None:
        reg_mult = np.full(n_p, INCOME_REGION_MULT_DEFAULT, dtype="float64")
        reg_vals = region[person_mask]
        for label, mult in INCOME_REGION_MULT.items():
            reg_mult[reg_vals == label] = mult
        raw *= reg_mult

    raw = np.clip(raw, INCOME_MIN, INCOME_MAX)
    income[person_mask] = np.round(raw / INCOME_ROUND_TO) * INCOME_ROUND_TO
    return income.astype("int64")


# ---------------------------------------------------------
# Helper: phone number generation
# ---------------------------------------------------------
def generate_phone_numbers(
    rng: np.random.Generator,
    N: int,
) -> np.ndarray:
    """Uniform 10-digit synthetic phone numbers, no country/region code.

    A consistent national-number format across all regions — the country is
    already carried by GeographyKey, so the phone column stays uniform for
    clean BI display. Stored as strings (10 digits, leading digit 2-9).
    """
    raw = rng.integers(2_000_000_000, 10_000_000_000, size=N, dtype="int64")
    return raw.astype(str).astype(object)


# ---------------------------------------------------------
# Helper: credit score generation
# ---------------------------------------------------------
def generate_credit_scores(
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
def generate_addresses(
    rng: np.random.Generator,
    customer_key: np.ndarray,
    N: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate street-line-only HomeAddress and WorkAddress strings.

    City/State/Country are not embedded — they are available via the customer's
    GeographyKey join, so the address column stays normalized (no "Tokyo, Tokyo").
    """
    _UNIT_LABELS = np.array(["Apt", "Suite", "Unit", "Fl", "#"])
    unit_num = customer_key.astype(str)

    def _street_line(unit_labels: np.ndarray) -> np.ndarray:
        street_num = rng.integers(1, 9999, size=N).astype(str)
        street_name = rng.choice(_STREET_NAMES, size=N)
        street_type = rng.choice(_STREET_TYPES, size=N)
        return (
            street_num.astype(object) + " " + street_name + " " + street_type
            + ", " + unit_labels + " " + unit_num
        )

    HomeAddress = _street_line(rng.choice(_UNIT_LABELS, size=N))
    WorkAddress = _street_line(rng.choice(_UNIT_LABELS, size=N))
    return HomeAddress, WorkAddress


# ---------------------------------------------------------
# Helper: per-country postal codes
# ---------------------------------------------------------
_PC_LETTERS = np.array(list("ABCDEFGHJKLMNPRSTUVWXYZ"))
_PC_CA_LETTERS = np.array(list("ABCEGHJKLMNPRSTVXY"))  # Canada-valid set
_PC_IE_ALNUM = np.array(list("0123456789ACDEFHKNPRTVWXY"))  # Eircode unique part


def _digits_postal(d: int):
    def gen(rng: np.random.Generator, n: int) -> np.ndarray:
        return np.char.zfill(
            rng.integers(0, 10 ** d, size=n).astype(str), d
        ).astype(object)
    return gen


def _postal_uk(rng: np.random.Generator, n: int) -> np.ndarray:
    prefix = rng.choice(np.array(["SW", "EC", "W", "SE", "N", "NW", "E", "WC"]), size=n)
    num = rng.integers(1, 19, size=n).astype(str)
    suffix = rng.integers(1, 9, size=n).astype(str)
    l1 = rng.choice(_PC_LETTERS, size=n)
    l2 = rng.choice(_PC_LETTERS, size=n)
    return (
        prefix.astype(object) + num.astype(object) + " "
        + suffix.astype(object) + l1.astype(object) + l2.astype(object)
    )


def _postal_jp(rng: np.random.Generator, n: int) -> np.ndarray:
    v = rng.integers(1000000, 9999999, size=n)
    hi = np.char.zfill((v // 10000).astype(str), 3).astype(object)
    lo = np.char.zfill((v % 10000).astype(str), 4).astype(object)
    return hi + "-" + lo


def _postal_ca(rng: np.random.Generator, n: int) -> np.ndarray:
    def L() -> np.ndarray:
        return rng.choice(_PC_CA_LETTERS, size=n).astype(object)

    def D() -> np.ndarray:
        return rng.integers(0, 10, size=n).astype(str).astype(object)

    return L() + D() + L() + " " + D() + L() + D()


def _postal_nl(rng: np.random.Generator, n: int) -> np.ndarray:
    digits4 = np.char.zfill(rng.integers(0, 10000, size=n).astype(str), 4).astype(object)
    l1 = rng.choice(_PC_LETTERS, size=n).astype(object)
    l2 = rng.choice(_PC_LETTERS, size=n).astype(object)
    return digits4 + " " + l1 + l2


def _postal_pl(rng: np.random.Generator, n: int) -> np.ndarray:
    v = rng.integers(0, 100000, size=n)
    hi = np.char.zfill((v // 1000).astype(str), 2).astype(object)
    lo = np.char.zfill((v % 1000).astype(str), 3).astype(object)
    return hi + "-" + lo


def _postal_pt(rng: np.random.Generator, n: int) -> np.ndarray:
    v = rng.integers(0, 10000000, size=n)
    hi = np.char.zfill((v // 1000).astype(str), 4).astype(object)
    lo = np.char.zfill((v % 1000).astype(str), 3).astype(object)
    return hi + "-" + lo


def _postal_se(rng: np.random.Generator, n: int) -> np.ndarray:
    v = rng.integers(10000, 100000, size=n)
    hi = np.char.zfill((v // 100).astype(str), 3).astype(object)
    lo = np.char.zfill((v % 100).astype(str), 2).astype(object)
    return hi + " " + lo


def _postal_ie(rng: np.random.Generator, n: int) -> np.ndarray:
    rk = (
        rng.choice(_PC_LETTERS, size=n).astype(object)
        + np.char.zfill(rng.integers(0, 100, size=n).astype(str), 2).astype(object)
    )
    uniq = (
        rng.choice(_PC_IE_ALNUM, size=n).astype(object)
        + rng.choice(_PC_IE_ALNUM, size=n).astype(object)
        + rng.choice(_PC_IE_ALNUM, size=n).astype(object)
        + rng.choice(_PC_IE_ALNUM, size=n).astype(object)
    )
    return rk + " " + uniq


# Registry: format key -> vectorized generator fn(rng, n) -> object array.
_POSTAL_GENERATORS = {
    4: _digits_postal(4),
    5: _digits_postal(5),
    6: _digits_postal(6),
    "uk": _postal_uk,
    "jp": _postal_jp,
    "ca": _postal_ca,
    "nl": _postal_nl,
    "pl": _postal_pl,
    "pt": _postal_pt,
    "se": _postal_se,
    "ie": _postal_ie,
}


def generate_postal_codes(
    rng: np.random.Generator,
    country: np.ndarray,
    N: int,
) -> np.ndarray:
    """Per-country synthetic postal codes, keyed off the customer's Country.

    The format matches the real country (UK alphanumeric, Japan NNN-NNNN, etc.);
    unmapped countries fall back to a numeric default. Vectorized by format group.
    """
    codes = np.empty(N, dtype=object)
    specs = np.array(
        [_POSTCODE_FMT_BY_COUNTRY.get(c, _POSTCODE_FMT_DEFAULT) for c in country],
        dtype=object,
    )
    for fmt, gen in _POSTAL_GENERATORS.items():
        mask = specs == fmt
        n = int(mask.sum())
        if n:
            codes[mask] = gen(rng, n)
    remaining = pd.isna(codes) | (codes == None)  # noqa: E711
    n_rem = int(remaining.sum())
    if n_rem:
        codes[remaining] = _POSTAL_GENERATORS[_POSTCODE_FMT_DEFAULT](rng, n_rem)
    return codes


def postal_code_for_country(rng: np.random.Generator, country: str) -> str:
    """Single postal code for a given country (scalar wrapper for SCD2 relocation)."""
    fmt = _POSTCODE_FMT_BY_COUNTRY.get(country, _POSTCODE_FMT_DEFAULT)
    gen = _POSTAL_GENERATORS.get(fmt, _POSTAL_GENERATORS[_POSTCODE_FMT_DEFAULT])
    return str(gen(rng, 1)[0])
