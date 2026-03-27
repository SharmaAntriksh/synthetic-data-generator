"""OrganizationProfile dimension generation for org-type customers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.exceptions import ValidationError
from src.utils.name_pools import PeopleNamePools, slugify_domain_label
from src.defaults import (
    CUSTOMER_STREET_NAMES as _STREET_NAMES,
    CUSTOMER_STREET_TYPES as _STREET_TYPES,
)

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

# Import-time validation for local probability arrays
for _pname, _parr in [
    ("_ORG_INDUSTRY_PROBS", _ORG_INDUSTRY_PROBS),
    ("_ORG_SIZE_PROBS", _ORG_SIZE_PROBS),
]:
    if abs(float(_parr.sum()) - 1.0) > 1e-6:
        raise ValidationError(f"org_profile.{_pname} sums to {float(_parr.sum())}, expected 1.0")
del _pname, _parr

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


def generate_org_profile(
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
    # Vectorised hq_city / hq_state: iterate 8 countries not M orgs
    _hq_countries_list = list(geo_lookup_hq.keys())
    _hq_cities_split = [geo_lookup_hq[c][0].split(",") for c in _hq_countries_list]
    _hq_states_split = [geo_lookup_hq[c][1].split(",") for c in _hq_countries_list]
    _hq_country_to_idx = {c: i for i, c in enumerate(_hq_countries_list)}
    _hq_mapped = np.array([_hq_country_to_idx.get(str(c), -1) for c in HeadquarterCountry])
    _hq_known_mask = _hq_mapped >= 0
    _hq_n_known = int(_hq_known_mask.sum())

    # Batch RNG draw for known-country orgs (all have 5 options)
    _hq_rand_all = rng.integers(0, 5, size=_hq_n_known)

    hq_city = np.empty(M, dtype=object)
    hq_state = np.empty(M, dtype=object)
    hq_city[:] = "New York"
    hq_state[:] = "NY"

    # Map batch RNG draws back to positions via cumulative indexing
    _hq_known_positions = np.flatnonzero(_hq_known_mask)
    _hq_rand_by_pos = np.empty(M, dtype=np.intp)
    _hq_rand_by_pos[_hq_known_positions] = _hq_rand_all

    for ci in range(len(_hq_countries_list)):
        mask = _hq_mapped == ci
        if not mask.any():
            continue
        n_opts = len(_hq_cities_split[ci])
        idxs = _hq_rand_by_pos[mask] % n_opts
        hq_city[mask] = np.array(_hq_cities_split[ci], dtype=object)[idxs]
        hq_state[mask] = np.array(_hq_states_split[ci], dtype=object)[idxs]

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
        "FoundedYear": pd.array(FoundedYear, dtype="Int32"),
        "IsPubliclyTraded": IsPubliclyTraded,
        "HeadquarterCountry": HeadquarterCountry,
        "Website": Website,
        "OrgDomain": OrgDomain,
        "OrgAddress": OrgAddress,
        "PrimaryContactName": PrimaryContactName,
        "PrimaryContactRole": PrimaryContactRole,
        "NumberOfEmployees": pd.array(NumberOfEmployees, dtype="Int32"),
        "NumberOfLocations": pd.array(NumberOfLocations, dtype="Int32"),
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
