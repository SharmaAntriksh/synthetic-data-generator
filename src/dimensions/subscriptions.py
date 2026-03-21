from __future__ import annotations

"""
Subscriptions (DimPlans + CustomerSubscriptions billing-period fact)

Writes:
  - plans.parquet                   (subscription plan dimension)
  - customer_subscriptions.parquet  (billing-period subscription fact)

The fact table has one row per subscription per billing period, enabling
revenue tracking, churn analysis, and cohort retention.

Written in parallel for large datasets (>200K eligible customers)
using a chunk-per-worker pattern.  Each worker generates period rows
IN MEMORY for its customer slice, writes a chunk parquet, and returns
its row count.  The main process merges chunks into the final parquet.

For small datasets (<=200K eligible customers) the original single-process
streaming writer is used as a fallback.

Schema:
  DimPlans (15 cols):
    PlanKey, PlanName, PlanType, Category, BillingCycle, CycleMonths,
    BaseMonthlyPrice, Discount, CyclePrice, AnnualPrice, Tier, MaxUsers,
    HasFreeTrial, LaunchDate, IsActiveFlag

  CustomerSubscriptions (10 cols, billing-period fact):
    SubscriptionKey, CustomerKey, PlanKey,
    PeriodStartDate, PeriodEndDate, PeriodPrice,
    IsFirstPeriod, IsChurnPeriod, IsTrialPeriod, BillingCycleNumber

Config (optional)
subscriptions:
  enabled: true
  generate_bridge: true
  participation_rate: 0.65
  avg_subscriptions_per_customer: 1.5
  max_subscriptions: 5
  churn_rate: 0.25
  trial_rate: 0.30
  seed: 700
  write_chunk_rows: 250000

Requires:
  defaults.dates.start/end in cfg.

Power BI:
  Customers (1) ──< CustomerSubscriptions >── (1) DimPlans
"""

import calendar
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.config_precedence import resolve_seed
from src.utils.logging_utils import info, skip, stage
from src.versioning.version_store import should_regenerate, save_version
from src.defaults import SUBSCRIPTION_PARALLEL_THRESHOLD


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

# Billing-cycle discount rates (off monthly price)
_CYCLE_DISCOUNT = {
    "Monthly":     0.00,
    "Quarterly":   0.05,
    "Half-Yearly": 0.10,
    "Annual":      0.17,
}
_CYCLE_MONTHS = {
    "Monthly": 1, "Quarterly": 3, "Half-Yearly": 6, "Annual": 12,
}

# Category mapping: PlanType -> higher-level Category for analytics
_CATEGORY_MAP = {
    "Streaming":     "Entertainment",
    "Gaming":        "Entertainment",
    "Fitness":       "Health",
    "Cloud Storage": "Productivity",
    "Education":     "Productivity",
    "News & Media":  "Information",
    "Music":         "Entertainment",
    "Productivity":  "Productivity",
}

# Base plan definitions: (PlanName, PlanType, BaseMonthlyPrice, Tier, MaxUsers, HasFreeTrial, LaunchDayOffset)
# LaunchDayOffset = days after global start date when the plan launches.
# Each base plan is a distinct product — customers pick across types, not within.
_BASE_PLANS: List[Tuple[str, str, float, str, int, int, int]] = [
    # Streaming — Netflix
    ("Netflix",                 "Streaming",      15.49, "Standard", 2, 1,   0),
    ("Netflix Premium",         "Streaming",      22.99, "Premium",  4, 1,   0),
    # Music — Spotify
    ("Spotify",                 "Music",          10.99, "Standard", 1, 1,  30),
    ("Spotify Family",          "Music",          16.99, "Premium",  6, 0, 120),
    # Cloud storage — Dropbox
    ("Dropbox Plus",            "Cloud Storage",  11.99, "Standard", 1, 1,   0),
    ("Dropbox Business",        "Cloud Storage",  20.00, "Premium",  5, 0,  90),
    # Fitness — Peloton
    ("Peloton",                 "Fitness",        12.99, "Standard", 1, 1,  15),
    ("Peloton All-Access",      "Fitness",        44.00, "Premium",  2, 1, 300),
    # Gaming — Xbox Game Pass
    ("Xbox Game Pass",          "Gaming",         10.99, "Standard", 1, 1,  60),
    ("Xbox Game Pass Ultimate", "Gaming",         19.99, "Premium",  1, 1, 365),
    # News & media — NYT
    ("NYT Digital",             "News & Media",    5.00, "Basic",    1, 1,   0),
    ("NYT All Access",          "News & Media",   12.50, "Premium",  5, 0, 180),
    # Education — Coursera / LinkedIn Learning
    ("Coursera Plus",           "Education",      59.00, "Standard", 1, 1, 150),
    ("LinkedIn Learning",       "Education",      29.99, "Premium", 10, 0, 365),
    # Productivity — Microsoft 365
    ("Microsoft 365",           "Productivity",    6.99, "Standard", 1, 0,  45),
    ("Microsoft 365 Business",  "Productivity",   12.50, "Premium", 25, 0, 210),
]

# Which billing cycles each base plan supports
_PLAN_CYCLES: Dict[str, List[str]] = {
    "Netflix":                 ["Monthly", "Annual"],
    "Netflix Premium":         ["Monthly", "Quarterly", "Annual"],
    "Spotify":                 ["Monthly", "Quarterly"],
    "Spotify Family":          ["Monthly", "Annual"],
    "Dropbox Plus":            ["Monthly"],
    "Dropbox Business":        ["Monthly", "Quarterly", "Annual"],
    "Peloton":                 ["Monthly", "Quarterly"],
    "Peloton All-Access":      ["Monthly", "Half-Yearly", "Annual"],
    "Xbox Game Pass":          ["Monthly", "Quarterly"],
    "Xbox Game Pass Ultimate": ["Monthly", "Annual"],
    "NYT Digital":             ["Monthly"],
    "NYT All Access":          ["Monthly", "Annual"],
    "Coursera Plus":           ["Monthly", "Half-Yearly"],
    "LinkedIn Learning":       ["Annual"],
    "Microsoft 365":           ["Monthly", "Annual"],
    "Microsoft 365 Business":  ["Monthly", "Quarterly", "Annual"],
}


def _expand_catalog() -> List[Tuple]:
    """
    Expand base plans × billing cycles into the full catalog.
    Returns list of:
      (PlanName, PlanType, Category, BillingCycle, CycleMonths,
       BaseMonthlyPrice, Discount, CyclePrice, AnnualPrice,
       Tier, MaxUsers, HasFreeTrial, LaunchDayOffset)
    """
    rows = []
    for name, ptype, mprice, tier, maxu, trial, launch in _BASE_PLANS:
        category = _CATEGORY_MAP.get(ptype, ptype)
        for cycle in _PLAN_CYCLES[name]:
            discount = _CYCLE_DISCOUNT[cycle]
            months = _CYCLE_MONTHS[cycle]
            cycle_price = round(mprice * months * (1 - discount), 2)
            annual_price = round(mprice * 12 * (1 - discount), 2)
            rows.append((
                name, ptype, category, cycle, months, mprice, discount,
                cycle_price, annual_price, tier, maxu, trial, launch,
            ))
    return rows


PLANS_CATALOG = _expand_catalog()

_PLAN_TYPE_WEIGHT = {
    "Streaming": 4.0,
    "Music": 3.5,
    "Cloud Storage": 3.0,
    "Gaming": 2.5,
    "News & Media": 2.0,
    "Fitness": 1.5,
    "Education": 1.5,
    "Productivity": 2.5,
}

PAYMENT_METHODS = ["Credit Card", "Debit Card", "PayPal", "Bank Transfer"]
_PAYMENT_WEIGHTS = np.array([0.45, 0.25, 0.20, 0.10])
if abs(_PAYMENT_WEIGHTS.sum() - 1.0) > 1e-9:
    raise ValueError(
        f"_PAYMENT_WEIGHTS must sum to 1.0, got {_PAYMENT_WEIGHTS.sum()}"
    )

CANCELLATION_REASONS = [
    "Too Expensive", "Not Using", "Switched Competitor",
    "Missing Features", "Poor Service", "Other",
]

_NS_PER_DAY: int = 86_400_000_000_000


def _ns_to_year_month(ns: int) -> Tuple[int, int]:
    """Convert nanosecond timestamp to (year, month) tuple."""
    dt = pd.Timestamp(ns, unit="ns")
    return dt.year, dt.month


def _months_between(y1: int, m1: int, y2: int, m2: int) -> int:
    """Number of months from (y1,m1) to (y2,m2) inclusive."""
    return (y2 - y1) * 12 + (m2 - m1) + 1


def _month_start_date(year: int, month: int) -> date:
    """Return date object for the 1st of the given month."""
    return date(year, month, 1)


def _month_end_date(year: int, month: int) -> date:
    """Return date object for the last day of the given month."""
    return date(year, month, calendar.monthrange(year, month)[1])


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubscriptionsCfg:
    enabled: bool = True
    generate_bridge: bool = True
    participation_rate: float = 0.65
    avg_subscriptions: float = 1.5
    max_subscriptions: int = 5
    churn_rate: float = 0.25
    trial_rate: float = 0.30
    trial_conversion_rate: float = 0.85
    seed: int = 700
    write_chunk_rows: int = 250_000


def _read_cfg(cfg: Any) -> SubscriptionsCfg:
    sc = cfg.subscriptions
    seed = resolve_seed(cfg, sc, fallback=700)
    return SubscriptionsCfg(
        enabled=bool(sc.enabled),
        generate_bridge=bool(sc.generate_bridge),
        participation_rate=float(sc.participation_rate),
        avg_subscriptions=float(sc.avg_subscriptions_per_customer),
        max_subscriptions=int(sc.max_subscriptions),
        churn_rate=float(sc.churn_rate),
        trial_rate=float(sc.trial_rate),
        trial_conversion_rate=float(sc.trial_conversion_rate),
        seed=seed,
        write_chunk_rows=int(sc.write_chunk_rows),
    )


def _parse_global_dates(cfg: Any) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Resolve timeline dates from (priority order):
      1) cfg.subscriptions.global_dates  (runner injected)
      2) cfg.defaults.dates
      3) cfg._defaults.dates             (backward compatibility)
    """
    sc = cfg.subscriptions
    gd = sc.global_dates if sc is not None else None
    if isinstance(gd, Mapping) and gd.get("start") and gd.get("end"):
        start = pd.to_datetime(gd["start"]).normalize()
        end = pd.to_datetime(gd["end"]).normalize()
        if end < start:
            raise ValueError("defaults.dates.end must be >= defaults.dates.start")
        return start, end

    defaults = cfg.defaults if hasattr(cfg, "defaults") else getattr(cfg, "_defaults", None)
    if defaults is None:
        raise ValueError("Missing defaults.dates.start/end (or _defaults.dates.start/end)")
    d = defaults.dates
    d_start = d.start if hasattr(d, "start") else None
    d_end = d.end if hasattr(d, "end") else None
    if not d_start or not d_end:
        raise ValueError("Missing defaults.dates.start/end (or _defaults.dates.start/end)")
    start = pd.to_datetime(d_start).normalize()
    end = pd.to_datetime(d_end).normalize()
    if end < start:
        raise ValueError("defaults.dates.end must be >= defaults.dates.start")
    return start, end


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_dim_plans(g_start: pd.Timestamp) -> pd.DataFrame:
    """Build the subscription plans dimension table (15 columns).

    LaunchDate for each plan is computed as g_start + LaunchDayOffset.
    """
    k = len(PLANS_CATALOG)
    launch_dates = pd.to_datetime([
        g_start + pd.Timedelta(days=int(r[12])) for r in PLANS_CATALOG
    ])
    return pd.DataFrame({
        "PlanKey":          np.arange(1, k + 1, dtype=np.int64),
        "PlanName":         [r[0] for r in PLANS_CATALOG],
        "PlanType":         [r[1] for r in PLANS_CATALOG],
        "Category":         [r[2] for r in PLANS_CATALOG],
        "BillingCycle":     [r[3] for r in PLANS_CATALOG],
        "CycleMonths":      np.array([r[4] for r in PLANS_CATALOG], dtype=np.int32),
        "BaseMonthlyPrice": pd.array([r[5] for r in PLANS_CATALOG], dtype="Float64"),
        "Discount":         pd.array([r[6] for r in PLANS_CATALOG], dtype="Float64"),
        "CyclePrice":       pd.array([r[7] for r in PLANS_CATALOG], dtype="Float64"),
        "AnnualPrice":      pd.array([r[8] for r in PLANS_CATALOG], dtype="Float64"),
        "Tier":             [r[9] for r in PLANS_CATALOG],
        "MaxUsers":         np.array([r[10] for r in PLANS_CATALOG], dtype=np.int32),
        "HasFreeTrial":     np.array([r[11] for r in PLANS_CATALOG], dtype=np.int8),
        "LaunchDate":       launch_dates,
        "IsActiveFlag":     np.ones(k, dtype=np.int8),
    })


def _build_type_groups(plan_types: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """Pre-compute per-type plan indices and type-level weights.

    Returns:
        unique_types  – array of distinct PlanType strings
        type_members  – list of arrays, each holding plan indices for that type
        type_weights  – normalised probability for picking each type
    """
    unique_types = np.unique(plan_types)
    type_members: List[np.ndarray] = []
    type_weights = np.empty(len(unique_types), dtype=np.float64)
    for i, t in enumerate(unique_types):
        members = np.where(plan_types == t)[0]
        type_members.append(members)
        type_weights[i] = _PLAN_TYPE_WEIGHT.get(t, 1.0)
    type_weights /= type_weights.sum()
    return unique_types, type_members, type_weights


def _choose_plans_diverse(
    rng: np.random.Generator,
    n_subs: int,
    unique_types: np.ndarray,
    type_members: List[np.ndarray],
    type_weights: np.ndarray,
) -> np.ndarray:
    """Pick n_subs plans from distinct PlanTypes.

    First selects n_subs unique types (weighted), then picks one plan
    uniformly from each selected type.
    """
    n_types = len(unique_types)
    n_subs = min(n_subs, n_types)
    chosen_type_idx = rng.choice(n_types, size=n_subs, replace=False, p=type_weights)
    plan_idx = np.empty(n_subs, dtype=np.intp)
    for i, ti in enumerate(chosen_type_idx):
        members = type_members[ti]
        plan_idx[i] = members[rng.integers(len(members))]
    return plan_idx


def _compute_customer_windows(
    customers: pd.DataFrame,
    g_start: pd.Timestamp,
    g_end: pd.Timestamp,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract sorted (CustomerKey, start_ns, end_ns) arrays from the customers
    DataFrame.  All dates are clamped to [g_start, g_end].
    """
    cust_keys = customers["CustomerKey"].astype(np.int64).to_numpy()
    order = np.argsort(cust_keys)
    cust_keys = cust_keys[order]

    g_start_ns = np.int64(g_start.value)
    g_end_ns = np.int64(g_end.value)

    has_start = "CustomerStartDate" in customers.columns
    has_end = "CustomerEndDate" in customers.columns

    if has_start:
        start_vals = customers["CustomerStartDate"].to_numpy().astype("datetime64[ns]")
        start_ns = start_vals.view(np.int64).copy()
        start_ns[np.isnat(start_vals)] = g_start_ns
        start_ns = np.clip(start_ns, g_start_ns, g_end_ns)
    else:
        start_ns = np.full(len(cust_keys), g_start_ns, dtype=np.int64)

    if has_end:
        end_vals = customers["CustomerEndDate"].to_numpy().astype("datetime64[ns]")
        end_ns = end_vals.view(np.int64).copy()
        end_ns[np.isnat(end_vals)] = g_end_ns
        end_ns = np.clip(end_ns, g_start_ns, g_end_ns)
    else:
        end_ns = np.full(len(cust_keys), g_end_ns, dtype=np.int64)

    start_ns = start_ns[order]
    end_ns = end_ns[order]
    end_ns = np.maximum(end_ns, start_ns)

    return cust_keys, start_ns, end_ns


def _bridge_schema() -> pa.Schema:
    return pa.schema([
        pa.field("SubscriptionKey", pa.int64()),
        pa.field("CustomerKey", pa.int64()),
        pa.field("PlanKey", pa.int32()),
        pa.field("PeriodStartDate", pa.date32()),
        pa.field("PeriodEndDate", pa.date32()),
        pa.field("PeriodPrice", pa.float64()),
        pa.field("IsFirstPeriod", pa.int8()),
        pa.field("IsChurnPeriod", pa.int8()),
        pa.field("IsTrialPeriod", pa.int8()),
        pa.field("BillingCycleNumber", pa.int32()),
    ])


# ---------------------------------------------------------------------------
# Bridge writer (streaming) -- serial path for small datasets
# ---------------------------------------------------------------------------

def _advance_months(y: int, m: int, n: int) -> Tuple[int, int]:
    """Advance (year, month) by n months."""
    m += n
    while m > 12:
        m -= 12
        y += 1
    return y, m


def _expand_subscription_periods(
    sub_key: int,
    ck: int,
    pk: int,
    sub_ns: int,
    cancel_ns: Optional[int],
    trial_end_ns: Optional[int],
    cycle_months: int,
    cycle_price: float,
    g_end_ns: int,
) -> Tuple[
    List[int], List[int], List[int],
    List[date], List[date], List[float],
    List[int], List[int], List[int], List[int],
]:
    """
    Expand a single subscription into billing-period rows.

    One row per billing cycle (monthly plans → 1 month, quarterly → 3 months, etc.).
    Returns parallel lists for each column (to be appended to buffers).
    """
    sub_y, sub_m = _ns_to_year_month(sub_ns)
    end_ref = cancel_ns if cancel_ns is not None else g_end_ns
    end_y, end_m = _ns_to_year_month(end_ref)
    n_months = _months_between(sub_y, sub_m, end_y, end_m)
    if n_months <= 0:
        n_months = 1

    sk_list: List[int] = []
    ck_list: List[int] = []
    pk_list: List[int] = []
    ps_list: List[date] = []       # PeriodStartDate
    pe_list: List[date] = []       # PeriodEndDate
    price_list: List[float] = []   # PeriodPrice
    first_list: List[int] = []
    churn_list: List[int] = []
    trial_list: List[int] = []
    cycle_list: List[int] = []     # BillingCycleNumber

    y, m = sub_y, sub_m
    period_idx = 0
    month_offset = 0
    while month_offset < n_months:
        # Period end: advance by cycle_months (clamped to subscription end)
        end_y_p, end_m_p = _advance_months(y, m, cycle_months - 1)

        is_first = 1 if period_idx == 0 else 0
        remaining = n_months - month_offset
        is_last_period = remaining <= cycle_months
        is_churn = 1 if (is_last_period and cancel_ns is not None) else 0
        is_trial = 1 if (trial_end_ns is not None and period_idx == 0) else 0

        period_price = 0.0 if is_trial else cycle_price

        sk_list.append(sub_key)
        ck_list.append(ck)
        pk_list.append(pk)
        ps_list.append(_month_start_date(y, m))
        pe_list.append(_month_end_date(end_y_p, end_m_p))
        price_list.append(period_price)
        first_list.append(is_first)
        churn_list.append(is_churn)
        trial_list.append(is_trial)
        cycle_list.append(period_idx + 1)

        # Advance by cycle_months
        y, m = _advance_months(y, m, cycle_months)
        month_offset += cycle_months
        period_idx += 1

    return (
        sk_list, ck_list, pk_list,
        ps_list, pe_list, price_list,
        first_list, churn_list, trial_list, cycle_list,
    )


def _write_bridge_streaming(
    customers: pd.DataFrame,
    dim_plans: pd.DataFrame,
    c: SubscriptionsCfg,
    g_start: pd.Timestamp,
    g_end: pd.Timestamp,
    out_bridge: Path,
) -> int:
    """
    Stream-write customer_subscriptions billing-period fact to parquet.
    Returns number of rows written.
    """
    if "CustomerKey" not in customers.columns:
        raise KeyError("customers must include CustomerKey")

    plan_keys = dim_plans["PlanKey"].astype(np.int32).to_numpy()
    plan_types = dim_plans["PlanType"].astype(str).to_numpy()
    plan_cycle_prices = dim_plans["CyclePrice"].to_numpy(dtype=np.float64, na_value=0.0)
    plan_cycle_months = dim_plans["CycleMonths"].astype(np.int32).to_numpy()
    unique_types, type_members, type_weights = _build_type_groups(plan_types)

    cust_keys, cust_start_ns, cust_end_ns = _compute_customer_windows(
        customers, g_start, g_end,
    )

    g_end_ns = np.int64(g_end.value)
    write_chunk_rows = max(c.write_chunk_rows, 10_000)
    max_subs = min(c.max_subscriptions, len(unique_types))

    rng = np.random.default_rng(c.seed)
    schema = _bridge_schema()

    n_cust = len(cust_keys)
    participate_mask = rng.random(n_cust) < c.participation_rate

    _part_idx = np.where(participate_mask)[0]
    _spans = np.maximum(0, (cust_end_ns[_part_idx] - cust_start_ns[_part_idx]) // _NS_PER_DAY)
    _ok = _spans >= 30
    eligible_idx = _part_idx[_ok]
    eligible_ck = cust_keys[eligible_idx]
    eligible_lo = cust_start_ns[eligible_idx]
    eligible_hi = cust_end_ns[eligible_idx]
    eligible_span = _spans[_ok]
    n_eligible = len(eligible_idx)

    # Accumulators for billing-period rows
    all_sk: List[int] = []
    all_ck: List[int] = []
    all_pk: List[int] = []
    all_ps: List[date] = []
    all_pe: List[date] = []
    all_price: List[float] = []
    all_first: List[int] = []
    all_churn: List[int] = []
    all_trial: List[int] = []
    all_cycle: List[int] = []

    total_rows = 0
    next_sub_key = 1

    def flush(writer: pq.ParquetWriter) -> None:
        nonlocal total_rows
        nonlocal all_sk, all_ck, all_pk
        nonlocal all_ps, all_pe, all_price
        nonlocal all_first, all_churn, all_trial, all_cycle
        n = len(all_sk)
        if n == 0:
            return

        arrays: List[pa.Array] = [
            pa.array(all_sk, type=pa.int64()),
            pa.array(all_ck, type=pa.int64()),
            pa.array(all_pk, type=pa.int32()),
            pa.array(all_ps, type=pa.date32()),
            pa.array(all_pe, type=pa.date32()),
            pa.array(all_price, type=pa.float64()),
            pa.array(all_first, type=pa.int8()),
            pa.array(all_churn, type=pa.int8()),
            pa.array(all_trial, type=pa.int8()),
            pa.array(all_cycle, type=pa.int32()),
        ]

        table = pa.Table.from_arrays(arrays, schema=schema)
        writer.write_table(table)
        total_rows += n

        # Reset accumulators
        all_sk = []
        all_ck = []
        all_pk = []
        all_ps = []
        all_pe = []
        all_price = []
        all_first = []
        all_churn = []
        all_trial = []
        all_cycle = []

    out_bridge.parent.mkdir(parents=True, exist_ok=True)
    if out_bridge.exists():
        out_bridge.unlink()

    with pq.ParquetWriter(out_bridge, schema=schema, compression="snappy") as writer:
        for i in range(n_eligible):
            ck = int(eligible_ck[i])
            lo_ns = int(eligible_lo[i])
            hi_ns = int(eligible_hi[i])
            span_days = int(eligible_span[i])

            n_subs = max(1, int(rng.poisson(c.avg_subscriptions)))
            n_subs = min(n_subs, max_subs)

            chosen_idx = _choose_plans_diverse(
                rng, n_subs, unique_types, type_members, type_weights,
            )
            n_subs = len(chosen_idx)  # may be clamped to n_types
            sub_offsets = np.sort(rng.integers(0, max(span_days - 30, 1), size=n_subs))

            for s in range(n_subs):
                pidx = chosen_idx[s]
                pk = int(plan_keys[pidx])

                sub_ns = lo_ns + int(sub_offsets[s]) * _NS_PER_DAY

                cycle_months = int(plan_cycle_months[pidx])
                n_periods = max(1, int(rng.geometric(0.3)))
                base_duration_days = cycle_months * 30 * n_periods

                cprice = float(plan_cycle_prices[pidx])

                has_trial = rng.random() < c.trial_rate
                if has_trial:
                    trial_end_ns = sub_ns + 14 * _NS_PER_DAY
                    converts = rng.random() < c.trial_conversion_rate
                else:
                    trial_end_ns = None
                    converts = True

                is_churned = rng.random() < c.churn_rate
                end_ns = sub_ns + int(base_duration_days) * _NS_PER_DAY

                if not converts:
                    cancel_ns = sub_ns
                elif is_churned and end_ns <= hi_ns:
                    cancel_ns = end_ns
                else:
                    cancel_ns = None

                # RNG consumption for payment method (maintain stream compatibility)
                rng.choice(len(PAYMENT_METHODS), p=_PAYMENT_WEIGHTS)

                # Expand into billing-period rows
                (
                    sk, ck_l, pk_l, ps, pe, pr,
                    first, churn, trial, cyc,
                ) = _expand_subscription_periods(
                    sub_key=next_sub_key,
                    ck=ck,
                    pk=pk,
                    sub_ns=sub_ns,
                    cancel_ns=cancel_ns,
                    trial_end_ns=trial_end_ns,
                    cycle_months=cycle_months,
                    cycle_price=cprice,
                    g_end_ns=int(g_end_ns),
                )

                all_sk.extend(sk)
                all_ck.extend(ck_l)
                all_pk.extend(pk_l)
                all_ps.extend(ps)
                all_pe.extend(pe)
                all_price.extend(pr)
                all_first.extend(first)
                all_churn.extend(churn)
                all_trial.extend(trial)
                all_cycle.extend(cyc)

                next_sub_key += 1

                if len(all_sk) >= write_chunk_rows:
                    flush(writer)

        flush(writer)

    return total_rows


# ---------------------------------------------------------------------------
# Parallel bridge writer -- top-level worker (must be importable for spawn)
# ---------------------------------------------------------------------------

def _subscription_worker_task(args: Tuple) -> Dict[str, Any]:
    """
    Worker entry point (must be top-level for Windows spawn pickling).

    Generates billing-period subscription fact rows IN MEMORY for a slice
    of eligible customers, writes a chunk parquet to disk, and returns its
    row count.

    SubscriptionKey values written here are LOCAL (1-based within chunk).
    """
    (
        chunk_idx, seed, n_chunks,
        eligible_ck, eligible_lo, eligible_hi, eligible_span,
        plan_keys, plan_cycle_prices, plan_cycle_months,
        unique_types, type_members, type_weights,
        g_end_ns, max_subs,
        avg_subscriptions, churn_rate, trial_rate, trial_conversion_rate,
        payment_weights,
        out_chunk_path,
    ) = args

    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(n_chunks)
    rng = np.random.default_rng(child_seeds[chunk_idx])

    n_eligible = len(eligible_ck)
    schema = _bridge_schema()

    # Accumulators for billing-period rows
    all_sk: List[int] = []
    all_ck: List[int] = []
    all_pk: List[int] = []
    all_ps: List[date] = []
    all_pe: List[date] = []
    all_price: List[float] = []
    all_first: List[int] = []
    all_churn: List[int] = []
    all_trial: List[int] = []
    all_cycle: List[int] = []

    local_sub_key = 1

    for i in range(n_eligible):
        ck = int(eligible_ck[i])
        lo_ns = int(eligible_lo[i])
        hi_ns = int(eligible_hi[i])
        span_days = int(eligible_span[i])

        n_subs = max(1, int(rng.poisson(avg_subscriptions)))
        n_subs = min(n_subs, max_subs)

        chosen_idx = _choose_plans_diverse(
            rng, n_subs, unique_types, type_members, type_weights,
        )
        n_subs = len(chosen_idx)
        sub_offsets = np.sort(rng.integers(0, max(span_days - 30, 1), size=n_subs))

        for s in range(n_subs):
            pidx = int(chosen_idx[s])
            pk = int(plan_keys[pidx])

            sub_ns = lo_ns + int(sub_offsets[s]) * _NS_PER_DAY

            cycle_months = int(plan_cycle_months[pidx])
            n_periods = max(1, int(rng.geometric(0.3)))
            base_duration_days = cycle_months * 30 * n_periods

            cprice = float(plan_cycle_prices[pidx])

            has_trial = rng.random() < trial_rate
            trial_end_ns: Optional[int] = int(sub_ns + 14 * _NS_PER_DAY) if has_trial else None
            converts = rng.random() < trial_conversion_rate if has_trial else True

            is_churned = rng.random() < churn_rate
            end_ns = sub_ns + int(base_duration_days) * _NS_PER_DAY

            if not converts:
                cancel_ns: Optional[int] = int(sub_ns)
            elif is_churned and end_ns <= hi_ns:
                cancel_ns = int(end_ns)
            else:
                cancel_ns = None

            # Consume RNG for payment method (maintain stream compatibility)
            rng.choice(len(PAYMENT_METHODS), p=payment_weights)

            (
                sk, ck_l, pk_l, ps, pe, pr,
                first, churn, trial, cyc,
            ) = _expand_subscription_periods(
                sub_key=local_sub_key,
                ck=ck,
                pk=pk,
                sub_ns=sub_ns,
                cancel_ns=cancel_ns,
                trial_end_ns=trial_end_ns,
                cycle_months=cycle_months,
                cycle_price=cprice,
                g_end_ns=int(g_end_ns),
            )

            all_sk.extend(sk)
            all_ck.extend(ck_l)
            all_pk.extend(pk_l)
            all_ps.extend(ps)
            all_pe.extend(pe)
            all_price.extend(pr)
            all_first.extend(first)
            all_churn.extend(churn)
            all_trial.extend(trial)
            all_cycle.extend(cyc)

            local_sub_key += 1

    n_rows = len(all_sk)

    if n_rows > 0:
        out_path = Path(out_chunk_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        table = pa.Table.from_arrays(
            [
                pa.array(all_sk, type=pa.int64()),
                pa.array(all_ck, type=pa.int64()),
                pa.array(all_pk, type=pa.int32()),
                pa.array(all_ps, type=pa.date32()),
                pa.array(all_pe, type=pa.date32()),
                pa.array(all_price, type=pa.float64()),
                pa.array(all_first, type=pa.int8()),
                pa.array(all_churn, type=pa.int8()),
                pa.array(all_trial, type=pa.int8()),
                pa.array(all_cycle, type=pa.int32()),
            ],
            schema=schema,
        )
        pq.write_table(
            table,
            out_chunk_path,
            compression="snappy",
            row_group_size=500_000,
        )

    return {"chunk_idx": chunk_idx, "rows": n_rows}



# ---------------------------------------------------------------------------
# Parallel bridge entry point
# ---------------------------------------------------------------------------

def _write_bridge_parallel(
    customers: pd.DataFrame,
    dim_plans: pd.DataFrame,
    c: SubscriptionsCfg,
    g_start: pd.Timestamp,
    g_end: pd.Timestamp,
    out_bridge: Path,
    workers: Optional[int] = None,
) -> int:
    """
    Parallel bridge writer for large datasets.

    Pre-filters eligible customers, splits into chunks, dispatches to worker
    pool, merges chunk parquets into final file.
    Returns total rows written.
    """
    from src.facts.sales.sales_worker.pool import PoolRunSpec, iter_imap_unordered

    if "CustomerKey" not in customers.columns:
        raise KeyError("customers must include CustomerKey")

    # --- Plan data (read-only, passed to every worker) ---
    plan_keys = dim_plans["PlanKey"].astype(np.int32).to_numpy()
    plan_types = dim_plans["PlanType"].astype(str).to_numpy()
    plan_cycle_prices = dim_plans["CyclePrice"].to_numpy(dtype=np.float64, na_value=0.0)
    plan_cycle_months = dim_plans["CycleMonths"].astype(np.int32).to_numpy()
    unique_types, type_members, type_weights = _build_type_groups(plan_types)

    # --- Customer windows ---
    cust_keys, cust_start_ns, cust_end_ns = _compute_customer_windows(
        customers, g_start, g_end
    )
    g_end_ns = np.int64(g_end.value)
    max_subs = min(c.max_subscriptions, len(unique_types))

    # --- Eligibility filter (same RNG as serial path for consistency) ---
    n_cust = len(cust_keys)
    rng_main = np.random.default_rng(c.seed)
    participate_mask = rng_main.random(n_cust) < c.participation_rate
    _part_idx = np.where(participate_mask)[0]
    _spans = np.maximum(0, (cust_end_ns[_part_idx] - cust_start_ns[_part_idx]) // _NS_PER_DAY)
    _ok = _spans >= 30
    eligible_ck = cust_keys[_part_idx[_ok]]
    eligible_lo = cust_start_ns[_part_idx[_ok]]
    eligible_hi = cust_end_ns[_part_idx[_ok]]
    eligible_span = _spans[_ok]
    n_eligible = len(eligible_ck)

    if n_eligible == 0:
        out_bridge.parent.mkdir(parents=True, exist_ok=True)
        schema = _bridge_schema()
        pq.write_table(pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema), str(out_bridge))
        return 0

    # --- Chunk planning ---
    n_cpus = max(1, cpu_count() - 1)
    if workers is not None and workers >= 1:
        n_cpus = min(n_cpus, workers)
    n_chunks = max(2, min(n_eligible, n_cpus * 2))
    n_workers = min(n_chunks, n_cpus)

    info(f"Subscriptions parallel: {n_eligible:,} eligible customers, "
         f"{n_chunks} chunks, {n_workers} workers")

    # Split eligible customers into chunks (round-robin by index for even distribution)
    chunk_boundaries = np.array_split(np.arange(n_eligible), n_chunks)

    # Scratch directory for chunk parquets
    scratch_dir = out_bridge.parent / "_sub_chunks"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    # Build task list
    tasks = []
    for idx, indices in enumerate(chunk_boundaries):
        if len(indices) == 0:
            continue
        chunk_path = str(scratch_dir / f"sub_chunk_{idx:05d}.parquet")
        tasks.append((
            idx,
            c.seed,
            n_chunks,
            eligible_ck[indices],
            eligible_lo[indices],
            eligible_hi[indices],
            eligible_span[indices],
            plan_keys,
            plan_cycle_prices,
            plan_cycle_months,
            unique_types,
            type_members,
            type_weights,
            int(g_end_ns),
            max_subs,
            c.avg_subscriptions,
            c.churn_rate,
            c.trial_rate,
            c.trial_conversion_rate,
            _PAYMENT_WEIGHTS,
            chunk_path,
        ))

    actual_n_chunks = len(tasks)

    pool_spec = PoolRunSpec(
        processes=n_workers,
        chunksize=1,
        label="subscriptions",
    )

    chunk_rows: List[int] = [0] * actual_n_chunks
    completed = 0
    for result in iter_imap_unordered(
        tasks=tasks,
        task_fn=_subscription_worker_task,
        spec=pool_spec,
    ):
        completed += 1
        chunk_rows[result["chunk_idx"]] = result["rows"]

    info(f"Subscriptions: {completed}/{actual_n_chunks} chunks done")

    # --- Merge chunks and reassign SubscriptionKey sequentially ---
    total_rows = _merge_subscription_chunks(
        scratch_dir=scratch_dir,
        out_bridge=out_bridge,
        n_chunks=actual_n_chunks,
        delete_chunks=True,
    )

    # Clean up scratch dir
    try:
        scratch_dir.rmdir()
    except OSError:
        pass

    return total_rows


def _merge_subscription_chunks(
    scratch_dir: Path,
    out_bridge: Path,
    n_chunks: int,
    delete_chunks: bool = True,
) -> int:
    """
    Read chunk parquets in order, write final merged parquet.
    Returns total rows written.
    """
    schema = _bridge_schema()
    chunk_files = sorted(scratch_dir.glob("sub_chunk_*.parquet"))

    if not chunk_files:
        out_bridge.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema), str(out_bridge))
        return 0

    out_bridge.parent.mkdir(parents=True, exist_ok=True)
    if out_bridge.exists():
        out_bridge.unlink()

    total_rows = 0

    with pq.ParquetWriter(out_bridge, schema=schema, compression="snappy") as writer:
        for chunk_path in chunk_files:
            tbl = pq.read_table(chunk_path)
            n = len(tbl)
            if n == 0:
                continue

            writer.write_table(tbl)
            total_rows += n

            if delete_chunks:
                try:
                    chunk_path.unlink()
                except OSError:
                    pass

    return total_rows


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_subscriptions(cfg: Any, parquet_folder: Path) -> Dict[str, Any]:
    parquet_folder = Path(parquet_folder)

    c = _read_cfg(cfg)
    if not c.enabled:
        skip("Subscriptions disabled; skipping.")
        return {"_regenerated": False, "reason": "disabled"}

    out_dim = parquet_folder / "plans.parquet"
    out_bridge = parquet_folder / "customer_subscriptions.parquet"

    customers_fp = parquet_folder / "customers.parquet"
    if not customers_fp.exists():
        alt = parquet_folder / "Customers.parquet"
        if alt.exists():
            customers_fp = alt
        else:
            raise FileNotFoundError(
                f"Customers parquet not found at {parquet_folder}. "
                "Expected customers.parquet (or Customers.parquet)."
            )

    from src.utils.config_helpers import as_dict
    st = os.stat(customers_fp)
    version_cfg = as_dict(cfg.subscriptions)
    version_cfg["_schema_version"] = 4
    version_cfg["_upstream_customers_sig"] = {
        "path": str(customers_fp),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }

    version_files_exist = out_dim.exists() and (out_bridge.exists() or not c.generate_bridge)
    if version_files_exist and (not should_regenerate("subscriptions", version_cfg, out_dim)):
        if not c.generate_bridge and out_bridge.exists():
            out_bridge.unlink()
            info("Removed stale customer_subscriptions bridge file.")
        skip("Subscriptions up-to-date")
        return {"_regenerated": False, "reason": "version"}

    g_start, g_end = _parse_global_dates(cfg)

    with stage("Generating Subscriptions"):
        dim = build_dim_plans(g_start)
        dim.to_parquet(out_dim, index=False)
        info(f"Plans written: {out_dim.name} ({len(dim):,} rows)")

        n_rows = 0
        if c.generate_bridge:
            customers = pd.read_parquet(customers_fp)

            # Estimate eligible customers to decide serial vs parallel path
            n_cust = len(customers)
            estimated_eligible = int(n_cust * c.participation_rate * 0.9)  # rough lower bound

            workers: Optional[int] = None
            w_attr = getattr(cfg, "scale", None) or getattr(cfg, "defaults", None)
            if w_attr is not None:
                workers = int(getattr(w_attr, "workers", 0) or 0) or None

            if estimated_eligible >= SUBSCRIPTION_PARALLEL_THRESHOLD:
                info(f"Subscriptions: {n_cust:,} customers -> parallel path "
                     f"(estimated {estimated_eligible:,} eligible)")
                n_rows = _write_bridge_parallel(
                    customers=customers,
                    dim_plans=dim,
                    c=c,
                    g_start=g_start,
                    g_end=g_end,
                    out_bridge=out_bridge,
                    workers=workers,
                )
            else:
                n_rows = _write_bridge_streaming(
                    customers=customers,
                    dim_plans=dim,
                    c=c,
                    g_start=g_start,
                    g_end=g_end,
                    out_bridge=out_bridge,
                )
            save_version("subscriptions", version_cfg, out_bridge)
            info(f"Customer subscriptions written: {out_bridge.name} ({n_rows:,} rows)")
        else:
            skip("customer_subscriptions bridge skipped (generate_bridge: false)")
            if out_bridge.exists():
                out_bridge.unlink()

    return {
        "_regenerated": True,
        "dim": str(out_dim),
        "bridge": str(out_bridge) if c.generate_bridge else None,
        "bridge_rows": n_rows,
    }
