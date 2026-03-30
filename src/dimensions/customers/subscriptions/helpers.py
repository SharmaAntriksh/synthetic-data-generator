"""Shared helpers for subscription generation."""
from __future__ import annotations

import calendar
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.exceptions import DimensionError
from src.utils.config_precedence import resolve_seed

from .catalog import PAYMENT_METHODS, _PAYMENT_WEIGHTS, PLANS_CATALOG, _PLAN_TYPE_WEIGHT


_NS_PER_DAY: int = 86_400_000_000_000


# ---------------------------------------------------------------------------
# Config dataclass
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
    trial_days: int = 14
    seed: int = 700
    write_chunk_rows: int = 250_000


def read_cfg(cfg: Any) -> SubscriptionsCfg:
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
        trial_days=int(sc.trial_days),
        seed=seed,
        write_chunk_rows=int(sc.write_chunk_rows),
    )


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------

def ns_to_year_month(ns: int) -> Tuple[int, int]:
    """Convert nanosecond timestamp to (year, month) tuple."""
    dt = pd.Timestamp(ns, unit="ns")
    return dt.year, dt.month


def months_between(y1: int, m1: int, y2: int, m2: int) -> int:
    """Number of months from (y1,m1) to (y2,m2) inclusive."""
    return (y2 - y1) * 12 + (m2 - m1) + 1


def month_start_date(year: int, month: int) -> date:
    return date(year, month, 1)


def month_end_date(year: int, month: int) -> date:
    return date(year, month, calendar.monthrange(year, month)[1])


def advance_months(y: int, m: int, n: int) -> Tuple[int, int]:
    """Advance (year, month) by n months."""
    m += n
    while m > 12:
        m -= 12
        y += 1
    return y, m


def parse_global_dates(cfg: Any) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Resolve timeline dates from config (priority: subscriptions.global_dates > defaults.dates)."""
    sc = cfg.subscriptions
    gd = sc.global_dates if sc is not None else None
    if isinstance(gd, Mapping) and gd.get("start") and gd.get("end"):
        start = pd.to_datetime(gd["start"]).normalize()
        end = pd.to_datetime(gd["end"]).normalize()
        if end < start:
            raise DimensionError("subscriptions.global_dates.end must be >= subscriptions.global_dates.start")
        return start, end

    defaults = cfg.defaults if hasattr(cfg, "defaults") else getattr(cfg, "_defaults", None)
    if defaults is None:
        raise DimensionError("Missing defaults.dates.start/end (or _defaults.dates.start/end)")
    d = defaults.dates
    d_start = d.start if hasattr(d, "start") else None
    d_end = d.end if hasattr(d, "end") else None
    if not d_start or not d_end:
        raise DimensionError("Missing defaults.dates.start/end (or _defaults.dates.start/end)")
    start = pd.to_datetime(d_start).normalize()
    end = pd.to_datetime(d_end).normalize()
    if end < start:
        raise DimensionError("defaults.dates.end must be >= defaults.dates.start")
    return start, end


# ---------------------------------------------------------------------------
# Plan builders
# ---------------------------------------------------------------------------

def build_dim_plans(g_start: pd.Timestamp) -> pd.DataFrame:
    """Build the subscription plans dimension table (15 columns)."""
    k = len(PLANS_CATALOG)
    launch_dates = pd.to_datetime([
        g_start + pd.Timedelta(days=int(r[12])) for r in PLANS_CATALOG
    ])
    return pd.DataFrame({
        "PlanKey":          np.arange(1, k + 1, dtype=np.int32),
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
        "HasFreeTrial":     np.array([r[11] for r in PLANS_CATALOG], dtype=bool),
        "LaunchDate":       launch_dates,
        "IsActiveFlag":     np.ones(k, dtype=bool),
    })


def build_type_groups(plan_types: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """Pre-compute per-type plan indices and type-level weights."""
    unique_types = np.unique(plan_types)
    type_members: List[np.ndarray] = []
    type_weights = np.empty(len(unique_types), dtype=np.float64)
    for i, t in enumerate(unique_types):
        members = np.where(plan_types == t)[0]
        type_members.append(members)
        type_weights[i] = _PLAN_TYPE_WEIGHT.get(t, 1.0)
    type_weights /= type_weights.sum()
    return unique_types, type_members, type_weights


def choose_plans_diverse(
    rng: np.random.Generator,
    n_subs: int,
    unique_types: np.ndarray,
    type_members: List[np.ndarray],
    type_weights: np.ndarray,
) -> np.ndarray:
    """Pick n_subs plans from distinct PlanTypes."""
    n_types = len(unique_types)
    n_subs = min(n_subs, n_types)
    chosen_type_idx = rng.choice(n_types, size=n_subs, replace=False, p=type_weights)
    plan_idx = np.empty(n_subs, dtype=np.intp)
    for i, ti in enumerate(chosen_type_idx):
        members = type_members[ti]
        plan_idx[i] = members[rng.integers(len(members))]
    return plan_idx


def compute_customer_windows(
    customers: pd.DataFrame,
    g_start: pd.Timestamp,
    g_end: pd.Timestamp,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract sorted (CustomerKey, start_ns, end_ns) arrays, clamped to [g_start, g_end]."""
    cust_keys = customers["CustomerKey"].astype(np.int32).to_numpy()
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


def bridge_schema() -> pa.Schema:
    return pa.schema([
        pa.field("SubscriptionKey", pa.int32()),
        pa.field("CustomerKey", pa.int32()),
        pa.field("PlanKey", pa.int32()),
        pa.field("PeriodStartDate", pa.date32()),
        pa.field("PeriodEndDate", pa.date32()),
        pa.field("PeriodPrice", pa.float64()),
        pa.field("IsFirstPeriod", pa.bool_()),
        pa.field("IsChurnPeriod", pa.bool_()),
        pa.field("IsTrialPeriod", pa.bool_()),
        pa.field("BillingCycleNumber", pa.int32()),
    ])


def write_empty_bridge(out_path: Path) -> None:
    """Write an empty bridge parquet with the correct schema."""
    schema = bridge_schema()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(schema.empty_table(), str(out_path))


# ---------------------------------------------------------------------------
# Vectorized bulk expansion
# ---------------------------------------------------------------------------

def _ns_to_year_month_arrays(ns_arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert nanosecond timestamps to (year, month) arrays without pd.Timestamp loop."""
    # Convert ns → days since epoch, then to datetime64[D] for year/month extraction
    dt = ns_arr.astype("datetime64[ns]")
    years = dt.astype("datetime64[Y]").astype(int) + 1970
    months = dt.astype("datetime64[M]").astype(int) % 12 + 1
    return years.astype(np.int32), months.astype(np.int32)


def _month_start_dates(years: np.ndarray, months: np.ndarray) -> np.ndarray:
    """Vectorized month-start dates as datetime64[D]."""
    # year*12 + (month-1) → months since epoch, then to datetime64[M] → [D]
    m_since_epoch = (years - 1970).astype("int64") * 12 + (months - 1).astype("int64")
    return m_since_epoch.astype("datetime64[M]").astype("datetime64[D]")


def _month_end_dates(years: np.ndarray, months: np.ndarray) -> np.ndarray:
    """Vectorized month-end dates as datetime64[D]."""
    # Start of next month minus 1 day
    next_m = months + 1
    next_y = years.copy()
    wrap = next_m > 12
    next_m[wrap] = 1
    next_y[wrap] += 1
    m_since_epoch = (next_y - 1970).astype("int64") * 12 + (next_m - 1).astype("int64")
    next_start = m_since_epoch.astype("datetime64[M]").astype("datetime64[D]")
    return next_start - np.timedelta64(1, "D")


def generate_subscriptions_bulk(
    eligible_ck: np.ndarray,
    eligible_lo: np.ndarray,
    eligible_hi: np.ndarray,
    eligible_span: np.ndarray,
    plan_keys: np.ndarray,
    plan_cycle_prices: np.ndarray,
    plan_cycle_months: np.ndarray,
    unique_types: np.ndarray,
    type_members: List[np.ndarray],
    type_weights: np.ndarray,
    g_end_ns: int,
    max_subs: int,
    avg_subscriptions: float,
    churn_rate: float,
    trial_rate: float,
    trial_conversion_rate: float,
    trial_days: int,
    rng: np.random.Generator,
    sub_key_start: int = 1,
) -> pa.Table:
    """Generate all subscription billing-period rows in vectorized bulk.

    Replaces the per-customer Python loop with array operations.
    Returns a PyArrow Table ready for writing.
    """
    n_eligible = len(eligible_ck)
    if n_eligible == 0:
        return bridge_schema().empty_table()

    # ------------------------------------------------------------------
    # 1. Generate subscription counts per customer (vectorized)
    # ------------------------------------------------------------------
    n_subs_arr = np.clip(
        rng.poisson(avg_subscriptions, size=n_eligible),
        1, max_subs,
    ).astype(np.int32)

    total_subs = int(n_subs_arr.sum())

    # Customer index for each subscription
    cust_idx = np.repeat(np.arange(n_eligible, dtype=np.int32), n_subs_arr)

    # ------------------------------------------------------------------
    # 2. Plan selection — must respect diverse type selection per customer
    #    Loop over customers but the inner work is minimal (RNG + indexing)
    # ------------------------------------------------------------------
    sub_plan_idx = np.empty(total_subs, dtype=np.int32)
    pos = 0
    for i in range(n_eligible):
        ns = int(n_subs_arr[i])
        chosen = choose_plans_diverse(rng, ns, unique_types, type_members, type_weights)
        sub_plan_idx[pos:pos + len(chosen)] = chosen
        pos += len(chosen)
    # Trim if choose_plans_diverse returned fewer than requested
    total_subs = pos
    cust_idx = cust_idx[:total_subs]
    sub_plan_idx = sub_plan_idx[:total_subs]

    # ------------------------------------------------------------------
    # 3. Subscription start offsets (vectorized per customer group)
    # ------------------------------------------------------------------
    # Generate random offsets then sort within each customer
    span_days = eligible_span[cust_idx]
    max_offset = np.maximum(span_days - 30, 1)
    raw_offsets = (rng.random(total_subs) * max_offset).astype(np.int64)

    # Sort offsets within each customer's contiguous block
    _starts = np.zeros(n_eligible + 1, dtype=np.int64)
    np.cumsum(n_subs_arr, out=_starts[1:])
    for i in range(n_eligible):
        lo, hi = int(_starts[i]), int(_starts[i + 1])
        if hi - lo > 1:
            raw_offsets[lo:hi] = np.sort(raw_offsets[lo:hi])

    # ------------------------------------------------------------------
    # 4. Subscription attributes (vectorized)
    # ------------------------------------------------------------------
    lo_ns = eligible_lo[cust_idx]
    hi_ns = eligible_hi[cust_idx]
    sub_ns = lo_ns + raw_offsets * _NS_PER_DAY

    cycle_months = plan_cycle_months[sub_plan_idx]
    cycle_prices = plan_cycle_prices[sub_plan_idx]
    pk_arr = plan_keys[sub_plan_idx]
    ck_arr = eligible_ck[cust_idx]

    # Duration: geometric distribution for number of billing periods
    n_billing_periods = np.clip(rng.geometric(0.3, size=total_subs), 1, 100).astype(np.int32)
    base_duration_days = cycle_months.astype(np.int64) * 30 * n_billing_periods.astype(np.int64)

    # Trial / churn decisions (vectorized)
    has_trial = rng.random(total_subs) < trial_rate
    converts = np.ones(total_subs, dtype=bool)
    trial_mask = has_trial
    converts[trial_mask] = rng.random(int(trial_mask.sum())) < trial_conversion_rate

    is_churned = rng.random(total_subs) < churn_rate
    end_ns = sub_ns + base_duration_days * _NS_PER_DAY

    # Cancel logic
    cancel_ns = np.full(total_subs, -1, dtype=np.int64)  # -1 = no cancel
    no_convert = ~converts
    cancel_ns[no_convert] = sub_ns[no_convert]
    churn_ok = is_churned & converts & (end_ns <= hi_ns)
    cancel_ns[churn_ok] = end_ns[churn_ok]
    has_cancel = cancel_ns >= 0

    # RNG consumption for payment method (maintain stream compatibility)
    rng.choice(len(PAYMENT_METHODS), size=total_subs, p=_PAYMENT_WEIGHTS)

    # ------------------------------------------------------------------
    # 5. Compute number of billing periods per subscription (vectorized)
    # ------------------------------------------------------------------
    # end_ref = cancel_ns if has_cancel else g_end_ns
    end_ref = np.where(has_cancel, cancel_ns, np.int64(g_end_ns))

    sub_y, sub_m = _ns_to_year_month_arrays(sub_ns)
    end_y, end_m = _ns_to_year_month_arrays(end_ref)

    n_months_span = (end_y - sub_y) * 12 + (end_m - sub_m) + 1
    n_months_span = np.maximum(n_months_span, 1)
    n_periods = (n_months_span + cycle_months - 1) // cycle_months  # ceil division
    n_periods = n_periods.astype(np.int32)

    total_rows = int(n_periods.sum())

    # ------------------------------------------------------------------
    # 6. Expand to billing-period rows using np.repeat
    # ------------------------------------------------------------------
    sub_keys = np.arange(sub_key_start, sub_key_start + total_subs, dtype=np.int32)

    r_sk = np.repeat(sub_keys, n_periods)
    r_ck = np.repeat(ck_arr, n_periods)
    r_pk = np.repeat(pk_arr, n_periods)
    r_price = np.repeat(cycle_prices, n_periods)
    r_sub_y = np.repeat(sub_y, n_periods)
    r_sub_m = np.repeat(sub_m, n_periods)
    r_cycle_months = np.repeat(cycle_months, n_periods)
    r_has_cancel = np.repeat(has_cancel, n_periods)
    r_has_trial = np.repeat(has_trial, n_periods)
    r_n_periods = np.repeat(n_periods, n_periods)

    # Period index within each subscription (0, 1, 2, ...)
    offsets = np.zeros(total_subs + 1, dtype=np.int64)
    np.cumsum(n_periods, out=offsets[1:])
    period_idx = np.arange(total_rows, dtype=np.int32) - np.repeat(offsets[:-1], n_periods).astype(np.int32)

    # Period start/end: advance sub_y/sub_m by period offsets
    total_month_offset = (period_idx * r_cycle_months).astype(np.int64)
    base_month = (r_sub_y.astype(np.int64) - 1970) * 12 + (r_sub_m.astype(np.int64) - 1)

    abs_month = base_month + total_month_offset
    ps_year = (abs_month // 12 + 1970).astype(np.int32)
    ps_month = (abs_month % 12 + 1).astype(np.int32)
    period_start = _month_start_dates(ps_year, ps_month)

    abs_end_month = base_month + total_month_offset + (r_cycle_months - 1).astype(np.int64)
    pe_year = (abs_end_month // 12 + 1970).astype(np.int32)
    pe_month = (abs_end_month % 12 + 1).astype(np.int32)
    period_end = _month_end_dates(pe_year, pe_month)

    # Flags
    billing_cycle = (period_idx + 1).astype(np.int32)
    is_first = (period_idx == 0).astype(np.int32)
    is_last = (period_idx == r_n_periods - 1)
    is_churn = (is_last & r_has_cancel).astype(np.int32)
    is_trial_period = (is_first.astype(bool) & r_has_trial).astype(np.int32)

    # Trial periods have price = 0
    r_price = np.where(is_trial_period, 0.0, r_price)

    # ------------------------------------------------------------------
    # 7. Build Arrow table
    # ------------------------------------------------------------------
    schema = bridge_schema()
    table = pa.Table.from_arrays(
        [
            pa.array(r_sk),
            pa.array(r_ck),
            pa.array(r_pk.astype(np.int32)),
            pa.array(period_start),
            pa.array(period_end),
            pa.array(r_price),
            pa.array(is_first),
            pa.array(is_churn),
            pa.array(is_trial_period),
            pa.array(billing_cycle),
        ],
        schema=schema,
    )

    return table
