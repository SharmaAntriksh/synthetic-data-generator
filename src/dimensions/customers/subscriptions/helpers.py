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

from src.utils.config_precedence import resolve_seed

from .catalog import PLANS_CATALOG, _PLAN_TYPE_WEIGHT


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
            raise ValueError("subscriptions.global_dates.end must be >= subscriptions.global_dates.start")
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
# Plan builders
# ---------------------------------------------------------------------------

def build_dim_plans(g_start: pd.Timestamp) -> pd.DataFrame:
    """Build the subscription plans dimension table (15 columns)."""
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


def bridge_schema() -> pa.Schema:
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


def write_empty_bridge(out_path: Path) -> None:
    """Write an empty bridge parquet with the correct schema."""
    schema = bridge_schema()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(schema.empty_table(), str(out_path))


def expand_subscription_periods(
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
    """Expand a single subscription into billing-period rows.

    Pre-computes the number of periods to avoid incremental list growth.
    """
    sub_y, sub_m = ns_to_year_month(sub_ns)
    end_ref = cancel_ns if cancel_ns is not None else g_end_ns
    end_y, end_m = ns_to_year_month(end_ref)
    n_months = months_between(sub_y, sub_m, end_y, end_m)
    if n_months <= 0:
        n_months = 1

    n_periods = -(-n_months // cycle_months)  # ceil division

    has_cancel = cancel_ns is not None
    has_trial = trial_end_ns is not None

    # Pre-allocate lists at final size
    sk_list = [sub_key] * n_periods
    ck_list = [ck] * n_periods
    pk_list = [pk] * n_periods
    ps_list: List[date] = [None] * n_periods  # type: ignore[list-item]
    pe_list: List[date] = [None] * n_periods  # type: ignore[list-item]
    price_list = [cycle_price] * n_periods
    first_list = [0] * n_periods
    churn_list = [0] * n_periods
    trial_list = [0] * n_periods
    cycle_list: List[int] = [0] * n_periods

    y, m = sub_y, sub_m
    for i in range(n_periods):
        end_y_p, end_m_p = advance_months(y, m, cycle_months - 1)

        ps_list[i] = month_start_date(y, m)
        pe_list[i] = month_end_date(end_y_p, end_m_p)
        cycle_list[i] = i + 1

        if i == 0:
            first_list[0] = 1
            if has_trial:
                trial_list[0] = 1
                price_list[0] = 0.0

        if i == n_periods - 1 and has_cancel:
            churn_list[i] = 1

        y, m = advance_months(y, m, cycle_months)

    return (
        sk_list, ck_list, pk_list,
        ps_list, pe_list, price_list,
        first_list, churn_list, trial_list, cycle_list,
    )
