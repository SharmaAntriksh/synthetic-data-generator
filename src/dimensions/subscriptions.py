from __future__ import annotations

"""
Subscriptions (DimPlans + CustomerSubscriptions bridge)

Writes:
  - plans.parquet          (subscription plan dimension)
  - customer_subscriptions.parquet  (many-to-many bridge)

Bridge is written in parallel for large datasets (>200K eligible customers)
using a chunk-per-worker pattern.  Each worker generates rows IN MEMORY for
its customer slice, writes a chunk parquet, and returns its row count.
The main process merges chunks and reassigns SubscriptionKey sequentially.

For small datasets (<=200K eligible customers) the original single-process
streaming writer is used as a fallback.

Schema:
  DimPlans (14 cols):
    PlanKey, PlanName, PlanType, Category, BillingCycle, MonthlyPrice,
    Discount, CyclePrice, AnnualPrice, Tier, MaxUsers, HasFreeTrial,
    LaunchDate, IsActiveFlag

  CustomerSubscriptions (11 cols):
    SubscriptionKey, CustomerKey, PlanKey, SubscribedDate, CancelledDate,
    Status, MonthlyPrice, AutoRenew, TrialEndDate, PaymentMethod,
    LoyaltyDiscount

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

import os
from collections.abc import Mapping
from dataclasses import dataclass
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
}

# Base plan definitions: (PlanName, PlanType, MonthlyPrice, Tier, MaxUsers, HasFreeTrial, LaunchDate)
# Each base plan is expanded into one row per billing cycle it supports.
_BASE_PLANS: List[Tuple[str, str, float, str, int, int, str]] = [
    ("Basic Streaming",   "Streaming",     9.99, "Basic",    1, 1, "2021-01-15"),
    ("Premium Streaming", "Streaming",    19.99, "Premium",  4, 1, "2021-03-01"),
    ("Family Streaming",  "Streaming",    14.99, "Standard", 6, 0, "2021-06-01"),
    ("Fitness",           "Fitness",      29.99, "Standard", 1, 1, "2021-02-01"),
    ("Fitness Premium",   "Fitness",      49.99, "Premium",  2, 1, "2022-01-10"),
    ("Cloud 100GB",       "Cloud Storage", 2.99, "Basic",    1, 1, "2021-01-15"),
    ("Cloud 1TB",         "Cloud Storage", 9.99, "Standard", 5, 0, "2021-04-01"),
    ("Cloud Unlimited",   "Cloud Storage",14.99, "Premium", 10, 0, "2021-09-15"),
    ("Gaming Pass",       "Gaming",       14.99, "Standard", 1, 1, "2021-05-01"),
    ("Gaming Ultimate",   "Gaming",       19.99, "Premium",  1, 1, "2022-03-01"),
    ("News Digital",      "News & Media",  4.99, "Basic",    1, 1, "2021-01-15"),
    ("News All-Access",   "News & Media", 12.99, "Premium",  5, 0, "2021-07-01"),
    ("Learn",             "Education",    19.99, "Standard", 1, 1, "2021-08-15"),
    ("Learn Teams",       "Education",    14.99, "Premium", 10, 0, "2022-02-01"),
    ("Music",             "Music",        10.99, "Standard", 1, 1, "2021-03-15"),
    ("Music Family",      "Music",        16.99, "Premium",  6, 0, "2021-11-01"),
]

# Which billing cycles each base plan supports (index into _BASE_PLANS)
_PLAN_CYCLES: Dict[str, List[str]] = {
    "Basic Streaming":   ["Monthly"],
    "Premium Streaming": ["Monthly", "Quarterly", "Annual"],
    "Family Streaming":  ["Annual"],
    "Fitness":           ["Monthly", "Quarterly"],
    "Fitness Premium":   ["Monthly", "Half-Yearly", "Annual"],
    "Cloud 100GB":       ["Monthly"],
    "Cloud 1TB":         ["Monthly", "Quarterly"],
    "Cloud Unlimited":   ["Annual"],
    "Gaming Pass":       ["Monthly", "Quarterly"],
    "Gaming Ultimate":   ["Monthly", "Annual"],
    "News Digital":      ["Monthly"],
    "News All-Access":   ["Annual"],
    "Learn":             ["Monthly", "Half-Yearly"],
    "Learn Teams":       ["Annual"],
    "Music":             ["Monthly", "Quarterly"],
    "Music Family":      ["Half-Yearly", "Annual"],
}


def _expand_catalog() -> List[Tuple]:
    """
    Expand base plans × billing cycles into the full catalog.
    Returns list of:
      (PlanName, PlanType, Category, BillingCycle, MonthlyPrice, Discount,
       CyclePrice, AnnualPrice, Tier, MaxUsers, HasFreeTrial, LaunchDate)
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
                name, ptype, category, cycle, mprice, discount,
                cycle_price, annual_price, tier, maxu, trial, launch,
            ))
    return rows


PLANS_CATALOG = _expand_catalog()

_PLAN_TYPE_WEIGHT = {
    "Streaming": 4.0,
    "Fitness": 1.5,
    "Cloud Storage": 3.0,
    "Gaming": 2.5,
    "News & Media": 2.0,
    "Education": 1.5,
    "Music": 3.0,
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

def build_dim_plans() -> pd.DataFrame:
    """Build the subscription plans dimension table (14 columns)."""
    k = len(PLANS_CATALOG)
    return pd.DataFrame({
        "PlanKey":       np.arange(1, k + 1, dtype=np.int64),
        "PlanName":      [r[0] for r in PLANS_CATALOG],
        "PlanType":      [r[1] for r in PLANS_CATALOG],
        "Category":      [r[2] for r in PLANS_CATALOG],
        "BillingCycle":  [r[3] for r in PLANS_CATALOG],
        "MonthlyPrice":  pd.array([r[4] for r in PLANS_CATALOG], dtype="Float64"),
        "Discount":      pd.array([r[5] for r in PLANS_CATALOG], dtype="Float64"),
        "CyclePrice":    pd.array([r[6] for r in PLANS_CATALOG], dtype="Float64"),
        "AnnualPrice":   pd.array([r[7] for r in PLANS_CATALOG], dtype="Float64"),
        "Tier":          [r[8] for r in PLANS_CATALOG],
        "MaxUsers":      np.array([r[9] for r in PLANS_CATALOG], dtype=np.int32),
        "HasFreeTrial":  np.array([r[10] for r in PLANS_CATALOG], dtype=np.int8),
        "LaunchDate":    pd.to_datetime([r[11] for r in PLANS_CATALOG]),
        "IsActiveFlag":  np.ones(k, dtype=np.int8),
    })


def _compute_plan_weights(plan_types: np.ndarray) -> np.ndarray:
    """Map plan types to normalised probability weights."""
    weights = np.ones(len(plan_types), dtype=np.float64)
    for ptype, w in _PLAN_TYPE_WEIGHT.items():
        weights[plan_types == ptype] = w
    total = weights.sum()
    if total > 0:
        weights /= total
    return weights


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
        pa.field("SubscribedDate", pa.timestamp("ns")),
        pa.field("CancelledDate", pa.timestamp("ns")),
        pa.field("Status", pa.utf8()),
        pa.field("MonthlyPrice", pa.float64()),
        pa.field("AutoRenew", pa.int8()),
        pa.field("TrialEndDate", pa.timestamp("ns")),
        pa.field("PaymentMethod", pa.utf8()),
        pa.field("LoyaltyDiscount", pa.float64()),
    ])


# ---------------------------------------------------------------------------
# Bridge writer (streaming) -- serial path for small datasets
# ---------------------------------------------------------------------------

def _write_bridge_streaming(
    customers: pd.DataFrame,
    dim_plans: pd.DataFrame,
    c: SubscriptionsCfg,
    g_start: pd.Timestamp,
    g_end: pd.Timestamp,
    out_bridge: Path,
) -> int:
    """
    Stream-write customer_subscriptions bridge to parquet.
    Returns number of rows written.
    """
    if "CustomerKey" not in customers.columns:
        raise KeyError("customers must include CustomerKey")

    plan_keys = dim_plans["PlanKey"].astype(np.int32).to_numpy()
    plan_types = dim_plans["PlanType"].astype(str).to_numpy()
    plan_monthly = dim_plans["MonthlyPrice"].to_numpy(dtype=np.float64, na_value=0.0)
    plan_cycles = dim_plans["BillingCycle"].astype(str).to_numpy()
    # Pre-compute cycle months as array for direct indexing (avoids dict lookup per sub)
    plan_cycle_months = np.array(
        [_CYCLE_MONTHS.get(c, 1) for c in plan_cycles], dtype=np.int32
    )
    n_plans = len(plan_keys)
    w = _compute_plan_weights(plan_types)

    cust_keys, cust_start_ns, cust_end_ns = _compute_customer_windows(
        customers, g_start, g_end,
    )

    g_end_ns = np.int64(g_end.value)
    write_chunk_rows = max(c.write_chunk_rows, 10_000)
    max_subs = min(c.max_subscriptions, n_plans)

    rng = np.random.default_rng(c.seed)
    schema = _bridge_schema()

    # Determine which customers participate
    n_cust = len(cust_keys)
    participate_mask = rng.random(n_cust) < c.participation_rate

    # Pre-filter: eligible customers (participating + sufficient span)
    _part_idx = np.where(participate_mask)[0]
    _spans = np.maximum(0, (cust_end_ns[_part_idx] - cust_start_ns[_part_idx]) // _NS_PER_DAY)
    _ok = _spans >= 30
    eligible_idx = _part_idx[_ok]
    eligible_ck = cust_keys[eligible_idx]
    eligible_lo = cust_start_ns[eligible_idx]
    eligible_hi = cust_end_ns[eligible_idx]
    eligible_span = _spans[_ok]
    n_eligible = len(eligible_idx)

    # Pre-allocate buffers
    buf_cap = write_chunk_rows + max_subs + 10
    buf_sk = np.empty(buf_cap, dtype=np.int64)     # SubscriptionKey
    buf_ck = np.empty(buf_cap, dtype=np.int64)     # CustomerKey
    buf_pk = np.empty(buf_cap, dtype=np.int32)     # PlanKey
    buf_sub_date = np.empty(buf_cap, dtype=np.int64)  # SubscribedDate ns
    buf_cancel = np.empty(buf_cap, dtype="object")    # CancelledDate (nullable)
    buf_status = np.empty(buf_cap, dtype="object")    # Status string
    buf_price = np.empty(buf_cap, dtype=np.float64)   # MonthlyPrice
    buf_renew = np.empty(buf_cap, dtype=np.int8)      # AutoRenew
    buf_trial = np.empty(buf_cap, dtype="object")      # TrialEndDate (nullable)
    buf_payment = np.empty(buf_cap, dtype="object")    # PaymentMethod
    buf_loyalty = np.empty(buf_cap, dtype=np.float64)  # LoyaltyDiscount

    total_rows = 0
    pos = 0
    next_sub_key = 1

    def flush(writer: pq.ParquetWriter, n: int) -> None:
        nonlocal total_rows, pos
        if n == 0:
            return

        # Convert nullable object buffers to lists for PyArrow
        cancel_ns_list = buf_cancel[:n].tolist()
        trial_ns_list = buf_trial[:n].tolist()

        arrays: List[pa.Array] = [
            pa.array(buf_sk[:n], type=pa.int64()),
            pa.array(buf_ck[:n], type=pa.int64()),
            pa.array(buf_pk[:n], type=pa.int32()),
            pa.array(buf_sub_date[:n].copy(), type=pa.timestamp("ns")),
            pa.array(cancel_ns_list, type=pa.timestamp("ns")),
            pa.array(buf_status[:n].tolist(), type=pa.utf8()),
            pa.array(buf_price[:n], type=pa.float64()),
            pa.array(buf_renew[:n], type=pa.int8()),
            pa.array(trial_ns_list, type=pa.timestamp("ns")),
            pa.array(buf_payment[:n].tolist(), type=pa.utf8()),
            pa.array(buf_loyalty[:n], type=pa.float64()),
        ]

        table = pa.Table.from_arrays(arrays, schema=schema)
        writer.write_table(table)
        total_rows += n
        pos = 0

    out_bridge.parent.mkdir(parents=True, exist_ok=True)
    if out_bridge.exists():
        out_bridge.unlink()

    with pq.ParquetWriter(out_bridge, schema=schema, compression="snappy") as writer:
        for i in range(n_eligible):
            ck = int(eligible_ck[i])
            lo_ns = int(eligible_lo[i])
            hi_ns = int(eligible_hi[i])
            span_days = int(eligible_span[i])

            # How many subscriptions for this customer
            n_subs = max(1, int(rng.poisson(c.avg_subscriptions)))
            n_subs = min(n_subs, max_subs)

            # Choose plans (allow repeat for re-subscriptions)
            chosen_idx = rng.choice(n_plans, size=n_subs, replace=True, p=w)

            # Generate subscription start dates spread across customer window
            sub_offsets = np.sort(rng.integers(0, max(span_days - 30, 1), size=n_subs))

            for s in range(n_subs):
                pidx = chosen_idx[s]
                pk = int(plan_keys[pidx])

                # Subscribed date
                sub_ns = lo_ns + int(sub_offsets[s]) * _NS_PER_DAY

                # Duration based on billing cycle
                cycle_months = int(plan_cycle_months[pidx])
                # Number of renewal periods the customer stays
                n_periods = max(1, int(rng.geometric(0.3)))  # geometric: most stay 1-3 periods
                base_duration_days = cycle_months * 30 * n_periods

                price = float(plan_monthly[pidx])

                # Trial?
                has_trial = rng.random() < c.trial_rate
                if has_trial:
                    trial_end_ns = sub_ns + 14 * _NS_PER_DAY  # 14-day trial
                else:
                    trial_end_ns = None

                # Churned?
                is_churned = rng.random() < c.churn_rate
                end_ns = sub_ns + int(base_duration_days) * _NS_PER_DAY

                if is_churned and end_ns <= hi_ns:
                    cancel_ns = end_ns
                    status = "Cancelled"
                    auto_renew = 0
                elif end_ns > g_end_ns:
                    cancel_ns = None
                    status = "Active"
                    auto_renew = 1
                else:
                    cancel_ns = None
                    status = "Expired"
                    auto_renew = 0

                # Loyalty discount based on cumulative tenure
                tenure_days = (int(cancel_ns if cancel_ns else g_end_ns) - sub_ns) // _NS_PER_DAY
                if tenure_days >= 730:       # 24+ months
                    loyalty_disc = 0.10
                elif tenure_days >= 365:     # 12-24 months
                    loyalty_disc = 0.05
                else:
                    loyalty_disc = 0.00

                # Payment method
                pm_idx = rng.choice(len(PAYMENT_METHODS), p=_PAYMENT_WEIGHTS)
                payment = PAYMENT_METHODS[pm_idx]

                buf_sk[pos] = next_sub_key
                buf_ck[pos] = ck
                buf_pk[pos] = pk
                buf_sub_date[pos] = sub_ns
                buf_cancel[pos] = cancel_ns
                buf_status[pos] = status
                buf_price[pos] = round(price * (1 - loyalty_disc), 2)
                buf_renew[pos] = auto_renew
                buf_trial[pos] = trial_end_ns
                buf_payment[pos] = payment
                buf_loyalty[pos] = loyalty_disc

                next_sub_key += 1
                pos += 1

                if pos >= write_chunk_rows:
                    flush(writer, pos)

        flush(writer, pos)

    return total_rows


# ---------------------------------------------------------------------------
# Parallel bridge writer -- top-level worker (must be importable for spawn)
# ---------------------------------------------------------------------------

def _subscription_worker_task(args: Tuple) -> Dict[str, Any]:
    """
    Worker entry point (must be top-level for Windows spawn pickling).

    Generates subscription rows IN MEMORY for a slice of eligible customers,
    writes a chunk parquet to disk, and returns its row count.

    SubscriptionKey values written here are LOCAL (1-based within chunk).
    The main process reassigns keys sequentially after merge.

    Args (tuple positional):
        chunk_idx          int   -- chunk sequence number
        seed               int   -- base seed; worker derives its own via SeedSequence
        n_chunks           int   -- total number of chunks (for SeedSequence spawn)
        eligible_ck        ndarray int64  -- CustomerKey for this chunk
        eligible_lo        ndarray int64  -- window start ns
        eligible_hi        ndarray int64  -- window end ns
        eligible_span      ndarray int64  -- span in days
        plan_keys          ndarray int32
        plan_monthly       ndarray float64
        plan_cycle_months  ndarray int32
        plan_weights       ndarray float64  -- normalised plan selection weights
        n_plans            int
        g_end_ns           int64
        max_subs           int
        avg_subscriptions  float
        churn_rate         float
        trial_rate         float
        payment_weights    ndarray float64
        out_chunk_path     str   -- full path to write chunk parquet

    Returns:
        dict with chunk_idx and rows
    """
    (
        chunk_idx, seed, n_chunks,
        eligible_ck, eligible_lo, eligible_hi, eligible_span,
        plan_keys, plan_monthly, plan_cycle_months, plan_weights,
        n_plans, g_end_ns, max_subs,
        avg_subscriptions, churn_rate, trial_rate,
        payment_weights,
        out_chunk_path,
    ) = args

    # Each worker gets its own independent RNG stream
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(n_chunks)
    rng = np.random.default_rng(child_seeds[chunk_idx])

    n_eligible = len(eligible_ck)
    schema = _bridge_schema()

    # Estimate capacity: n_eligible * max_subs rows at most
    cap = max(n_eligible * max_subs + 10, 100)

    arr_sk = np.empty(cap, dtype=np.int64)
    arr_ck = np.empty(cap, dtype=np.int64)
    arr_pk = np.empty(cap, dtype=np.int32)
    arr_sub_date = np.empty(cap, dtype=np.int64)
    arr_cancel: List[Optional[int]] = []
    arr_status: List[str] = []
    arr_price = np.empty(cap, dtype=np.float64)
    arr_renew = np.empty(cap, dtype=np.int8)
    arr_trial: List[Optional[int]] = []
    arr_payment: List[str] = []
    arr_loyalty = np.empty(cap, dtype=np.float64)

    pos = 0
    local_key = 1  # local key, reassigned by main process after merge

    for i in range(n_eligible):
        ck = int(eligible_ck[i])
        lo_ns = int(eligible_lo[i])
        hi_ns = int(eligible_hi[i])
        span_days = int(eligible_span[i])

        n_subs = max(1, int(rng.poisson(avg_subscriptions)))
        n_subs = min(n_subs, max_subs)

        chosen_idx = rng.choice(n_plans, size=n_subs, replace=True, p=plan_weights)
        sub_offsets = np.sort(rng.integers(0, max(span_days - 30, 1), size=n_subs))

        for s in range(n_subs):
            pidx = int(chosen_idx[s])
            pk = int(plan_keys[pidx])

            sub_ns = lo_ns + int(sub_offsets[s]) * _NS_PER_DAY

            cycle_months = int(plan_cycle_months[pidx])
            n_periods = max(1, int(rng.geometric(0.3)))
            base_duration_days = cycle_months * 30 * n_periods

            price = float(plan_monthly[pidx])

            has_trial = rng.random() < trial_rate
            trial_end_ns: Optional[int] = int(sub_ns + 14 * _NS_PER_DAY) if has_trial else None

            is_churned = rng.random() < churn_rate
            end_ns = sub_ns + int(base_duration_days) * _NS_PER_DAY

            if is_churned and end_ns <= hi_ns:
                cancel_ns: Optional[int] = int(end_ns)
                status = "Cancelled"
                auto_renew = np.int8(0)
            elif end_ns > g_end_ns:
                cancel_ns = None
                status = "Active"
                auto_renew = np.int8(1)
            else:
                cancel_ns = None
                status = "Expired"
                auto_renew = np.int8(0)

            ref_ns = cancel_ns if cancel_ns is not None else g_end_ns
            tenure_days = (int(ref_ns) - sub_ns) // _NS_PER_DAY
            if tenure_days >= 730:
                loyalty_disc = 0.10
            elif tenure_days >= 365:
                loyalty_disc = 0.05
            else:
                loyalty_disc = 0.00

            pm_idx = int(rng.choice(len(PAYMENT_METHODS), p=payment_weights))
            payment = PAYMENT_METHODS[pm_idx]

            if pos >= cap:
                # Grow buffers (shouldn't happen often given the estimate)
                new_cap = cap * 2
                arr_sk = _grow(arr_sk, new_cap)
                arr_ck = _grow(arr_ck, new_cap)
                arr_pk = _grow(arr_pk, new_cap)
                arr_sub_date = _grow(arr_sub_date, new_cap)
                arr_price = _grow(arr_price, new_cap)
                arr_renew = _grow(arr_renew, new_cap)
                arr_loyalty = _grow(arr_loyalty, new_cap)
                cap = new_cap

            arr_sk[pos] = local_key
            arr_ck[pos] = ck
            arr_pk[pos] = pk
            arr_sub_date[pos] = sub_ns
            arr_cancel.append(cancel_ns)
            arr_status.append(status)
            arr_price[pos] = round(price * (1 - loyalty_disc), 2)
            arr_renew[pos] = auto_renew
            arr_trial.append(trial_end_ns)
            arr_payment.append(payment)
            arr_loyalty[pos] = loyalty_disc

            local_key += 1
            pos += 1

    n_rows = pos

    if n_rows > 0:
        out_path = Path(out_chunk_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        table = pa.Table.from_arrays(
            [
                pa.array(arr_sk[:n_rows], type=pa.int64()),
                pa.array(arr_ck[:n_rows], type=pa.int64()),
                pa.array(arr_pk[:n_rows], type=pa.int32()),
                pa.array(arr_sub_date[:n_rows].copy(), type=pa.timestamp("ns")),
                pa.array(arr_cancel, type=pa.timestamp("ns")),
                pa.array(arr_status, type=pa.utf8()),
                pa.array(arr_price[:n_rows], type=pa.float64()),
                pa.array(arr_renew[:n_rows], type=pa.int8()),
                pa.array(arr_trial, type=pa.timestamp("ns")),
                pa.array(arr_payment, type=pa.utf8()),
                pa.array(arr_loyalty[:n_rows], type=pa.float64()),
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


def _grow(arr: np.ndarray, new_cap: int) -> np.ndarray:
    """Return a new array with size new_cap, copying existing data."""
    new_arr = np.empty(new_cap, dtype=arr.dtype)
    new_arr[: len(arr)] = arr
    return new_arr


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
    pool, merges chunk parquets, reassigns SubscriptionKey, writes final file.
    Returns total rows written.
    """
    from src.facts.sales.sales_worker.pool import PoolRunSpec, iter_imap_unordered

    if "CustomerKey" not in customers.columns:
        raise KeyError("customers must include CustomerKey")

    # --- Plan data (read-only, passed to every worker) ---
    plan_keys = dim_plans["PlanKey"].astype(np.int32).to_numpy()
    plan_types = dim_plans["PlanType"].astype(str).to_numpy()
    plan_monthly = dim_plans["MonthlyPrice"].to_numpy(dtype=np.float64, na_value=0.0)
    plan_cycles = dim_plans["BillingCycle"].astype(str).to_numpy()
    plan_cycle_months = np.array(
        [_CYCLE_MONTHS.get(cyc, 1) for cyc in plan_cycles], dtype=np.int32
    )
    n_plans = len(plan_keys)
    plan_weights = _compute_plan_weights(plan_types)

    # --- Customer windows ---
    cust_keys, cust_start_ns, cust_end_ns = _compute_customer_windows(
        customers, g_start, g_end
    )
    g_end_ns = np.int64(g_end.value)
    max_subs = min(c.max_subscriptions, n_plans)

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
            plan_monthly,
            plan_cycle_months,
            plan_weights,
            n_plans,
            int(g_end_ns),
            max_subs,
            c.avg_subscriptions,
            c.churn_rate,
            c.trial_rate,
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
    Read chunk parquets in order, reassign SubscriptionKey 1..N, write final parquet.
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
    next_key = np.int64(1)

    with pq.ParquetWriter(out_bridge, schema=schema, compression="snappy") as writer:
        for chunk_path in chunk_files:
            tbl = pq.read_table(chunk_path)
            n = len(tbl)
            if n == 0:
                continue

            # Reassign SubscriptionKey to be globally sequential
            new_keys = pa.array(
                np.arange(next_key, next_key + n, dtype=np.int64),
                type=pa.int64(),
            )
            tbl = tbl.set_column(
                tbl.schema.get_field_index("SubscriptionKey"),
                "SubscriptionKey",
                new_keys,
            )
            writer.write_table(tbl)
            next_key = np.int64(next_key + n)
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
    version_cfg["_schema_version"] = 2
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

    with stage("Generating Subscriptions"):
        dim = build_dim_plans()
        dim.to_parquet(out_dim, index=False)
        info(f"Plans written: {out_dim.name} ({len(dim):,} rows)")

        n_rows = 0
        if c.generate_bridge:
            customers = pd.read_parquet(customers_fp)
            g_start, g_end = _parse_global_dates(cfg)

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
