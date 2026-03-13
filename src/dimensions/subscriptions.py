from __future__ import annotations

"""
Subscriptions (DimPlans + CustomerSubscriptions bridge)

Writes:
  - plans.parquet          (subscription plan dimension)
  - customer_subscriptions.parquet  (many-to-many bridge)

Bridge is written STREAMING with pyarrow ParquetWriter (does not hold the
whole bridge in RAM).

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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.config_precedence import resolve_seed
from src.utils.logging_utils import info, skip, stage
from src.versioning.version_store import should_regenerate, save_version


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
# Bridge writer (streaming)
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

        # Build CancelledDate array with nulls
        cancel_ns_list = []
        for j in range(n):
            v = buf_cancel[j]
            cancel_ns_list.append(v if v is not None else None)

        trial_ns_list = []
        for j in range(n):
            v = buf_trial[j]
            trial_ns_list.append(v if v is not None else None)

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
        for i in range(n_cust):
            if not participate_mask[i]:
                continue

            ck = int(cust_keys[i])
            lo_ns = int(cust_start_ns[i])
            hi_ns = int(cust_end_ns[i])
            span_days = max(0, (hi_ns - lo_ns) // _NS_PER_DAY)

            if span_days < 30:
                continue

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
                cycle = plan_cycles[pidx]

                # Subscribed date
                sub_ns = lo_ns + int(sub_offsets[s]) * _NS_PER_DAY

                # Duration based on billing cycle
                cycle_months = _CYCLE_MONTHS.get(cycle, 1)
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
        info(f"Plans written: {out_dim} ({len(dim):,} rows)")

        n_rows = 0
        if c.generate_bridge:
            customers = pd.read_parquet(customers_fp)
            g_start, g_end = _parse_global_dates(cfg)
            n_rows = _write_bridge_streaming(
                customers=customers,
                dim_plans=dim,
                c=c,
                g_start=g_start,
                g_end=g_end,
                out_bridge=out_bridge,
            )
            save_version("subscriptions", version_cfg, out_bridge)
            info(f"Customer subscriptions written: {out_bridge} ({n_rows:,} rows)")
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
