from __future__ import annotations

"""
Customer Segments (kept for backward compatibility; improved + easier to reason about)

This module produces two datasets:

1) customer_segment.parquet (DimCustomerSegment)
2) customer_segment_membership.parquet (BridgeCustomerSegmentMembership)

Improvements vs the older implementation:
- Adds a "mode" switch:
    - mode="scd2"   (DEFAULT): retains your existing month-grain churn simulation
    - mode="simple": demo-friendly, rule-based tags derived from Customers attributes
- Major perf improvement when include_validity=false:
    - emits a single interval per membership (no month-by-month loop)
- Cleaner, more explicit segment catalog for simple mode (tags grouped by SegmentType)
- More selective parquet reads (loads only required Customer columns)

Config (backward compatible)
customer_segments:
  enabled: true
  mode: scd2                 # "scd2" (default) or "simple"
  segment_count: 12
  segments_per_customer_min: 1
  segments_per_customer_max: 4
  include_score: true
  include_primary_flag: true
  include_validity: true
  validity:
    grain: month              # "month" or "day" (day not implemented for scd2; kept for config compatibility)
    churn_rate_qtr: 0.08
    new_customer_months: 2
  seed: 123
  override:
    seed: 123
    dates:
      start: "2010-01-01"
      end: "2012-12-31"
    paths:
      customers: "customers.parquet"
      customer_segment: "customer_segment.parquet"
      customer_segment_membership: "customer_segment_membership.parquet"
Notes:
- "simple" mode is recommended for demos: stable, explainable, low volume.
- The bridge output always includes ValidFromDate/ValidToDate as datetime64[ns].
"""

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, Tuple

import os
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.logging_utils import done, skip, info
from src.versioning.version_store import should_regenerate, save_version


# -----------------------------------------------------------------------------
# Segment catalogs
# -----------------------------------------------------------------------------

# Legacy SCD2-ish segments (kept as-is)
DEFAULT_SEGMENTS: List[Tuple[str, str, str]] = [
    ("High Value", "RFM", "Top spend/margin customers"),
    ("Frequent Shopper", "RFM", "High order frequency"),
    ("Discount Seeker", "Behavioral", "High promo/discount usage"),
    ("Online First", "Channel", "Majority purchases online"),
    ("Store Loyalist", "Channel", "Majority purchases in-store"),
    ("Returns Prone", "Risk", "Return rate above threshold"),
    ("New Customer", "Lifecycle", "Recently acquired (first N months)"),
    ("Bulk Buyer", "Behavioral", "Often buys higher quantities"),
    ("Premium Buyer", "Behavioral", "Buys premium assortments"),
    ("Seasonal Shopper", "Lifecycle", "Spikes in seasonal periods"),
    ("Lapsed", "Lifecycle", "No purchases recently (approximate)"),
    ("Price Sensitive", "Behavioral", "Responds strongly to discounts"),
]

# Demo-friendly segments (simple mode): concrete, explainable tags
SIMPLE_SEGMENTS: List[Tuple[str, str, str]] = [
    # Value (exclusive)
    ("Budget", "Value", "CustomerSegment = 'Budget' (or lower spend propensity)"),
    ("Mainstream", "Value", "CustomerSegment = 'Mainstream'"),
    ("Premium", "Value", "CustomerSegment = 'Premium' (or higher spend propensity)"),
    # Type (exclusive)
    ("Individual", "Type", "CustomerType = 'Individual'"),
    ("Organization", "Type", "CustomerType = 'Organization'"),
    # Lifecycle (exclusive-ish)
    ("New Customer", "Lifecycle", "Joined within last N months (validity.new_customer_months)"),
    ("Established", "Lifecycle", "Not new and not lapsed"),
    ("Lapsed", "Lifecycle", "CustomerEndDate is set (churned)"),
    # Optional tags
    ("VIP", "Loyalty", "Top loyalty tiers"),
    ("High Value", "Behavior", "CustomerWeight in top quantile"),
    ("Frequent Shopper", "Behavior", "CustomerWeight in top frequency quantile"),
]


# -----------------------------------------------------------------------------
# Helpers: deterministic hashing
# -----------------------------------------------------------------------------

def _to_int_key(x: Any) -> int:
    if x is None:
        return 0
    if isinstance(x, (int, np.integer)):
        return int(x)
    s = str(x)
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) & 0x7FFFFFFF
    return h


def _stable_u32(key: Any, seed: int, salt: int = 0) -> int:
    k = _to_int_key(key)
    x = (k ^ (seed * 0x9E3779B1) ^ (salt * 0x85EBCA6B)) & 0xFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= (x >> 17) & 0xFFFFFFFF
    x ^= (x << 5) & 0xFFFFFFFF
    return x & 0x7FFFFFFF


def _stable_float01(key: Any, seed: int, salt: int = 0) -> float:
    return _stable_u32(key, seed, salt) / float(0x7FFFFFFF)


# -----------------------------------------------------------------------------
# Helpers: dates/months
# -----------------------------------------------------------------------------

def _parse_iso_date(s: Optional[str]) -> date:
    """Parse an ISO date string, raising a clear error when *s* is None."""
    if s is None:
        raise ValueError(
            "Date string is None.  Ensure cfg.defaults.dates.start/end and any "
            "override dates are set before calling this function."
        )
    return pd.to_datetime(s).date()


def _month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def _month_end(d: date) -> date:
    """Return the last day of *d*'s month as a ``datetime.date``."""
    if d.month == 12:
        nm = date(d.year + 1, 1, 1)
    else:
        nm = date(d.year, d.month + 1, 1)
    # Use timedelta to stay in pure-date land (avoids returning Timestamp).
    return nm - timedelta(days=1)


def _iter_month_starts(start: date, end: date) -> List[date]:
    cur = _month_start(start)
    out: List[date] = []
    while cur <= end:
        out.append(cur)
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)
    return out


def _is_quarter_start(month_start_dt: date) -> bool:
    return month_start_dt.month in (1, 4, 7, 10)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class CustomerSegmentsCfg:
    enabled: bool = True
    mode: str = "scd2"  # "scd2" (default) | "simple"
    segment_count: int = 12
    segs_per_cust_min: int = 1
    segs_per_cust_max: int = 4
    include_score: bool = True
    include_primary_flag: bool = True
    include_validity: bool = True
    grain: str = "month"
    churn_rate_qtr: float = 0.08
    new_customer_months: int = 2
    seed: int = 123
    override_seed: Optional[int] = None
    override_start: Optional[str] = None
    override_end: Optional[str] = None


def _read_cfg(cfg: Dict[str, Any], global_dates: Dict[str, str]) -> CustomerSegmentsCfg:
    seg = cfg.customer_segments

    mode = str(seg.mode).strip().lower()
    if mode not in ("scd2", "simple"):
        mode = "scd2"

    override = seg.override
    override_dates: Dict[str, Any] = {}
    if override is not None:
        override_dates = override.dates or {}

    validity = seg.validity

    # Keep backwards compatibility: if include_validity not specified, default to True (since the cfg block usually includes it)
    include_validity = bool(seg.include_validity)

    grain = str(validity.grain if validity is not None else "month").lower()
    if grain not in ("month", "day"):
        grain = "month"

    seed = int(seg.seed if seg.seed is not None else 123)
    override_seed = override.seed if override is not None else None
    if override_seed is not None:
        try:
            override_seed = int(override_seed)
        except (TypeError, ValueError):
            override_seed = None

    start = override_dates.get("start") or global_dates.get("start")
    end = override_dates.get("end") or global_dates.get("end")

    # Validate that we actually have resolvable dates — fail early with a
    # clear message instead of deferring to _parse_iso_date(None).
    if not start or not end:
        raise ValueError(
            "Cannot resolve start/end dates for customer_segments.  "
            "Provide customer_segments.override.dates.start/end or "
            "ensure cfg.defaults.dates.start/end are set."
        )

    return CustomerSegmentsCfg(
        enabled=bool(seg.enabled),
        mode=mode,
        segment_count=int(seg.segment_count),
        segs_per_cust_min=int(seg.segments_per_customer_min),
        segs_per_cust_max=int(seg.segments_per_customer_max),
        include_score=bool(seg.include_score),
        include_primary_flag=bool(seg.include_primary_flag),
        include_validity=include_validity,
        grain=grain,
        churn_rate_qtr=float(validity.churn_rate_qtr if validity is not None else 0.08),
        new_customer_months=int(validity.new_customer_months if validity is not None else 2),
        seed=seed,
        override_seed=override_seed,
        override_start=str(start),
        override_end=str(end),
    )


# -----------------------------------------------------------------------------
# Public builders
# -----------------------------------------------------------------------------

def build_dim_customer_segment(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Output columns:
      SegmentKey (int), SegmentName (str), SegmentType (str), Definition (str), IsActiveFlag (int8)
    """
    seg_cfg = cfg.customer_segments

    mode = str(seg_cfg.mode).strip().lower()
    segment_count = int(seg_cfg.segment_count)

    base = SIMPLE_SEGMENTS if mode == "simple" else DEFAULT_SEGMENTS
    segs = list(base[:segment_count])

    # If user asks for more segments than we have defaults, generate filler.
    if segment_count > len(segs):
        for i in range(len(segs) + 1, segment_count + 1):
            segs.append((f"Segment {i}", "Custom", f"Auto-generated segment {i}"))

    rows = [
        {
            "SegmentKey": i,
            "SegmentName": name,
            "SegmentType": stype,
            "Definition": desc,
            "IsActiveFlag": np.int8(1),
        }
        for i, (name, stype, desc) in enumerate(segs, start=1)
    ]

    return pd.DataFrame(rows)


def build_bridge_customer_segment_membership(customers: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Output columns:
      CustomerKey, SegmentKey, ValidFromDate, ValidToDate,
      Score (optional float32), IsPrimaryFlag (optional int8)

    ValidFromDate/ValidToDate are ALWAYS present and are datetime64[ns].
    """
    if "CustomerKey" not in customers.columns:
        raise KeyError("customers DataFrame must contain 'CustomerKey'")

    defaults_dates = cfg.defaults.dates
    global_dates = {"start": defaults_dates.start, "end": defaults_dates.end}
    if not global_dates.get("start") or not global_dates.get("end"):
        raise KeyError("cfg.defaults.dates.start/end required")

    c = _read_cfg(cfg, global_dates)
    if not c.enabled:
        return pd.DataFrame(columns=["CustomerKey", "SegmentKey", "ValidFromDate", "ValidToDate"])

    seed = c.override_seed if c.override_seed is not None else c.seed
    start_dt = _parse_iso_date(c.override_start)
    end_dt = _parse_iso_date(c.override_end)

    # Build the dimension once and share with bridge builders.
    dim_seg = build_dim_customer_segment(cfg)

    if c.mode == "simple":
        return _build_bridge_simple(customers=customers, c=c, seed=seed, start_dt=start_dt, end_dt=end_dt, dim_seg=dim_seg)

    # Legacy SCD2-ish
    return _build_bridge_scd2(customers=customers, c=c, seed=seed, start_dt=start_dt, end_dt=end_dt, dim_seg=dim_seg)


# -----------------------------------------------------------------------------
# SIMPLE mode (demo-friendly, rule-based)
#
# Replaced iterrows with direct numpy-array iteration for ~5-10x speedup.
# All segment assignment logic is still column-derived; the inner loop is
# lightweight scalar/list operations only.
# -----------------------------------------------------------------------------

def _score_for_segment_name(seg_name: str, ck: Any, seed: int) -> float:
    """Deterministic score for a (customer, segment) pair in simple mode."""
    if seg_name in ("Budget", "Mainstream", "Premium"):
        return float(np.float32(0.70))
    if seg_name == "VIP":
        return float(np.float32(0.85))
    if seg_name == "High Value":
        return float(np.float32(0.80))
    if seg_name == "Frequent Shopper":
        return float(np.float32(0.78))
    if seg_name in ("New Customer", "Established", "Lapsed"):
        return float(np.float32(0.65))
    return float(np.float32(0.60 + (_stable_u32((ck, seg_name), seed, 4001) % 20) / 100.0))


def _build_bridge_simple(
    customers: pd.DataFrame,
    c: CustomerSegmentsCfg,
    seed: int,
    start_dt: date,
    end_dt: date,
    dim_seg: pd.DataFrame,
) -> pd.DataFrame:
    name_to_key: Dict[str, int] = dim_seg.set_index("SegmentName")["SegmentKey"].astype(int).to_dict()

    start_ts = pd.to_datetime(start_dt).normalize()
    end_ts = pd.to_datetime(end_dt).normalize()

    df = customers.copy()
    N = len(df)
    if N == 0:
        return _finalize_bridge_df(
            pd.DataFrame(columns=["CustomerKey", "SegmentKey", "ValidFromDate", "ValidToDate"]),
            include_score=c.include_score,
            include_primary=c.include_primary_flag,
        )

    ck_arr = df["CustomerKey"].to_numpy()

    # --- Value segment from Customers.CustomerSegment ---
    if "CustomerSegment" in df.columns:
        value_name = df["CustomerSegment"].astype(str).str.strip().str.title()
        value_name = value_name.where(value_name.isin(["Budget", "Mainstream", "Premium"]), "Mainstream")
    else:
        value_name = pd.Series("Mainstream", index=df.index)

    # --- Type from Customers.CustomerType ---
    if "CustomerType" in df.columns:
        type_name = df["CustomerType"].astype(str).str.strip().str.title()
        type_name = type_name.where(type_name.isin(["Individual", "Organization"]), "Individual")
    else:
        type_name = pd.Series("Individual", index=df.index)

    # --- Start/End for lifecycle + validity windows ---
    cust_start = pd.to_datetime(df.get("CustomerStartDate", start_ts), errors="coerce").dt.normalize().fillna(start_ts)
    cust_end_raw = pd.to_datetime(df.get("CustomerEndDate", pd.NaT), errors="coerce").dt.normalize()
    has_end = cust_end_raw.notna()
    cust_end = cust_end_raw.fillna(end_ts)

    # Clamp to global window
    cust_start = cust_start.clip(lower=start_ts, upper=end_ts)
    cust_end = cust_end.clip(lower=start_ts, upper=end_ts)
    cust_end = pd.Series(
        np.where(cust_end.to_numpy() < cust_start.to_numpy(), cust_start.to_numpy(), cust_end.to_numpy()),
        index=df.index,
    )
    cust_end = pd.to_datetime(cust_end).dt.normalize()

    # --- Lifecycle ---
    months_old = (end_ts.to_period("M").ordinal - cust_start.dt.to_period("M").ordinal).astype("int64")
    is_new = (~has_end) & (months_old <= max(int(c.new_customer_months) - 1, 0))
    lifecycle_name = pd.Series("Established", index=df.index)
    lifecycle_name[is_new] = "New Customer"
    lifecycle_name[has_end] = "Lapsed"

    # --- Optional extra tags ---
    weight = pd.to_numeric(df.get("CustomerWeight", np.nan), errors="coerce")

    w_q = float(weight.dropna().quantile(0.85)) if weight.notna().any() else np.inf
    # "Frequent Shopper" uses a slightly lower quantile of the same weight
    freq_q = float(weight.dropna().quantile(0.80)) if weight.notna().any() else np.inf

    is_high_value = (weight.notna() & (weight >= w_q)).to_numpy()
    is_frequent = (weight.notna() & (weight >= freq_q)).to_numpy()

    vip = pd.Series(False, index=df.index)
    if "LoyaltyTierKey" in df.columns:
        tiers = pd.to_numeric(df["LoyaltyTierKey"], errors="coerce").dropna().astype("int64")
        if len(tiers) > 0:
            uniq = np.sort(tiers.unique())
            top = set(uniq[-min(2, len(uniq)):].tolist())
            vip = pd.to_numeric(df["LoyaltyTierKey"], errors="coerce").fillna(-1).astype("int64").isin(top)
    vip_arr = vip.to_numpy()

    # --- Per-customer k (how many segments) ---
    min_k = max(int(c.segs_per_cust_min), 1)
    max_k = max(min_k, int(c.segs_per_cust_max))

    # Pre-extract numpy arrays to avoid iloc/loc inside the loop.
    value_arr = value_name.to_numpy()
    type_arr = type_name.to_numpy()
    lifecycle_arr = lifecycle_name.to_numpy()
    cust_start_arr = cust_start.to_numpy()
    cust_end_arr = cust_end.to_numpy()

    out_rows: List[Dict[str, Any]] = []

    for i in range(N):
        ck = ck_arr[i]

        base = [str(value_arr[i]), str(type_arr[i]), str(lifecycle_arr[i])]

        extras: List[str] = []
        if vip_arr[i]:
            extras.append("VIP")
        if is_high_value[i]:
            extras.append("High Value")
        if is_frequent[i]:
            extras.append("Frequent Shopper")

        # Deterministic shuffle of extras so it doesn't look too repetitive
        extras.sort(key=lambda nm: _stable_u32((ck, nm), seed, 5001))

        # Determine k for this customer deterministically
        span = max_k - min_k + 1
        k = min_k + (int(_stable_u32(ck, seed, 6001) % span) if span > 1 else 0)

        chosen_names = base + extras
        # Deduplicate while preserving order
        seen: set = set()
        deduped: List[str] = []
        for x in chosen_names:
            if x not in seen:
                seen.add(x)
                deduped.append(x)
        chosen_names = deduped[:k]

        primary_name = base[0]  # value segment is primary
        primary_sk = name_to_key.get(primary_name)

        # Validity: in simple mode, we keep it intuitive but cheap:
        # - default validity: customer start->end if include_validity else global start->end
        if c.include_validity:
            v_from_base = cust_start_arr[i]
            v_to_base = cust_end_arr[i]
        else:
            v_from_base = start_ts
            v_to_base = end_ts

        for seg_name in chosen_names:
            sk = name_to_key.get(seg_name)
            if sk is None:
                continue

            # Special-case: New Customer covers only first N months if validity enabled
            v_from, v_to = v_from_base, v_to_base
            if c.include_validity and seg_name == "New Customer":
                n = max(int(c.new_customer_months), 0)
                v_from = cust_start_arr[i]
                v_to = (pd.Timestamp(v_from) + pd.DateOffset(months=n) - pd.Timedelta(days=1)).normalize()
                if v_to > v_to_base:
                    v_to = v_to_base
            elif c.include_validity and seg_name == "Lapsed":
                v_from = cust_end_arr[i]
                v_to = end_ts

            out_rows.append(
                _membership_row(
                    ck=ck,
                    sk=int(sk),
                    from_date=v_from,
                    to_date=v_to,
                    primary_sk=int(primary_sk) if primary_sk is not None else int(sk),
                    score=_score_for_segment_name(seg_name, ck, seed) if c.include_score else None,
                    include_primary=c.include_primary_flag,
                    include_score=c.include_score,
                )
            )

    return _finalize_bridge_df(pd.DataFrame(out_rows), include_score=c.include_score, include_primary=c.include_primary_flag)


# -----------------------------------------------------------------------------
# SCD2-ish mode (legacy) with a fast-path when include_validity=False
# -----------------------------------------------------------------------------

def _build_bridge_scd2(
    customers: pd.DataFrame,
    c: CustomerSegmentsCfg,
    seed: int,
    start_dt: date,
    end_dt: date,
    dim_seg: pd.DataFrame,
) -> pd.DataFrame:
    month_starts = _iter_month_starts(start_dt, end_dt)
    if not month_starts:
        return pd.DataFrame(columns=["CustomerKey", "SegmentKey", "ValidFromDate", "ValidToDate"])

    # month boundaries as datetime64[ns] arrays (fast indexing)
    ms_ts = pd.to_datetime(pd.Series(month_starts)).dt.normalize().to_numpy(dtype="datetime64[ns]")
    me_ts = pd.to_datetime(pd.Series([_month_end(ms) for ms in month_starts])).dt.normalize().to_numpy(dtype="datetime64[ns]")

    name_to_key: Dict[str, int] = dim_seg.set_index("SegmentName")["SegmentKey"].astype(int).to_dict()
    new_customer_key = name_to_key.get("New Customer")

    cust_keys = customers["CustomerKey"].tolist()

    # -----------------------------------------------------------------
    # Fast-path: include_validity=False → single interval per segment
    # -----------------------------------------------------------------
    if not c.include_validity:
        seg_keys_all = list(range(1, c.segment_count + 1))
        k_min = max(0, c.segs_per_cust_min)
        k_max = max(k_min, min(c.segs_per_cust_max, c.segment_count))

        fp_start_ts = pd.to_datetime(start_dt).normalize()
        fp_end_ts = pd.to_datetime(end_dt).normalize()

        fast_rows: List[Dict[str, Any]] = []
        for ck in cust_keys:
            base_h = _stable_u32(ck, seed, 100)
            k = k_min + (base_h % (k_max - k_min + 1)) if k_max >= k_min else k_min

            chosen: List[int] = []
            for i in range(1, k + 1):
                idx = (base_h + i * 17) % c.segment_count
                sk = seg_keys_all[idx]
                if sk not in chosen:
                    chosen.append(sk)

            if not chosen:
                continue

            primary_sk = chosen[0]
            base_set = set(chosen)
            base_set.add(primary_sk)

            for sk in base_set:
                fast_rows.append(
                    _membership_row(
                        ck=ck,
                        sk=int(sk),
                        from_date=fp_start_ts,
                        to_date=fp_end_ts,
                        primary_sk=int(primary_sk),
                        score=(float(np.float32(0.50 + (_stable_u32((ck, sk), seed, 777) % 50) / 100.0)) if c.include_score else None),
                        include_primary=c.include_primary_flag,
                        include_score=c.include_score,
                    )
                )

        return _finalize_bridge_df(pd.DataFrame(fast_rows), include_score=c.include_score, include_primary=c.include_primary_flag)

    # -----------------------------------------------------------------
    # include_validity=True: legacy churn simulation
    # -----------------------------------------------------------------
    join_candidates = ["JoinDateKey", "CustomerStartDateKey", "StartDateKey", "CreatedDateKey"]
    join_col = next((col for col in join_candidates if col in customers.columns), None)

    # Build year-month -> month index map for faster join month lookup
    ym_to_idx = {(ms.year * 100 + ms.month): i for i, ms in enumerate(month_starts)}

    join_month_idx: Dict[Any, int] = {}
    if join_col is not None:
        join_keys = customers.set_index("CustomerKey")[join_col].to_dict()
        for ck in cust_keys:
            jk = join_keys.get(ck)
            if pd.isna(jk) or jk is None:
                join_month_idx[ck] = int(_stable_u32(ck, seed, 9001) % len(month_starts))
                continue
            jk = int(jk)
            ym = (jk // 100)  # YYYYMM
            join_month_idx[ck] = ym_to_idx.get(ym, int(_stable_u32(ck, seed, 9001) % len(month_starts)))
    else:
        for ck in cust_keys:
            join_month_idx[ck] = int(_stable_u32(ck, seed, 9001) % len(month_starts))

    scd2_rows: List[Dict[str, Any]] = []

    seg_keys_all = list(range(1, c.segment_count + 1))
    k_min = max(0, c.segs_per_cust_min)
    k_max = max(k_min, min(c.segs_per_cust_max, c.segment_count))

    for ck in cust_keys:
        base_h = _stable_u32(ck, seed, 100)
        k = k_min + (base_h % (k_max - k_min + 1))

        chosen: List[int] = []
        for i in range(1, k + 1):
            idx = (base_h + i * 17) % c.segment_count
            sk = seg_keys_all[idx]
            if sk not in chosen:
                chosen.append(sk)

        primary_sk = chosen[0] if chosen else 1
        base_set = set(chosen)
        base_set.add(primary_sk)

        score_by_seg: Dict[int, float] = {}
        if c.include_score:
            extra = {int(new_customer_key)} if new_customer_key is not None else set()
            for sk in base_set.union(extra):
                score_by_seg[int(sk)] = float(np.float32(0.50 + (_stable_u32((ck, sk), seed, 777) % 50) / 100.0))

        active_set: set[int] = set()
        start_month_for_seg: Dict[int, int] = {}
        cur_base_set = set(base_set)

        jm0 = join_month_idx[ck]
        new_window = set(range(jm0, min(len(month_starts), jm0 + max(0, c.new_customer_months)))) if (new_customer_key and c.new_customer_months > 0) else set()

        for mi, ms in enumerate(month_starts):
            if mi > 0 and _is_quarter_start(ms):
                if _stable_float01(ck, seed, salt=10_000 + mi) < c.churn_rate_qtr:
                    secondaries = [s for s in cur_base_set if s != primary_sk]
                    if secondaries:
                        victim = secondaries[int(_stable_u32(ck, seed, 20_000 + mi) % len(secondaries))]
                        cand_start = int(_stable_u32(ck, seed, 30_000 + mi) % c.segment_count)
                        repl = None
                        for j in range(c.segment_count):
                            sk = seg_keys_all[(cand_start + j) % c.segment_count]
                            if sk != primary_sk and sk not in cur_base_set:
                                repl = sk
                                break
                        if repl is not None:
                            cur_base_set.remove(victim)
                            cur_base_set.add(repl)
                            if c.include_score and repl not in score_by_seg:
                                score_by_seg[repl] = float(np.float32(0.50 + (_stable_u32((ck, repl), seed, 888) % 50) / 100.0))

            desired = set(cur_base_set)

            if new_customer_key is not None:
                if mi in new_window:
                    desired.add(int(new_customer_key))
                else:
                    desired.discard(int(new_customer_key))

            for sk in desired - active_set:
                start_month_for_seg[sk] = mi

            for sk in active_set - desired:
                smi = start_month_for_seg.pop(sk, None)
                if smi is None:
                    continue
                scd2_rows.append(
                    _membership_row(
                        ck=ck,
                        sk=sk,
                        from_date=ms_ts[smi],
                        to_date=me_ts[mi - 1] if mi - 1 >= 0 else me_ts[smi],
                        primary_sk=primary_sk,
                        score=score_by_seg.get(sk) if c.include_score else None,
                        include_primary=c.include_primary_flag,
                        include_score=c.include_score,
                    )
                )

            active_set = desired

        last_mi = len(month_starts) - 1
        for sk in list(active_set):
            smi = start_month_for_seg.get(sk)
            if smi is None:
                continue
            scd2_rows.append(
                _membership_row(
                    ck=ck,
                    sk=sk,
                    from_date=ms_ts[smi],
                    to_date=me_ts[last_mi],
                    primary_sk=primary_sk,
                    score=score_by_seg.get(sk) if c.include_score else None,
                    include_primary=c.include_primary_flag,
                    include_score=c.include_score,
                )
            )

    return _finalize_bridge_df(pd.DataFrame(scd2_rows), include_score=c.include_score, include_primary=c.include_primary_flag)


# -----------------------------------------------------------------------------
# Row builder + finalization
# -----------------------------------------------------------------------------

def _membership_row(
    ck: Any,
    sk: int,
    from_date,
    to_date,
    primary_sk: int,
    score: Optional[float],
    include_primary: bool,
    include_score: bool,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "CustomerKey": int(_to_int_key(ck)),
        "SegmentKey": int(sk),
        "ValidFromDate": pd.to_datetime(from_date).normalize(),
        "ValidToDate": pd.to_datetime(to_date).normalize(),
    }
    if include_primary:
        row["IsPrimaryFlag"] = np.int8(1 if int(sk) == int(primary_sk) else 0)
    if include_score:
        row["Score"] = float(score) if score is not None else float(np.float32(0.60))
    return row


def _finalize_bridge_df(df: pd.DataFrame, *, include_score: bool, include_primary: bool) -> pd.DataFrame:
    if df.empty:
        cols = ["CustomerKey", "SegmentKey", "ValidFromDate", "ValidToDate"]
        if include_primary:
            cols.append("IsPrimaryFlag")
        if include_score:
            cols.append("Score")
        return pd.DataFrame(columns=cols)

    df["CustomerKey"] = df["CustomerKey"].astype("int64")
    df["SegmentKey"] = df["SegmentKey"].astype("int32")
    df["ValidFromDate"] = pd.to_datetime(df["ValidFromDate"]).dt.normalize()
    df["ValidToDate"] = pd.to_datetime(df["ValidToDate"]).dt.normalize()

    if include_primary and "IsPrimaryFlag" in df.columns:
        df["IsPrimaryFlag"] = df["IsPrimaryFlag"].astype("int8")
    if include_score and "Score" in df.columns:
        df["Score"] = df["Score"].astype("float32")
    return df


# -----------------------------------------------------------------------------
# Runner (I/O + skip logic) — unchanged output names
# -----------------------------------------------------------------------------

def _pick_existing_file(folder: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        p = folder / name
        if p.exists():
            return p
    return None


def _resolve_out_path(parquet_dims_folder: Path, p: str | None, default_name: str) -> Path:
    if not p:
        return parquet_dims_folder / default_name
    pp = Path(p)
    return pp if pp.is_absolute() else (parquet_dims_folder / pp)


def _safe_read_customers(parquet_path: Path, desired_cols: List[str]) -> pd.DataFrame:
    """
    Read only available desired_cols from a Customers parquet.
    Uses pyarrow schema probe (metadata-only) to determine available columns,
    falling back to a full read only when pyarrow is unavailable.
    """
    desired_cols = [c for c in desired_cols if isinstance(c, str) and c]
    if not desired_cols:
        return pd.read_parquet(parquet_path)

    try:
        import pyarrow.parquet as pq  # type: ignore
        schema = pq.read_schema(parquet_path)
        available = set(schema.names)
    except Exception:
        # pyarrow schema probe failed; fall back to reading everything.
        return pd.read_parquet(parquet_path)

    cols = [c for c in desired_cols if c in available]
    if not cols:
        return pd.read_parquet(parquet_path)
    return pd.read_parquet(parquet_path, columns=cols)


def _columns_needed_for_bridge(mode: str) -> List[str]:
    cols = ["CustomerKey"]
    if mode == "simple":
        cols += [
            "CustomerType",
            "CustomerStartDate",
            "CustomerEndDate",
            "LoyaltyTierKey",
            "CustomerWeight",
        ]
    else:  # scd2
        cols += [
            # join date key candidates
            "JoinDateKey",
            "CustomerStartDateKey",
            "StartDateKey",
            "CreatedDateKey",
        ]
    # de-dupe while preserving order
    seen: set = set()
    out: List[str] = []
    for c in cols:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def run_customer_segments(cfg: dict, parquet_dims_folder: Path) -> dict:
    """
    Generates:
      - customer_segment.parquet
      - customer_segment_membership.parquet (unless generate_bridge: false)

    Depends on:
      - Customers parquet already generated in parquet_dims_folder.
    """
    parquet_dims_folder = Path(parquet_dims_folder)

    seg_cfg = cfg.customer_segments
    enabled = bool(seg_cfg.enabled)
    generate_bridge = bool(seg_cfg.generate_bridge)

    if not enabled:
        skip("Customer segments disabled; skipping.")
        return {"_regenerated": False, "reason": "disabled"}

    override = seg_cfg.override
    if override is not None:
        override_paths = override.paths or {}
    else:
        override_paths = {}

    customers_path = override_paths.get("customers")
    if customers_path:
        customers_fp = Path(customers_path)
        if not customers_fp.is_absolute():
            customers_fp = parquet_dims_folder / customers_fp
    else:
        customers_fp = _pick_existing_file(
            parquet_dims_folder,
            candidates=[
                "Customers.parquet",
                "customers.parquet",
                "DimCustomer.parquet",
                "dim_customer.parquet",
            ],
        )

    if not customers_fp or not customers_fp.exists():
        raise FileNotFoundError(
            "Customers parquet not found. "
            "Expected one of: Customers.parquet / customers.parquet / DimCustomer.parquet. "
            "Or set customer_segments.override.paths.customers"
        )

    dim_out = _resolve_out_path(
        parquet_dims_folder,
        override_paths.get("customer_segment") or override_paths.get("dim_customer_segment"),
        "customer_segment.parquet",
    )
    bridge_out = _resolve_out_path(
        parquet_dims_folder,
        override_paths.get("customer_segment_membership") or override_paths.get("bridge_customer_segment_membership"),
        "customer_segment_membership.parquet",
    )

    # Version signature includes config + upstream Customers file sig
    seg_cfg_for_version = dict(seg_cfg)
    st = os.stat(customers_fp)
    seg_cfg_for_version["upstream_customers_sig"] = {"path": str(customers_fp), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}
    dd = cfg.defaults.dates
    seg_cfg_for_version["upstream_global_dates"] = {"start": dd.start, "end": dd.end}
    seg_cfg_for_version["_schema_version"] = 2  # mode support + scd2 fast-path + simple mode

    # --- Version skip (bridge-aware) ---
    version_files_exist = dim_out.exists() and (bridge_out.exists() or not generate_bridge)
    if version_files_exist and (not should_regenerate("customer_segments", seg_cfg_for_version, dim_out)):
        # Clean up stale bridge from a previous run where generate_bridge was true
        if not generate_bridge and bridge_out.exists():
            bridge_out.unlink()
            info("Removed stale customer_segment_membership bridge file.")
        skip("Customer segments up-to-date")
        return {"_regenerated": False, "reason": "version"}

    # Parse global dates + cfg to decide which customer columns to read
    gd = cfg.defaults.dates
    global_dates = {"start": gd.start, "end": gd.end}
    if not global_dates.get("start") or not global_dates.get("end"):
        raise KeyError("cfg.defaults.dates.start/end required")

    c = _read_cfg(cfg, global_dates)
    cols = _columns_needed_for_bridge(c.mode)

    info(f"Loading Customers from: {customers_fp}")
    customers = _safe_read_customers(customers_fp, cols)

    # Filter to customers active in sales (if column exists)
    customers_for_bridge = customers
    if "IsActiveInSales" in customers_for_bridge.columns:
        customers_for_bridge = customers_for_bridge[customers_for_bridge["IsActiveInSales"].fillna(0).astype("int64") == 1].copy()

    dim_seg = build_dim_customer_segment(cfg)

    dim_out.parent.mkdir(parents=True, exist_ok=True)
    dim_seg.to_parquet(dim_out, index=False)
    done(f"Wrote customer_segment: {dim_out.name} ({len(dim_seg):,} rows)")

    if generate_bridge:
        bridge = build_bridge_customer_segment_membership(customers=customers_for_bridge, cfg=cfg)
        bridge_out.parent.mkdir(parents=True, exist_ok=True)
        bridge.to_parquet(bridge_out, index=False)
        save_version("customer_segments", seg_cfg_for_version, bridge_out)
        done(f"Wrote customer_segment_membership: {bridge_out.name} ({len(bridge):,} rows)")
    else:
        skip("customer_segment_membership bridge skipped (generate_bridge: false)")
        if bridge_out.exists():
            bridge_out.unlink()

    return {"_regenerated": True, "dim": str(dim_out), "bridge": str(bridge_out) if generate_bridge else None}