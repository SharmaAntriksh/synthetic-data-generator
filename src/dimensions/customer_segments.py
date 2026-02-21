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
  _force_regenerate: false

Notes:
- "simple" mode is recommended for demos: stable, explainable, low volume.
- The bridge output always includes ValidFromDate/ValidToDate as datetime64[ns].
"""

from dataclasses import dataclass
from datetime import date
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
    ("Frequent Shopper", "Behavior", "CustomerTemperature in top quantile"),
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

def _parse_iso_date(s: str) -> date:
    return pd.to_datetime(s).date()


def _month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def _month_end(d: date) -> date:
    if d.month == 12:
        nm = date(d.year + 1, 1, 1)
    else:
        nm = date(d.year, d.month + 1, 1)
    return nm - pd.Timedelta(days=1)  # type: ignore[arg-type]


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
    seg = cfg.get("customer_segments") or {}
    if not isinstance(seg, dict):
        seg = {}

    mode = str(seg.get("mode", "scd2")).strip().lower()
    if mode not in ("scd2", "simple"):
        mode = "scd2"

    override = seg.get("override") or {}
    if not isinstance(override, dict):
        override = {}

    override_dates = override.get("dates") or {}
    if not isinstance(override_dates, dict):
        override_dates = {}

    validity = seg.get("validity") or {}
    if not isinstance(validity, dict):
        validity = {}

    # Keep backwards compatibility: if include_validity not specified, default to True (since the cfg block usually includes it)
    include_validity = bool(seg.get("include_validity", True))

    grain = str(validity.get("grain", "month")).lower()
    if grain not in ("month", "day"):
        grain = "month"

    seed = int(seg.get("seed", 123))
    override_seed = override.get("seed")
    if override_seed is not None:
        try:
            override_seed = int(override_seed)
        except Exception:
            override_seed = None

    start = override_dates.get("start") or global_dates.get("start")
    end = override_dates.get("end") or global_dates.get("end")

    return CustomerSegmentsCfg(
        enabled=bool(seg.get("enabled", True)),
        mode=mode,
        segment_count=int(seg.get("segment_count", 12)),
        segs_per_cust_min=int(seg.get("segments_per_customer_min", 1)),
        segs_per_cust_max=int(seg.get("segments_per_customer_max", 4)),
        include_score=bool(seg.get("include_score", True)),
        include_primary_flag=bool(seg.get("include_primary_flag", True)),
        include_validity=include_validity,
        grain=grain,
        churn_rate_qtr=float(validity.get("churn_rate_qtr", 0.08)),
        new_customer_months=int(validity.get("new_customer_months", 2)),
        seed=seed,
        override_seed=override_seed,
        override_start=str(start) if start else None,
        override_end=str(end) if end else None,
    )


# -----------------------------------------------------------------------------
# Public builders
# -----------------------------------------------------------------------------

def build_dim_customer_segment(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Output columns:
      SegmentKey (int), SegmentName (str), SegmentType (str), Definition (str), IsActiveFlag (int8)
    """
    seg_cfg = cfg.get("customer_segments") or {}
    if not isinstance(seg_cfg, dict):
        seg_cfg = {}

    mode = str(seg_cfg.get("mode", "scd2")).strip().lower()
    segment_count = int(seg_cfg.get("segment_count", 12))

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

    global_dates = (cfg.get("defaults", {}) or {}).get("dates", {})
    if not isinstance(global_dates, dict) or not global_dates.get("start") or not global_dates.get("end"):
        raise KeyError("cfg.defaults.dates.start/end required")

    c = _read_cfg(cfg, global_dates)
    if not c.enabled:
        return pd.DataFrame(columns=["CustomerKey", "SegmentKey", "ValidFromDate", "ValidToDate"])

    seed = c.override_seed if c.override_seed is not None else c.seed
    start_dt = _parse_iso_date(c.override_start)  # type: ignore[arg-type]
    end_dt = _parse_iso_date(c.override_end)      # type: ignore[arg-type]

    if c.mode == "simple":
        return _build_bridge_simple(customers=customers, c=c, seed=seed, start_dt=start_dt, end_dt=end_dt)

    # Legacy SCD2-ish
    return _build_bridge_scd2(customers=customers, c=c, seed=seed, start_dt=start_dt, end_dt=end_dt)


# -----------------------------------------------------------------------------
# SIMPLE mode (demo-friendly, rule-based)
# -----------------------------------------------------------------------------

def _build_bridge_simple(
    customers: pd.DataFrame,
    c: CustomerSegmentsCfg,
    seed: int,
    start_dt: date,
    end_dt: date,
) -> pd.DataFrame:
    dim_seg = build_dim_customer_segment({"customer_segments": {"mode": "simple", "segment_count": c.segment_count}})
    name_to_key = {r["SegmentName"]: int(r["SegmentKey"]) for _, r in dim_seg.iterrows()}

    start_ts = pd.to_datetime(start_dt).normalize()
    end_ts = pd.to_datetime(end_dt).normalize()

    df = customers.copy()

    # Value segment from Customers.CustomerSegment (Budget/Mainstream/Premium)
    if "CustomerSegment" in df.columns:
        value_name = df["CustomerSegment"].astype(str).str.strip().str.title()
        value_name = value_name.where(value_name.isin(["Budget", "Mainstream", "Premium"]), "Mainstream")
    else:
        value_name = pd.Series("Mainstream", index=df.index)

    # Type from Customers.CustomerType
    if "CustomerType" in df.columns:
        type_name = df["CustomerType"].astype(str).str.strip().str.title()
        type_name = type_name.where(type_name.isin(["Individual", "Organization"]), "Individual")
    else:
        type_name = pd.Series("Individual", index=df.index)

    # Start/End for lifecycle + validity windows
    cust_start = pd.to_datetime(df.get("CustomerStartDate", start_ts), errors="coerce").dt.normalize().fillna(start_ts)
    cust_end_raw = pd.to_datetime(df.get("CustomerEndDate", pd.NaT), errors="coerce").dt.normalize()
    has_end = cust_end_raw.notna()
    cust_end = cust_end_raw.fillna(end_ts)

    # Clamp to global window
    cust_start = cust_start.clip(lower=start_ts, upper=end_ts)
    cust_end = cust_end.clip(lower=start_ts, upper=end_ts)
    cust_end = pd.Series(np.where(cust_end.to_numpy() < cust_start.to_numpy(), cust_start.to_numpy(), cust_end.to_numpy()), index=df.index)
    cust_end = pd.to_datetime(cust_end).dt.normalize()

    # Lifecycle: New/Established/Lapsed (as-of end date)
    months_old = (end_ts.to_period("M").ordinal - cust_start.dt.to_period("M").ordinal).astype("int64")
    is_new = (~has_end) & (months_old <= max(int(c.new_customer_months) - 1, 0))
    lifecycle_name = pd.Series("Established", index=df.index)
    lifecycle_name[is_new] = "New Customer"
    lifecycle_name[has_end] = "Lapsed"

    # Optional extra tags (derive thresholds from present columns)
    weight = pd.to_numeric(df.get("CustomerWeight", np.nan), errors="coerce")
    temp = pd.to_numeric(df.get("CustomerTemperature", np.nan), errors="coerce")

    w_q = float(weight.dropna().quantile(0.85)) if weight.notna().any() else np.inf
    t_q = float(temp.dropna().quantile(0.80)) if temp.notna().any() else np.inf

    is_high_value = weight.notna() & (weight >= w_q)
    is_frequent = temp.notna() & (temp >= t_q)

    vip = pd.Series(False, index=df.index)
    if "LoyaltyTierKey" in df.columns:
        tiers = pd.to_numeric(df["LoyaltyTierKey"], errors="coerce").dropna().astype("int64")
        if len(tiers) > 0:
            uniq = np.sort(tiers.unique())
            top = set(uniq[-min(2, len(uniq)) :].tolist())
            vip = pd.to_numeric(df["LoyaltyTierKey"], errors="coerce").fillna(-1).astype("int64").isin(top)

    # Decide how many tags per customer (bounded)
    min_k = max(int(c.segs_per_cust_min), 1)
    max_k = max(min_k, int(c.segs_per_cust_max))

    out_rows: List[Dict[str, Any]] = []

    # Score helpers: intuitive, stable
    def score_for(seg_name: str, ck: Any) -> float:
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

    for i, row in df.reset_index(drop=True).iterrows():
        ck = row["CustomerKey"]

        base = [
            str(value_name.iloc[i]),
            str(type_name.iloc[i]),
            str(lifecycle_name.iloc[i]),
        ]

        extras: List[str] = []
        if bool(vip.iloc[i]):
            extras.append("VIP")
        if bool(is_high_value.iloc[i]):
            extras.append("High Value")
        if bool(is_frequent.iloc[i]):
            extras.append("Frequent Shopper")

        # Deterministic shuffle of extras so it doesn't look too repetitive
        extras.sort(key=lambda nm: _stable_u32((ck, nm), seed, 5001))

        # Determine k for this customer deterministically
        span = max_k - min_k + 1
        k = min_k + (int(_stable_u32(ck, seed, 6001) % span) if span > 1 else 0)

        chosen_names = base + extras
        # Deduplicate while preserving order
        seen = set()
        chosen_names = [x for x in chosen_names if not (x in seen or seen.add(x))]  # type: ignore[arg-type]
        chosen_names = chosen_names[:k]

        primary_name = base[0]  # value segment is primary

        # Validity: in simple mode, we keep it intuitive but cheap:
        # - default validity: customer start->end if include_validity else global start->end
        if c.include_validity:
            v_from_base = cust_start.iloc[i]
            v_to_base = cust_end.iloc[i]
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
                v_from = cust_start.iloc[i]
                v_to = (v_from + pd.DateOffset(months=n) - pd.Timedelta(days=1)).normalize()
                if v_to > v_to_base:
                    v_to = v_to_base
            elif c.include_validity and seg_name == "Lapsed":
                v_from = cust_end.iloc[i]
                v_to = end_ts

            out_rows.append(
                _membership_row(
                    ck=ck,
                    sk=int(sk),
                    from_date=v_from,
                    to_date=v_to,
                    primary_sk=int(name_to_key[primary_name]),
                    score=score_for(seg_name, ck) if c.include_score else None,
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
) -> pd.DataFrame:
    month_starts = _iter_month_starts(start_dt, end_dt)
    if not month_starts:
        return pd.DataFrame(columns=["CustomerKey", "SegmentKey", "ValidFromDate", "ValidToDate"])

    # month boundaries as datetime64[ns] arrays (fast indexing)
    ms_ts = pd.to_datetime(pd.Series(month_starts)).dt.normalize().to_numpy(dtype="datetime64[ns]")
    me_ts = pd.to_datetime(pd.Series([_month_end(ms) for ms in month_starts])).dt.normalize().to_numpy(dtype="datetime64[ns]")

    dim_seg = build_dim_customer_segment({"customer_segments": {"mode": "scd2", "segment_count": c.segment_count}})
    name_to_key = {r["SegmentName"]: int(r["SegmentKey"]) for _, r in dim_seg.iterrows()}
    new_customer_key = name_to_key.get("New Customer")

    cust_keys = customers["CustomerKey"].tolist()

    # If include_validity is False: emit one interval per segment (no month loop)
    if not c.include_validity:
        seg_keys_all = list(range(1, c.segment_count + 1))
        k_min = max(0, c.segs_per_cust_min)
        k_max = max(k_min, min(c.segs_per_cust_max, c.segment_count))

        start_ts = pd.to_datetime(start_dt).normalize()
        end_ts = pd.to_datetime(end_dt).normalize()

        out_rows: List[Dict[str, Any]] = []
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
                out_rows.append(
                    _membership_row(
                        ck=ck,
                        sk=int(sk),
                        from_date=start_ts,
                        to_date=end_ts,
                        primary_sk=int(primary_sk),
                        score=(float(np.float32(0.50 + (_stable_u32((ck, sk), seed, 777) % 50) / 100.0)) if c.include_score else None),
                        include_primary=c.include_primary_flag,
                        include_score=c.include_score,
                    )
                )

        return _finalize_bridge_df(pd.DataFrame(out_rows), include_score=c.include_score, include_primary=c.include_primary_flag)

    # include_validity=True legacy churn simulation
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

    out_rows: List[Dict[str, Any]] = []

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
                out_rows.append(
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
            out_rows.append(
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

    return _finalize_bridge_df(pd.DataFrame(out_rows), include_score=c.include_score, include_primary=c.include_primary_flag)


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
    Uses pyarrow schema if available to avoid reading full data twice.
    """
    desired_cols = [c for c in desired_cols if isinstance(c, str) and c]
    if not desired_cols:
        return pd.read_parquet(parquet_path)

    try:
        import pyarrow.parquet as pq  # type: ignore
        schema = pq.read_schema(parquet_path)
        available = set(schema.names)
        cols = [c for c in desired_cols if c in available]
        if not cols:
            return pd.read_parquet(parquet_path)
        return pd.read_parquet(parquet_path, columns=cols)
    except Exception:
        # fallback: try columns directly; pandas will raise if missing, so we filter by a cheap read of zero-row
        try:
            df0 = pd.read_parquet(parquet_path, engine="pyarrow").head(0)
            available = set(df0.columns)
            cols = [c for c in desired_cols if c in available]
            if not cols:
                return pd.read_parquet(parquet_path)
            return pd.read_parquet(parquet_path, columns=cols)
        except Exception:
            return pd.read_parquet(parquet_path)
def _columns_needed_for_bridge(mode: str) -> List[str]:
    cols = ["CustomerKey", "IsActiveInSales"]
    if mode == "simple":
        cols += [
            "CustomerSegment",
            "CustomerType",
            "CustomerStartDate",
            "CustomerEndDate",
            "LoyaltyTierKey",
            "CustomerWeight",
            "CustomerTemperature",
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
    seen = set()
    out = []
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
      - customer_segment_membership.parquet

    Depends on:
      - Customers parquet already generated in parquet_dims_folder.
    """
    parquet_dims_folder = Path(parquet_dims_folder)

    seg_cfg = cfg.get("customer_segments") if isinstance(cfg.get("customer_segments"), dict) else {}
    enabled = bool(seg_cfg.get("enabled", False))
    force = bool(seg_cfg.get("_force_regenerate", False))

    if not enabled:
        skip("Customer segments disabled; skipping.")
        return {"_regenerated": False, "reason": "disabled"}

    override = seg_cfg.get("override") if isinstance(seg_cfg.get("override"), dict) else {}
    override_paths = override.get("paths") if isinstance(override.get("paths"), dict) else {}

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
    defaults_dates = (cfg.get("defaults") or {}).get("dates") or {}
    if isinstance(defaults_dates, dict):
        seg_cfg_for_version["upstream_global_dates"] = {"start": defaults_dates.get("start"), "end": defaults_dates.get("end")}
    seg_cfg_for_version["_schema_version"] = 2  # mode support + scd2 fast-path + simple mode

    if not force:
        if dim_out.exists() and bridge_out.exists() and (not should_regenerate("customer_segments", seg_cfg_for_version, bridge_out)):
            skip("Customer segments up-to-date; skipping.")
            return {"_regenerated": False, "reason": "version"}

    # Parse global dates + cfg to decide which customer columns to read
    global_dates = (cfg.get("defaults", {}) or {}).get("dates", {})
    if not isinstance(global_dates, dict) or not global_dates.get("start") or not global_dates.get("end"):
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
    bridge = build_bridge_customer_segment_membership(customers=customers_for_bridge, cfg=cfg)

    dim_out.parent.mkdir(parents=True, exist_ok=True)
    bridge_out.parent.mkdir(parents=True, exist_ok=True)

    dim_seg.to_parquet(dim_out, index=False)
    bridge.to_parquet(bridge_out, index=False)

    save_version("customer_segments", seg_cfg_for_version, bridge_out)

    done(f"Wrote customer_segment: {dim_out.name} ({len(dim_seg):,} rows)")
    done(f"Wrote customer_segment_membership: {bridge_out.name} ({len(bridge):,} rows)")
    return {"_regenerated": True, "dim": str(dim_out), "bridge": str(bridge_out)}
