from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from pathlib import Path
import os 

from src.utils.logging_utils import done, skip, info
from src.versioning.version_store import should_regenerate
from src.versioning.version_store import save_version
# -----------------------------
# Segment definitions
# -----------------------------

# You can extend/replace this list later; SegmentKey will be 1..segment_count
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


# -----------------------------
# Helpers: deterministic hashing
# -----------------------------

def _to_int_key(x: Any) -> int:
    """Best-effort stable conversion for keys that might be int-like or string-like."""
    if x is None:
        return 0
    if isinstance(x, (int, np.integer)):
        return int(x)
    s = str(x)
    # simple stable polynomial rolling hash (32-bit)
    h = 0
    for ch in s:
        h = (h * 131 + ord(ch)) & 0x7FFFFFFF
    return h


def _stable_u32(key: Any, seed: int, salt: int = 0) -> int:
    """Deterministic 32-bit pseudo-random int derived from (key, seed, salt)."""
    k = _to_int_key(key)
    x = (k ^ (seed * 0x9E3779B1) ^ (salt * 0x85EBCA6B)) & 0xFFFFFFFF
    # xorshift32
    x ^= (x << 13) & 0xFFFFFFFF
    x ^= (x >> 17) & 0xFFFFFFFF
    x ^= (x << 5) & 0xFFFFFFFF
    return x & 0x7FFFFFFF


def _stable_float01(key: Any, seed: int, salt: int = 0) -> float:
    return _stable_u32(key, seed, salt) / float(0x7FFFFFFF)


# -----------------------------
# Helpers: dates/months/datekeys
# -----------------------------

def _parse_iso_date(s: str) -> date:
    return pd.to_datetime(s).date()


def _datekey(d: date) -> int:
    return int(d.strftime("%Y%m%d"))


def _month_start(d: date) -> date:
    return date(d.year, d.month, 1)


def _month_end(d: date) -> date:
    # next month start minus one day
    if d.month == 12:
        nm = date(d.year + 1, 1, 1)
    else:
        nm = date(d.year, d.month + 1, 1)
    return nm - pd.Timedelta(days=1)  # type: ignore[arg-type]


def _iter_month_starts(start: date, end: date) -> List[date]:
    """Return list of month-start dates between start..end (inclusive), aligned to 1st."""
    cur = _month_start(start)
    out: List[date] = []
    while cur <= end:
        out.append(cur)
        # increment month
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)
    return out


def _is_quarter_start(month_start_dt: date) -> bool:
    return month_start_dt.month in (1, 4, 7, 10)


# -----------------------------
# Config access (dict-based)
# -----------------------------

@dataclass(frozen=True)
class CustomerSegmentsCfg:
    enabled: bool = True
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

    override = seg.get("override") or {}
    if not isinstance(override, dict):
        override = {}

    override_dates = override.get("dates") or {}
    if not isinstance(override_dates, dict):
        override_dates = {}

    validity = seg.get("validity") or {}
    if not isinstance(validity, dict):
        validity = {}

    include_validity = bool(seg.get("include_validity", False))
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


# -----------------------------
# Public builders
# -----------------------------

def build_dim_customer_segment(cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Build DimCustomerSegment.

    Output columns:
      SegmentKey (int), SegmentName (str), SegmentType (str),
      Definition (str), IsActiveFlag (int8)
    """
    seg_cfg = cfg.get("customer_segments") or {}
    if not isinstance(seg_cfg, dict):
        seg_cfg = {}
    segment_count = int(seg_cfg.get("segment_count", 12))

    segs = DEFAULT_SEGMENTS[:segment_count]
    # If user asks for more segments than we have defaults, generate filler.
    if segment_count > len(segs):
        for i in range(len(segs) + 1, segment_count + 1):
            segs.append((f"Segment {i}", "Custom", f"Auto-generated segment {i}"))

    rows = []
    for i, (name, stype, desc) in enumerate(segs, start=1):
        rows.append(
            {
                "SegmentKey": i,
                "SegmentName": name,
                "SegmentType": stype,
                "Definition": desc,
                "IsActiveFlag": np.int8(1),
            }
        )

    return pd.DataFrame(rows)


def build_bridge_customer_segment_membership(
    customers: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Build BridgeCustomerSegmentMembership (interval/SCD2-like).

    Required input:
      customers: DataFrame with at least CustomerKey

    Config input:
      cfg['defaults']['dates'] must exist (start/end ISO strings)

    Output columns (depending on cfg):
      CustomerKey, SegmentKey,
      ValidFromDate, ValidToDate (nullable int),
      Score (optional float32),
      IsPrimaryFlag (optional int8)
    """
    if "CustomerKey" not in customers.columns:
        raise KeyError("customers DataFrame must contain 'CustomerKey'")

    global_dates = cfg.get("defaults", {}).get("dates", {})
    if not isinstance(global_dates, dict) or not global_dates.get("start") or not global_dates.get("end"):
        raise KeyError("cfg.defaults.dates.start/end required")

    c = _read_cfg(cfg, global_dates)
    if not c.enabled:
        return pd.DataFrame(columns=["CustomerKey", "SegmentKey", "ValidFromDate", "ValidToDate"])

    # Resolve seed + date range
    seed = c.override_seed if c.override_seed is not None else c.seed
    start_dt = _parse_iso_date(c.override_start)  # type: ignore[arg-type]
    end_dt = _parse_iso_date(c.override_end)      # type: ignore[arg-type]

    month_starts = _iter_month_starts(start_dt, end_dt)

    # date fields (new) - use datetime64[ns] so parquet has proper logical type
    month_start_dates = pd.to_datetime(pd.Series(month_starts)).dt.normalize().to_numpy(dtype="datetime64[ns]")
    month_end_dates = pd.to_datetime(pd.Series([_month_end(ms) for ms in month_starts])).dt.normalize().to_numpy(dtype="datetime64[ns]")

    # Build segments table (local) to locate special segments by name
    dim_seg = build_dim_customer_segment({"customer_segments": {"segment_count": c.segment_count}})
    name_to_key = {r["SegmentName"]: int(r["SegmentKey"]) for _, r in dim_seg.iterrows()}
    new_customer_key = name_to_key.get("New Customer")

    # Try to find a join month for New Customer validity, if customers has a usable column.
    join_candidates = ["JoinDateKey", "CustomerStartDateKey", "StartDateKey", "CreatedDateKey"]
    join_col = next((col for col in join_candidates if col in customers.columns), None)

    # Precompute per-customer join month index (deterministic fallback)
    cust_keys = customers["CustomerKey"].tolist()
    join_month_idx: Dict[Any, int] = {}
    if join_col is not None:
        # Convert join date keys like 20240115 to month start index
        join_keys = customers.set_index("CustomerKey")[join_col].to_dict()
        for ck in cust_keys:
            jk = join_keys.get(ck)
            if pd.isna(jk) or jk is None:
                # fallback: deterministic spread across range
                join_month_idx[ck] = int(_stable_u32(ck, seed, 9001) % max(1, len(month_starts)))
                continue
            jk = int(jk)
            y = jk // 10000
            m = (jk // 100) % 100
            jm = date(y, m, 1)
            # clamp to global range
            if jm < month_starts[0]:
                join_month_idx[ck] = 0
            elif jm > month_starts[-1]:
                join_month_idx[ck] = len(month_starts) - 1
            else:
                # find index
                # (month_starts is small; linear scan is fine)
                join_month_idx[ck] = next(i for i, ms in enumerate(month_starts) if ms.year == jm.year and ms.month == jm.month)
    else:
        for ck in cust_keys:
            join_month_idx[ck] = int(_stable_u32(ck, seed, 9001) % max(1, len(month_starts)))

    # Main generation
    out_rows: List[Dict[str, Any]] = []

    seg_keys_all = list(range(1, c.segment_count + 1))
    k_min = max(0, c.segs_per_cust_min)
    k_max = max(k_min, min(c.segs_per_cust_max, c.segment_count))

    for ck in cust_keys:
        # choose base set size deterministically
        base_h = _stable_u32(ck, seed, 100)
        k = k_min + (base_h % (k_max - k_min + 1))

        # pick k segment keys deterministically
        chosen: List[int] = []
        for i in range(1, k + 1):
            idx = (base_h + i * 17) % c.segment_count
            sk = seg_keys_all[idx]
            if sk not in chosen:
                chosen.append(sk)

        # Ensure exactly 1 primary segment if requested
        primary_sk = chosen[0] if chosen else (1 if c.segment_count > 0 else 0)
        if primary_sk == 0:
            continue

        # Prevent churn from removing primary
        base_set = set(chosen)
        if primary_sk not in base_set:
            base_set.add(primary_sk)

        # Precompute per-segment score (stable per customer-segment)
        score_by_seg: Dict[int, float] = {}
        if c.include_score:
            for sk in base_set.union({new_customer_key} if new_customer_key else set()):
                if sk is None:
                    continue
                score_by_seg[int(sk)] = float(
                    np.float32(0.50 + (_stable_u32((ck, sk), seed, 777) % 50) / 100.0)
                )

        # Track current membership at month granularity, then emit intervals
        active_set: set[int] = set()
        start_month_for_seg: Dict[int, int] = {}

        # A mutable copy for churn
        cur_base_set = set(base_set)

        # New customer active window
        jm0 = join_month_idx[ck]
        new_window = set(range(jm0, min(len(month_starts), jm0 + max(0, c.new_customer_months)))) if (c.include_validity and new_customer_key and c.new_customer_months > 0) else set()

        for mi, ms in enumerate(month_starts):
            # quarterly churn applied at quarter starts (except month 0)
            if c.include_validity and mi > 0 and _is_quarter_start(ms):
                # decide churn
                if _stable_float01(ck, seed, salt=10_000 + mi) < c.churn_rate_qtr:
                    # replace ONE secondary segment (not primary) with a new one not in set
                    secondaries = sorted([s for s in cur_base_set if s != primary_sk])
                    if secondaries:
                        victim_idx = int(_stable_u32(ck, seed, 20_000 + mi) % len(secondaries))
                        victim = secondaries[victim_idx]
                        # find replacement
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

            # apply New Customer validity window
            if c.include_validity and new_customer_key is not None:
                if mi in new_window:
                    desired.add(int(new_customer_key))
                else:
                    desired.discard(int(new_customer_key))

            # open intervals
            for sk in desired - active_set:
                start_month_for_seg[sk] = mi

            # close intervals
            for sk in active_set - desired:
                smi = start_month_for_seg.pop(sk, None)
                if smi is None:
                    continue
                out_rows.append(
                    _membership_row(
                        ck=ck,
                        sk=sk,
                        from_date=month_start_dates[smi],
                        to_date=month_end_dates[mi - 1] if mi - 1 >= 0 else month_end_dates[smi],
                        primary_sk=primary_sk,
                        score=score_by_seg.get(sk),
                        include_primary=c.include_primary_flag,
                        include_score=c.include_score,
                    )
                )

            active_set = desired

        # close remaining intervals to end
        last_mi = len(month_starts) - 1
        for sk in list(active_set):
            smi = start_month_for_seg.get(sk)
            if smi is None:
                continue
            out_rows.append(
                _membership_row(
                    ck=ck,
                    sk=sk,
                    from_date=month_start_dates[smi],
                    to_date=month_end_dates[last_mi],
                    primary_sk=primary_sk,
                    score=score_by_seg.get(sk),
                    include_primary=c.include_primary_flag,
                    include_score=c.include_score,
                )
            )

    df = pd.DataFrame(out_rows)

    # Dtypes (keep Arrow-friendly)
    if not df.empty:
        df["CustomerKey"] = df["CustomerKey"].astype("int64")
        df["SegmentKey"] = df["SegmentKey"].astype("int32")

        df["ValidFromDate"] = pd.to_datetime(df["ValidFromDate"]).dt.normalize()
        df["ValidToDate"] = pd.to_datetime(df["ValidToDate"]).dt.normalize()

        if "IsPrimaryFlag" in df.columns:
            df["IsPrimaryFlag"] = df["IsPrimaryFlag"].astype("int8")
        if "Score" in df.columns:
            df["Score"] = df["Score"].astype("float32")

    return df


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
        row["IsPrimaryFlag"] = np.int8(1 if sk == primary_sk else 0)
    if include_score:
        row["Score"] = float(score) if score is not None else float(np.float32(0.60))
    return row


# -----------------------------
# Runner (I/O + skip logic)
# -----------------------------
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

    # INPUT: Customers parquet (support common naming variants)
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

    # OUTPUT paths (allow override) — keep backward compatibility with your old keys
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

    # ------------------------------------------------------------------
    # Versioning: regenerate if customer_segments config OR Customers input changed
    # ------------------------------------------------------------------
    # Make a copy so we don't mutate cfg in-place in surprising ways
    seg_cfg_for_version = dict(seg_cfg)

    # Include a cheap but reliable signature of the upstream Customers input
    st = os.stat(customers_fp)
    seg_cfg_for_version["upstream_customers_sig"] = {
        "path": str(customers_fp),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }

    # If you want validity to be date-range sensitive even when defaults live elsewhere:
    defaults_dates = (cfg.get("defaults") or {}).get("dates") or {}
    if isinstance(defaults_dates, dict):
        seg_cfg_for_version["upstream_global_dates"] = {
            "start": defaults_dates.get("start"),
            "end": defaults_dates.get("end"),
        }

    # Skip only if BOTH outputs exist AND version says nothing relevant changed
    if not force:
        if dim_out.exists() and bridge_out.exists() and (not should_regenerate("customer_segments", seg_cfg_for_version, bridge_out)):
            skip("Customer segments up-to-date; skipping.")
            return {"_regenerated": False, "reason": "version"}

    info(f"Loading Customers from: {customers_fp}")
    customers = pd.read_parquet(customers_fp)

    # Only generate memberships for customers that participate in Sales
    customers_for_bridge = customers
    if "IsActiveInSales" in customers_for_bridge.columns:
        customers_for_bridge = customers_for_bridge[
            customers_for_bridge["IsActiveInSales"].fillna(0).astype("int64") == 1
        ].copy()

    dim_seg = build_dim_customer_segment(cfg)
    bridge = build_bridge_customer_segment_membership(customers=customers_for_bridge, cfg=cfg)

    dim_out.parent.mkdir(parents=True, exist_ok=True)
    bridge_out.parent.mkdir(parents=True, exist_ok=True)

    dim_seg.to_parquet(dim_out, index=False)
    bridge.to_parquet(bridge_out, index=False)

    # Save version (tie it to the bridge output path)
    save_version("customer_segments", seg_cfg_for_version, bridge_out)

    done(f"Wrote customer_segment: {dim_out.name} ({len(dim_seg):,} rows)")
    done(f"Wrote customer_segment_membership: {bridge_out.name} ({len(bridge):,} rows)")
    return {"_regenerated": True, "dim": str(dim_out), "bridge": str(bridge_out)}
