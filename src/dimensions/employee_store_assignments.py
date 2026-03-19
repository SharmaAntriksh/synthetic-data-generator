"""Employee Store Assignment bridge table generator.

Produces a non-overlapping timeline of store assignments for each employee.
Each employee's assignments form a contiguous sequence: home segments
interleaved with away episodes, guaranteed non-overlapping by construction
(single monotonic cursor per employee).

Output columns:
  EmployeeKey, StoreKey, StartDate, EndDate, FTE, RoleAtStore,
  IsPrimary, TransferReason, AssignmentSequence, Status, MaxAssignments
"""
from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.dimensions.employees import (
    STORE_MGR_KEY_BASE,
    STAFF_KEY_BASE,
    STAFF_KEY_STORE_MULT,
)

from src.defaults import (
    EMPLOYEE_TRANSFER_REASON_LABELS,
    EMPLOYEE_TRANSFER_REASON_PROBS,
    ONLINE_EMP_KEY_BASE,
    ONLINE_STORE_KEY_BASE,
)
from src.utils.logging_utils import info, skip, stage, warn
from src.utils.output_utils import write_parquet_with_date32
from src.versioning import should_regenerate, save_version
from src.utils.config_helpers import (
    as_dict,
    int_or,
    float_or,
    str_or,
    parse_global_dates,
    rand_single_date,
)
from src.utils.config_precedence import resolve_seed


# ---------------------------------------------------------------------------
# Immutable slot for one contiguous assignment
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True, slots=True)
class TimelineSlot:
    """One contiguous store assignment for an employee. Immutable."""
    store_key: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    fte: float
    role: str
    is_primary: bool
    transfer_reason: Optional[str]


# ---------------------------------------------------------------------------
# Config helpers  (kept from original)
# ---------------------------------------------------------------------------

def _store_assignments_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preferred path: ``cfg.employees.store_assignments``
    Legacy path:    ``cfg.employee_store_assignments``

    Merge rule: nested overrides legacy.
    """
    cfg = cfg or {}
    emp_cfg = cfg.employees
    nested = dict(emp_cfg.store_assignments) if emp_cfg.store_assignments is not None else {}
    legacy = as_dict(cfg.employee_store_assignments)

    _RUNNER_KEYS = {"global_dates"}
    user_legacy = {k: v for k, v in legacy.items() if k not in _RUNNER_KEYS}
    if user_legacy:
        warn(
            "Top-level 'employee_store_assignments' config is deprecated. "
            "Move settings under 'employees.store_assignments'. "
            "Nested keys take precedence over legacy keys."
        )
    out = dict(legacy)
    out.update(nested)
    return out


def _infer_home_store_key(employees: pd.DataFrame) -> pd.Series:
    """Derive the home StoreKey from the EmployeeKey encoding."""
    if "StoreKey" in employees.columns:
        return employees["StoreKey"].astype("Int32")

    ek = employees["EmployeeKey"].astype(np.int64)
    out = pd.Series([pd.NA] * len(employees), dtype="Int32")

    # Online employees (50M+) — check before staff (40M+) since range overlaps
    online_mask = ek >= ONLINE_EMP_KEY_BASE
    if online_mask.any():
        out.loc[online_mask] = (ek.loc[online_mask] - ONLINE_EMP_KEY_BASE).astype("Int32")

    mgr_mask = (ek >= STORE_MGR_KEY_BASE) & (ek < STAFF_KEY_BASE) & ~online_mask
    if mgr_mask.any():
        out.loc[mgr_mask] = (ek.loc[mgr_mask] - STORE_MGR_KEY_BASE).astype("Int32")

    staff_mask = (ek >= STAFF_KEY_BASE) & ~online_mask
    if staff_mask.any():
        out.loc[staff_mask] = (
            (ek.loc[staff_mask] - STAFF_KEY_BASE) // STAFF_KEY_STORE_MULT
        ).astype("Int32")

    return out


def _build_store_by_district(emp: pd.DataFrame) -> Dict[Any, List[int]]:
    """``DistrictId -> [StoreKey, ...]``"""
    emp2 = emp[emp["DistrictId"].notna() & emp["HomeStoreKey"].notna()].copy()
    if emp2.empty:
        return {}

    is_mgr = emp2["Title"].astype(str) == "Store Manager"
    mgrs = emp2[is_mgr]
    if not mgrs.empty:
        src = mgrs
    else:
        warn(
            "No Store Managers found when building district->store mapping; "
            "falling back to all store-attached employees. "
            "District store pools may be inaccurate."
        )
        src = emp2

    pairs = src[["DistrictId", "HomeStoreKey"]].dropna().drop_duplicates()
    if pairs.empty:
        return {}

    return {
        did: sorted({int(x) for x in skeys})
        for did, skeys in pairs.groupby("DistrictId")["HomeStoreKey"]
    }


# ---------------------------------------------------------------------------
# Movement-profile helpers  (kept from original)
# ---------------------------------------------------------------------------

def _normalize_profile_keys(prof: Dict[str, Any]) -> Dict[str, Any]:
    """Map shorthand config keys to long-form keys."""
    out = dict(prof)

    if "mult" in out and "role_multiplier" not in out:
        out["role_multiplier"] = out.pop("mult")
    else:
        out.pop("mult", None)

    if "episodes" in out:
        ep = out.pop("episodes")
        if isinstance(ep, (list, tuple)) and len(ep) >= 2:
            if "episodes_min" not in out:
                out["episodes_min"] = int(ep[0])
            if "episodes_max" not in out:
                out["episodes_max"] = int(ep[1])
        elif isinstance(ep, (int, float)):
            if "episodes_min" not in out:
                out["episodes_min"] = int(ep)
            if "episodes_max" not in out:
                out["episodes_max"] = int(ep)

    if "duration" in out:
        dur = out.pop("duration")
        if isinstance(dur, (list, tuple)) and len(dur) >= 2:
            if "duration_days_min" not in out:
                out["duration_days_min"] = int(dur[0])
            if "duration_days_max" not in out:
                out["duration_days_max"] = int(dur[1])
        elif isinstance(dur, (int, float)):
            if "duration_days_min" not in out:
                out["duration_days_min"] = int(dur)
            if "duration_days_max" not in out:
                out["duration_days_max"] = int(dur)

    return out


def _get_profile(
    role: str,
    *,
    primary_sales_role: str,
    default_profile: Dict[str, Any],
    per_role_profile: Dict[str, Dict[str, Any]],
    sales_profile: Dict[str, Any],
    max_non_sales_multiplier_frac: float,
    max_non_sales_episodes_frac: float,
) -> Dict[str, Any]:
    """Merge ``default_profile <- per_role_profile[role]``, enforce non-sales caps."""
    prof = dict(default_profile)
    prof.update(per_role_profile.get(role, {}))

    if role != primary_sales_role:
        sales_mult = float_or(sales_profile.get("role_multiplier"), 1.0)
        role_mult = float_or(prof.get("role_multiplier"), 1.0)
        prof["role_multiplier"] = min(role_mult, sales_mult * max_non_sales_multiplier_frac)

        sales_emax = int_or(sales_profile.get("episodes_max"), 2)
        emax = int_or(prof.get("episodes_max"), 1)
        prof["episodes_max"] = min(emax, int(np.floor(sales_emax * max_non_sales_episodes_frac)))

        emin = int_or(prof.get("episodes_min"), 0)
        prof["episodes_min"] = min(emin, int_or(prof.get("episodes_max"), 0))

    return prof


# ---------------------------------------------------------------------------
# Store operational window helpers  (kept from original)
# ---------------------------------------------------------------------------

def _store_operational_window_pd(
    store: int,
    ws: pd.Timestamp,
    we: pd.Timestamp,
    store_opening_dates: Optional[Dict[int, pd.Timestamp]],
    store_closing_dates: Optional[Dict[int, pd.Timestamp]],
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Return (start, end) of a store's operational window, or None."""
    one_day = pd.Timedelta(days=1)
    s_start = ws
    s_end = we
    if store_opening_dates and store in store_opening_dates:
        od = pd.to_datetime(store_opening_dates[store]).normalize()
        if od > s_start:
            s_start = od
    if store_closing_dates and store in store_closing_dates:
        cd = pd.to_datetime(store_closing_dates[store]).normalize()
        last_op = cd - one_day
        if last_op < s_end:
            s_end = last_op
    if s_end < s_start:
        return None
    return s_start, s_end


def _store_operational_window_np(
    store: int,
    ws: np.datetime64,
    we: np.datetime64,
    store_opening_dates: Optional[Dict[int, pd.Timestamp]],
    store_closing_dates: Optional[Dict[int, pd.Timestamp]],
) -> Optional[Tuple[np.datetime64, np.datetime64]]:
    """Return (start, end) as np.datetime64[D], or None."""
    one_day = np.timedelta64(1, "D")
    s_ws = ws
    s_we = we
    if store_opening_dates and store in store_opening_dates:
        _od = np.datetime64(store_opening_dates[store], "D")
        if _od > s_ws:
            s_ws = _od
    if store_closing_dates and store in store_closing_dates:
        _cd = np.datetime64(store_closing_dates[store], "D") - one_day
        if _cd < s_we:
            s_we = _cd
    if s_we < s_ws:
        return None
    return s_ws, s_we


def _build_emp_date_lookup(
    ek_arr: np.ndarray,
    date_arr: np.ndarray,
) -> Dict[int, pd.Timestamp]:
    """Build {EmployeeKey: normalized Timestamp} from parallel arrays."""
    out: Dict[int, pd.Timestamp] = {}
    for i in range(len(ek_arr)):
        d = date_arr[i]
        if isinstance(d, (np.datetime64,)):
            if np.isnat(d):
                continue
            out[int(ek_arr[i])] = pd.Timestamp(d).normalize()
        else:
            if pd.isna(d):
                continue
            out[int(ek_arr[i])] = pd.Timestamp(d).normalize()
    return out


def _merge_adjacent_intervals(starts, ends, tolerance):
    """Merge overlapping/adjacent intervals.

    Works with both numpy datetime64 and pandas Timestamp arrays/lists.
    ``tolerance`` is one day in the caller's dtype (np.timedelta64 or pd.Timedelta).
    Returns (merged_starts, merged_ends) as lists.
    """
    if len(starts) == 0:
        return [], []
    ms = [starts[0]]
    me = [ends[0]]
    for j in range(1, len(starts)):
        if starts[j] <= me[-1] + tolerance:
            me[-1] = max(me[-1], ends[j])
        else:
            ms.append(starts[j])
            me.append(ends[j])
    return ms, me


# ---------------------------------------------------------------------------
# Episode sampling  (extracted from original inner function — logic unchanged)
# ---------------------------------------------------------------------------

def _sample_non_overlapping_episodes(
    rng: np.random.Generator,
    start_min: pd.Timestamp,
    end_max: pd.Timestamp,
    other_stores: List[int],
    k: int,
    tgt_fte: float,
    dmin: int,
    dmax: int,
    allow_store_revisit: bool,
    store_opening_dates: Optional[Dict[int, pd.Timestamp]],
    store_closing_dates: Optional[Dict[int, pd.Timestamp]],
) -> List[Tuple[pd.Timestamp, pd.Timestamp, int, float]]:
    """Sample up to *k* non-overlapping away episodes within [start_min, end_max].

    Returns sorted list of (start, end, store_key, fte) tuples, guaranteed
    non-overlapping and within each destination store's operational window.
    """
    if k <= 0 or not other_stores:
        return []

    # Pick stores ensuring no two consecutive picks are the same
    if not allow_store_revisit:
        chosen = rng.choice(
            other_stores, size=min(int(k), len(other_stores)), replace=False,
        )
    else:
        chosen_list: List[int] = []
        for _ in range(int(k)):
            pool = (
                [s for s in other_stores if s != chosen_list[-1]]
                if chosen_list
                else list(other_stores)
            )
            if not pool:
                break
            chosen_list.append(int(rng.choice(pool)))
        chosen = np.array(chosen_list, dtype=np.int32)

    one_day = pd.Timedelta(days=1)

    # (start, dur, store, fte, store_end_limit)
    raw: List[Tuple[pd.Timestamp, int, int, float, pd.Timestamp]] = []
    for store in chosen:
        ep_start_min = start_min
        ep_end_max = end_max
        _sk = int(store)
        if store_opening_dates and _sk in store_opening_dates:
            _sod = pd.to_datetime(store_opening_dates[_sk]).normalize()
            if _sod > ep_start_min:
                ep_start_min = _sod
        if store_closing_dates and _sk in store_closing_dates:
            # Last operational day = close_date - 1
            _scd = (pd.to_datetime(store_closing_dates[_sk]).normalize() - one_day)
            if _scd < ep_end_max:
                ep_end_max = _scd
        if ep_end_max < ep_start_min:
            continue

        dur = int(rng.integers(dmin, dmax + 1))
        latest_start = (ep_end_max - pd.Timedelta(days=dur - 1)).normalize()
        if latest_start < ep_start_min:
            continue
        s = rand_single_date(rng, ep_start_min, latest_start)
        raw.append((s, dur, int(store), tgt_fte, ep_end_max))

    if not raw:
        return []

    raw.sort(key=lambda x: x[0])
    placed: List[Tuple[pd.Timestamp, pd.Timestamp, int, float]] = []
    last_end: Optional[pd.Timestamp] = None

    for s, dur, store, sec_fte, store_limit in raw:
        if last_end is not None and s <= last_end:
            s = (last_end + one_day).normalize()
        if s > end_max or s > store_limit:
            continue
        e = (s + pd.Timedelta(days=dur - 1)).normalize()
        # Clamp to both global end and per-store operational limit
        e = min(e, end_max, store_limit)
        if e < s:
            continue
        placed.append((s, e, store, sec_fte))
        last_end = e

    # Merge consecutive same-store episodes
    if len(placed) <= 1:
        return placed

    out: List[Tuple[pd.Timestamp, pd.Timestamp, int, float]] = [placed[0]]
    for s, e, store, sec_fte in placed[1:]:
        prev_s, prev_e, prev_store, prev_fte = out[-1]
        if store == prev_store:
            out[-1] = (prev_s, e, prev_store, prev_fte)
        else:
            out.append((s, e, store, sec_fte))

    return out


# ---------------------------------------------------------------------------
# Core: build one employee's complete timeline
# ---------------------------------------------------------------------------

MIN_HOME_GAP_DAYS = 3


def _build_employee_timeline(
    *,
    rng: np.random.Generator,
    home_store: int,
    role: str,
    fte: float,
    effective_start: pd.Timestamp,
    effective_end: pd.Timestamp,
    candidate_stores: List[int],
    episodes_min: int,
    episodes_max: int,
    duration_days_min: int,
    duration_days_max: int,
    allow_store_revisit: bool,
    store_opening_dates: Optional[Dict[int, pd.Timestamp]],
    store_closing_dates: Optional[Dict[int, pd.Timestamp]],
    is_forced_mover: bool = False,
    transfer_info: Optional[Tuple[pd.Timestamp, int, pd.Timestamp]] = None,
) -> List[TimelineSlot]:
    """Build a complete non-overlapping assignment timeline for one employee.

    Parameters
    ----------
    transfer_info : optional (close_date, dest_store, transfer_end)
        If provided, the employee's home-store window is truncated at
        close_date - 1, and a transfer assignment is appended at dest_store
        from close_date to transfer_end.

    Returns
    -------
    List of TimelineSlot, sorted by start_date, guaranteed non-overlapping.
    """
    slots: List[TimelineSlot] = []

    # If transferred, truncate the home-store window before generating episodes
    home_end = effective_end
    if transfer_info is not None:
        close_date, dest_store, transfer_end = transfer_info
        home_end = min(effective_end, (close_date - pd.Timedelta(days=1)).normalize())

    if home_end < effective_start:
        # No room for any assignment at home store; just emit transfer if any
        if transfer_info is not None:
            close_date, dest_store, transfer_end = transfer_info
            t_start = max(effective_start, close_date)
            if t_start <= transfer_end:
                slots.append(TimelineSlot(
                    store_key=dest_store,
                    start_date=t_start,
                    end_date=transfer_end,
                    fte=fte,
                    role=role,
                    is_primary=True,
                    transfer_reason=None,
                ))
        return slots

    # Sample away episodes within [effective_start, home_end]
    episodes: List[Tuple[pd.Timestamp, pd.Timestamp, int, float]] = []
    if candidate_stores and episodes_max > 0:
        k = (
            int(rng.integers(episodes_min, episodes_max + 1))
            if episodes_max >= episodes_min
            else int(episodes_min)
        )
        if is_forced_mover and k < 1:
            k = 1
        if k > 0:
            episodes = _sample_non_overlapping_episodes(
                rng, effective_start, home_end, candidate_stores, k, fte,
                duration_days_min, duration_days_max,
                allow_store_revisit, store_opening_dates, store_closing_dates,
            )

    # Walk a monotonic cursor to interleave home and away segments
    cursor = effective_start

    for (ep_start, ep_end, ep_store, _ep_fte) in episodes:
        ep_start = pd.to_datetime(ep_start).normalize()
        ep_end = pd.to_datetime(ep_end).normalize()

        before_end = (ep_start - pd.Timedelta(days=1)).normalize()
        gap_days = (before_end - cursor).days + 1 if cursor <= before_end else 0

        if gap_days >= MIN_HOME_GAP_DAYS:
            # Emit home-store segment for the gap
            slots.append(TimelineSlot(
                store_key=home_store,
                start_date=cursor,
                end_date=before_end,
                fte=fte,
                role=role,
                is_primary=True,
                transfer_reason=None,
            ))
        elif gap_days > 0:
            # Gap too short — absorb by extending away episode backward
            ep_start = cursor

        # Emit away episode
        reason = str(rng.choice(
            EMPLOYEE_TRANSFER_REASON_LABELS, p=EMPLOYEE_TRANSFER_REASON_PROBS,
        ))
        slots.append(TimelineSlot(
            store_key=ep_store,
            start_date=ep_start,
            end_date=ep_end,
            fte=fte,
            role=role,
            is_primary=False,
            transfer_reason=reason,
        ))
        cursor = (ep_end + pd.Timedelta(days=1)).normalize()

    # Final home segment after all episodes
    if cursor <= home_end:
        slots.append(TimelineSlot(
            store_key=home_store,
            start_date=cursor,
            end_date=home_end,
            fte=fte,
            role=role,
            is_primary=True,
            transfer_reason=None,
        ))

    # Append transfer assignment if employee moved from closing store
    if transfer_info is not None:
        close_date, dest_store, transfer_end = transfer_info
        t_start = close_date
        # Cursor may have advanced past close_date if last away episode ended
        # after the close date; clamp to the day after the last slot
        if slots:
            last_slot_end = slots[-1].end_date
            candidate_start = (last_slot_end + pd.Timedelta(days=1)).normalize()
            if candidate_start > t_start:
                t_start = candidate_start
        if t_start <= transfer_end:
            slots.append(TimelineSlot(
                store_key=dest_store,
                start_date=t_start,
                end_date=transfer_end,
                fte=fte,
                role=role,
                is_primary=True,
                transfer_reason=None,
            ))

    return slots


# ---------------------------------------------------------------------------
# Metadata derivation
# ---------------------------------------------------------------------------

def _derive_status(df: pd.DataFrame, window_end: pd.Timestamp) -> pd.Series:
    """Derive Status: Active / Transferred / Completed.

    - Active:      EndDate >= window_end (still ongoing at dataset boundary)
    - Transferred: EndDate < window_end AND next assignment (by sequence) is
                   at a *different* store
    - Completed:   everything else (assignment ended, no cross-store transfer)
    """
    we = pd.to_datetime(window_end).normalize()
    end_dates = pd.to_datetime(df["EndDate"]).dt.normalize()

    is_active = end_dates >= we

    # Shift StoreKey forward within each employee to detect store changes
    next_store = df.groupby("EmployeeKey")["StoreKey"].shift(-1)
    is_transferred = (~is_active) & (next_store.notna()) & (next_store != df["StoreKey"])

    status = pd.Series("Completed", index=df.index, dtype="object")
    status[is_active] = "Active"
    status[is_transferred] = "Transferred"
    return status


# ---------------------------------------------------------------------------
# Comprehensive validation
# ---------------------------------------------------------------------------

def _validate_assignments(
    df: pd.DataFrame,
    *,
    ps_role: str,
    all_stores: List[int],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    store_opening_dates: Optional[Dict[int, pd.Timestamp]],
    store_closing_dates: Optional[Dict[int, pd.Timestamp]],
    emp_hire: Dict[int, pd.Timestamp],
    emp_term: Dict[int, pd.Timestamp],
    coverage_mode: str = "warn",
) -> None:
    """Single comprehensive validation of all assignment invariants.

    Checks:
    1. No per-employee overlaps (across ALL stores)
    2. Dates within hire/termination window
    3. Dates within store open/close window
    4. SA coverage per store (warn or raise)
    """
    if df.empty:
        return

    one_day = pd.Timedelta(days=1)

    # --- Check 1: no per-employee overlaps across all stores ---
    chk = df.sort_values(["EmployeeKey", "StartDate", "EndDate"]).copy()
    chk_end = pd.to_datetime(chk["EndDate"]).dt.normalize()
    chk_start = pd.to_datetime(chk["StartDate"]).dt.normalize()
    prev_end = chk_end.groupby(chk["EmployeeKey"]).shift(1)
    bad = (prev_end.notna()) & (chk_start <= prev_end)
    if bool(bad.any()):
        ex = chk.loc[bad, ["EmployeeKey", "StoreKey", "StartDate", "EndDate"]].head(10)
        raise RuntimeError(
            f"Overlapping assignments detected across stores (sample):\n{ex}"
        )

    # --- Check 2: dates within hire/termination window ---
    violations: List[str] = []
    for ek_val, grp in df.groupby("EmployeeKey"):
        ek = int(ek_val)
        hire_dt = emp_hire.get(ek)
        term_dt = emp_term.get(ek)
        first_start = pd.to_datetime(grp["StartDate"].iloc[0]).normalize()
        last_end = pd.to_datetime(grp["EndDate"].iloc[-1]).normalize()
        if hire_dt is not None and first_start < hire_dt - one_day:
            violations.append(
                f"EK={ek}: StartDate {first_start.date()} before HireDate {hire_dt.date()}"
            )
        if term_dt is not None and last_end > term_dt + one_day:
            violations.append(
                f"EK={ek}: EndDate {last_end.date()} after TerminationDate {term_dt.date()}"
            )
    if violations:
        sample = "\n".join(violations[:10])
        warn(f"Assignment date/employment window mismatches (sample):\n{sample}")

    # --- Check 3: dates within store operational window ---
    ws = pd.to_datetime(window_start).normalize()
    we = pd.to_datetime(window_end).normalize()
    store_violations: List[str] = []
    for sk_val, grp in df.groupby("StoreKey"):
        sk = int(sk_val)
        win = _store_operational_window_pd(
            sk, ws, we, store_opening_dates, store_closing_dates,
        )
        if win is None:
            continue
        s_start, s_end = win
        grp_starts = pd.to_datetime(grp["StartDate"]).dt.normalize()
        grp_ends = pd.to_datetime(grp["EndDate"]).dt.normalize()
        early = grp_starts < s_start - one_day
        late = grp_ends > s_end + one_day
        if early.any() or late.any():
            store_violations.append(f"StoreKey={sk}: assignments outside operational window")
    if store_violations:
        sample = "\n".join(store_violations[:10])
        warn(f"Store operational window mismatches (sample):\n{sample}")

    # --- Check 4: SA coverage per store ---
    ws_np = np.datetime64(window_start, "D")
    we_np = np.datetime64(window_end, "D")
    one_day_np = np.timedelta64(1, "D")

    df_cov = df[df["RoleAtStore"].astype(str) == ps_role].copy()
    if df_cov.empty:
        return

    df_cov["StartDate"] = pd.to_datetime(df_cov["StartDate"]).dt.normalize()
    df_cov["EndDate"] = pd.to_datetime(df_cov["EndDate"]).dt.normalize()

    covered_stores = set(df_cov["StoreKey"].unique())
    gaps: List[str] = []

    # Check stores with no SA at all
    for s in all_stores:
        win = _store_operational_window_np(
            s, ws_np, we_np, store_opening_dates, store_closing_dates,
        )
        if win is None:
            continue
        if s not in covered_stores:
            gaps.append(f"StoreKey={s}: no '{ps_role}' assignments")

    # Check coverage continuity per store
    df_cov_sorted = df_cov.sort_values(["StoreKey", "StartDate", "EndDate"]).reset_index(drop=True)
    sk_arr = df_cov_sorted["StoreKey"].to_numpy()
    starts_arr = df_cov_sorted["StartDate"].to_numpy().astype("datetime64[D]")
    ends_arr = df_cov_sorted["EndDate"].to_numpy().astype("datetime64[D]")

    store_breaks = np.flatnonzero(np.r_[True, sk_arr[1:] != sk_arr[:-1]])
    group_ends = np.r_[store_breaks[1:], len(sk_arr)]

    for gs, ge in zip(store_breaks, group_ends):
        store = int(sk_arr[gs])
        win = _store_operational_window_np(
            store, ws_np, we_np, store_opening_dates, store_closing_dates,
        )
        if win is None:
            continue
        store_ws, store_we = win

        seg_starts = starts_arr[gs:ge]
        seg_ends = ends_arr[gs:ge]

        merged_s, merged_e = _merge_adjacent_intervals(seg_starts, seg_ends, one_day_np)

        # Walk merged intervals to find gaps within [store_ws, store_we]
        cursor = store_ws
        for ms, me in zip(merged_s, merged_e):
            if ms > cursor:
                gap_end = min(ms - one_day_np, store_we)
                if gap_end >= cursor:
                    gaps.append(f"StoreKey={store} gap {cursor}..{gap_end}")
                    break
            cursor = me + one_day_np
            if cursor > store_we:
                break
        else:
            if cursor <= store_we:
                gaps.append(f"StoreKey={store} gap {cursor}..{store_we}")

    if gaps:
        sample = "\n".join(gaps[:10])
        msg = (
            "EmployeeStoreAssignments coverage gaps detected "
            f"(sales-eligible role '{ps_role}'). Sample:\n{sample}"
        )
        if coverage_mode == "raise":
            raise RuntimeError(msg)
        else:
            warn(msg)


# ---------------------------------------------------------------------------
# Safe coverage gap fill
# ---------------------------------------------------------------------------

def _fill_coverage_gaps_safe(
    df: pd.DataFrame,
    *,
    ps_role: str,
    all_stores: List[int],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    store_opening_dates: Optional[Dict[int, pd.Timestamp]],
    store_closing_dates: Optional[Dict[int, pd.Timestamp]],
    emp_hire: Optional[Dict[int, pd.Timestamp]] = None,
    emp_term: Optional[Dict[int, pd.Timestamp]] = None,
) -> pd.DataFrame:
    """Fill SA coverage gaps by extending the nearest same-store SA assignment.

    Safety rules:
    - Only extend if the employee has NO other assignment at ANY store during
      the gap period (no cross-store overlaps).
    - Never extend past the employee's termination date.
    - Never extend before the employee's hire date.

    Returns the DataFrame with gap-filling rows appended (or extended).
    """
    if df.empty:
        return df

    one_day = pd.Timedelta(days=1)
    ws = pd.to_datetime(window_start).normalize()
    we = pd.to_datetime(window_end).normalize()

    # Build per-employee full timeline lookup: {ek: [(start, end), ...]}
    emp_timeline: Dict[int, List[Tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for ek_val, grp in df.groupby("EmployeeKey"):
        intervals = list(zip(
            pd.to_datetime(grp["StartDate"]).dt.normalize(),
            pd.to_datetime(grp["EndDate"]).dt.normalize(),
        ))
        emp_timeline[int(ek_val)] = sorted(intervals, key=lambda x: x[0])

    def _employee_free_during(ek: int, gap_start: pd.Timestamp, gap_end: pd.Timestamp) -> bool:
        """Return True if employee has NO assignment overlapping [gap_start, gap_end]."""
        for s, e in emp_timeline.get(ek, []):
            if s <= gap_end and e >= gap_start:
                return False
        return True

    def _clamp_to_employment(ek: int, start: pd.Timestamp, end: pd.Timestamp
                             ) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
        """Clamp [start, end] to employee's hire/term window. Return None if empty."""
        s, e = start, end
        if emp_hire and ek in emp_hire:
            s = max(s, emp_hire[ek])
        if emp_term and ek in emp_term:
            e = min(e, emp_term[ek])
        return (s, e) if e >= s else None

    # Filter to SA assignments only
    df_sa = df[df["RoleAtStore"].astype(str) == ps_role].copy()
    df_sa["StartDate"] = pd.to_datetime(df_sa["StartDate"]).dt.normalize()
    df_sa["EndDate"] = pd.to_datetime(df_sa["EndDate"]).dt.normalize()

    new_rows: List[Dict[str, Any]] = []
    extensions: Dict[int, pd.Timestamp] = {}  # df index -> new EndDate

    def _make_gap_fill_row(
        ek: int, store: int, start: pd.Timestamp, end: pd.Timestamp,
        fte: float, role: str,
    ) -> Dict[str, Any]:
        return {
            "EmployeeKey": ek, "StoreKey": store,
            "StartDate": start, "EndDate": end,
            "FTE": fte, "RoleAtStore": role,
            "IsPrimary": True, "TransferReason": pd.NA,
        }

    for store in all_stores:
        win = _store_operational_window_pd(
            store, ws, we, store_opening_dates, store_closing_dates,
        )
        if win is None:
            continue
        store_ws, store_we = win

        # Get SA assignments at this store sorted by start
        store_sa = df_sa[df_sa["StoreKey"] == store].sort_values("StartDate")
        if store_sa.empty:
            continue

        # Merge overlapping/adjacent intervals to find gaps
        _ms, _me = _merge_adjacent_intervals(
            store_sa["StartDate"].tolist(), store_sa["EndDate"].tolist(), one_day,
        )
        merged = list(zip(_ms, _me))

        # Identify gaps within [store_ws, store_we]
        gaps: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
        cursor = store_ws
        for ms, me in merged:
            if ms > cursor:
                gap_end = min(ms - one_day, store_we)
                if gap_end >= cursor:
                    gaps.append((cursor, gap_end))
            cursor = me + one_day
            if cursor > store_we:
                break
        if cursor <= store_we:
            gaps.append((cursor, store_we))

        if not gaps:
            continue

        # Try to fill each gap by extending nearest SA
        for gap_start, gap_end in gaps:
            filled = False

            # Strategy 1: extend an existing adjacent SA assignment forward
            # Find SA whose EndDate is just before the gap
            sa_ends = store_sa["EndDate"]
            before = store_sa[(sa_ends >= gap_start - one_day) & (sa_ends < gap_start)]
            if not before.empty:
                best_idx = before["EndDate"].idxmax()
                best_row = df.loc[best_idx]
                ek = int(best_row["EmployeeKey"])
                clamped = _clamp_to_employment(ek, gap_start, gap_end)
                if clamped is not None:
                    c_start, c_end = clamped
                    if _employee_free_during(ek, c_start, c_end):
                        current_end = extensions.get(best_idx, pd.to_datetime(best_row["EndDate"]).normalize())
                        new_end = max(current_end, c_end)
                        extensions[best_idx] = new_end
                        timeline = emp_timeline[ek]
                        for ti, (ts, te) in enumerate(timeline):
                            if te == current_end or te == pd.to_datetime(best_row["EndDate"]).normalize():
                                timeline[ti] = (ts, new_end)
                                break
                        filled = c_end >= gap_end  # fully filled?

            if filled:
                continue

            # Strategy 2: extend an SA whose StartDate is just after the gap backward
            sa_starts = store_sa["StartDate"]
            after = store_sa[(sa_starts <= gap_end + one_day) & (sa_starts > gap_end)]
            if not after.empty:
                best_idx = after["StartDate"].idxmin()
                best_row = df.loc[best_idx]
                ek = int(best_row["EmployeeKey"])
                clamped = _clamp_to_employment(ek, gap_start, gap_end)
                if clamped is not None:
                    c_start, c_end = clamped
                    if _employee_free_during(ek, c_start, c_end):
                        new_rows.append(_make_gap_fill_row(
                            ek, store, c_start, c_end,
                            float(best_row["FTE"]), str(best_row["RoleAtStore"]),
                        ))
                        emp_timeline[ek].append((c_start, c_end))
                        emp_timeline[ek].sort(key=lambda x: x[0])
                        filled = c_start <= gap_start and c_end >= gap_end

            if filled:
                continue

            # Strategy 3: pick ANY SA at this store who is free during the gap
            for _, sa_row in store_sa.iterrows():
                ek = int(sa_row["EmployeeKey"])
                clamped = _clamp_to_employment(ek, gap_start, gap_end)
                if clamped is None:
                    continue
                c_start, c_end = clamped
                if _employee_free_during(ek, c_start, c_end):
                    new_rows.append(_make_gap_fill_row(
                        ek, store, c_start, c_end,
                        float(sa_row["FTE"]), str(sa_row["RoleAtStore"]),
                    ))
                    emp_timeline[ek].append((c_start, c_end))
                    emp_timeline[ek].sort(key=lambda x: x[0])
                    filled = c_start <= gap_start and c_end >= gap_end
                    break

            if not filled:
                warn(
                    f"Cannot safely fill coverage gap at StoreKey={store} "
                    f"{gap_start.date()}..{gap_end.date()} — no SA is free "
                    f"during that period without creating cross-store overlaps"
                )

    # Apply forward extensions
    if extensions:
        for idx, new_end in extensions.items():
            df.at[idx, "EndDate"] = new_end

    # Append new gap-fill rows
    if new_rows:
        out_cols = [
            "EmployeeKey", "StoreKey", "StartDate", "EndDate",
            "FTE", "RoleAtStore", "IsPrimary", "TransferReason",
        ]
        fill_df = pd.DataFrame(new_rows, columns=out_cols)
        df = pd.concat([df, fill_df], ignore_index=True)

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_employee_store_assignments(
    employees: pd.DataFrame,
    seed: int,
    global_start: pd.Timestamp,
    global_end: pd.Timestamp,
    *,
    enabled: bool = True,
    ensure_store_sales_coverage: bool = False,
    mover_share: float = 0.20,
    pool_scope: str = "district",
    allow_store_revisit: bool = True,
    part_time_multiplier: float = 1.0,
    secondary_fte_min: float = 0.10,
    secondary_fte_max: float = 0.40,
    primary_sales_role: str = "Sales Associate",
    role_profiles: Optional[Dict[str, Any]] = None,
    max_non_sales_multiplier_frac: float = 0.35,
    max_non_sales_episodes_frac: float = 0.50,
    movable_sales_per_store: int = 1,
    multi_store_share: Optional[float] = None,
    store_opening_dates: Optional[Dict[int, pd.Timestamp]] = None,
    store_closing_dates: Optional[Dict[int, pd.Timestamp]] = None,
    # Transfer integration (new)
    transfers_df: Optional[pd.DataFrame] = None,
    stores_emp_count: Optional[Dict[int, int]] = None,
    stores_district: Optional[Dict[int, Any]] = None,
    prefer_same_district: bool = True,
) -> pd.DataFrame:
    """Build non-overlapping store assignments for all employees.

    Timeline is constructed in a single pass per employee — no post-hoc
    repair or gap-fill.  Transfers from closed stores are integrated
    during construction.
    """
    out_cols = [
        "EmployeeKey", "StoreKey", "StartDate", "EndDate",
        "FTE", "RoleAtStore", "IsPrimary", "TransferReason",
    ]
    if not enabled:
        empty = pd.DataFrame(columns=out_cols + [
            "AssignmentSequence", "Status", "MaxAssignments",
        ])
        return empty

    if multi_store_share is not None:
        warn(
            "'multi_store_share' is deprecated; use 'mover_share' instead. "
            "Remove 'multi_store_share' from config.yaml to silence this warning."
        )
        mover_share = float(multi_store_share)

    rng = np.random.default_rng(int(seed))

    required = {"EmployeeKey", "HireDate", "TerminationDate", "Title", "DistrictId"}
    missing = [c for c in required if c not in employees.columns]
    if missing:
        raise ValueError(f"employees missing required columns: {missing}")

    emp = employees.copy()
    emp["EmployeeKey"] = emp["EmployeeKey"].astype(np.int32)
    emp["DistrictId"] = emp["DistrictId"].astype("Int16")
    emp["HomeStoreKey"] = _infer_home_store_key(emp)

    # Store-attached only
    emp = emp[emp["HomeStoreKey"].notna()].copy()
    if emp.empty:
        return pd.DataFrame(columns=out_cols + [
            "AssignmentSequence", "Status", "MaxAssignments",
        ])

    hire = pd.to_datetime(emp["HireDate"]).dt.normalize()
    term = pd.to_datetime(emp["TerminationDate"]).dt.normalize()

    window_start = pd.to_datetime(global_start).normalize()
    window_end = pd.to_datetime(global_end).normalize()
    if window_end < window_start:
        window_start, window_end = window_end, window_start

    term_filled = term.fillna(window_end)
    title_arr = emp["Title"].astype(str).to_numpy()

    if "FTE" in emp.columns:
        target_fte = emp["FTE"].astype(np.float64).to_numpy()
    else:
        target_fte = np.ones(len(emp), dtype=np.float64)

    # Store pools
    store_by_district = _build_store_by_district(emp)
    # Exclude online stores from the away-episode pool — only physical stores
    all_stores = sorted({
        int(x) for x in emp["HomeStoreKey"].dropna().astype(int).tolist()
        if int(x) <= ONLINE_STORE_KEY_BASE
    })
    scope = str_or(pool_scope, "district").lower()

    ps_role = str(primary_sales_role or "Sales Associate")

    # Pre-extract per-employee arrays
    ek_arr = emp["EmployeeKey"].astype(np.int32).to_numpy()
    did_arr = emp["DistrictId"].to_numpy()
    home_arr = emp["HomeStoreKey"].fillna(-1).astype(np.int32).to_numpy()
    role_arr = title_arr
    hire_arr = hire.to_numpy()
    term_arr = term.to_numpy()
    term_filled_arr = term_filled.to_numpy()

    # -----------------------------------------------------------------
    # Build transfer destination map UPFRONT
    # -----------------------------------------------------------------
    # transfer_map: {EmployeeKey: (close_date, dest_store, transfer_end)}
    transfer_map: Dict[int, Tuple[pd.Timestamp, int, pd.Timestamp]] = {}
    if transfers_df is not None and not transfers_df.empty and stores_emp_count:
        rng_t = np.random.default_rng(int(seed) ^ 0xDEAD_BEEF)
        # Exclude online stores as transfer destinations
        open_stores = sorted(sk for sk in stores_emp_count if sk <= ONLINE_STORE_KEY_BASE)
        _hire_lookup = _build_emp_date_lookup(ek_arr, hire_arr)
        _term_lookup = _build_emp_date_lookup(ek_arr, term_arr)

        def _store_open_at(sk: int, dt: pd.Timestamp) -> bool:
            if store_opening_dates and sk in store_opening_dates:
                if dt < pd.to_datetime(store_opening_dates[sk]).normalize():
                    return False
            if store_closing_dates and sk in store_closing_dates:
                if dt >= pd.to_datetime(store_closing_dates[sk]).normalize():
                    return False
            return True

        for _, trow in transfers_df.iterrows():
            ek = int(trow["EmployeeKey"])
            orig_sk = int(trow["OriginalStoreKey"])
            t_date = pd.to_datetime(trow["TransferDate"]).normalize()
            did = trow.get("DistrictId")

            # Pick destination store — must be open at the transfer date
            if prefer_same_district and stores_district and pd.notna(did):
                orig_district = stores_district.get(orig_sk, "")
                same_dist = [
                    sk for sk in open_stores
                    if stores_district.get(sk, "") == orig_district
                    and sk != orig_sk and _store_open_at(sk, t_date)
                ]
                candidates = same_dist if same_dist else [
                    sk for sk in open_stores
                    if sk != orig_sk and _store_open_at(sk, t_date)
                ]
            else:
                candidates = [
                    sk for sk in open_stores
                    if sk != orig_sk and _store_open_at(sk, t_date)
                ]

            if not candidates:
                candidates = [sk for sk in open_stores if sk != orig_sk]
            if not candidates:
                candidates = open_stores

            weights = np.array(
                [max(1, stores_emp_count.get(sk, 1)) for sk in candidates],
                dtype=np.float64,
            )
            weights /= weights.sum()
            dest_sk = int(rng_t.choice(candidates, p=weights))

            # Clamp transfer start to HireDate
            hire_date = _hire_lookup.get(ek)
            if hire_date is not None and t_date < hire_date:
                t_date = hire_date

            # Clamp transfer end to employee's termination date
            transfer_end = window_end
            emp_term_dt = _term_lookup.get(ek)
            if emp_term_dt is not None and emp_term_dt < transfer_end:
                transfer_end = emp_term_dt

            # Use the store's actual closing date (not the transfer date)
            # so the SA stays at the home store until its last operational day.
            actual_close = t_date
            if store_closing_dates and orig_sk in store_closing_dates:
                actual_close = pd.to_datetime(store_closing_dates[orig_sk]).normalize()

            transfer_map[ek] = (actual_close, dest_sk, transfer_end)

    # -----------------------------------------------------------------
    # Coverage anchors + explicit mover pool
    # -----------------------------------------------------------------
    anchor_keys: set[int] = set()
    mover_keys: set[int] = set()
    anchor_by_store: Dict[int, int] = {}

    if bool(ensure_store_sales_coverage):
        staff_idx = np.full(len(emp), -1, dtype=np.int32)
        staff_mask = ek_arr >= STAFF_KEY_BASE
        if bool(np.any(staff_mask)):
            staff_idx[staff_mask] = (ek_arr[staff_mask] - STAFF_KEY_BASE) % STAFF_KEY_STORE_MULT

        n_movable = max(0, int(movable_sales_per_store))

        _active_stores = {int(s) for s in all_stores if int(s) <= ONLINE_STORE_KEY_BASE}
        if store_closing_dates:
            for sk, cd in store_closing_dates.items():
                _effective_open = window_start
                if store_opening_dates and sk in store_opening_dates:
                    _od = store_opening_dates[sk]
                    if _od > window_start:
                        _effective_open = _od
                if pd.to_datetime(cd).normalize() < _effective_open:
                    _active_stores.discard(int(sk))

        _MIN_TENURE_FOR_TRANSFER_DAYS = 365
        ws_np = np.datetime64(window_start, "ns")
        we_np = np.datetime64(window_end, "ns")

        def _anchors_cover(
            anch_starts: np.ndarray,
            anch_ends: np.ndarray,
            target_start: np.int64,
            target_end: np.int64,
        ) -> bool:
            mask = (anch_ends >= target_start) & (anch_starts <= target_end)
            if not np.any(mask):
                return False
            starts = anch_starts[mask]
            ends = anch_ends[mask]
            order = np.argsort(starts)
            starts = starts[order]
            ends = ends[order]
            covered_until = starts[0]
            if covered_until > target_start:
                return False
            covered_until = ends[0]
            for k in range(1, len(starts)):
                if starts[k] > covered_until + np.timedelta64(1, "D"):
                    break
                if ends[k] > covered_until:
                    covered_until = ends[k]
            return bool(covered_until >= target_end)

        for store in all_stores:
            if int(store) not in _active_stores:
                continue
            m_store_role = (home_arr == int(store)) & (role_arr == ps_role)
            if not bool(np.any(m_store_role)):
                continue

            cand_pos = np.where(m_store_role)[0]
            cand_ek = ek_arr[cand_pos]
            cand_hire_ns = np.asarray(hire_arr[cand_pos], dtype="datetime64[ns]")
            cand_term_ns = np.asarray(term_arr[cand_pos], dtype="datetime64[ns]")

            eff_start = np.maximum(cand_hire_ns, ws_np)
            eff_end = np.where(
                np.isnat(cand_term_ns), we_np,
                np.minimum(cand_term_ns, we_np),
            )
            tenure_days = (eff_end - eff_start) / np.timedelta64(1, "D")

            store_anchor_set: set[int] = set(cand_ek.astype(int))
            store_mover_set: set[int] = set()

            order = np.argsort(-tenure_days)
            for idx in order:
                if tenure_days[idx] < _MIN_TENURE_FOR_TRANSFER_DAYS:
                    continue
                ek_val = int(cand_ek[idx])
                other_mask = np.array(
                    [int(cand_ek[j]) in store_anchor_set and j != idx
                     for j in range(len(cand_ek))],
                    dtype=bool,
                )
                if not np.any(other_mask):
                    continue
                if _anchors_cover(
                    eff_start[other_mask], eff_end[other_mask],
                    eff_start[idx], eff_end[idx],
                ):
                    store_anchor_set.remove(ek_val)
                    store_mover_set.add(ek_val)

            anchor_keys.update(store_anchor_set)
            mover_keys.update(store_mover_set)
            if store_anchor_set:
                anchor_by_store[int(store)] = min(store_anchor_set)

        missing_stores = [int(s) for s in _active_stores if int(s) not in anchor_by_store]
        if missing_stores:
            raise RuntimeError(
                f"ensure_store_sales_coverage=true but no '{ps_role}' employees "
                f"found for stores: {missing_stores}. "
                f"Increase employees.min_staff_per_store or ensure "
                f"min_primary_sales_per_store>=1."
            )

    # -----------------------------------------------------------------
    # Parse role profiles
    # -----------------------------------------------------------------
    rp = as_dict(role_profiles) if role_profiles is not None else {}
    default_profile = _normalize_profile_keys(as_dict(rp.get("default")))
    per_role_profile = {
        k: _normalize_profile_keys(dict(v))
        for k, v in rp.items() if k != "default" and isinstance(v, Mapping)
    }

    sales_profile = dict(default_profile)
    sales_profile.update(per_role_profile.get(primary_sales_role, {}))
    sales_profile.setdefault("role_multiplier", 2.0)
    sales_profile.setdefault("episodes_max", 4)
    sales_profile.setdefault("episodes_min", 1)
    sales_profile.setdefault("duration_days_min", 14)
    sales_profile.setdefault("duration_days_max", 120)

    base = float(np.clip(float_or(mover_share, 0.03), 0.0, 1.0))
    pt_mult = float(max(1.0, float_or(part_time_multiplier, 1.0)))

    clamped_mult_frac = float(np.clip(max_non_sales_multiplier_frac, 0.0, 1.0))
    clamped_ep_frac = float(np.clip(max_non_sales_episodes_frac, 0.0, 1.0))

    def _candidate_stores(did: Any, home_store: int) -> List[int]:
        if scope == "all":
            return [s for s in all_stores if s != home_store]
        if pd.isna(did):
            return []
        pool = store_by_district.get(did, [])
        return [int(s) for s in pool if int(s) != home_store]

    # Build per-role profile cache
    _role_cache: Dict[str, Dict[str, Any]] = {}
    hire_ns = np.asarray(hire_arr, dtype="datetime64[ns]")
    term_ns = np.asarray(term_arr, dtype="datetime64[ns]")
    term_filled_ns = np.asarray(term_filled_arr, dtype="datetime64[ns]")

    valid = ~np.isnat(hire_ns) & ~np.isnat(term_filled_ns) & (term_filled_ns >= hire_ns)
    unique_roles = set(role_arr[valid])
    for r in unique_roles:
        if r == "Store Manager":
            _role_cache[r] = {
                "role_multiplier": 0.0, "episodes_min": 0,
                "episodes_max": 0, "duration_days_min": 30,
                "duration_days_max": 180,
            }
        else:
            prof = _get_profile(
                r,
                primary_sales_role=primary_sales_role,
                default_profile=default_profile,
                per_role_profile=per_role_profile,
                sales_profile=sales_profile,
                max_non_sales_multiplier_frac=clamped_mult_frac,
                max_non_sales_episodes_frac=clamped_ep_frac,
            )
            _role_cache[r] = {
                "role_multiplier": float(max(0.0, float_or(prof.get("role_multiplier"), 1.0))),
                "episodes_min": max(0, int_or(prof.get("episodes_min"), 0)),
                "episodes_max": max(
                    max(0, int_or(prof.get("episodes_min"), 0)),
                    int_or(prof.get("episodes_max"), 0),
                ),
                "duration_days_min": max(1, int_or(prof.get("duration_days_min"), 30)),
                "duration_days_max": max(
                    max(1, int_or(prof.get("duration_days_min"), 30)),
                    int_or(prof.get("duration_days_max"), 180),
                ),
            }

    # Vectorized movement probability
    role_mult_arr = np.array(
        [_role_cache.get(r, {"role_multiplier": 1.0})["role_multiplier"] for r in role_arr],
        dtype=np.float64,
    )
    emax_arr = np.array(
        [_role_cache.get(r, {"episodes_max": 0})["episodes_max"] for r in role_arr],
        dtype=np.int32,
    )

    pt_factor = np.where(target_fte < 1.0, pt_mult, 1.0)
    p_move_all = np.clip(base * role_mult_arr * pt_factor, 0.0, 0.90)

    move_draws = rng.random(len(emp))

    if bool(ensure_store_sales_coverage) and mover_keys:
        forced_mover = np.isin(ek_arr, np.array(sorted(mover_keys), dtype=np.int32))
    else:
        forced_mover = np.zeros(len(emp), dtype=bool)

    # -----------------------------------------------------------------
    # Per-employee store opening/closing dates (for clamping)
    # -----------------------------------------------------------------
    _store_open_ns = None
    _store_close_ns = None
    if store_opening_dates:
        _store_open_ns = np.array([
            np.datetime64(store_opening_dates.get(int(sk), window_start), "ns")
            for sk in home_arr
        ], dtype="datetime64[ns]")
    if store_closing_dates:
        # Last operational day = closing_date - 1 day (only for stores that actually close)
        _one_day_ns = np.timedelta64(1, "D")
        _far_future = np.datetime64("2262-04-11", "ns")
        _store_close_ns = np.array([
            np.datetime64(store_closing_dates[int(sk)], "ns") - _one_day_ns
            if int(sk) in store_closing_dates
            else _far_future
            for sk in home_arr
        ], dtype="datetime64[ns]")

    # Compute effective start/end for all employees
    ws_np = np.datetime64(window_start, "ns")
    we_np = np.datetime64(window_end, "ns")
    gen_start_all = np.maximum(hire_ns, ws_np)
    plan_end_all = np.minimum(term_filled_ns, we_np)

    if _store_open_ns is not None:
        gen_start_all = np.maximum(gen_start_all, _store_open_ns)
    if _store_close_ns is not None:
        plan_end_all = np.minimum(plan_end_all, _store_close_ns)

    valid &= (plan_end_all >= gen_start_all) & (gen_start_all <= we_np)

    # Exclude anchors and online employees from mover classification
    _online_emp_mask = ek_arr >= ONLINE_EMP_KEY_BASE
    if bool(ensure_store_sales_coverage) and anchor_keys:
        valid_for_move = valid & ~np.isin(ek_arr, np.array(sorted(anchor_keys), dtype=np.int32))
    else:
        valid_for_move = valid.copy()
    valid_for_move &= ~_online_emp_mask  # online employees never move

    will_move = valid_for_move & (emax_arr > 0) & (forced_mover | (move_draws < p_move_all))
    non_mover = valid & ~will_move
    # Anchors and online employees are non-movers
    if bool(ensure_store_sales_coverage) and anchor_keys:
        anchor_mask = np.isin(ek_arr, np.array(sorted(anchor_keys), dtype=np.int32))
        non_mover |= (valid & anchor_mask)
        will_move &= ~anchor_mask
    non_mover |= (valid & _online_emp_mask)

    open_ended_all = np.isnat(term_ns) | (term_ns > we_np)
    end_date_all = np.where(open_ended_all, we_np, np.minimum(term_ns, we_np))
    if _store_close_ns is not None:
        end_date_all = np.minimum(end_date_all, _store_close_ns)

    # -----------------------------------------------------------------
    # Phase 2: Build timelines
    # -----------------------------------------------------------------
    batch_dfs: List[pd.DataFrame] = []
    all_slots: List[Dict[str, Any]] = []

    # --- Batch: non-movers (single assignment, no loop) ---
    # Separate non-movers into: those with transfers and those without
    nm_idx_all = np.where(non_mover)[0]
    nm_transferred = []
    nm_simple = []
    for i in nm_idx_all:
        ek = int(ek_arr[i])
        if ek in transfer_map:
            nm_transferred.append(i)
        else:
            nm_simple.append(i)

    nm_simple = np.array(nm_simple, dtype=np.intp) if nm_simple else np.array([], dtype=np.intp)

    if nm_simple.size > 0:
        nm_start = gen_start_all[nm_simple]
        nm_end = end_date_all[nm_simple]
        ok = nm_end >= nm_start
        nm_simple = nm_simple[ok]
        if nm_simple.size > 0:
            batch_dfs.append(pd.DataFrame({
                "EmployeeKey": ek_arr[nm_simple],
                "StoreKey": home_arr[nm_simple],
                "StartDate": gen_start_all[nm_simple],
                "EndDate": end_date_all[nm_simple],
                "FTE": target_fte[nm_simple],
                "RoleAtStore": role_arr[nm_simple],
                "IsPrimary": True,
                "TransferReason": pd.NA,
            }))

    # --- Non-movers with transfers: build per-employee timeline ---
    for i in nm_transferred:
        ek = int(ek_arr[i])
        home_store = int(home_arr[i])
        role = str(role_arr[i])
        fte = float(target_fte[i])
        eff_start = pd.Timestamp(gen_start_all[i])
        eff_end = pd.Timestamp(end_date_all[i])

        t_info = transfer_map.get(ek)
        slots = _build_employee_timeline(
            rng=rng,
            home_store=home_store,
            role=role,
            fte=fte,
            effective_start=eff_start,
            effective_end=eff_end,
            candidate_stores=[],
            episodes_min=0,
            episodes_max=0,
            duration_days_min=30,
            duration_days_max=180,
            allow_store_revisit=allow_store_revisit,
            store_opening_dates=store_opening_dates,
            store_closing_dates=store_closing_dates,
            transfer_info=t_info,
        )
        for sl in slots:
            all_slots.append({
                "EmployeeKey": ek,
                "StoreKey": sl.store_key,
                "StartDate": sl.start_date,
                "EndDate": sl.end_date,
                "FTE": sl.fte,
                "RoleAtStore": sl.role,
                "IsPrimary": sl.is_primary,
                "TransferReason": sl.transfer_reason if sl.transfer_reason else pd.NA,
            })

    # --- Loop: movers ---
    mover_indices = np.where(will_move)[0]
    for i in mover_indices:
        ek = int(ek_arr[i])
        did = did_arr[i]
        home_store = int(home_arr[i])
        role = str(role_arr[i])
        fte = float(target_fte[i])

        eff_start = pd.Timestamp(gen_start_all[i])
        eff_end = pd.Timestamp(end_date_all[i])

        rp_cached = _role_cache.get(role, {})
        e_min = int(rp_cached.get("episodes_min", 0))
        e_max = int(rp_cached.get("episodes_max", 0))
        dmin = int(rp_cached.get("duration_days_min", 30))
        dmax = int(rp_cached.get("duration_days_max", 180))

        is_fm = bool(forced_mover[i])
        other = _candidate_stores(did, home_store)
        t_info = transfer_map.get(ek)

        slots = _build_employee_timeline(
            rng=rng,
            home_store=home_store,
            role=role,
            fte=fte,
            effective_start=eff_start,
            effective_end=eff_end,
            candidate_stores=other,
            episodes_min=e_min,
            episodes_max=e_max,
            duration_days_min=dmin,
            duration_days_max=dmax,
            allow_store_revisit=allow_store_revisit,
            store_opening_dates=store_opening_dates,
            store_closing_dates=store_closing_dates,
            is_forced_mover=is_fm,
            transfer_info=t_info,
        )
        for sl in slots:
            all_slots.append({
                "EmployeeKey": ek,
                "StoreKey": sl.store_key,
                "StartDate": sl.start_date,
                "EndDate": sl.end_date,
                "FTE": sl.fte,
                "RoleAtStore": sl.role,
                "IsPrimary": sl.is_primary,
                "TransferReason": sl.transfer_reason if sl.transfer_reason else pd.NA,
            })

    # -----------------------------------------------------------------
    # Assemble output
    # -----------------------------------------------------------------
    parts = list(batch_dfs)
    if all_slots:
        parts.append(pd.DataFrame(all_slots, columns=out_cols))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=out_cols)

    if out.empty:
        out["AssignmentSequence"] = pd.Series(dtype=np.int32)
        out["Status"] = pd.Series(dtype="object")
        out["MaxAssignments"] = pd.Series(dtype=np.int32)
        return out

    out["EmployeeKey"] = out["EmployeeKey"].astype(np.int32)
    out["StoreKey"] = out["StoreKey"].astype(np.int32)
    out["FTE"] = out["FTE"].astype(np.float64)
    out["IsPrimary"] = out["IsPrimary"].astype(bool)
    out["RoleAtStore"] = out["RoleAtStore"].astype(str)
    out["StartDate"] = pd.to_datetime(out["StartDate"]).dt.normalize()
    out["EndDate"] = pd.to_datetime(out["EndDate"]).dt.normalize()
    out["EndDate"] = out["EndDate"].fillna(window_end)

    # -----------------------------------------------------------------
    # Fill SA coverage gaps (safe — only extends when no cross-store overlap)
    # -----------------------------------------------------------------
    _emp_hire_lookup = _build_emp_date_lookup(ek_arr, hire_ns)
    _emp_term_lookup = _build_emp_date_lookup(ek_arr, term_ns)

    if ensure_store_sales_coverage:
        out = _fill_coverage_gaps_safe(
            out,
            ps_role=ps_role,
            all_stores=all_stores,
            window_start=window_start,
            window_end=window_end,
            store_opening_dates=store_opening_dates,
            store_closing_dates=store_closing_dates,
            emp_hire=_emp_hire_lookup,
            emp_term=_emp_term_lookup,
        )
        # Re-normalize after gap-fill additions
        out["EmployeeKey"] = out["EmployeeKey"].astype(np.int32)
        out["StoreKey"] = out["StoreKey"].astype(np.int32)
        out["StartDate"] = pd.to_datetime(out["StartDate"]).dt.normalize()
        out["EndDate"] = pd.to_datetime(out["EndDate"]).dt.normalize()

    # Sort and assign sequence
    out = out.sort_values(["EmployeeKey", "StartDate", "EndDate"]).reset_index(drop=True)
    out["AssignmentSequence"] = out.groupby("EmployeeKey").cumcount().astype(np.int32) + 1

    # Derive Status and MaxAssignments
    out["Status"] = _derive_status(out, window_end)
    max_asgn = out.groupby("EmployeeKey")["AssignmentSequence"].transform("max").astype(np.int32)
    out["MaxAssignments"] = max_asgn

    # -----------------------------------------------------------------
    # Final validation
    # -----------------------------------------------------------------
    _validate_assignments(
        out,
        ps_role=ps_role,
        all_stores=all_stores,
        window_start=window_start,
        window_end=window_end,
        store_opening_dates=store_opening_dates,
        store_closing_dates=store_closing_dates,
        emp_hire=_emp_hire_lookup,
        emp_term=_emp_term_lookup,
        coverage_mode="warn",
    )

    return out


# ---------------------------------------------------------------------------
# Pipeline entrypoint
# ---------------------------------------------------------------------------

def run_employee_store_assignments(cfg: Dict[str, Any], parquet_folder: Path) -> None:
    cfg = cfg or {}
    a_cfg = _store_assignments_cfg(cfg)

    parquet_folder = Path(parquet_folder)
    parquet_folder.mkdir(parents=True, exist_ok=True)

    employees_path = parquet_folder / "employees.parquet"
    out_path = parquet_folder / "employee_store_assignments.parquet"

    if not employees_path.exists():
        raise FileNotFoundError(f"Missing employees parquet: {employees_path}")

    seed = resolve_seed(cfg, a_cfg, fallback=42)
    global_start, global_end = parse_global_dates(
        cfg, a_cfg,
        allow_override=True,
        dimension_name="employee_store_assignments",
    )

    employees = pd.read_parquet(
        employees_path,
        columns=["EmployeeKey", "HireDate", "TerminationDate", "Title", "DistrictId", "FTE"],
    )

    version_cfg = dict(a_cfg)
    version_cfg["schema_version"] = 12  # bumped for online store key pattern
    version_cfg["_stores_cfg"] = dict(cfg.stores)
    version_cfg["_rows_employees"] = int(len(employees))
    if "EmployeeKey" in employees.columns and len(employees) > 0:
        ek = pd.to_numeric(employees["EmployeeKey"], errors="coerce").dropna().astype(np.int32)
        if len(ek) > 0:
            version_cfg["_emp_key_min"] = int(ek.min())
            version_cfg["_emp_key_max"] = int(ek.max())
            version_cfg["_emp_key_sum"] = int(ek.sum())
    if "HireDate" in employees.columns and len(employees) > 0:
        hd = pd.to_datetime(employees["HireDate"], errors="coerce").dropna()
        if len(hd) > 0:
            version_cfg["_hire_min"] = str(hd.min().date())
            version_cfg["_hire_max"] = str(hd.max().date())
    version_cfg["_global_dates"] = {
        "start": str(global_start.date()),
        "end": str(global_end.date()),
    }

    if not should_regenerate("employee_store_assignments", version_cfg, out_path):
        skip("Employee Store Assignments up-to-date")
        return

    role_profiles = as_dict(a_cfg.get("role_profiles"))

    # Read store opening/closing dates
    stores_path = parquet_folder / "stores.parquet"
    store_opening_dates: Optional[Dict[int, pd.Timestamp]] = None
    store_closing_dates: Optional[Dict[int, pd.Timestamp]] = None
    if stores_path.exists():
        try:
            _stores_df = pd.read_parquet(stores_path, columns=["StoreKey", "OpeningDate", "ClosingDate"])
        except (KeyError, ValueError):
            _stores_df = None
        if _stores_df is not None:
            _sk_arr = _stores_df["StoreKey"].astype(np.int32).to_numpy()
            if "OpeningDate" in _stores_df.columns:
                _od = pd.to_datetime(_stores_df["OpeningDate"], errors="coerce").dt.normalize()
                store_opening_dates = {
                    int(sk): ts for sk, ts in zip(_sk_arr, _od) if pd.notna(ts)
                }
            if "ClosingDate" in _stores_df.columns:
                _cd = pd.to_datetime(_stores_df["ClosingDate"], errors="coerce").dt.normalize()
                store_closing_dates = {
                    int(sk): ts for sk, ts in zip(_sk_arr, _cd) if pd.notna(ts)
                }

    # Read employee transfers sidecar
    transfers_path = parquet_folder / "employee_transfers.parquet"
    transfers_df: Optional[pd.DataFrame] = None
    if transfers_path.exists():
        transfers_df = pd.read_parquet(transfers_path)

    # Read store info for transfer destination selection
    stores_emp_count: Optional[Dict[int, int]] = None
    stores_district: Optional[Dict[int, Any]] = None
    if stores_path.exists():
        try:
            _st_info = pd.read_parquet(
                stores_path,
                columns=["StoreKey", "EmployeeCount", "StoreDistrict", "Status"],
            )
            _sk = _st_info["StoreKey"].astype(np.int32).to_numpy()
            _ec = _st_info["EmployeeCount"].fillna(0).astype(int).to_numpy()
            _sd = _st_info["StoreDistrict"].astype(str).to_numpy()
            _ss = _st_info["Status"].astype(str).to_numpy()
            _eligible = (_ss == "Open") | (_ss == "Renovating")
            stores_emp_count = {int(k): int(c) for k, c, o in zip(_sk, _ec, _eligible) if o}
            stores_district = {int(k): str(d) for k, d in zip(_sk, _sd)}
        except (KeyError, ValueError):
            pass

    _closing_cfg = as_dict(cfg.stores.closing) if hasattr(cfg.stores, "closing") and cfg.stores.closing is not None else {}
    _prefer_same_district = bool(_closing_cfg.get("prefer_same_district", True))

    with stage("Generating Employee Store Assignments"):
        df = generate_employee_store_assignments(
            employees=employees,
            seed=seed,
            global_start=global_start,
            global_end=global_end,
            enabled=bool(a_cfg.get("enabled", True)),
            ensure_store_sales_coverage=bool(a_cfg.get("ensure_store_sales_coverage", False)),
            mover_share=float_or(
                a_cfg.get("mover_share", a_cfg.get("multi_store_share", 0.03)), 0.03,
            ),
            pool_scope=str_or(a_cfg.get("pool_scope"), "district"),
            allow_store_revisit=bool(a_cfg.get("allow_store_revisit", True)),
            part_time_multiplier=float_or(a_cfg.get("part_time_multiplier", 1.0), 1.0),
            secondary_fte_min=float_or(a_cfg.get("secondary_fte_min", 0.10), 0.10),
            secondary_fte_max=float_or(a_cfg.get("secondary_fte_max", 0.40), 0.40),
            primary_sales_role=str_or(a_cfg.get("primary_sales_role"), "Sales Associate"),
            role_profiles=role_profiles,
            max_non_sales_multiplier_frac=float_or(
                a_cfg.get("max_non_sales_multiplier_frac", 0.35), 0.35,
            ),
            max_non_sales_episodes_frac=float_or(
                a_cfg.get("max_non_sales_episodes_frac", 0.50), 0.50,
            ),
            movable_sales_per_store=int_or(a_cfg.get("movable_sales_per_store"), 1),
            multi_store_share=(
                a_cfg.get("multi_store_share") if "multi_store_share" in a_cfg else None
            ),
            store_opening_dates=store_opening_dates,
            store_closing_dates=store_closing_dates,
            # Transfer integration — all handled inside generate now
            transfers_df=transfers_df,
            stores_emp_count=stores_emp_count,
            stores_district=stores_district,
            prefer_same_district=_prefer_same_district,
        )

        # Update employees.parquet StoreKey for transferred employees
        _td = _transfer_dest_from_df(df, transfers_df)
        if _td:
            emp_full = pd.read_parquet(employees_path)
            updated = 0
            for ek, dest in _td.items():
                mask = emp_full["EmployeeKey"] == ek
                if mask.any():
                    emp_full.loc[mask, "StoreKey"] = np.int32(dest)
                    updated += 1
            if updated > 0:
                emp_full.to_parquet(employees_path, index=False)
                info(f"Updated StoreKey for {updated} transferred employees in employees.parquet")

                if stores_path.exists():
                    sk_col = emp_full["StoreKey"].dropna()
                    sk_col = sk_col[sk_col > 0].astype(np.int32)
                    actual_counts = sk_col.value_counts().to_dict()

                    stores_full = pd.read_parquet(stores_path)
                    stores_full["EmployeeCount"] = stores_full["StoreKey"].map(
                        lambda sk: actual_counts.get(int(sk), 0)
                    ).astype(np.int64)

                    write_parquet_with_date32(
                        stores_full, stores_path,
                        date_cols=["OpeningDate", "ClosingDate"],
                        cast_all_datetime=False,
                        compression="snappy", compression_level=None,
                        force_date32=True,
                    )
                    info("Re-synced stores.parquet EmployeeCount after transfers")

        # Remove transfers sidecar
        if transfers_path.exists():
            transfers_path.unlink()

        write_parquet_with_date32(
            df,
            out_path,
            date_cols=["StartDate", "EndDate"],
            cast_all_datetime=False,
            compression=str_or(a_cfg.get("parquet_compression"), "snappy"),
            compression_level=(
                int(a_cfg["parquet_compression_level"])
                if "parquet_compression_level" in a_cfg
                else None
            ),
            force_date32=True,
        )

    save_version("employee_store_assignments", version_cfg, out_path)
    info(f"Employee Store Assignments written: {out_path.name}")


def _transfer_dest_from_df(
    df: pd.DataFrame,
    transfers_df: Optional[pd.DataFrame],
) -> Dict[int, int]:
    """Extract {EmployeeKey: destination StoreKey} for transferred employees.

    Uses the last assignment per transferred employee as the destination.
    """
    if transfers_df is None or transfers_df.empty:
        return {}

    transferred_eks = set(transfers_df["EmployeeKey"].astype(int).tolist())
    if not transferred_eks:
        return {}

    sub = df[df["EmployeeKey"].isin(transferred_eks)].copy()
    if sub.empty:
        return {}

    # Last assignment per employee (highest AssignmentSequence)
    sub = sub.sort_values(["EmployeeKey", "AssignmentSequence"])
    last = sub.groupby("EmployeeKey").last()
    return {int(ek): int(sk) for ek, sk in zip(last.index, last["StoreKey"].values)}
