from __future__ import annotations

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


# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------

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

    # The dimensions runner injects internal keys (global_dates)
    # into cfg["employee_store_assignments"]; exclude those when checking for
    # actual user-supplied legacy config.
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
    """
    Derive the home StoreKey from the EmployeeKey encoding when the column
    is not already present.
    """
    if "StoreKey" in employees.columns:
        return employees["StoreKey"].astype("Int32")

    ek = employees["EmployeeKey"].astype(np.int32)
    out = pd.Series([pd.NA] * len(employees), dtype="Int32")

    mgr_mask = (ek >= STORE_MGR_KEY_BASE) & (ek < STAFF_KEY_BASE)
    if mgr_mask.any():
        out.loc[mgr_mask] = (ek.loc[mgr_mask] - STORE_MGR_KEY_BASE).astype("Int32")

    staff_mask = ek >= STAFF_KEY_BASE
    if staff_mask.any():
        out.loc[staff_mask] = (
            (ek.loc[staff_mask] - STAFF_KEY_BASE) // STAFF_KEY_STORE_MULT
        ).astype("Int32")

    return out


def _build_store_by_district(emp: pd.DataFrame) -> Dict[Any, List[int]]:
    """``DistrictId → [StoreKey, ...]``"""
    emp2 = emp[emp["DistrictId"].notna() & emp["HomeStoreKey"].notna()].copy()
    if emp2.empty:
        return {}

    is_mgr = emp2["Title"].astype(str) == "Store Manager"
    mgrs = emp2[is_mgr]
    if not mgrs.empty:
        src = mgrs
    else:
        warn(
            "No Store Managers found when building district→store mapping; "
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


# -----------------------------------------------------------------------------
# Movement-profile helpers
# -----------------------------------------------------------------------------

def _normalize_profile_keys(prof: Dict[str, Any]) -> Dict[str, Any]:
    """
    Map shorthand config keys to the long-form keys consumed by the generator.

    Shorthand → long-form:
      mult             → role_multiplier
      episodes: [a, b] → episodes_min, episodes_max
      duration: [a, b] → duration_days_min, duration_days_max

    Long-form keys take precedence when both are present.
    """
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
    """
    Merge ``default_profile ← per_role_profile[role]``, then enforce
    non-sales caps relative to the sales profile.
    """
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


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

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
) -> pd.DataFrame:
    """
    Exclusive (non-overlapping) store assignments per employee, with
    role-weighted movement.

    Output columns:
      EmployeeKey, StoreKey, StartDate, EndDate, FTE, RoleAtStore,
      IsPrimary, AssignmentSequence
    """
    out_cols = [
        "EmployeeKey", "StoreKey", "StartDate", "EndDate",
        "FTE", "RoleAtStore", "IsPrimary", "TransferReason",
    ]
    if not enabled:
        return pd.DataFrame(columns=out_cols + ["AssignmentSequence"])

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
        return pd.DataFrame(columns=out_cols + ["AssignmentSequence"])

    hire = pd.to_datetime(emp["HireDate"]).dt.normalize()
    term = pd.to_datetime(emp["TerminationDate"]).dt.normalize()

    window_start = pd.to_datetime(global_start).normalize()
    window_end = pd.to_datetime(global_end).normalize()
    if window_end < window_start:
        window_start, window_end = window_end, window_start

    term_filled = term.fillna(window_end)
    title_arr = emp["Title"].astype(str).to_numpy()

    # FTE from employee dimension (determined at hire)
    if "FTE" in emp.columns:
        target_fte = emp["FTE"].astype(np.float64).to_numpy()
    else:
        target_fte = np.ones(len(emp), dtype=np.float64)

    # Store pools
    store_by_district = _build_store_by_district(emp)
    all_stores = sorted({int(x) for x in emp["HomeStoreKey"].dropna().astype(int).tolist()})
    scope = str_or(pool_scope, "district").lower()

    ps_role = str(primary_sales_role or "Sales Associate")

    # -----------------------------------------------------------------
    # Pre-extract all per-employee arrays (avoid .iloc inside the loop)
    # -----------------------------------------------------------------
    ek_arr = emp["EmployeeKey"].astype(np.int32).to_numpy()
    did_arr = emp["DistrictId"].to_numpy()
    home_arr = emp["HomeStoreKey"].fillna(-1).astype(np.int32).to_numpy()
    role_arr = title_arr
    hire_arr = hire.to_numpy()
    term_arr = term.to_numpy()
    term_filled_arr = term_filled.to_numpy()

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

        # Determine which stores are active during the dataset window
        _active_stores = set(all_stores)
        if store_closing_dates:
            for sk, cd in store_closing_dates.items():
                _effective_open = window_start
                if store_opening_dates and sk in store_opening_dates:
                    _od = store_opening_dates[sk]
                    if _od > window_start:
                        _effective_open = _od
                if pd.to_datetime(cd).normalize() < _effective_open:
                    _active_stores.discard(int(sk))

        # Build anchor/mover sets per store.  An SA can safely become a
        # mover if the remaining anchors at the store fully cover the
        # mover's tenure (so the store is never left unstaffed during
        # away episodes).  Short-tenure SAs (< 1 year) stay anchors.
        _MIN_TENURE_FOR_TRANSFER_DAYS = 365
        ws_np = np.datetime64(window_start, "ns")
        we_np = np.datetime64(window_end, "ns")

        def _anchors_cover(
            anch_starts: np.ndarray,
            anch_ends: np.ndarray,
            target_start: np.int64,
            target_end: np.int64,
        ) -> bool:
            """Check merged anchor intervals fully cover [target_start, target_end]."""
            # Filter to anchors overlapping the target window
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

            # Effective tenure per SA (clamped to data window)
            eff_start = np.maximum(cand_hire_ns, ws_np)
            eff_end = np.where(
                np.isnat(cand_term_ns), we_np,
                np.minimum(cand_term_ns, we_np),
            )
            tenure_days = (eff_end - eff_start) / np.timedelta64(1, "D")

            # Start with everyone as anchor
            store_anchor_set: set[int] = set(cand_ek.astype(int))
            store_mover_set: set[int] = set()

            # Try to convert long-tenure SAs to movers (longest first)
            order = np.argsort(-tenure_days)
            for idx in order:
                if tenure_days[idx] < _MIN_TENURE_FOR_TRANSFER_DAYS:
                    continue
                ek_val = int(cand_ek[idx])
                # Build arrays of remaining anchors (excluding this SA)
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

    fmin = float(max(0.0, float_or(secondary_fte_min, 0.10)))
    fmax = float(max(fmin, float_or(secondary_fte_max, 0.40)))

    # Clip these once (they are constant for the entire run)
    clamped_mult_frac = float(np.clip(max_non_sales_multiplier_frac, 0.0, 1.0))
    clamped_ep_frac = float(np.clip(max_non_sales_episodes_frac, 0.0, 1.0))

    def _candidate_stores(did: Any, home_store: int) -> List[int]:
        if scope == "all":
            return [s for s in all_stores if s != home_store]
        if pd.isna(did):
            return []
        pool = store_by_district.get(did, [])
        return [int(s) for s in pool if int(s) != home_store]

    def _sample_non_overlapping_episodes(
        start_min: pd.Timestamp,
        end_max: pd.Timestamp,
        other_stores: List[int],
        k: int,
        tgt_fte: float,
        dmin: int,
        dmax: int,
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp, int, float]]:
        if k <= 0 or not other_stores:
            return []

        # Pick stores ensuring no two consecutive picks are the same.
        # When allow_store_revisit is False, numpy handles uniqueness;
        # otherwise we re-roll consecutive duplicates so that each
        # transfer is to a genuinely different location.
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

        raw: List[Tuple[pd.Timestamp, int, int, float]] = []
        for store in chosen:
            # Clamp episode window to the target store's open/close dates
            ep_start_min = start_min
            ep_end_max = end_max
            _sk = int(store)
            if store_opening_dates and _sk in store_opening_dates:
                _sod = pd.to_datetime(store_opening_dates[_sk]).normalize()
                if _sod > ep_start_min:
                    ep_start_min = _sod
            if store_closing_dates and _sk in store_closing_dates:
                _scd = pd.to_datetime(store_closing_dates[_sk]).normalize()
                if _scd < ep_end_max:
                    ep_end_max = _scd
            if ep_end_max < ep_start_min:
                continue  # store not active during employee's window

            dur = int(rng.integers(dmin, dmax + 1))
            latest_start = (ep_end_max - pd.Timedelta(days=dur - 1)).normalize()
            if latest_start < ep_start_min:
                continue
            s = rand_single_date(rng, ep_start_min, latest_start)
            raw.append((s, dur, int(store), tgt_fte))

        if not raw:
            return []

        raw.sort(key=lambda x: x[0])
        placed: List[Tuple[pd.Timestamp, pd.Timestamp, int, float]] = []
        last_end: Optional[pd.Timestamp] = None

        for s, dur, store, sec_fte in raw:
            if last_end is not None and s <= last_end:
                s = (last_end + pd.Timedelta(days=1)).normalize()
            if s > end_max:
                continue
            e = (s + pd.Timedelta(days=dur - 1)).normalize()
            if e > end_max:
                e = end_max
            if e < s:
                continue
            placed.append((s, e, store, sec_fte))
            last_end = e

        # Safety net: merge consecutive same-store episodes that can
        # still arise after date-sorting reorders the picks.
        if len(placed) <= 1:
            return placed

        out: List[Tuple[pd.Timestamp, pd.Timestamp, int, float]] = [placed[0]]
        for s, e, store, sec_fte in placed[1:]:
            prev_s, prev_e, prev_store, prev_fte = out[-1]
            if store == prev_store:
                # Extend the previous episode to cover this one
                out[-1] = (prev_s, e, prev_store, prev_fte)
            else:
                out.append((s, e, store, sec_fte))

        return out

    # -----------------------------------------------------------------
    # Emit assignment rows
    # -----------------------------------------------------------------
    batch_dfs: List[pd.DataFrame] = []
    mover_rows: List[Dict[str, Any]] = []

    def _emit_mover(
        ek: int, role: str, store_key: int,
        seg_start: pd.Timestamp, seg_end: Any,
        is_primary: bool, fte_val: float,
        transfer_reason: Any = pd.NA,
    ) -> None:
        mover_rows.append(dict(
            EmployeeKey=int(ek),
            StoreKey=int(store_key),
            StartDate=pd.to_datetime(seg_start).normalize(),
            EndDate=(pd.to_datetime(seg_end).normalize() if pd.notna(seg_end) else pd.NaT),
            FTE=float(fte_val),
            RoleAtStore=str(role),
            IsPrimary=bool(is_primary),
            TransferReason=transfer_reason,
        ))

    # -----------------------------------------------------------------
    # Per-employee store opening/closing dates (for clamping assignment windows)
    # -----------------------------------------------------------------
    _store_open_ns = None
    _store_close_ns = None
    if store_opening_dates:
        _store_open_ns = np.array([
            np.datetime64(store_opening_dates.get(int(sk), window_start), "ns")
            for sk in home_arr
        ], dtype="datetime64[ns]")
    if store_closing_dates:
        _store_close_ns = np.array([
            np.datetime64(store_closing_dates.get(int(sk), window_end), "ns")
            for sk in home_arr
        ], dtype="datetime64[ns]")

    # -----------------------------------------------------------------
    # Vectorized: anchors get tenure-bounded rows (chained for attrition)
    # -----------------------------------------------------------------
    if bool(ensure_store_sales_coverage) and anchor_keys:
        anchor_mask = np.isin(ek_arr, np.array(sorted(anchor_keys), dtype=np.int32))
        anc_idx = np.where(anchor_mask)[0]
        if anc_idx.size > 0:
            # Start = max(hire_date, window_start, store_open)
            anc_start = np.maximum(
                np.asarray(hire_arr[anc_idx], dtype="datetime64[ns]"),
                np.datetime64(window_start, "ns"),
            )
            if _store_open_ns is not None:
                anc_start = np.maximum(anc_start, _store_open_ns[anc_idx])

            # End = min(term_date or window_end, window_end, store_close)
            anc_end = np.asarray(term_filled_arr[anc_idx], dtype="datetime64[ns]")
            anc_end = np.minimum(anc_end, np.datetime64(window_end, "ns"))
            if _store_close_ns is not None:
                anc_end = np.minimum(anc_end, _store_close_ns[anc_idx])

            # Filter out anchors whose effective window is invalid
            anc_valid = anc_end >= anc_start
            if not np.all(anc_valid):
                anc_idx = anc_idx[anc_valid]
                anc_start = anc_start[anc_valid]
                anc_end = anc_end[anc_valid]
            if anc_idx.size > 0:
                batch_dfs.append(pd.DataFrame({
                    "EmployeeKey": ek_arr[anc_idx],
                    "StoreKey": home_arr[anc_idx],
                    "StartDate": anc_start,
                    "EndDate": anc_end,
                    "FTE": target_fte[anc_idx],
                    "RoleAtStore": ps_role,
                    "IsPrimary": True,
                    "TransferReason": pd.NA,
                }))

    # -----------------------------------------------------------------
    # Vectorized: pre-compute per-employee validity and movement
    # -----------------------------------------------------------------
    hire_ns = np.asarray(hire_arr, dtype="datetime64[ns]")
    term_ns = np.asarray(term_arr, dtype="datetime64[ns]")
    term_filled_ns = np.asarray(term_filled_arr, dtype="datetime64[ns]")

    valid = ~np.isnat(hire_ns) & ~np.isnat(term_filled_ns) & (term_filled_ns >= hire_ns)

    # Exclude anchors from further processing
    if bool(ensure_store_sales_coverage) and anchor_keys:
        valid &= ~np.isin(ek_arr, np.array(sorted(anchor_keys), dtype=np.int32))

    ws_np = np.datetime64(window_start, "ns")
    we_np = np.datetime64(window_end, "ns")
    gen_start_all = np.maximum(hire_ns, ws_np)
    plan_end_all = np.minimum(term_filled_ns, we_np)

    # Clamp assignment start to store opening, end to store closing
    if _store_open_ns is not None:
        gen_start_all = np.maximum(gen_start_all, _store_open_ns)
    if _store_close_ns is not None:
        plan_end_all = np.minimum(plan_end_all, _store_close_ns)

    valid &= (plan_end_all >= gen_start_all) & (gen_start_all <= we_np)

    open_ended_all = np.isnat(term_ns) | (term_ns > we_np)
    end_date_all = np.where(open_ended_all, we_np, np.minimum(term_ns, we_np))
    if _store_close_ns is not None:
        end_date_all = np.minimum(end_date_all, _store_close_ns)

    # Build per-role profile lookup (few unique roles, so dict lookup is fine)
    _role_cache: Dict[str, Dict[str, Any]] = {}
    unique_roles = set(role_arr[valid])
    for r in unique_roles:
        if r == "Store Manager":
            _role_cache[r] = {"role_multiplier": 0.0, "episodes_min": 0,
                              "episodes_max": 0, "duration_days_min": 30,
                              "duration_days_max": 180}
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

    # Vectorized per-employee movement probability
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

    will_move = valid & (emax_arr > 0) & (forced_mover | (move_draws < p_move_all))
    non_mover = valid & ~will_move

    # -----------------------------------------------------------------
    # Vectorized: batch-emit all non-movers as single-assignment rows
    # -----------------------------------------------------------------
    nm_idx = np.where(non_mover)[0]
    if nm_idx.size > 0:
        nm_start = gen_start_all[nm_idx]
        nm_end = end_date_all[nm_idx]
        ok = nm_end >= nm_start
        nm_idx = nm_idx[ok]
        if nm_idx.size > 0:
            batch_dfs.append(pd.DataFrame({
                "EmployeeKey": ek_arr[nm_idx],
                "StoreKey": home_arr[nm_idx],
                "StartDate": gen_start_all[nm_idx],
                "EndDate": end_date_all[nm_idx],
                "FTE": target_fte[nm_idx],
                "RoleAtStore": role_arr[nm_idx],
                "IsPrimary": True,
                "TransferReason": pd.NA,
            }))

    # -----------------------------------------------------------------
    # Loop only over movers (small fraction of employees)
    # -----------------------------------------------------------------
    mover_indices = np.where(will_move)[0]
    for i in mover_indices:
        ek = int(ek_arr[i])
        did = did_arr[i]
        home_store = int(home_arr[i])
        role = str(role_arr[i])
        tgt = float(target_fte[i])

        emp_term = term_arr[i]

        gen_start = pd.Timestamp(gen_start_all[i])
        plan_end = pd.Timestamp(plan_end_all[i])
        open_ended = bool(open_ended_all[i])

        rp_cached = _role_cache.get(role, {})
        e_min = int(rp_cached.get("episodes_min", 0))
        e_max = int(rp_cached.get("episodes_max", 0))
        dmin = int(rp_cached.get("duration_days_min", 30))
        dmax = int(rp_cached.get("duration_days_max", 180))

        is_mover = bool(forced_mover[i])

        episodes: List[Tuple[pd.Timestamp, pd.Timestamp, int, float]] = []
        other = _candidate_stores(did, home_store)
        if other and e_max > 0:
            k = (
                int(rng.integers(e_min, e_max + 1))
                if e_max >= e_min
                else int(e_min)
            )
            if is_mover and k < 1:
                k = 1
            if k > 0:
                episodes = _sample_non_overlapping_episodes(
                    gen_start, plan_end, other, k, tgt, dmin, dmax,
                )

        cur = pd.to_datetime(gen_start).normalize()

        MIN_HOME_GAP_DAYS = 3

        for (s, e, store, _sec_fte) in episodes:
            s = pd.to_datetime(s).normalize()
            e = pd.to_datetime(e).normalize()

            before_end = (s - pd.Timedelta(days=1)).normalize()
            gap_days = (before_end - cur).days + 1 if cur <= before_end else 0

            if gap_days >= MIN_HOME_GAP_DAYS:
                # Emit a home-store segment for the gap
                _emit_mover(ek, role, home_store, cur, before_end, True, tgt)
            elif gap_days > 0:
                # Gap too short for a realistic home stint;
                # extend the away episode backwards to absorb it.
                s = cur

            # Away episode: employee FTE, random transfer reason
            _reason = str(rng.choice(
                EMPLOYEE_TRANSFER_REASON_LABELS, p=EMPLOYEE_TRANSFER_REASON_PROBS,
            ))
            _emit_mover(ek, role, store, s, e, False, tgt, transfer_reason=_reason)
            cur = (e + pd.Timedelta(days=1)).normalize()

        # Final home segment
        if cur <= plan_end:
            if open_ended:
                _emit_mover(ek, role, home_store, cur, window_end, True, tgt)
            else:
                final_end = pd.to_datetime(emp_term).normalize()
                if final_end > window_end:
                    final_end = window_end
                if final_end >= cur:
                    _emit_mover(ek, role, home_store, cur, final_end, True, tgt)

    # -----------------------------------------------------------------
    # Assemble output from batched DataFrames + mover rows
    # -----------------------------------------------------------------
    parts = list(batch_dfs)
    if mover_rows:
        parts.append(pd.DataFrame(mover_rows, columns=out_cols))
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=out_cols)

    if out.empty:
        out["AssignmentSequence"] = pd.Series(dtype=np.int32)
        return out

    out["EmployeeKey"] = out["EmployeeKey"].astype(np.int32)
    out["StoreKey"] = out["StoreKey"].astype(np.int32)
    out["FTE"] = out["FTE"].astype(np.float64)
    out["IsPrimary"] = out["IsPrimary"].astype(bool)
    out["RoleAtStore"] = out["RoleAtStore"].astype(str)

    # Close any open-ended segments at the dataset window end
    out["EndDate"] = pd.to_datetime(out["EndDate"]).dt.normalize()
    out["EndDate"] = out["EndDate"].fillna(window_end)

    # Non-overlap validation
    FAR_FUTURE = pd.Timestamp("2262-04-11")
    chk = out.sort_values(["EmployeeKey", "StartDate", "EndDate"]).copy()
    chk_end = pd.to_datetime(chk["EndDate"]).fillna(FAR_FUTURE)
    chk_start = pd.to_datetime(chk["StartDate"])
    prev_end = chk_end.groupby(chk["EmployeeKey"]).shift(1)
    bad = (prev_end.notna()) & (chk_start <= prev_end)
    if bool(bad.any()):
        ex = chk.loc[bad, ["EmployeeKey", "StoreKey", "StartDate", "EndDate"]].head(10)
        raise RuntimeError(f"Overlapping assignments detected (sample):\n{ex}")

    out = out.sort_values(["EmployeeKey", "StartDate", "EndDate"]).reset_index(drop=True)
    out["AssignmentSequence"] = out.groupby("EmployeeKey").cumcount().astype(np.int32) + 1

    if bool(ensure_store_sales_coverage):
        # Build hire/term lookups for gap-fill clamping
        _emp_hire_lookup = {
            int(ek_arr[i]): pd.Timestamp(hire_ns[i])
            for i in range(len(ek_arr)) if not np.isnat(hire_ns[i])
        }
        _emp_term_lookup = {
            int(ek_arr[i]): pd.Timestamp(term_ns[i])
            for i in range(len(ek_arr)) if not np.isnat(term_ns[i])
        }
        out = _fill_coverage_gaps(
            out, ps_role, all_stores, window_start, window_end,
            store_opening_dates=store_opening_dates,
            store_closing_dates=store_closing_dates,
            emp_hire=_emp_hire_lookup,
            emp_term=_emp_term_lookup,
        )
        _validate_store_coverage(
            out, ps_role, all_stores, window_start, window_end,
            store_opening_dates=store_opening_dates,
            store_closing_dates=store_closing_dates,
        )

    return out


def _fill_coverage_gaps(
    df: pd.DataFrame,
    ps_role: str,
    all_stores: List[int],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    store_opening_dates: Optional[Dict[int, pd.Timestamp]] = None,
    store_closing_dates: Optional[Dict[int, pd.Timestamp]] = None,
    emp_hire: Optional[Dict[int, pd.Timestamp]] = None,
    emp_term: Optional[Dict[int, pd.Timestamp]] = None,
) -> pd.DataFrame:
    """
    Safety-net repair: scan each store's SA coverage and extend assignment
    dates to fill any gaps.  Extensions are clamped to each employee's
    HireDate/TerminationDate so we never create assignments outside the
    employment window.
    """
    one_day = pd.Timedelta(days=1)
    ws = pd.to_datetime(window_start).normalize()
    we = pd.to_datetime(window_end).normalize()
    _hire = emp_hire or {}
    _term = emp_term or {}
    repairs = 0

    for store in all_stores:
        # Determine the store's operational window
        s_start = ws
        s_end = we
        if store_opening_dates and store in store_opening_dates:
            od = pd.to_datetime(store_opening_dates[store]).normalize()
            if od > s_start:
                s_start = od
        if store_closing_dates and store in store_closing_dates:
            cd = pd.to_datetime(store_closing_dates[store]).normalize()
            last_op = cd - one_day  # ClosingDate = first non-operational day
            if last_op < s_end:
                s_end = last_op
        if s_end < s_start:
            continue  # store not active in dataset window

        # Find SA assignments at this store
        sa_mask = (
            (df["StoreKey"] == store)
            & (df["RoleAtStore"].astype(str) == ps_role)
        )
        if not sa_mask.any():
            continue

        sa_idx = df.loc[sa_mask].sort_values("StartDate").index
        starts = pd.to_datetime(df.loc[sa_idx, "StartDate"]).dt.normalize()
        ends = pd.to_datetime(df.loc[sa_idx, "EndDate"]).dt.normalize()

        # Leading gap: extend first SA's StartDate back, clamped to HireDate
        if starts.iloc[0] > s_start:
            ek = int(df.at[sa_idx[0], "EmployeeKey"])
            hire_dt = _hire.get(ek)
            target = s_start
            if hire_dt is not None:
                hire_dt = pd.to_datetime(hire_dt).normalize()
                target = max(target, hire_dt)
            if target < starts.iloc[0]:
                df.at[sa_idx[0], "StartDate"] = target
                repairs += 1

        # Trailing gap: extend last SA's EndDate, clamped to TerminationDate
        if ends.iloc[-1] < s_end:
            ek = int(df.at[sa_idx[-1], "EmployeeKey"])
            term_dt = _term.get(ek)
            target = s_end
            if term_dt is not None:
                term_dt = pd.to_datetime(term_dt).normalize()
                target = min(target, term_dt)
            if target > ends.iloc[-1]:
                df.at[sa_idx[-1], "EndDate"] = target
                repairs += 1

        # Intermediate gaps: extend preceding SA's EndDate, clamped to their
        # TerminationDate.  If that doesn't close the gap, try extending the
        # next SA's StartDate back to its HireDate.
        for k in range(len(sa_idx) - 1):
            cur_end = pd.to_datetime(df.at[sa_idx[k], "EndDate"]).normalize()
            nxt_start = pd.to_datetime(df.at[sa_idx[k + 1], "StartDate"]).normalize()
            if cur_end + one_day >= nxt_start:
                continue  # no gap

            # Try extending current SA forward
            ek_cur = int(df.at[sa_idx[k], "EmployeeKey"])
            term_cur = _term.get(ek_cur)
            extend_to = nxt_start - one_day
            if term_cur is not None:
                extend_to = min(extend_to, pd.to_datetime(term_cur).normalize())
            if extend_to > cur_end:
                df.at[sa_idx[k], "EndDate"] = extend_to
                repairs += 1

            # If gap remains, try pulling next SA backward
            new_gap_start = extend_to + one_day
            if new_gap_start < nxt_start:
                ek_nxt = int(df.at[sa_idx[k + 1], "EmployeeKey"])
                hire_nxt = _hire.get(ek_nxt)
                pull_to = new_gap_start
                if hire_nxt is not None:
                    pull_to = max(pull_to, pd.to_datetime(hire_nxt).normalize())
                if pull_to < nxt_start:
                    df.at[sa_idx[k + 1], "StartDate"] = pull_to
                    repairs += 1

    # Post-repair: trim any same-employee-same-store overlaps created by
    # gap-fill extensions.  If two assignments for the same employee at
    # the same store overlap, trim the earlier one's EndDate to the day
    # before the later one's StartDate.
    overlap_fixes = 0
    for (ek, sk), grp in df.groupby(["EmployeeKey", "StoreKey"]):
        if len(grp) < 2:
            continue
        grp_sorted = grp.sort_values("StartDate")
        idxs = grp_sorted.index.tolist()
        for i in range(len(idxs) - 1):
            cur_end = pd.to_datetime(df.at[idxs[i], "EndDate"]).normalize()
            nxt_start = pd.to_datetime(df.at[idxs[i + 1], "StartDate"]).normalize()
            if cur_end >= nxt_start:
                df.at[idxs[i], "EndDate"] = nxt_start - one_day
                overlap_fixes += 1

    total_fixes = repairs + overlap_fixes
    if total_fixes > 0:
        parts = []
        if repairs > 0:
            parts.append(f"extended {repairs} SA assignment(s) to fill gaps")
        if overlap_fixes > 0:
            parts.append(f"trimmed {overlap_fixes} overlap(s)")
        info("Coverage repair: " + ", ".join(parts))

    return df


def _validate_store_coverage(
    out: pd.DataFrame,
    ps_role: str,
    all_stores: List[int],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
    store_opening_dates: Optional[Dict[int, pd.Timestamp]] = None,
    store_closing_dates: Optional[Dict[int, pd.Timestamp]] = None,
) -> None:
    """Raise if any store has a coverage gap for the sales-eligible role."""
    ws = np.datetime64(window_start, "D")
    we = np.datetime64(window_end, "D")
    one_day = np.timedelta64(1, "D")

    # Build set of stores active during the dataset window
    # (exclude stores that closed before the window started).
    # ClosingDate = first non-operational day; last operational day = close - 1.
    active_stores = []
    for s in all_stores:
        s_ws = ws
        s_we = we
        if store_opening_dates and s in store_opening_dates:
            _od = np.datetime64(store_opening_dates[s], "D")
            if _od > s_ws:
                s_ws = _od
        if store_closing_dates and s in store_closing_dates:
            _cd = np.datetime64(store_closing_dates[s], "D") - one_day
            if _cd < s_we:
                s_we = _cd
        if s_we >= s_ws:
            active_stores.append(s)

    df_cov = out[out["RoleAtStore"].astype(str) == ps_role].copy()

    if df_cov.empty:
        if active_stores:
            sample = "\n".join(
                f"StoreKey={s}: no '{ps_role}' assignments" for s in active_stores[:10]
            )
            raise RuntimeError(
                "EmployeeStoreAssignments coverage gaps detected (sales-eligible role). "
                "Sample:\n" + sample
            )
        return

    df_cov["StartDate"] = pd.to_datetime(df_cov["StartDate"]).dt.normalize()
    df_cov["EndDate"] = pd.to_datetime(df_cov["EndDate"]).dt.normalize()

    covered = set(df_cov["StoreKey"].unique())
    gaps: list[str] = [
        f"StoreKey={s}: no '{ps_role}' assignments"
        for s in active_stores if s not in covered
    ]

    # Sort once, then walk through pre-grouped segments
    df_cov = df_cov.sort_values(["StoreKey", "StartDate", "EndDate"]).reset_index(drop=True)
    sk = df_cov["StoreKey"].to_numpy()
    starts = df_cov["StartDate"].to_numpy().astype("datetime64[D]")
    ends = df_cov["EndDate"].to_numpy().astype("datetime64[D]")

    store_breaks = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1]])
    group_ends = np.r_[store_breaks[1:], len(sk)]

    for gs, ge in zip(store_breaks, group_ends):
        store = int(sk[gs])
        seg_s = starts[gs:ge]
        seg_e = ends[gs:ge]

        # Per-store expected window: use store opening/closing dates if available.
        # ClosingDate = first non-operational day; last operational = close - 1.
        store_ws = ws
        store_we = we
        if store_opening_dates and store in store_opening_dates:
            _od = np.datetime64(store_opening_dates[store], "D")
            if _od > store_ws:
                store_ws = _od
        if store_closing_dates and store in store_closing_dates:
            _cd = np.datetime64(store_closing_dates[store], "D") - one_day
            if _cd < store_we:
                store_we = _cd

        # Skip stores that closed before the dataset window started
        if store_we < store_ws:
            continue

        # Merge overlapping/adjacent, then check coverage
        cur_ms, cur_me = seg_s[0], seg_e[0]
        cur = store_ws
        gap_found = False
        for j in range(1, ge - gs):
            if seg_s[j] <= cur_me + one_day:
                cur_me = max(cur_me, seg_e[j])
            else:
                # Flush merged segment
                if cur_me < cur:
                    pass
                elif cur_ms > cur:
                    gap_end = min(cur_ms - one_day, store_we)
                    if gap_end >= cur:
                        gaps.append(
                            f"StoreKey={store} gap {cur}..{gap_end}"
                        )
                    gap_found = True
                    break
                else:
                    cur = cur_me + one_day
                    if cur > store_we:
                        gap_found = True  # fully covered
                        break
                cur_ms, cur_me = seg_s[j], seg_e[j]

        if not gap_found:
            # Check final merged segment
            if cur_me >= cur and cur_ms <= cur:
                cur = cur_me + one_day
            elif cur_ms > cur:
                gap_end = min(cur_ms - one_day, store_we)
                if gap_end >= cur:
                    gaps.append(f"StoreKey={store} gap {cur}..{gap_end}")
                continue

            if cur <= store_we:
                gaps.append(f"StoreKey={store} gap {cur}..{store_we}")

    if gaps:
        sample = "\n".join(gaps[:10])
        raise RuntimeError(
            "EmployeeStoreAssignments coverage gaps detected (sales-eligible role). "
            "Sample:\n" + sample
        )


# -----------------------------------------------------------------------------
# Pipeline entrypoint
# -----------------------------------------------------------------------------

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
    version_cfg["schema_version"] = 10
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

    # Read store opening/closing dates for assignment window clamping
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

    # Read employee transfers sidecar (generated by employees.py)
    transfers_path = parquet_folder / "employee_transfers.parquet"
    transfers_df: Optional[pd.DataFrame] = None
    if transfers_path.exists():
        transfers_df = pd.read_parquet(transfers_path)

    # Read store employee counts for capacity-weighted destination selection
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
            # Open and Renovating stores are eligible destinations
            _eligible = (_ss == "Open") | (_ss == "Renovating")
            stores_emp_count = {int(k): int(c) for k, c, o in zip(_sk, _ec, _eligible) if o}
            stores_district = {int(k): str(d) for k, d in zip(_sk, _sd)}
        except (KeyError, ValueError):
            pass

    # Resolve closing config for prefer_same_district
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
        )

        # ---------------------------------------------------------------
        # Append transfer assignments for employees from closed stores
        # ---------------------------------------------------------------
        if transfers_df is not None and not transfers_df.empty and stores_emp_count:
            rng_t = np.random.default_rng(int(seed) ^ 0xDEAD_BEEF)
            transfer_rows: list[dict] = []
            open_stores = sorted(stores_emp_count.keys())

            _ps_role = str_or(a_cfg.get("primary_sales_role"), "Sales Associate")

            # Build HireDate lookup for clamping transfer StartDate
            _hire_lookup: dict[int, pd.Timestamp] = {}
            for _, erow in employees.iterrows():
                _hire_lookup[int(erow["EmployeeKey"])] = pd.to_datetime(erow["HireDate"])

            # Track EmployeeKey -> dest_sk for updating employees.parquet
            _transfer_dest: dict[int, int] = {}

            for _, trow in transfers_df.iterrows():
                ek = int(trow["EmployeeKey"])
                orig_sk = int(trow["OriginalStoreKey"])
                t_date = pd.to_datetime(trow["TransferDate"]).normalize()
                role = str(trow["Title"])
                did = trow.get("DistrictId")

                # Trim existing assignments for this employee at the old store
                trim_end = (t_date - pd.Timedelta(days=1)).normalize()
                mask = (df["EmployeeKey"] == ek) & (df["StoreKey"] == orig_sk)
                if mask.any():
                    idx = df.index[mask]
                    for i in idx:
                        end = pd.to_datetime(df.at[i, "EndDate"])
                        if pd.isna(end) or end >= t_date:
                            df.at[i, "EndDate"] = trim_end

                # Pick destination store: prefer same district, weight by capacity
                if _prefer_same_district and stores_district and pd.notna(did):
                    orig_district = stores_district.get(orig_sk, "")
                    same_dist = [
                        sk for sk in open_stores
                        if stores_district.get(sk, "") == orig_district and sk != orig_sk
                    ]
                    candidates = same_dist if same_dist else [sk for sk in open_stores if sk != orig_sk]
                else:
                    candidates = [sk for sk in open_stores if sk != orig_sk]

                if not candidates:
                    candidates = open_stores

                # Weight by employee count (larger stores absorb more)
                weights = np.array(
                    [max(1, stores_emp_count.get(sk, 1)) for sk in candidates],
                    dtype=np.float64,
                )
                weights /= weights.sum()
                dest_sk = int(rng_t.choice(candidates, p=weights))

                # Create transfer assignment with ramp-up period
                ramp_days = int_or(_closing_cfg.get("ramp_days"), 30)
                ramp_start_factor = float_or(_closing_cfg.get("ramp_start_factor"), 0.50)
                ramp_end = (t_date + pd.Timedelta(days=ramp_days)).normalize()

                # Look up employee's FTE from the employees data
                _emp_fte = float(trow.get("FTE", 1.0)) if "FTE" in trow.index else 1.0

                # Clamp StartDate to HireDate (transfer can't start before hire)
                hire_date = _hire_lookup.get(ek)
                if hire_date is not None and t_date < hire_date:
                    t_date = hire_date

                _transfer_dest[ek] = dest_sk

                transfer_rows.append({
                    "EmployeeKey": np.int32(ek),
                    "StoreKey": np.int32(dest_sk),
                    "StartDate": t_date,
                    "EndDate": global_end,
                    "FTE": _emp_fte,
                    "RoleAtStore": role,
                    "IsPrimary": True,
                    "TransferReason": pd.NA,
                })

            if transfer_rows:
                out_cols = ["EmployeeKey", "StoreKey", "StartDate", "EndDate", "FTE", "RoleAtStore", "IsPrimary", "TransferReason"]
                t_df = pd.DataFrame(transfer_rows, columns=out_cols)
                df = pd.concat([df, t_df], ignore_index=True)
                info(f"Added {len(transfer_rows)} transfer assignments from closed stores")

            # Update employees.parquet: set StoreKey to destination for transferred employees
            if _transfer_dest:
                emp_full = pd.read_parquet(employees_path)
                updated = 0
                for ek, dest in _transfer_dest.items():
                    mask = emp_full["EmployeeKey"] == ek
                    if mask.any():
                        emp_full.loc[mask, "StoreKey"] = np.int32(dest)
                        updated += 1
                if updated > 0:
                    emp_full.to_parquet(employees_path, index=False)
                    info(f"Updated StoreKey for {updated} transferred employees in employees.parquet")

                    # Re-sync stores.parquet EmployeeCount after transfers
                    # Use actual StoreKey column (not EmployeeKey encoding)
                    stores_path = parquet_folder / "stores.parquet"
                    if stores_path.exists():
                        sk_col = emp_full["StoreKey"].dropna()
                        sk_col = sk_col[sk_col > 0].astype(np.int32)
                        actual_counts = sk_col.value_counts().to_dict()

                        stores_full = pd.read_parquet(stores_path)
                        stores_full["EmployeeCount"] = stores_full["StoreKey"].map(
                            lambda sk: actual_counts.get(int(sk), 0)
                        ).astype(np.int64)

                        from src.dimensions.employees import write_parquet_with_date32
                        write_parquet_with_date32(
                            stores_full, stores_path,
                            date_cols=["OpeningDate", "ClosingDate"],
                            cast_all_datetime=False,
                            compression="snappy", compression_level=None,
                            force_date32=True,
                        )
                        info("Re-synced stores.parquet EmployeeCount after transfers")

            # ---------------------------------------------------------------
            # Post-transfer coverage repair: comprehensive gap-fill for all
            # stores (not just closing ones).  Extends SA assignments to
            # cover any gaps caused by transfer staggering, attrition chain
            # timing, or closing-date boundary mismatches.
            # ---------------------------------------------------------------
            _all_stores_for_repair = sorted({
                int(sk) for sk in df["StoreKey"].unique()
            })
            # Build hire/term lookups from employees for clamping
            _eh = pd.to_datetime(employees["HireDate"], errors="coerce")
            _et = pd.to_datetime(employees["TerminationDate"], errors="coerce")
            _ek_vals = employees["EmployeeKey"].astype(np.int32).to_numpy()
            _emp_hire_l = {int(_ek_vals[i]): _eh.iloc[i] for i in range(len(_ek_vals)) if pd.notna(_eh.iloc[i])}
            _emp_term_l = {int(_ek_vals[i]): _et.iloc[i] for i in range(len(_ek_vals)) if pd.notna(_et.iloc[i])}
            df = _fill_coverage_gaps(
                df, _ps_role, _all_stores_for_repair,
                global_start, global_end,
                store_opening_dates=store_opening_dates,
                store_closing_dates=store_closing_dates,
                emp_hire=_emp_hire_l,
                emp_term=_emp_term_l,
            )

            # Re-sort and recompute AssignmentSequence
            df = df.sort_values(["EmployeeKey", "StartDate", "EndDate"]).reset_index(drop=True)
            df["AssignmentSequence"] = df.groupby("EmployeeKey").cumcount().astype(np.int32) + 1

        # Remove transfers sidecar so it doesn't get copied to final output
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
