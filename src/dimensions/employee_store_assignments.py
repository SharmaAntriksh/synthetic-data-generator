from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.dimensions.employees import (
    STORE_MGR_KEY_BASE,
    STAFF_KEY_BASE,
    STAFF_KEY_STORE_MULT,
)

from src.utils.logging_utils import info, skip, stage, warn
from src.utils.output_utils import write_parquet_with_date32
from src.versioning import should_regenerate, save_version
from src.utils.config_helpers import (
    as_dict,
    int_or,
    float_or,
    str_or,
    pick_seed_nested,
    parse_global_dates,
    rand_single_date,
)


# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------

def _store_assignments_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preferred path: ``cfg["employees"]["store_assignments"]``
    Legacy path:    ``cfg["employee_store_assignments"]``

    Merge rule: nested overrides legacy.
    """
    cfg = cfg or {}
    emp_cfg = as_dict(cfg.get("employees"))
    nested = as_dict(emp_cfg.get("store_assignments"))
    legacy = as_dict(cfg.get("employee_store_assignments"))

    # The dimensions runner injects internal keys (global_dates, _force_regenerate)
    # into cfg["employee_store_assignments"]; exclude those when checking for
    # actual user-supplied legacy config.
    _RUNNER_KEYS = {"global_dates", "_force_regenerate"}
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
    mover_share: float = 0.03,
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
        "FTE", "RoleAtStore", "IsPrimary",
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

    # Part-time probability by role
    pt_prob = np.where(
        np.isin(title_arr, ["Cashier"]),
        0.45,
        np.where(np.isin(title_arr, ["Sales Associate"]), 0.25, 0.10),
    )
    is_part_time = rng.random(len(emp)) < pt_prob
    target_fte = np.where(
        is_part_time, rng.uniform(0.50, 0.80, size=len(emp)), 1.0,
    ).astype(np.float64)

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

        for store in all_stores:
            m_store_role = (home_arr == int(store)) & (role_arr == ps_role)
            if not bool(np.any(m_store_role)):
                continue

            cand_ek = ek_arr[m_store_role]
            cand_si = staff_idx[m_store_role]
            order = np.argsort(cand_si)
            cand_ek = cand_ek[order]

            n_sa_store = len(cand_ek)
            n_anchor = max(1, n_sa_store - n_movable)
            n_anchor = min(n_anchor, n_sa_store)
            n_mover = min(n_movable, n_sa_store - n_anchor)

            for j in range(n_anchor):
                ek_j = int(cand_ek[j])
                anchor_keys.add(ek_j)
                if j == 0:
                    anchor_by_store[int(store)] = ek_j

            for j in range(n_anchor, n_anchor + n_mover):
                mover_keys.add(int(cand_ek[j]))

        missing_stores = [int(s) for s in all_stores if int(s) not in anchor_by_store]
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
        for k, v in rp.items() if k != "default" and isinstance(v, dict)
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

        chosen = rng.choice(other_stores, size=int(k), replace=bool(allow_store_revisit))

        raw: List[Tuple[pd.Timestamp, int, int, float]] = []
        for store in chosen:
            dur = int(rng.integers(dmin, dmax + 1))
            latest_start = (end_max - pd.Timedelta(days=dur - 1)).normalize()
            if latest_start < start_min:
                continue
            s = rand_single_date(rng, start_min, latest_start)
            sec_fte = float(min(tgt_fte, rng.uniform(fmin, fmax)))
            raw.append((s, dur, int(store), sec_fte))

        if not raw:
            return []

        raw.sort(key=lambda x: x[0])
        out: List[Tuple[pd.Timestamp, pd.Timestamp, int, float]] = []
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
            out.append((s, e, store, sec_fte))
            last_end = e

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
    ) -> None:
        mover_rows.append(dict(
            EmployeeKey=int(ek),
            StoreKey=int(store_key),
            StartDate=pd.to_datetime(seg_start).normalize(),
            EndDate=(pd.to_datetime(seg_end).normalize() if pd.notna(seg_end) else pd.NaT),
            FTE=float(fte_val),
            RoleAtStore=str(role),
            IsPrimary=bool(is_primary),
        ))

    # -----------------------------------------------------------------
    # Vectorized: anchors get a single full-window row
    # -----------------------------------------------------------------
    if bool(ensure_store_sales_coverage) and anchor_keys:
        anchor_mask = np.isin(ek_arr, np.array(sorted(anchor_keys), dtype=np.int32))
        anc_idx = np.where(anchor_mask)[0]
        if anc_idx.size > 0:
            batch_dfs.append(pd.DataFrame({
                "EmployeeKey": ek_arr[anc_idx],
                "StoreKey": home_arr[anc_idx],
                "StartDate": window_start,
                "EndDate": window_end,
                "FTE": target_fte[anc_idx],
                "RoleAtStore": ps_role,
                "IsPrimary": True,
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

    valid &= (plan_end_all >= gen_start_all) & (gen_start_all <= we_np)

    open_ended_all = np.isnat(term_ns) | (term_ns > we_np)
    end_date_all = np.where(open_ended_all, we_np, np.minimum(term_ns, we_np))

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

        for (s, e, store, sec_fte) in episodes:
            s = pd.to_datetime(s).normalize()
            e = pd.to_datetime(e).normalize()

            before_end = (s - pd.Timedelta(days=1)).normalize()
            if cur <= before_end:
                _emit_mover(ek, role, home_store, cur, before_end, True, tgt)

            _emit_mover(ek, role, store, s, e, False, sec_fte)
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
        _validate_store_coverage(out, ps_role, all_stores, window_start, window_end)

    return out


def _validate_store_coverage(
    out: pd.DataFrame,
    ps_role: str,
    all_stores: List[int],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> None:
    """Raise if any store has a coverage gap for the sales-eligible role."""
    df_cov = out[out["RoleAtStore"].astype(str) == ps_role].copy()

    if df_cov.empty:
        if all_stores:
            sample = "\n".join(
                f"StoreKey={s}: no '{ps_role}' assignments" for s in all_stores[:10]
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
        for s in all_stores if s not in covered
    ]

    # Sort once, then walk through pre-grouped segments
    df_cov = df_cov.sort_values(["StoreKey", "StartDate", "EndDate"]).reset_index(drop=True)
    sk = df_cov["StoreKey"].to_numpy()
    starts = df_cov["StartDate"].to_numpy().astype("datetime64[D]")
    ends = df_cov["EndDate"].to_numpy().astype("datetime64[D]")

    ws = np.datetime64(window_start, "D")
    we = np.datetime64(window_end, "D")
    one_day = np.timedelta64(1, "D")

    store_breaks = np.flatnonzero(np.r_[True, sk[1:] != sk[:-1]])
    group_ends = np.r_[store_breaks[1:], len(sk)]

    for gs, ge in zip(store_breaks, group_ends):
        store = int(sk[gs])
        seg_s = starts[gs:ge]
        seg_e = ends[gs:ge]

        # Merge overlapping/adjacent, then check coverage
        cur_ms, cur_me = seg_s[0], seg_e[0]
        cur = ws
        gap_found = False
        for j in range(1, ge - gs):
            if seg_s[j] <= cur_me + one_day:
                cur_me = max(cur_me, seg_e[j])
            else:
                # Flush merged segment
                if cur_me < cur:
                    pass
                elif cur_ms > cur:
                    gaps.append(
                        f"StoreKey={store} gap "
                        f"{cur}..{min(cur_ms - one_day, we)}"
                    )
                    gap_found = True
                    break
                else:
                    cur = cur_me + one_day
                cur_ms, cur_me = seg_s[j], seg_e[j]

        if not gap_found:
            # Check final merged segment
            if cur_me >= cur and cur_ms <= cur:
                cur = cur_me + one_day
            elif cur_ms > cur:
                gaps.append(f"StoreKey={store} gap {cur}..{min(cur_ms - one_day, we)}")
                continue

            if cur <= we:
                gaps.append(f"StoreKey={store} gap {cur}..{we}")

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

    force = bool(a_cfg.get("_force_regenerate", False))
    seed = pick_seed_nested(cfg, a_cfg, fallback=42)
    global_start, global_end = parse_global_dates(
        cfg, a_cfg,
        allow_override=True,
        dimension_name="employee_store_assignments",
    )

    employees = pd.read_parquet(
        employees_path,
        columns=["EmployeeKey", "HireDate", "TerminationDate", "Title", "DistrictId"],
    )

    version_cfg = dict(a_cfg)
    version_cfg.pop("_force_regenerate", None)
    version_cfg["schema_version"] = 9
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

    if not force and not should_regenerate("employee_store_assignments", version_cfg, out_path):
        skip("EmployeeStoreAssignments up-to-date; skipping.")
        return

    role_profiles = as_dict(a_cfg.get("role_profiles"))

    with stage("Generating EmployeeStoreAssignments"):
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
        )

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
    info(f"EmployeeStoreAssignments written: {out_path}")
