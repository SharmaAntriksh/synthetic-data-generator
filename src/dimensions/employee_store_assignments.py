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

from src.utils.logging_utils import info, skip, stage
from src.utils.output_utils import write_parquet_with_date32
from src.versioning import should_regenerate, save_version
from src.utils.config_helpers import (
    as_dict,
    int_or,
    float_or,
    str_or,
    pick_seed_nested,
    parse_global_dates,
    rand_dates_between,
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
    out = dict(legacy)
    out.update(nested)
    return out


def _infer_home_store_key(employees: pd.DataFrame) -> pd.Series:
    """
    Derive the home StoreKey from the EmployeeKey encoding when the column
    is not already present.
    """
    if "StoreKey" in employees.columns:
        return employees["StoreKey"].astype("Int64")

    ek = employees["EmployeeKey"].astype(np.int64)
    out = pd.Series([pd.NA] * len(employees), dtype="Int64")

    mgr_mask = (ek >= STORE_MGR_KEY_BASE) & (ek < STAFF_KEY_BASE)
    if mgr_mask.any():
        out.loc[mgr_mask] = (ek.loc[mgr_mask] - STORE_MGR_KEY_BASE).astype("Int64")

    staff_mask = ek >= STAFF_KEY_BASE
    if staff_mask.any():
        out.loc[staff_mask] = (
            (ek.loc[staff_mask] - STAFF_KEY_BASE) // STAFF_KEY_STORE_MULT
        ).astype("Int64")

    return out


def _build_store_by_district(emp: pd.DataFrame) -> Dict[Any, List[int]]:
    """``DistrictId → [StoreKey, ...]``"""
    emp2 = emp[emp["DistrictId"].notna() & emp["HomeStoreKey"].notna()].copy()
    if emp2.empty:
        return {}

    is_mgr = emp2["Title"].astype(str) == "Store Manager"
    mgrs = emp2[is_mgr]
    src = mgrs if not mgrs.empty else emp2

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
        mover_share = float(multi_store_share)

    rng = np.random.default_rng(int(seed))

    required = {"EmployeeKey", "HireDate", "TerminationDate", "Title", "DistrictId"}
    missing = [c for c in required if c not in employees.columns]
    if missing:
        raise ValueError(f"employees missing required columns: {missing}")

    emp = employees.copy()
    emp["EmployeeKey"] = emp["EmployeeKey"].astype(np.int64)
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
    ek_arr = emp["EmployeeKey"].astype(np.int64).to_numpy()
    did_arr = emp["DistrictId"].to_numpy()
    home_arr = emp["HomeStoreKey"].fillna(-1).astype(np.int64).to_numpy()
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
        staff_idx = np.full(len(emp), -1, dtype=np.int64)
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
    default_profile = as_dict(rp.get("default"))
    per_role_profile = {
        k: dict(v) for k, v in rp.items() if k != "default" and isinstance(v, dict)
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
    rows: List[Dict[str, Any]] = []

    def _emit(
        ek: int, role: str, store_key: int,
        seg_start: pd.Timestamp, seg_end: Any,
        is_primary: bool, fte_val: float,
    ) -> None:
        rows.append(dict(
            EmployeeKey=int(ek),
            StoreKey=int(store_key),
            StartDate=pd.to_datetime(seg_start).normalize(),
            EndDate=(pd.to_datetime(seg_end).normalize() if pd.notna(seg_end) else pd.NaT),
            FTE=float(fte_val),
            RoleAtStore=str(role),
            IsPrimary=bool(is_primary),
        ))

    # Vectorized emission: anchors get a single full-window row
    if bool(ensure_store_sales_coverage) and anchor_keys:
        anchor_mask = np.isin(ek_arr, np.array(sorted(anchor_keys), dtype=np.int64))
        n_anc = int(anchor_mask.sum())
        if n_anc > 0:
            for idx in np.where(anchor_mask)[0]:
                _emit(
                    int(ek_arr[idx]), ps_role, int(home_arr[idx]),
                    window_start, window_end, True, float(target_fte[idx]),
                )

    for i in range(len(emp)):
        ek = int(ek_arr[i])
        did = did_arr[i]
        home_store = int(home_arr[i])
        role = str(role_arr[i])
        tgt = float(target_fte[i])

        emp_hire = hire_arr[i]
        emp_term = term_arr[i]
        plan_end_raw = term_filled_arr[i]

        if pd.isna(emp_hire) or pd.isna(plan_end_raw):
            continue
        emp_hire_ts = pd.Timestamp(emp_hire)
        plan_end_raw_ts = pd.Timestamp(plan_end_raw)
        if plan_end_raw_ts < emp_hire_ts:
            continue

        # Anchors already emitted above
        if bool(ensure_store_sales_coverage) and (ek in anchor_keys):
            continue

        open_ended = pd.isna(emp_term) or (
            pd.notna(emp_term) and pd.Timestamp(emp_term).normalize() > window_end
        )

        plan_end = plan_end_raw_ts.normalize()
        if plan_end > window_end:
            plan_end = window_end

        gen_start = emp_hire_ts.normalize()
        if gen_start < window_start:
            gen_start = window_start

        if plan_end < gen_start or gen_start > window_end:
            continue

        # Store Managers: home store only, no movement
        if role == "Store Manager":
            prof = dict(default_profile)
            prof["role_multiplier"] = 0.0
            prof["episodes_min"] = 0
            prof["episodes_max"] = 0
        else:
            prof = _get_profile(
                role,
                primary_sales_role=primary_sales_role,
                default_profile=default_profile,
                per_role_profile=per_role_profile,
                sales_profile=sales_profile,
                max_non_sales_multiplier_frac=clamped_mult_frac,
                max_non_sales_episodes_frac=clamped_ep_frac,
            )

        role_mult = float(max(0.0, float_or(prof.get("role_multiplier"), 1.0)))
        p_move = base * role_mult * (pt_mult if tgt < 1.0 else 1.0)
        p_move = float(np.clip(p_move, 0.0, 0.90))

        e_min = max(0, int_or(prof.get("episodes_min"), 0))
        e_max = max(e_min, int_or(prof.get("episodes_max"), 0))

        dmin = max(1, int_or(prof.get("duration_days_min"), 30))
        dmax = max(dmin, int_or(prof.get("duration_days_max"), 180))

        is_mover = bool(ensure_store_sales_coverage) and (ek in mover_keys)

        episodes: List[Tuple[pd.Timestamp, pd.Timestamp, int, float]] = []
        if e_max > 0 and (is_mover or rng.random() < p_move):
            other = _candidate_stores(did, home_store)
            if other:
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
                _emit(ek, role, home_store, cur, before_end, True, tgt)

            _emit(ek, role, store, s, e, False, sec_fte)
            cur = (e + pd.Timedelta(days=1)).normalize()

        # Final home segment
        if cur <= plan_end:
            if open_ended:
                _emit(ek, role, home_store, cur, window_end, True, tgt)
            else:
                final_end = pd.to_datetime(emp_term).normalize()
                if final_end > window_end:
                    final_end = window_end
                if final_end >= cur:
                    _emit(ek, role, home_store, cur, final_end, True, tgt)

    out = pd.DataFrame(rows, columns=out_cols)
    if out.empty:
        out["AssignmentSequence"] = pd.Series(dtype=np.int32)
        return out

    out["EmployeeKey"] = out["EmployeeKey"].astype(np.int64)
    out["StoreKey"] = out["StoreKey"].astype(np.int64)
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
    df_cov["StartDate"] = pd.to_datetime(df_cov["StartDate"]).dt.normalize()
    df_cov["EndDate"] = pd.to_datetime(df_cov["EndDate"]).dt.normalize()

    gaps: list[str] = []
    for store in all_stores:
        segs = (
            df_cov[df_cov["StoreKey"] == int(store)][["StartDate", "EndDate"]]
            .sort_values("StartDate")
        )
        if segs.empty:
            gaps.append(f"StoreKey={int(store)}: no '{ps_role}' assignments")
            continue

        cur = window_start
        for s, e in segs.itertuples(index=False, name=None):
            s = pd.to_datetime(s).normalize()
            e = pd.to_datetime(e).normalize()
            if e < cur:
                continue
            if s > cur:
                gaps.append(
                    f"StoreKey={int(store)} gap "
                    f"{cur.date()}..{(s - pd.Timedelta(days=1)).date()}"
                )
                break
            cur = (max(cur, e) + pd.Timedelta(days=1)).normalize()
            if cur > window_end:
                break

        if cur <= window_end:
            gaps.append(f"StoreKey={int(store)} gap {cur.date()}..{window_end.date()}")

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
        ek = pd.to_numeric(employees["EmployeeKey"], errors="coerce").dropna().astype(np.int64)
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
