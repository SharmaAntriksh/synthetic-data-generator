from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional

import numpy as np
import pandas as pd

from src.utils.logging_utils import info, skip, stage
from src.utils.output_utils import write_parquet_with_date32
from src.versioning.version_store import should_regenerate, save_version


# -----------------------------------------------------------------------------
# Config helpers
# -----------------------------------------------------------------------------

def _as_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _int_or(v: Any, default: int) -> int:
    try:
        if v is None or v == "":
            return int(default)
        return int(v)
    except (TypeError, ValueError):
        return int(default)


def _float_or(v: Any, default: float) -> float:
    try:
        if v is None or v == "":
            return float(default)
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _str_or(v: Any, default: str) -> str:
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def _store_assignments_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Preferred:
      cfg["employees"]["store_assignments"]

    Legacy:
      cfg["employee_store_assignments"]

    Merge rule: legacy overrides nested.
    """
    cfg = cfg or {}
    emp_cfg = _as_dict(cfg.get("employees"))
    nested = _as_dict(emp_cfg.get("store_assignments"))
    legacy = _as_dict(cfg.get("employee_store_assignments"))
    out = dict(nested)
    out.update(legacy)
    return out


def _pick_seed(cfg: Dict[str, Any], a_cfg: Dict[str, Any], fallback: int = 42) -> int:
    override = _as_dict(a_cfg.get("override"))
    seed = override.get("seed")
    if seed is None:
        seed = a_cfg.get("seed")
    if seed is None:
        seed = _as_dict(cfg.get("defaults")).get("seed")
    return _int_or(seed, fallback)


def _parse_global_dates(cfg: Dict[str, Any], a_cfg: Dict[str, Any]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Resolve dataset window for assignments.

    Priority:
      1) a_cfg.global_dates {start,end}
      2) a_cfg.override.dates {start,end}
      3) cfg.defaults.dates {start,end}
      4) fallback (2021-01-01..2026-12-31)
    """
    gd = _as_dict(a_cfg.get("global_dates"))
    if gd and gd.get("start") and gd.get("end"):
        gs = pd.to_datetime(gd["start"]).normalize()
        ge = pd.to_datetime(gd["end"]).normalize()
        if ge < gs:
            gs, ge = ge, gs
        return gs, ge

    override = _as_dict(a_cfg.get("override"))
    od = _as_dict(override.get("dates"))
    if od and od.get("start") and od.get("end"):
        gs = pd.to_datetime(od["start"]).normalize()
        ge = pd.to_datetime(od["end"]).normalize()
        if ge < gs:
            gs, ge = ge, gs
        return gs, ge

    defaults = _as_dict(cfg.get("defaults"))
    dd = _as_dict(defaults.get("dates"))
    start = dd.get("start") or "2021-01-01"
    end = dd.get("end") or "2026-12-31"

    gs = pd.to_datetime(start).normalize()
    ge = pd.to_datetime(end).normalize()
    if ge < gs:
        gs, ge = ge, gs
    return gs, ge


# -----------------------------------------------------------------------------
# Core utilities
# -----------------------------------------------------------------------------

def _rand_dates_between(rng: np.random.Generator, start: pd.Timestamp, end: pd.Timestamp, n: int) -> pd.Series:
    start = pd.to_datetime(start).normalize()
    end = pd.to_datetime(end).normalize()
    if end < start:
        start, end = end, start

    start_i = start.value // 10**9
    end_i = end.value // 10**9
    secs = rng.integers(start_i, end_i + 1, size=int(n), dtype=np.int64)

    dt = pd.to_datetime(secs, unit="s").normalize()  # DatetimeIndex
    return pd.Series(dt)


def _infer_home_store_key(employees: pd.DataFrame) -> pd.Series:
    """
    If employees has StoreKey, use it.
    Else decode from EmployeeKey:
      - Store managers: 30_000_000 + StoreKey
      - Staff:         40_000_000 + StoreKey*1000 + idx
    """
    if "StoreKey" in employees.columns:
        return employees["StoreKey"].astype("Int64")

    ek = employees["EmployeeKey"].astype(np.int64)
    out = pd.Series([pd.NA] * len(employees), dtype="Int64")

    mgr_mask = (ek >= 30_000_000) & (ek < 40_000_000)
    if mgr_mask.any():
        out.loc[mgr_mask] = (ek.loc[mgr_mask] - 30_000_000).astype("Int64")

    staff_mask = ek >= 40_000_000
    if staff_mask.any():
        out.loc[staff_mask] = ((ek.loc[staff_mask] - 40_000_000) // 1000).astype("Int64")

    return out


def _build_store_by_district(emp: pd.DataFrame) -> Dict[Any, List[int]]:
    """
    DistrictId -> [StoreKey,...]

    Prefer store managers, but fall back to all employees if needed.
    """
    emp2 = emp[emp["DistrictId"].notna() & emp["HomeStoreKey"].notna()].copy()
    if emp2.empty:
        return {}

    is_mgr = (emp2["Title"].astype(str) == "Store Manager")
    mgrs = emp2[is_mgr]
    src = mgrs if not mgrs.empty else emp2

    pairs = src[["DistrictId", "HomeStoreKey"]].dropna().drop_duplicates()
    if pairs.empty:
        return {}

    return (
        pairs.groupby("DistrictId")["HomeStoreKey"]
        .apply(lambda s: sorted({int(x) for x in s.tolist()}))
        .to_dict()
    )


# -----------------------------------------------------------------------------
# Movement profile parsing (guarantee Sales Associate dominates)
# -----------------------------------------------------------------------------

def _role_profiles(a_cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any], Dict[str, Dict[str, Any]], float, float]:
    """
    Returns:
      primary_sales_role,
      default_profile,
      per_role_profile,
      max_non_sales_multiplier_frac,
      max_non_sales_episodes_frac
    """
    primary = _str_or(a_cfg.get("primary_sales_role"), "Sales Associate")

    rp = _as_dict(a_cfg.get("role_profiles"))
    default_profile = _as_dict(rp.get("default"))

    per_role: Dict[str, Dict[str, Any]] = {}
    for k, v in rp.items():
        if k == "default":
            continue
        if isinstance(v, dict):
            per_role[str(k)] = dict(v)

    max_mult_frac = _float_or(a_cfg.get("max_non_sales_multiplier_frac"), 0.35)
    max_ep_frac = _float_or(a_cfg.get("max_non_sales_episodes_frac"), 0.50)

    # clamp
    max_mult_frac = float(np.clip(max_mult_frac, 0.0, 1.0))
    max_ep_frac = float(np.clip(max_ep_frac, 0.0, 1.0))

    return primary, default_profile, per_role, max_mult_frac, max_ep_frac


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
    Merge: default_profile <- per_role_profile[role]
    Then enforce: non-sales <= sales caps.
    """
    prof = dict(default_profile)
    prof.update(per_role_profile.get(role, {}))

    # Sales profile is the upper bound reference
    if role != primary_sales_role:
        # Clamp role_multiplier
        sales_mult = _float_or(sales_profile.get("role_multiplier"), 1.0)
        role_mult = _float_or(prof.get("role_multiplier"), 1.0)
        prof["role_multiplier"] = min(role_mult, sales_mult * max_non_sales_multiplier_frac)

        # Clamp episodes_max
        sales_emax = _int_or(sales_profile.get("episodes_max"), 2)
        emax = _int_or(prof.get("episodes_max"), 1)
        prof["episodes_max"] = min(emax, int(np.floor(sales_emax * max_non_sales_episodes_frac)))

        # Keep episodes_min <= episodes_max
        emin = _int_or(prof.get("episodes_min"), 0)
        prof["episodes_min"] = min(emin, _int_or(prof.get("episodes_max"), 0))

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

    # global knobs
    mover_share: float = 0.03,
    pool_scope: str = "district",  # district | all
    allow_store_revisit: bool = False,
    part_time_multiplier: float = 1.0,

    # assignment weighting for transfers
    secondary_fte_min: float = 0.10,
    secondary_fte_max: float = 0.40,

    # movement profiles / caps
    primary_sales_role: str = "Sales Associate",
    role_profiles: Optional[Dict[str, Any]] = None,
    max_non_sales_multiplier_frac: float = 0.35,
    max_non_sales_episodes_frac: float = 0.50,

    # legacy aliases (accepted)
    multi_store_share: Optional[float] = None,
) -> pd.DataFrame:
    """
    Exclusive (non-overlapping) store assignments per employee, with role-weighted movement.

    Output columns:
      EmployeeKey, StoreKey, StartDate, EndDate, FTE, RoleAtStore, IsPrimary, AssignmentSequence
    """
    cols = ["EmployeeKey", "StoreKey", "StartDate", "EndDate", "FTE", "RoleAtStore", "IsPrimary"]
    if not enabled:
        out = pd.DataFrame(columns=cols + ["AssignmentSequence"])
        return out

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

    # store-attached only
    emp = emp[emp["HomeStoreKey"].notna()].copy()
    if emp.empty:
        out = pd.DataFrame(columns=cols + ["AssignmentSequence"])
        return out

    hire = pd.to_datetime(emp["HireDate"]).dt.normalize()
    term = pd.to_datetime(emp["TerminationDate"]).dt.normalize()

    window_start = pd.to_datetime(global_start).normalize()
    window_end = pd.to_datetime(global_end).normalize()
    if window_end < window_start:
        window_start, window_end = window_end, window_start

    # For planning we cap at window_end; open-ended semantics are handled later.
    term_filled = term.fillna(window_end)

    title = emp["Title"].astype(str).to_numpy()

    # Part-time probability by role (keep your existing distribution stable)
    pt_prob = np.where(
        np.isin(title, ["Cashier"]),
        0.45,
        np.where(np.isin(title, ["Sales Associate"]), 0.25, 0.10),
    )
    is_part_time = rng.random(len(emp)) < pt_prob
    target_fte = np.where(is_part_time, rng.uniform(0.50, 0.80, size=len(emp)), 1.0).astype(np.float64)

    # store pools
    store_by_district = _build_store_by_district(emp)
    all_stores = sorted({int(x) for x in emp["HomeStoreKey"].dropna().astype(int).tolist()})
    scope = _str_or(pool_scope, "district").lower()

    # movement profile parsing (from cfg role_profiles dict)
    rp = _as_dict(role_profiles) if role_profiles is not None else {}
    default_profile = _as_dict(rp.get("default"))
    per_role_profile = {k: dict(v) for k, v in rp.items() if k != "default" and isinstance(v, dict)}

    # Determine sales profile for clamping reference
    sales_profile = dict(default_profile)
    sales_profile.update(per_role_profile.get(primary_sales_role, {}))
    if "role_multiplier" not in sales_profile:
        sales_profile["role_multiplier"] = 2.0
    if "episodes_max" not in sales_profile:
        sales_profile["episodes_max"] = 4
    if "episodes_min" not in sales_profile:
        sales_profile["episodes_min"] = 1
    if "duration_days_min" not in sales_profile:
        sales_profile["duration_days_min"] = 14
    if "duration_days_max" not in sales_profile:
        sales_profile["duration_days_max"] = 120

    base = float(np.clip(_float_or(mover_share, 0.03), 0.0, 1.0))
    pt_mult = float(max(1.0, _float_or(part_time_multiplier, 1.0)))

    fmin = float(max(0.0, _float_or(secondary_fte_min, 0.10)))
    fmax = float(max(fmin, _float_or(secondary_fte_max, 0.40)))

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
            s = _rand_dates_between(rng, start_min, latest_start, 1).iloc[0]
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

    rows: List[Dict[str, Any]] = []

    for i in range(len(emp)):
        ek = int(emp.iloc[i]["EmployeeKey"])
        did = emp.iloc[i]["DistrictId"]
        home_store = int(emp.iloc[i]["HomeStoreKey"])
        role = str(emp.iloc[i]["Title"])
        tgt = float(target_fte[i])

        emp_hire = hire.iloc[i]
        emp_term = term.iloc[i]              # may be NaT
        plan_end_raw = term_filled.iloc[i]   # NaT -> window_end

        if pd.isna(emp_hire) or pd.isna(plan_end_raw) or plan_end_raw < emp_hire:
            continue

        # Open-ended if still employed beyond dataset end (or unknown end)
        open_ended = pd.isna(emp_term) or (pd.notna(emp_term) and pd.to_datetime(emp_term).normalize() > window_end)

        # Plan end is capped to dataset end (and to termination if within window)
        plan_end = pd.to_datetime(plan_end_raw).normalize()
        if plan_end > window_end:
            plan_end = window_end

        # Assignments must start within dataset window
        gen_start = pd.to_datetime(emp_hire).normalize()
        if gen_start < window_start:
            gen_start = window_start

        # No overlap with dataset window -> skip
        if plan_end < gen_start or gen_start > window_end:
            continue

        # never move Store Manager (but still emit home store row)
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
                max_non_sales_multiplier_frac=float(np.clip(max_non_sales_multiplier_frac, 0.0, 1.0)),
                max_non_sales_episodes_frac=float(np.clip(max_non_sales_episodes_frac, 0.0, 1.0)),
            )

        role_mult = float(max(0.0, _float_or(prof.get("role_multiplier"), 1.0)))
        p_move = base * role_mult * (pt_mult if tgt < 1.0 else 1.0)
        p_move = float(np.clip(p_move, 0.0, 0.90))

        e_min = max(0, _int_or(prof.get("episodes_min"), 0))
        e_max = max(e_min, _int_or(prof.get("episodes_max"), 0))

        dmin = max(1, _int_or(prof.get("duration_days_min"), 30))
        dmax = max(dmin, _int_or(prof.get("duration_days_max"), 180))

        episodes: List[Tuple[pd.Timestamp, pd.Timestamp, int, float]] = []
        if e_max > 0 and rng.random() < p_move:
            other = _candidate_stores(did, home_store)
            if other:
                k = int(rng.integers(e_min, e_max + 1)) if e_max >= e_min else int(e_min)
                if k > 0:
                    # NOTE: episodes are sampled only inside dataset window (gen_start..plan_end)
                    episodes = _sample_non_overlapping_episodes(gen_start, plan_end, other, k, tgt, dmin, dmax)

        cur = pd.to_datetime(gen_start).normalize()

        def _emit(store_key: int, seg_start: pd.Timestamp, seg_end: Any, is_primary: bool, fte_val: float) -> None:
            rows.append(
                dict(
                    EmployeeKey=ek,
                    StoreKey=int(store_key),
                    StartDate=pd.to_datetime(seg_start).normalize(),
                    EndDate=(pd.to_datetime(seg_end).normalize() if pd.notna(seg_end) else pd.NaT),
                    FTE=float(fte_val),
                    RoleAtStore=role,
                    IsPrimary=bool(is_primary),
                )
            )

        # split home assignment around transfer episodes (exclusive)
        for (s, e, store, sec_fte) in episodes:
            s = pd.to_datetime(s).normalize()
            e = pd.to_datetime(e).normalize()

            before_end = (s - pd.Timedelta(days=1)).normalize()
            if cur <= before_end:
                _emit(home_store, cur, before_end, True, tgt)

            _emit(store, s, e, False, sec_fte)
            cur = (e + pd.Timedelta(days=1)).normalize()

        # final home segment
        if cur <= plan_end:
            if open_ended:
                # employment continues beyond dataset end (or unknown end): keep open-ended validity
                _emit(home_store, cur, pd.NaT, True, tgt)
            else:
                # terminated within dataset window
                final_end = pd.to_datetime(emp_term).normalize()
                if final_end > window_end:
                    final_end = window_end
                if final_end >= cur:
                    _emit(home_store, cur, final_end, True, tgt)

    out = pd.DataFrame(rows, columns=cols)
    if out.empty:
        out["AssignmentSequence"] = pd.Series(dtype=np.int32)
        return out

    out["EmployeeKey"] = out["EmployeeKey"].astype(np.int64)
    out["StoreKey"] = out["StoreKey"].astype(np.int64)
    out["FTE"] = out["FTE"].astype(np.float64)
    out["IsPrimary"] = out["IsPrimary"].astype(bool)
    out["RoleAtStore"] = out["RoleAtStore"].astype(str)

    # Safety: non-overlap per employee (treat NaT EndDate as far future)
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
    return out


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
    seed = _pick_seed(cfg, a_cfg, fallback=42)
    global_start, global_end = _parse_global_dates(cfg, a_cfg)

    employees = pd.read_parquet(
        employees_path,
        columns=["EmployeeKey", "HireDate", "TerminationDate", "Title", "DistrictId"],
    )

    version_cfg = dict(a_cfg)
    version_cfg.pop("_force_regenerate", None)
    version_cfg["schema_version"] = 7  # bump: dataset-window clamping + open-ended semantics
    version_cfg["_rows_employees"] = int(len(employees))
    version_cfg["_global_dates"] = {"start": str(global_start.date()), "end": str(global_end.date())}

    if not force and not should_regenerate("employee_store_assignments", version_cfg, out_path):
        skip("EmployeeStoreAssignments up-to-date; skipping.")
        return

    role_profiles = _as_dict(a_cfg.get("role_profiles"))

    with stage("Generating EmployeeStoreAssignments"):
        df = generate_employee_store_assignments(
            employees=employees,
            seed=seed,
            global_start=global_start,
            global_end=global_end,
            enabled=bool(a_cfg.get("enabled", True)),
            mover_share=_float_or(a_cfg.get("mover_share", a_cfg.get("multi_store_share", 0.03)), 0.03),
            pool_scope=_str_or(a_cfg.get("pool_scope"), "district"),
            allow_store_revisit=bool(a_cfg.get("allow_store_revisit", False)),
            part_time_multiplier=_float_or(a_cfg.get("part_time_multiplier", 1.0), 1.0),
            secondary_fte_min=_float_or(a_cfg.get("secondary_fte_min", 0.10), 0.10),
            secondary_fte_max=_float_or(a_cfg.get("secondary_fte_max", 0.40), 0.40),
            primary_sales_role=_str_or(a_cfg.get("primary_sales_role"), "Sales Associate"),
            role_profiles=role_profiles,
            max_non_sales_multiplier_frac=_float_or(a_cfg.get("max_non_sales_multiplier_frac", 0.35), 0.35),
            max_non_sales_episodes_frac=_float_or(a_cfg.get("max_non_sales_episodes_frac", 0.50), 0.50),
            multi_store_share=(a_cfg.get("multi_store_share") if "multi_store_share" in a_cfg else None),
        )

        write_parquet_with_date32(
            df,
            out_path,
            date_cols=["StartDate", "EndDate"],
            cast_all_datetime=False,
            compression=_str_or(a_cfg.get("parquet_compression"), "snappy"),
            compression_level=(int(a_cfg["parquet_compression_level"]) if "parquet_compression_level" in a_cfg else None),
            force_date32=True,
        )

    save_version("employee_store_assignments", version_cfg, out_path)
    info(f"EmployeeStoreAssignments written: {out_path}")