# src/dimensions/employee_store_assignments.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.utils.logging_utils import info, skip, stage
from src.utils.output_utils import write_parquet_with_date32
from src.versioning.version_store import should_regenerate, save_version


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


def _pick_seed(cfg: Dict[str, Any], a_cfg: Dict[str, Any], fallback: int = 42) -> int:
    override = _as_dict(a_cfg.get("override"))
    seed = override.get("seed")
    if seed is None:
        seed = a_cfg.get("seed")
    if seed is None:
        seed = _as_dict(cfg.get("defaults")).get("seed")
    return _int_or(seed, fallback)


def _parse_global_dates(cfg: Dict[str, Any], a_cfg: Dict[str, Any]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    gd = _as_dict(a_cfg.get("global_dates"))
    if gd:
        return pd.to_datetime(gd["start"]).normalize(), pd.to_datetime(gd["end"]).normalize()
    # fallback
    return pd.Timestamp("2021-01-01"), pd.Timestamp("2026-12-31")


def _rand_dates_between(rng: np.random.Generator, start: pd.Timestamp, end: pd.Timestamp, n: int) -> pd.Series:
    start_i = start.value // 10**9
    end_i = end.value // 10**9
    if end_i < start_i:
        start_i, end_i = end_i, start_i
    secs = rng.integers(start_i, end_i + 1, size=n, dtype=np.int64)
    dt = pd.to_datetime(secs, unit="s")
    if isinstance(dt, pd.DatetimeIndex):
        return pd.Series(dt.normalize())
    return dt.dt.normalize()


def _infer_home_store_key(employees: pd.DataFrame) -> pd.Series:
    """
    If employees has StoreKey column, use it.
    Else decode from EmployeeKey:
      - Store managers: 30_000_000 + StoreKey
      - Staff:         40_000_000 + StoreKey*1000 + idx
    Returns pandas nullable Int64.
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


def generate_employee_store_assignments(
    employees: pd.DataFrame,
    seed: int,
    global_end: pd.Timestamp,
    *,
    enabled: bool = True,
    multi_store_share: float = 0.03,
    max_extra_stores: int = 2,
    secondary_fte_min: float = 0.10,
    secondary_fte_max: float = 0.40,
    secondary_duration_days_min: int = 30,
    secondary_duration_days_max: int = 180,
) -> pd.DataFrame:
    """
    Time-bounded bridge Employee <-> Store.

    Output columns:
      EmployeeKey, StoreKey, StartDate, EndDate, FTE, RoleAtStore, IsPrimary
    """
    cols = ["EmployeeKey", "StoreKey", "StartDate", "EndDate", "FTE", "RoleAtStore", "IsPrimary"]
    if not enabled:
        return pd.DataFrame(columns=cols)

    rng = np.random.default_rng(int(seed))

    # Required columns (StoreKey is NOT required)
    required = {"EmployeeKey", "HireDate", "TerminationDate", "Title", "DistrictId"}
    missing = [c for c in required if c not in employees.columns]
    if missing:
        raise ValueError(f"employees missing required columns: {missing}")

    emp = employees.copy()
    emp["EmployeeKey"] = emp["EmployeeKey"].astype(np.int64)
    emp["DistrictId"] = emp["DistrictId"].astype("Int16")

    emp["HomeStoreKey"] = _infer_home_store_key(emp)

    # only store-attached employees get assignments
    emp = emp[emp["HomeStoreKey"].notna()].copy()
    if emp.empty:
        return pd.DataFrame(columns=cols)

    hire = pd.to_datetime(emp["HireDate"]).dt.normalize()
    term = pd.to_datetime(emp["TerminationDate"]).dt.normalize()
    term_filled = term.fillna(global_end)

    title = emp["Title"].astype(str).to_numpy()

    # Part-time probability by role
    pt_prob = np.where(np.isin(title, ["Cashier"]), 0.45, np.where(np.isin(title, ["Sales Associate"]), 0.25, 0.10))
    is_part_time = rng.random(len(emp)) < pt_prob
    target_fte = np.where(is_part_time, rng.uniform(0.50, 0.80, size=len(emp)), 1.0)

    rows = []

    # Primary assignment: employee's home store
    for i in range(len(emp)):
        rows.append(
            dict(
                EmployeeKey=int(emp.iloc[i]["EmployeeKey"]),
                StoreKey=int(emp.iloc[i]["HomeStoreKey"]),
                StartDate=hire.iloc[i],
                EndDate=term.iloc[i],
                FTE=float(target_fte[i]),
                RoleAtStore=str(emp.iloc[i]["Title"]),
                IsPrimary=True,
            )
        )

    # Build store pool per district for secondaries.
    # Use Store Manager rows as the authoritative list of stores in each district.
    is_store_mgr = (emp["Title"].astype(str) == "Store Manager")
    store_mgrs = emp[is_store_mgr & emp["DistrictId"].notna()].copy()

    if store_mgrs.empty:
        out = pd.DataFrame(rows)
        out["EmployeeKey"] = out["EmployeeKey"].astype(np.int64)
        out["StoreKey"] = out["StoreKey"].astype(np.int64)
        out["FTE"] = out["FTE"].astype(np.float64)
        out["IsPrimary"] = out["IsPrimary"].astype(bool)
        out["RoleAtStore"] = out["RoleAtStore"].astype(str)
        return out

    store_by_district = (
        store_mgrs[["DistrictId", "HomeStoreKey"]]
        .dropna()
        .drop_duplicates()
        .groupby("DistrictId")["HomeStoreKey"]
        .apply(lambda s: [int(x) for x in s.tolist()])
        .to_dict()
    )

    # Multi-store candidates: exclude store managers
    can_multi = emp["Title"].astype(str) != "Store Manager"
    idx_candidates = np.where(can_multi.to_numpy())[0]
    rng.shuffle(idx_candidates)

    share = max(0.0, min(1.0, float(multi_store_share)))
    n_multi = int(round(len(emp) * share))
    idx_multi = set(idx_candidates[:n_multi].tolist())

    for i in idx_multi:
        did = emp.iloc[i]["DistrictId"]
        if pd.isna(did):
            continue
        pool = store_by_district.get(did, [])
        if len(pool) <= 1:
            continue

        home_store = int(emp.iloc[i]["HomeStoreKey"])
        other_stores = [s for s in pool if int(s) != home_store]
        if not other_stores:
            continue

        k = rng.integers(1, max(2, _int_or(max_extra_stores, 2)) + 1)
        k = min(k, len(other_stores))
        chosen = rng.choice(other_stores, size=k, replace=False)

        sec_ftes = rng.uniform(float(secondary_fte_min), float(secondary_fte_max), size=k)
        sec_total = float(sec_ftes.sum())
        tgt = float(target_fte[i])

        # keep at least 0.20 on primary
        max_sec_total = max(0.0, tgt - 0.20)
        if max_sec_total <= 0:
            continue
        if sec_total > max_sec_total:
            sec_ftes = sec_ftes * (max_sec_total / sec_total)

        # secondary assignments time-bounded within employment
        start_min = hire.iloc[i]
        start_max = term_filled.iloc[i]

        for j in range(k):
            sec_start = _rand_dates_between(rng, start_min, start_max, 1).iloc[0]
            dur = rng.integers(int(secondary_duration_days_min), int(secondary_duration_days_max) + 1)
            sec_end = (sec_start + pd.Timedelta(days=int(dur))).normalize()
            if pd.notna(term.iloc[i]):
                sec_end = min(sec_end, term.iloc[i])
            if sec_end < sec_start:
                sec_end = sec_start

            rows.append(
                dict(
                    EmployeeKey=int(emp.iloc[i]["EmployeeKey"]),
                    StoreKey=int(chosen[j]),
                    StartDate=sec_start,
                    EndDate=sec_end,
                    FTE=float(sec_ftes[j]),
                    RoleAtStore=str(emp.iloc[i]["Title"]),
                    IsPrimary=False,
                )
            )

        # reduce primary FTE (rows index matches the primary row for this i because primaries were appended first)
        rows[i]["FTE"] = float(max(0.20, tgt - float(sec_ftes.sum())))

    out = pd.DataFrame(rows)
    out["EmployeeKey"] = out["EmployeeKey"].astype(np.int64)
    out["StoreKey"] = out["StoreKey"].astype(np.int64)
    out["FTE"] = out["FTE"].astype(np.float64)
    out["IsPrimary"] = out["IsPrimary"].astype(bool)
    out["RoleAtStore"] = out["RoleAtStore"].astype(str)
    return out


def run_employee_store_assignments(cfg: Dict[str, Any], parquet_folder: Path) -> None:
    cfg = cfg or {}
    a_cfg = _as_dict(cfg.get("employee_store_assignments"))

    parquet_folder = Path(parquet_folder)
    parquet_folder.mkdir(parents=True, exist_ok=True)

    employees_path = parquet_folder / "employees.parquet"
    out_path = parquet_folder / "employee_store_assignments.parquet"

    if not employees_path.exists():
        raise FileNotFoundError(f"Missing employees parquet: {employees_path}")

    force = bool(a_cfg.get("_force_regenerate", False))
    seed = _pick_seed(cfg, a_cfg, fallback=42)
    _, global_end = _parse_global_dates(cfg, a_cfg)

    # IMPORTANT: do NOT request StoreKey here; it may not exist in employees.parquet
    employees = pd.read_parquet(
        employees_path,
        columns=["EmployeeKey", "HireDate", "TerminationDate", "Title", "DistrictId"],
    )

    version_cfg = dict(a_cfg)
    version_cfg.pop("_force_regenerate", None)
    version_cfg["schema_version"] = 2  # bump to force regen after storekey inference change
    version_cfg["_rows_employees"] = int(len(employees))

    if not force and not should_regenerate("employee_store_assignments", version_cfg, out_path):
        skip("EmployeeStoreAssignments up-to-date; skipping.")
        return

    with stage("Generating EmployeeStoreAssignments"):
        df = generate_employee_store_assignments(
            employees=employees,
            seed=seed,
            global_end=global_end,
            enabled=bool(a_cfg.get("enabled", True)),
            multi_store_share=_float_or(a_cfg.get("multi_store_share"), 0.03),
            max_extra_stores=_int_or(a_cfg.get("max_extra_stores"), 2),
            secondary_fte_min=_float_or(a_cfg.get("secondary_fte_min"), 0.10),
            secondary_fte_max=_float_or(a_cfg.get("secondary_fte_max"), 0.40),
            secondary_duration_days_min=_int_or(a_cfg.get("secondary_duration_days_min"), 30),
            secondary_duration_days_max=_int_or(a_cfg.get("secondary_duration_days_max"), 180),
        )

        write_parquet_with_date32(
            df,
            out_path,
            date_cols=["StartDate", "EndDate"],
            cast_all_datetime=False,
            compression=str(a_cfg.get("parquet_compression", "snappy")),
            compression_level=(int(a_cfg["parquet_compression_level"]) if "parquet_compression_level" in a_cfg else None),
            force_date32=True,
        )

    save_version("employee_store_assignments", version_cfg, out_path)
    info(f"EmployeeStoreAssignments written: {out_path}")
