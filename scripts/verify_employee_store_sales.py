"""Comprehensive data-quality checks for stores, employees,
employee_store_assignments bridge, and sales alignment.

Usage:
    python scripts/verify_employee_store_sales.py <dataset_folder>

    # Examples:
    python scripts/verify_employee_store_sales.py "generated_datasets/2026-04-05 05_03_16 PM Customers 43K Sales 1M PARQUET"
    python scripts/verify_employee_store_sales.py generated_datasets/latest

Exit code 0 = all passed, 1 = failures found.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Key ranges (mirror src/defaults.py and employees/generator.py)
# ---------------------------------------------------------------------------
ONLINE_STORE_KEY_BASE = 10_000
STORE_MGR_KEY_BASE = 30_000_000
STAFF_KEY_BASE = 40_000_000
ONLINE_EMP_KEY_BASE = 50_000_000


# ============================================================================
# Result model
# ============================================================================

class Check:
    __slots__ = ("category", "name", "passed", "message", "detail")

    def __init__(
        self,
        category: str,
        name: str,
        passed: bool,
        message: str,
        detail: str = "",
    ):
        self.category = category
        self.name = name
        self.passed = passed
        self.message = message
        self.detail = detail


results: List[Check] = []


def _add(
    category: str,
    name: str,
    passed: bool,
    message: str,
    detail: str = "",
) -> None:
    results.append(Check(category, name, passed, message, detail))
    tag = "\033[32mPASS\033[0m" if passed else "\033[31mFAIL\033[0m"
    print(f"  [{tag}] {name}: {message}")
    if detail and not passed:
        for line in detail.strip().splitlines():
            print(f"         {line}")


# ============================================================================
# Loaders
# ============================================================================

def _load(folder: Path, sub: str, name: str) -> Optional[pd.DataFrame]:
    p = folder / sub / f"{name}.parquet"
    if not p.exists():
        print(f"  [SKIP] {name}: file not found ({p})")
        return None
    return pd.read_parquet(p)


def _to_dt(series: pd.Series) -> pd.Series:
    """Coerce any date-like column (date objects, timestamps, strings) to datetime64[ns]."""
    return pd.to_datetime(series, errors="coerce")


# ============================================================================
# Checks: STORES
# ============================================================================

def check_stores(stores: pd.DataFrame) -> None:
    cat = "Stores"
    n = len(stores)

    # --- Basic counts ---
    n_phys = int((stores["StoreKey"] < ONLINE_STORE_KEY_BASE).sum())
    n_online = n - n_phys
    status_vc = stores["Status"].astype(str).value_counts().to_dict()
    _add(cat, "Store counts",
         n > 0,
         f"{n} stores ({n_phys} physical, {n_online} online). "
         f"Status: {status_vc}")

    # --- Renovation date sanity ---
    has_reno = stores["RenovationStartDate"].notna() & stores["RenovationEndDate"].notna()
    reno = stores[has_reno]
    n_reno = len(reno)
    _add(cat, "Renovation count", True, f"{n_reno} store(s) with renovation dates")

    if n_reno > 0:
        rs = _to_dt(reno["RenovationStartDate"])
        re = _to_dt(reno["RenovationEndDate"])
        od = _to_dt(reno["OpeningDate"])

        inv = (re < rs).sum()
        _add(cat, "RenovationEnd >= RenovationStart",
             inv == 0,
             f"{inv} violation(s)",
             _sample(reno[re < rs], ["StoreKey", "RenovationStartDate", "RenovationEndDate"]))

        before_open = (rs < od).sum()
        _add(cat, "RenovationStart >= OpeningDate",
             before_open == 0,
             f"{before_open} violation(s)",
             _sample(reno[rs < od], ["StoreKey", "OpeningDate", "RenovationStartDate"]))

    # --- ClosingDate sanity ---
    has_close = stores["ClosingDate"].notna()
    if has_close.any():
        closed_stores = stores[has_close]
        status_mismatch = closed_stores[closed_stores["Status"].astype(str) != "Closed"]
        _add(cat, "ClosingDate implies Status=Closed",
             len(status_mismatch) == 0,
             f"{len(status_mismatch)} store(s) with ClosingDate but Status != Closed",
             _sample(status_mismatch, ["StoreKey", "Status", "ClosingDate"]))

        cd = _to_dt(closed_stores["ClosingDate"])
        od = _to_dt(closed_stores["OpeningDate"])
        bad = (cd <= od).sum()
        _add(cat, "ClosingDate > OpeningDate",
             bad == 0,
             f"{bad} violation(s)")


# ============================================================================
# Checks: EMPLOYEES
# ============================================================================

def check_employees(employees: pd.DataFrame) -> None:
    cat = "Employees"
    n = len(employees)

    hd = _to_dt(employees["HireDate"])
    td = _to_dt(employees["TerminationDate"])
    active = employees["IsActive"].astype(int)

    # --- HireDate > TerminationDate ---
    has_both = hd.notna() & td.notna()
    inv = (hd[has_both] > td[has_both]).sum()
    _add(cat, "HireDate <= TerminationDate",
         inv == 0,
         f"{inv} violation(s)",
         _sample(employees[has_both & (hd > td)], ["EmployeeKey", "HireDate", "TerminationDate"]))

    # --- IsActive consistency ---
    active_with_term = ((active == 1) & td.notna()).sum()
    _add(cat, "IsActive=1 has no TerminationDate",
         active_with_term == 0,
         f"{active_with_term} violation(s)")

    inactive_no_term = ((active == 0) & td.isna()).sum()
    _add(cat, "IsActive=0 has TerminationDate",
         inactive_no_term == 0,
         f"{inactive_no_term} violation(s)")

    # --- Hierarchy (non-CEO parent exists) ---
    if "ParentEmployeeKey" in employees.columns:
        all_ek = set(employees["EmployeeKey"].astype(int))
        non_root = employees[employees["OrgLevel"].astype(int) > 1]
        if len(non_root) > 0:
            parent_keys = non_root["ParentEmployeeKey"].astype(int)
            orphans = (~parent_keys.isin(all_ek)).sum()
            _add(cat, "Employee hierarchy valid",
                 orphans == 0,
                 f"{orphans} broken parent link(s)")

    # --- No managers should have SalesPersonFlag ---
    if "SalesPersonFlag" in employees.columns:
        mgr_mask = (employees["EmployeeKey"].astype(np.int64) >= STORE_MGR_KEY_BASE) & \
                   (employees["EmployeeKey"].astype(np.int64) < STAFF_KEY_BASE)
        mgr_sp = (mgr_mask & (employees["SalesPersonFlag"].astype(int) == 1)).sum()
        _add(cat, "No managers flagged as salespeople",
             mgr_sp == 0,
             f"{mgr_sp} manager(s) with SalesPersonFlag=1")


# ============================================================================
# Checks: EMPLOYEE STORE ASSIGNMENTS
# ============================================================================

def check_esa(
    esa: pd.DataFrame,
    stores: pd.DataFrame,
    employees: pd.DataFrame,
) -> None:
    cat = "ESA-Bridge"

    sd = _to_dt(esa["StartDate"])
    ed = _to_dt(esa["EndDate"])

    # --- StartDate > EndDate ---
    inv = (sd > ed).sum()
    _add(cat, "StartDate <= EndDate",
         inv == 0,
         f"{inv} violation(s)",
         _sample(esa[sd > ed], ["EmployeeKey", "StoreKey", "StartDate", "EndDate", "TransferReason"]))

    # --- TransferReason / IsPrimary / Status distribution ---
    tr_vc = esa["TransferReason"].value_counts().to_dict()
    _add(cat, "TransferReason distribution", True, str(tr_vc))

    ip_vc = esa["IsPrimary"].value_counts().to_dict()
    _add(cat, "IsPrimary distribution", True, str(ip_vc))

    st_vc = esa["Status"].astype(str).value_counts().to_dict()
    _add(cat, "Status distribution", True, str(st_vc))

    # --- Shared renovation lookups for IsPrimary + segment checks ---
    reno_store_keys = set(
        stores.loc[
            stores["RenovationStartDate"].notna(), "StoreKey"
        ].astype(int)
    )
    closed_reno_keys = set(
        stores.loc[
            stores["RenovationStartDate"].notna() & stores["ClosingDate"].notna(),
            "StoreKey",
        ].astype(int)
    )

    def _find_source_reno_store(ek: int) -> int:
        """Find which renovating store an employee came from."""
        emp_rows = esa[esa["EmployeeKey"] == ek]
        at_reno = emp_rows[emp_rows["StoreKey"].astype(int).isin(reno_store_keys)]
        if len(at_reno) > 0:
            return int(at_reno.iloc[0]["StoreKey"])
        # Manager key encoding: 30_000_000 + StoreKey
        if STORE_MGR_KEY_BASE <= ek < STAFF_KEY_BASE:
            return ek - STORE_MGR_KEY_BASE
        # Staff key encoding: 40_000_000 + StoreKey * multiplier
        if STAFF_KEY_BASE <= ek < ONLINE_EMP_KEY_BASE:
            return (ek - STAFF_KEY_BASE) // 1000
        return -1

    # --- Renovation Reassignment IsPrimary rules ---
    # Temporary renovation (store reopens): IsPrimary must be False.
    # Permanent closure renovation (store closed): IsPrimary must be True.
    rr = esa[esa["TransferReason"] == "Renovation Reassignment"]
    if len(rr) > 0:
        rr_source = rr["EmployeeKey"].astype(int).apply(_find_source_reno_store)
        rr_from_closed = rr_source.isin(closed_reno_keys)

        # Temp reno (store reopens): IsPrimary must be False
        temp_rr = rr[~rr_from_closed]
        bad_temp = (temp_rr["IsPrimary"].astype(int) == 1).sum() if len(temp_rr) > 0 else 0
        # Permanent reno (store closed): IsPrimary must be True
        perm_rr = rr[rr_from_closed]
        bad_perm = (perm_rr["IsPrimary"].astype(int) == 0).sum() if len(perm_rr) > 0 else 0

        _add(cat, "Renovation Reassignment IsPrimary rules",
             bad_temp == 0 and bad_perm == 0,
             f"{len(temp_rr)} temp (should be False): {bad_temp} violation(s), "
             f"{len(perm_rr)} permanent (should be True): {bad_perm} violation(s)")

    # --- Renovation segment completeness ---
    # Stores that reopened: employees need pre-reno + temp + return (3 segments).
    # Stores that closed permanently: employees need pre-reno + temp (2 segments, no return).
    reno_stores = set(reno_store_keys)  # reuse shared lookup
    if reno_stores:
        reopened_reno = reno_stores - closed_reno_keys
        closed_reno = reno_stores & closed_reno_keys

        reassigned = set(
            esa.loc[esa["TransferReason"] == "Renovation Reassignment", "EmployeeKey"].astype(int)
        )
        returned = set(
            esa.loc[esa["TransferReason"] == "Renovation Return", "EmployeeKey"].astype(int)
        )

        # Classify each reassigned employee by source store
        reopen_emps: set[int] = set()
        closed_emps: set[int] = set()
        for ek in reassigned:
            src = _find_source_reno_store(ek)
            if src in closed_reno:
                closed_emps.add(ek)
            elif src in reopened_reno:
                reopen_emps.add(ek)

        # Reopened stores: need reassign + return
        miss_reassign_reopen = reopen_emps - reassigned
        miss_return_reopen = reopen_emps - returned

        # Closed stores: need reassign only, no return
        miss_reassign_closed = closed_emps - reassigned
        unexpected_return = closed_emps & returned

        ok = (
            len(miss_reassign_reopen) == 0
            and len(miss_return_reopen) == 0
            and len(miss_reassign_closed) == 0
            and len(unexpected_return) == 0
        )
        _add(cat, "Renovation segments complete",
             ok,
             f"Reopened stores ({len(reopened_reno)}): {len(reopen_emps)} emp(s), "
             f"missing reassign={len(miss_reassign_reopen)}, missing return={len(miss_return_reopen)}. "
             f"Closed stores ({len(closed_reno)}): {len(closed_emps)} emp(s), "
             f"missing reassign={len(miss_reassign_closed)}, unexpected return={len(unexpected_return)}")

    # --- No overlapping same-store assignments ---
    esa_sorted = esa.sort_values(["EmployeeKey", "StoreKey", "StartDate"])
    grp = esa_sorted.groupby(["EmployeeKey", "StoreKey"])
    overlap_count = 0
    for _, g in grp:
        if len(g) < 2:
            continue
        ends = _to_dt(g["EndDate"]).values[:-1]
        starts = _to_dt(g["StartDate"]).values[1:]
        overlap_count += int((starts <= ends).sum())

    _add(cat, "No overlapping same-store assignments",
         overlap_count == 0,
         f"{overlap_count} overlap(s)")

    # --- ESA EmployeeKey exists in Employees ---
    all_emp_keys = set(employees["EmployeeKey"].astype(int))
    esa_emp_keys = set(esa["EmployeeKey"].astype(int))
    orphan_ek = esa_emp_keys - all_emp_keys
    _add(cat, "All ESA EmployeeKeys exist in Employees",
         len(orphan_ek) == 0,
         f"{len(orphan_ek)} orphan key(s)")

    # --- ESA StoreKey exists in Stores ---
    all_store_keys = set(stores["StoreKey"].astype(int))
    esa_store_keys = set(esa["StoreKey"].astype(int))
    orphan_sk = esa_store_keys - all_store_keys
    _add(cat, "All ESA StoreKeys exist in Stores",
         len(orphan_sk) == 0,
         f"{len(orphan_sk)} orphan key(s)")

    # --- Assignment dates within employment window ---
    merged = esa.merge(
        employees[["EmployeeKey", "HireDate", "TerminationDate"]],
        on="EmployeeKey", how="left", suffixes=("", "_emp"),
    )
    a_start = _to_dt(merged["StartDate"])
    hire = _to_dt(merged["HireDate"])
    before_hire = (a_start < hire).sum()
    _add(cat, "Assignment StartDate >= HireDate",
         before_hire == 0,
         f"{before_hire} violation(s)")

    # --- ESA rows past store ClosingDate ---
    store_close = stores[stores["ClosingDate"].notna()][["StoreKey", "ClosingDate"]].copy()
    store_close["ClosingDate"] = _to_dt(store_close["ClosingDate"])
    if len(store_close) > 0:
        esa_c = esa.merge(store_close, on="StoreKey", how="inner")
        if len(esa_c) > 0:
            past_close = (_to_dt(esa_c["StartDate"]) >= esa_c["ClosingDate"]).sum()
            _add(cat, "No ESA rows start after store ClosingDate",
                 past_close == 0,
                 f"{past_close} violation(s)")

    # --- Salesperson coverage per open physical store-month ---
    _check_salesperson_coverage(esa, stores, cat)


def _check_salesperson_coverage(
    esa: pd.DataFrame,
    stores: pd.DataFrame,
    cat: str,
) -> None:
    """Every open, non-renovating physical store must have >= 1 salesperson per month."""
    sp = esa[esa["RoleAtStore"].astype(str) == "Sales Associate"]
    if sp.empty:
        _add(cat, "Salesperson coverage", True, "No Sales Associate assignments (skipped)")
        return

    phys = stores[stores["StoreKey"].astype(int) < ONLINE_STORE_KEY_BASE]
    sp_start = _to_dt(sp["StartDate"]).values.astype("datetime64[ns]")
    sp_end = _to_dt(sp["EndDate"]).values.astype("datetime64[ns]")
    sp_sk = sp["StoreKey"].astype(int).values

    open_dates = {}
    close_dates = {}
    reno_start_dates = {}
    reno_end_dates = {}

    for _, row in phys.iterrows():
        sk = int(row["StoreKey"])
        od = _to_dt(pd.Series([row["OpeningDate"]])).iloc[0]
        if pd.notna(od):
            open_dates[sk] = od
        cd = _to_dt(pd.Series([row["ClosingDate"]])).iloc[0]
        if pd.notna(cd):
            close_dates[sk] = cd
        rs = _to_dt(pd.Series([row["RenovationStartDate"]])).iloc[0] if pd.notna(row.get("RenovationStartDate")) else pd.NaT
        re = _to_dt(pd.Series([row["RenovationEndDate"]])).iloc[0] if pd.notna(row.get("RenovationEndDate")) else pd.NaT
        if pd.notna(rs) and pd.notna(re):
            reno_start_dates[sk] = rs
            reno_end_dates[sk] = re

    # Determine date range from ESA
    all_starts = _to_dt(esa["StartDate"])
    all_ends = _to_dt(esa["EndDate"])
    global_start = all_starts.min()
    global_end = all_ends.max()

    months = pd.date_range(global_start.replace(day=1), global_end, freq="MS")
    violations: list[tuple[int, str]] = []

    for check_date in months:
        cd_ns = np.datetime64(check_date, "ns")
        active_mask = (sp_start <= cd_ns) & (sp_end >= cd_ns)
        covered = set(sp_sk[active_mask])

        for sk in phys["StoreKey"].astype(int):
            # Skip if not yet open
            od = open_dates.get(sk)
            if od is not None and check_date < od:
                continue
            # Skip if closed
            cd = close_dates.get(sk)
            if cd is not None and check_date >= cd:
                continue
            # Skip if renovating
            rs = reno_start_dates.get(sk)
            re = reno_end_dates.get(sk)
            if rs is not None and re is not None and rs <= check_date < re:
                continue

            if sk not in covered:
                violations.append((sk, check_date.strftime("%Y-%m")))

    _add(cat, "Salesperson coverage (every open store-month)",
         len(violations) == 0,
         f"{len(violations)} store-month gap(s)",
         "\n".join(f"  Store {sk}, {ym}" for sk, ym in violations[:15]))


# ============================================================================
# Checks: SALES ALIGNMENT
# ============================================================================

def check_sales(
    sales: pd.DataFrame,
    esa: pd.DataFrame,
    stores: pd.DataFrame,
) -> None:
    cat = "Sales"
    n = len(sales)
    order_date = _to_dt(sales["OrderDate"])

    _add(cat, "Sales row count", True,
         f"{n:,} rows, {order_date.min().date()} to {order_date.max().date()}, "
         f"{sales['StoreKey'].nunique()} stores, "
         f"{sales['EmployeeKey'].nunique()} employees")

    # --- No manager keys in sales (30M-40M range) ---
    ek = sales["EmployeeKey"].astype(np.int64)
    mgr_in_sales = ((ek >= STORE_MGR_KEY_BASE) & (ek < STAFF_KEY_BASE)).sum()
    _add(cat, "No manager keys in sales",
         mgr_in_sales == 0,
         f"{mgr_in_sales} sale(s) with manager EmployeeKey")

    # --- 100% sales-to-ESA bridge match ---
    _check_sales_esa_match(sales, esa, cat)

    # --- No sales during renovation ---
    _check_no_sales_during_renovation(sales, stores, cat)

    # --- No sales after store closure ---
    _check_no_sales_after_closure(sales, stores, cat)

    # --- Post-transfer leakage ---
    _check_post_transfer_leakage(sales, esa, cat)


def _check_sales_esa_match(
    sales: pd.DataFrame,
    esa: pd.DataFrame,
    cat: str,
) -> None:
    """Every sale must match an ESA row on EmployeeKey + StoreKey + OrderDate."""
    # Build arrays for vectorized matching
    s_ek = sales["EmployeeKey"].astype(np.int64).values
    s_sk = sales["StoreKey"].astype(np.int32).values
    s_od = _to_dt(sales["OrderDate"]).values.astype("datetime64[ns]")

    e_ek = esa["EmployeeKey"].astype(np.int64).values
    e_sk = esa["StoreKey"].astype(np.int32).values
    e_sd = _to_dt(esa["StartDate"]).values.astype("datetime64[ns]")
    e_ed = _to_dt(esa["EndDate"]).values.astype("datetime64[ns]")

    # Build an index: (EmployeeKey, StoreKey) -> list of (start, end)
    from collections import defaultdict
    esa_idx: dict[tuple[int, int], list[tuple]] = defaultdict(list)
    for i in range(len(e_ek)):
        esa_idx[(int(e_ek[i]), int(e_sk[i]))].append((e_sd[i], e_ed[i]))

    # Check in batches for memory efficiency
    batch = 200_000
    unmatched = 0
    unmatched_samples: list[dict] = []

    for start in range(0, len(s_ek), batch):
        end = min(start + batch, len(s_ek))
        for j in range(start, end):
            key = (int(s_ek[j]), int(s_sk[j]))
            windows = esa_idx.get(key)
            if windows is None:
                unmatched += 1
                if len(unmatched_samples) < 10:
                    unmatched_samples.append({
                        "EmployeeKey": int(s_ek[j]),
                        "StoreKey": int(s_sk[j]),
                        "OrderDate": str(s_od[j])[:10],
                    })
                continue
            od = s_od[j]
            matched = False
            for ws, we in windows:
                if ws <= od <= we:
                    matched = True
                    break
            if not matched:
                unmatched += 1
                if len(unmatched_samples) < 10:
                    unmatched_samples.append({
                        "EmployeeKey": int(s_ek[j]),
                        "StoreKey": int(s_sk[j]),
                        "OrderDate": str(s_od[j])[:10],
                    })

    total = len(s_ek)
    pct = 100.0 * (total - unmatched) / total if total > 0 else 100.0
    detail = ""
    if unmatched_samples:
        detail = "Samples:\n" + "\n".join(
            f"  EK={s['EmployeeKey']}, SK={s['StoreKey']}, Date={s['OrderDate']}"
            for s in unmatched_samples
        )

    _add(cat, "Sales-to-ESA 100% match",
         unmatched == 0,
         f"{pct:.2f}% ({total - unmatched:,}/{total:,}). {unmatched:,} unmatched.",
         detail)


def _check_no_sales_during_renovation(
    sales: pd.DataFrame,
    stores: pd.DataFrame,
    cat: str,
) -> None:
    reno = stores[
        stores["RenovationStartDate"].notna() & stores["RenovationEndDate"].notna()
    ]
    if reno.empty:
        _add(cat, "No sales during renovation", True, "No renovating stores")
        return

    total_violations = 0
    detail_lines: list[str] = []

    for _, row in reno.iterrows():
        sk = int(row["StoreKey"])
        rs = _to_dt(pd.Series([row["RenovationStartDate"]])).iloc[0]
        re = _to_dt(pd.Series([row["RenovationEndDate"]])).iloc[0]

        store_sales = sales[sales["StoreKey"].astype(int) == sk]
        od = _to_dt(store_sales["OrderDate"])
        during = ((od >= rs) & (od < re)).sum()

        if during > 0:
            total_violations += during
            detail_lines.append(f"  Store {sk}: {during} sale(s) in [{rs.date()}, {re.date()})")

    _add(cat, "No sales during renovation",
         total_violations == 0,
         f"{total_violations} violation(s) across {len(reno)} renovating store(s)",
         "\n".join(detail_lines[:10]))


def _check_no_sales_after_closure(
    sales: pd.DataFrame,
    stores: pd.DataFrame,
    cat: str,
) -> None:
    closed = stores[stores["ClosingDate"].notna()]
    if closed.empty:
        _add(cat, "No sales after store closure", True, "No closed stores")
        return

    total_violations = 0
    detail_lines: list[str] = []

    for _, row in closed.iterrows():
        sk = int(row["StoreKey"])
        cd = _to_dt(pd.Series([row["ClosingDate"]])).iloc[0]

        store_sales = sales[sales["StoreKey"].astype(int) == sk]
        od = _to_dt(store_sales["OrderDate"])
        after = (od >= cd).sum()

        if after > 0:
            total_violations += after
            detail_lines.append(f"  Store {sk}: {after} sale(s) on/after {cd.date()}")

    _add(cat, "No sales after store closure",
         total_violations == 0,
         f"{total_violations} violation(s)",
         "\n".join(detail_lines[:10]))


def _check_post_transfer_leakage(
    sales: pd.DataFrame,
    esa: pd.DataFrame,
    cat: str,
) -> None:
    """Employees must not appear in sales at a store after their LAST ESA
    window there has ended.  (The sales-to-ESA match covers this, but this
    check gives a more specific diagnostic.)"""
    transferred = esa[esa["Status"].astype(str) == "Transferred"]
    if transferred.empty:
        _add(cat, "No post-transfer sales leakage", True, "No transferred assignments")
        return

    # Build last EndDate per (EmployeeKey, StoreKey) from ALL ESA rows
    esa_copy = esa.copy()
    esa_copy["_end"] = _to_dt(esa_copy["EndDate"])
    last_end = (
        esa_copy.groupby(["EmployeeKey", "StoreKey"])["_end"]
        .max()
        .reset_index()
        .rename(columns={"_end": "LastEndDate"})
    )

    merged = sales.merge(last_end, on=["EmployeeKey", "StoreKey"], how="inner")
    if merged.empty:
        _add(cat, "No post-transfer sales leakage", True, "No matching sales to check")
        return

    od = _to_dt(merged["OrderDate"])
    leaked = (od > merged["LastEndDate"]).sum()

    _add(cat, "No post-transfer sales leakage",
         leaked == 0,
         f"{leaked} sale(s) after last ESA EndDate for that employee+store")


# ============================================================================
# Helpers
# ============================================================================

def _sample(df: pd.DataFrame, cols: list[str], n: int = 5) -> str:
    if df.empty:
        return ""
    show = df.head(n)
    available = [c for c in cols if c in show.columns]
    return show[available].to_string(index=False)


# ============================================================================
# Main
# ============================================================================

def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/verify_employee_store_sales.py <dataset_folder>")
        return 2

    folder = Path(sys.argv[1])
    if not folder.is_absolute():
        folder = Path.cwd() / folder
    if not folder.exists():
        print(f"Folder not found: {folder}")
        return 2

    dims = folder / "dimensions"
    facts = folder / "facts"

    t0 = time.perf_counter()

    # --- Load ---
    print("\nLoading parquet files...")
    stores = _load(folder, "dimensions", "stores")
    employees = _load(folder, "dimensions", "employees")
    esa = _load(folder, "dimensions", "employee_store_assignments")
    sales = _load(folder, "facts", "sales")

    if stores is None or employees is None or esa is None or sales is None:
        print("\nCannot proceed: required files missing.")
        return 2

    print(f"  Loaded in {time.perf_counter() - t0:.1f}s "
          f"(stores={len(stores)}, employees={len(employees)}, "
          f"ESA={len(esa)}, sales={len(sales):,})\n")

    # --- Run checks ---
    print("=" * 60)
    print("STORES")
    print("=" * 60)
    check_stores(stores)

    print()
    print("=" * 60)
    print("EMPLOYEES")
    print("=" * 60)
    check_employees(employees)

    print()
    print("=" * 60)
    print("ESA BRIDGE TABLE")
    print("=" * 60)
    check_esa(esa, stores, employees)

    print()
    print("=" * 60)
    print("SALES ALIGNMENT")
    print("=" * 60)
    check_sales(sales, esa, stores)

    # --- Summary ---
    elapsed = time.perf_counter() - t0
    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)
    total = len(results)

    print()
    print("=" * 60)
    tag = "\033[32mALL PASSED\033[0m" if failed == 0 else f"\033[31m{failed} FAILED\033[0m"
    print(f"SUMMARY: {passed}/{total} passed, {tag}  ({elapsed:.1f}s)")
    print("=" * 60)

    if failed > 0:
        print("\nFailed checks:")
        for r in results:
            if not r.passed:
                print(f"  [{r.category}] {r.name}: {r.message}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
