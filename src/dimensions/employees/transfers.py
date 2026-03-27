"""Employee transfer engine.

Post-processes the EmployeeStoreAssignments bridge table to add
inter-store transfers for staff employees.  Config-gated: only runs
when ``employees.transfers.enabled = True``.

Design invariants:
  - No store drops below 1 salesperson on any date after a transfer.
  - No date gaps: old assignment ends the day before new one starts.
  - Managers (30M), online reps (50M), and corporate hierarchy never transfer.
  - Only staff in ``salesperson_roles`` are eligible.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.defaults import (
    EMPLOYEE_TRANSFER_REASON_LABELS,
    EMPLOYEE_TRANSFER_REASON_PROBS,
    ONLINE_STORE_KEY_BASE,
    ONLINE_EMP_KEY_BASE,
)
from src.dimensions.employees.generator import STAFF_KEY_BASE
from src.utils.logging_utils import info, warn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _store_is_open(
    store_key: int,
    on_date: pd.Timestamp,
    open_dates: Dict[int, pd.Timestamp],
    close_dates: Dict[int, pd.Timestamp],
) -> bool:
    """True if *store_key* is open on *on_date*."""
    opened = open_dates.get(store_key)
    if opened is not None and on_date < opened:
        return False
    closed = close_dates.get(store_key)
    if closed is not None and on_date >= closed:
        return False
    return True


def _build_region_store_map(
    stores: pd.DataFrame,
) -> Dict[str, List[int]]:
    """Map StoreRegion -> list of physical StoreKeys."""
    physical = stores[stores["StoreKey"] <= ONLINE_STORE_KEY_BASE]
    if "StoreRegion" not in physical.columns:
        return {}
    result: Dict[str, List[int]] = {}
    for region, grp in physical.groupby("StoreRegion"):
        result[str(region)] = grp["StoreKey"].tolist()
    return result


def _build_store_sp_index(
    sp_mask: np.ndarray,
    skeys: np.ndarray,
) -> Dict[int, List[int]]:
    """Build store_key -> [row indices] for salesperson assignments."""
    store_sp_idx: Dict[int, List[int]] = defaultdict(list)
    idxs = np.where(sp_mask)[0]
    for i in idxs:
        store_sp_idx[int(skeys[i])].append(i)
    return store_sp_idx


def _build_open_store_candidates(
    region_stores: Dict[str, List[int]],
    all_physical_keys: List[int],
    year_start: pd.Timestamp,
    year_end: pd.Timestamp,
    open_dates: Dict[int, pd.Timestamp],
    close_dates: Dict[int, pd.Timestamp],
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Pre-compute open store arrays for the year.

    Returns (region_open, all_open).  A store is included only if open on
    both year_start and year_end — stores that close mid-year are excluded
    entirely (they will never appear as transfer destinations).
    """
    all_open = np.array([
        sk for sk in all_physical_keys
        if _store_is_open(sk, year_start, open_dates, close_dates)
        and _store_is_open(sk, year_end, open_dates, close_dates)
    ], dtype=np.int32)
    all_open_set = set(all_open)

    region_open: Dict[str, np.ndarray] = {}
    for region, sks in region_stores.items():
        arr = np.array([sk for sk in sks if sk in all_open_set], dtype=np.int32)
        if len(arr) > 0:
            region_open[region] = arr

    return region_open, all_open


def _count_salespeople_fast(
    store_key: int,
    on_date_ns: np.datetime64,
    start_ns: np.ndarray,
    end_ns: np.ndarray,
    store_sp_idx: Dict[int, List[int]],
) -> int:
    """Count salespeople actively assigned to *store_key* on *on_date_ns*."""
    count = 0
    for i in store_sp_idx.get(store_key, []):
        if start_ns[i] <= on_date_ns <= end_ns[i]:
            count += 1
    return count


# ---------------------------------------------------------------------------
# Transfer engine
# ---------------------------------------------------------------------------

def apply_transfers(
    assignments: pd.DataFrame,
    stores: pd.DataFrame,
    *,
    seed: int,
    global_start: pd.Timestamp,
    global_end: pd.Timestamp,
    annual_rate: float = 0.05,
    min_tenure_months: int = 6,
    same_region_pref: float = 0.7,
    salesperson_roles: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Apply employee transfers to the assignments bridge table.

    For each year in the dataset range, a fraction of eligible staff
    employees are randomly transferred to another store.  The source
    assignment is ended and a new assignment is created at the
    destination store.

    Returns the modified assignments DataFrame with transfer rows added.
    If no transfers are possible, returns the original DataFrame unchanged.
    """
    if assignments.empty or annual_rate <= 0:
        return assignments

    rng = np.random.default_rng(seed)

    if salesperson_roles is None:
        salesperson_roles = ["Sales Associate"]

    sp_roles_set = set(salesperson_roles)

    # --- Store metadata ---------------------------------------------------
    sk_arr = stores["StoreKey"].astype(np.int32).to_numpy()
    open_dates: Dict[int, pd.Timestamp] = {}
    close_dates: Dict[int, pd.Timestamp] = {}
    store_region: Dict[int, str] = {}

    if "OpeningDate" in stores.columns:
        od = pd.to_datetime(stores["OpeningDate"], errors="coerce")
        for sk, dt in zip(sk_arr, od):
            if pd.notna(dt):
                open_dates[int(sk)] = dt

    if "ClosingDate" in stores.columns:
        cd = pd.to_datetime(stores["ClosingDate"], errors="coerce")
        for sk, dt in zip(sk_arr, cd):
            if pd.notna(dt):
                close_dates[int(sk)] = dt

    if "StoreRegion" in stores.columns:
        sr = stores["StoreRegion"].astype(str).to_numpy()
        for sk, region in zip(sk_arr, sr):
            store_region[int(sk)] = region

    region_stores = _build_region_store_map(stores)

    # All physical store keys (for cross-region fallback)
    all_physical_keys = [
        int(sk) for sk in sk_arr if sk <= ONLINE_STORE_KEY_BASE
    ]

    # --- Work on a mutable copy -------------------------------------------
    df = assignments.copy()
    transfer_count = 0
    skipped_guard = 0
    skipped_no_dest = 0

    start_year = global_start.year
    end_year = global_end.year

    for year in range(start_year, end_year + 1):
        year_start = pd.Timestamp(f"{year}-01-01")
        year_end = pd.Timestamp(f"{year}-12-31")

        # Eligible: staff employees with active primary assignments
        # and sufficient tenure at current store
        tenure_cutoff = year_start - pd.DateOffset(months=min_tenure_months)

        eligible_mask = (
            (df["IsPrimary"] == 1)
            & (df["Status"] == "Active")
            & (df["RoleAtStore"].isin(salesperson_roles))
            & (df["EmployeeKey"] >= STAFF_KEY_BASE)
            & (df["EmployeeKey"] < ONLINE_EMP_KEY_BASE)
            & (df["StartDate"] <= tenure_cutoff)
            & (df["EndDate"] >= year_start)
        )
        eligible_idx = df.index[eligible_mask].to_numpy()

        if len(eligible_idx) == 0:
            continue

        n_transfers = max(1, int(round(len(eligible_idx) * annual_rate)))
        n_transfers = min(n_transfers, len(eligible_idx))
        chosen_idx = rng.choice(eligible_idx, size=n_transfers, replace=False)

        # Compute transfer dates upfront, then process chronologically
        # so the guard sees earlier transfers' effects
        t_earliest = max(year_start, global_start + pd.Timedelta(days=1))
        t_latest = min(
            year_end - pd.Timedelta(days=30),
            global_end - pd.Timedelta(days=30),
        )
        if t_earliest >= t_latest:
            continue
        t_range_days = (t_latest - t_earliest).days

        transfer_dates = [
            t_earliest + pd.Timedelta(days=int(rng.integers(0, t_range_days + 1)))
            for _ in chosen_idx
        ]
        chronological = sorted(zip(transfer_dates, chosen_idx))

        # Writable copies for columns mutated in the transfer loop
        col_emp_key = df["EmployeeKey"].values
        col_store_key = df["StoreKey"].values
        col_fte = df["FTE"].values
        col_role = df["RoleAtStore"].values
        col_is_primary = np.array(df["IsPrimary"].values, copy=True)
        col_status = np.array(df["Status"].values, copy=True)

        start_ns = df["StartDate"].values.astype("datetime64[ns]")
        end_ns = df["EndDate"].values.astype("datetime64[ns]").copy()

        sp_mask = np.isin(col_role, list(sp_roles_set))
        store_sp_idx = _build_store_sp_index(sp_mask, col_store_key)

        region_open, all_open = _build_open_store_candidates(
            region_stores, all_physical_keys, year_start, year_end,
            open_dates, close_dates,
        )

        year_new_rows: list[dict] = []

        for transfer_date, idx in chronological:
            emp_key = int(col_emp_key[idx])
            source_sk = int(col_store_key[idx])
            source_region = store_region.get(source_sk)

            # Guard: source store must retain >= 1 salesperson
            td_ns = np.datetime64(transfer_date, "ns")
            source_count = _count_salespeople_fast(
                source_sk, td_ns, start_ns, end_ns, store_sp_idx,
            )
            if source_count <= 1:
                skipped_guard += 1
                continue

            # Pick destination store
            dest_sk = _pick_destination(
                rng=rng,
                source_sk=source_sk,
                source_region=source_region,
                region_open=region_open,
                all_open=all_open,
                same_region_pref=same_region_pref,
            )
            if dest_sk is None:
                skipped_no_dest += 1
                continue

            # --- Execute transfer ---
            original_end_ns = end_ns[idx]
            new_end_ns = np.datetime64(transfer_date - pd.Timedelta(days=1), "ns")
            end_ns[idx] = new_end_ns
            col_is_primary[idx] = np.int32(0)
            col_status[idx] = "Transferred"

            reason = rng.choice(
                EMPLOYEE_TRANSFER_REASON_LABELS,
                p=EMPLOYEE_TRANSFER_REASON_PROBS,
            )

            # Clamp EndDate to day before destination store closes (last open day),
            # matching the convention used by generate_employee_store_assignments
            new_end = pd.Timestamp(original_end_ns)
            new_status = "Active"
            dest_close = close_dates.get(dest_sk)
            if dest_close is not None:
                last_open_day = dest_close - pd.Timedelta(days=1)
                if last_open_day < new_end:
                    new_end = last_open_day
                    new_status = "Completed"

            year_new_rows.append({
                "EmployeeKey": np.int32(emp_key),
                "StoreKey": np.int32(dest_sk),
                "StartDate": transfer_date,
                "EndDate": new_end,
                "FTE": col_fte[idx],
                "RoleAtStore": col_role[idx],
                "IsPrimary": np.int32(1),
                "TransferReason": str(reason),
                "Status": new_status,
            })
            transfer_count += 1

        df["EndDate"] = end_ns
        df["IsPrimary"] = col_is_primary
        df["Status"] = col_status

        # Concat inside the year loop so subsequent years see earlier transfers
        if year_new_rows:
            df = pd.concat([df, pd.DataFrame(year_new_rows)], ignore_index=True)

    if transfer_count == 0:
        info(
            f"Transfers: 0 transfers generated "
            f"(skipped: {skipped_guard} guard, {skipped_no_dest} no destination)"
        )
        return assignments

    # Sort and reassign keys/sequences
    df = df.sort_values(["EmployeeKey", "StartDate"]).reset_index(drop=True)
    df["AssignmentKey"] = np.arange(1, len(df) + 1, dtype=np.int32)
    df["AssignmentSequence"] = df.groupby("EmployeeKey").cumcount().astype(np.int32) + 1

    # Normalize dates
    df["StartDate"] = pd.to_datetime(df["StartDate"]).dt.normalize()
    df["EndDate"] = pd.to_datetime(df["EndDate"]).dt.normalize()

    info(
        f"Transfers: {transfer_count} transfers across "
        f"{df['EmployeeKey'].nunique()} employees, "
        f"{df['StoreKey'].nunique()} stores "
        f"(skipped: {skipped_guard} guard, {skipped_no_dest} no destination)"
    )

    # --- Invariant check: every open store has >= 1 salesperson -----------
    violations = _check_coverage_invariant(
        df, stores, salesperson_roles, global_start, global_end,
        open_dates, close_dates,
    )
    if violations:
        warn(
            f"Transfer invariant violated: {len(violations)} store-month(s) "
            f"with 0 salespeople. Falling back to pre-transfer assignments."
        )
        return assignments

    return df


def _pick_destination(
    *,
    rng: np.random.Generator,
    source_sk: int,
    source_region: Optional[str],
    region_open: Dict[str, np.ndarray],
    all_open: np.ndarray,
    same_region_pref: float,
) -> Optional[int]:
    """Pick a destination store from pre-built open-store arrays."""
    use_same_region = rng.random() < same_region_pref

    candidates = None
    if use_same_region and source_region and source_region in region_open:
        arr = region_open[source_region]
        filtered = arr[arr != source_sk]
        if len(filtered) > 0:
            candidates = filtered

    if candidates is None:
        filtered = all_open[all_open != source_sk]
        if len(filtered) > 0:
            candidates = filtered

    if candidates is None:
        return None

    return int(rng.choice(candidates))


def _check_coverage_invariant(
    assignments: pd.DataFrame,
    stores: pd.DataFrame,
    salesperson_roles: List[str],
    global_start: pd.Timestamp,
    global_end: pd.Timestamp,
    open_dates: Dict[int, pd.Timestamp],
    close_dates: Dict[int, pd.Timestamp],
) -> List[Tuple[int, str]]:
    """Check that every open physical store has >= 1 salesperson each month.

    Returns a list of (StoreKey, 'YYYY-MM') violations, empty if OK.
    """
    sp = assignments[assignments["RoleAtStore"].isin(salesperson_roles)]
    if sp.empty:
        return []

    physical_stores = stores[stores["StoreKey"] <= ONLINE_STORE_KEY_BASE]
    physical_sks = physical_stores["StoreKey"].astype(int).to_numpy()

    sp_start = sp["StartDate"].values.astype("datetime64[ns]")
    sp_end = sp["EndDate"].values.astype("datetime64[ns]")
    sp_sk = sp["StoreKey"].astype(int).values

    violations: List[Tuple[int, str]] = []
    month_starts = pd.date_range(global_start, global_end, freq="MS")

    for check_date in month_starts:
        cd_ns = np.datetime64(check_date, "ns")
        active_mask = (sp_start <= cd_ns) & (sp_end >= cd_ns)
        covered_stores = set(sp_sk[active_mask])

        for sk in physical_sks:
            if not _store_is_open(sk, check_date, open_dates, close_dates):
                continue
            if sk not in covered_stores:
                violations.append((sk, check_date.strftime("%Y-%m")))
                if len(violations) >= 10:
                    return violations

    return violations
