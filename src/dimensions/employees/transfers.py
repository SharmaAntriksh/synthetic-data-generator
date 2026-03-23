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
    """Map StoreRegion → list of physical StoreKeys."""
    physical = stores[stores["StoreKey"] <= ONLINE_STORE_KEY_BASE]
    if "StoreRegion" not in physical.columns:
        return {}
    result: Dict[str, List[int]] = {}
    for region, grp in physical.groupby("StoreRegion"):
        result[str(region)] = grp["StoreKey"].tolist()
    return result


def _count_salespeople_at_store(
    store_key: int,
    on_date: pd.Timestamp,
    active_assignments: pd.DataFrame,
    salesperson_roles: List[str],
) -> int:
    """Count salespeople actively assigned to *store_key* on *on_date*.

    Uses date range as source of truth — a "Transferred" row was active
    during its [StartDate, EndDate] window.
    """
    mask = (
        (active_assignments["StoreKey"] == store_key)
        & (active_assignments["StartDate"] <= on_date)
        & (active_assignments["EndDate"] >= on_date)
        & (active_assignments["RoleAtStore"].isin(salesperson_roles))
    )
    return int(mask.sum())


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
        # Sort by date so earlier transfers are processed first
        chronological = sorted(zip(transfer_dates, chosen_idx))

        year_new_rows: list[dict] = []

        for transfer_date, idx in chronological:
            row = df.loc[idx]
            emp_key = int(row["EmployeeKey"])
            source_sk = int(row["StoreKey"])
            source_region = store_region.get(source_sk)

            # Guard: source store must retain >= 1 salesperson
            source_count = _count_salespeople_at_store(
                source_sk, transfer_date, df, salesperson_roles,
            )
            if source_count <= 1:
                continue

            # Pick destination store
            dest_sk = _pick_destination(
                rng=rng,
                source_sk=source_sk,
                source_region=source_region,
                region_stores=region_stores,
                all_physical_keys=all_physical_keys,
                same_region_pref=same_region_pref,
                transfer_date=transfer_date,
                open_dates=open_dates,
                close_dates=close_dates,
            )
            if dest_sk is None:
                continue

            # --- Execute transfer ---
            # End current assignment
            df.at[idx, "EndDate"] = transfer_date - pd.Timedelta(days=1)
            df.at[idx, "IsPrimary"] = np.int8(0)
            df.at[idx, "Status"] = "Transferred"

            # Preserve original end date (e.g. termination) for the new assignment
            original_end = row["EndDate"]

            reason = rng.choice(
                EMPLOYEE_TRANSFER_REASON_LABELS,
                p=EMPLOYEE_TRANSFER_REASON_PROBS,
            )

            year_new_rows.append({
                "EmployeeKey": np.int32(emp_key),
                "StoreKey": np.int32(dest_sk),
                "StartDate": transfer_date,
                "EndDate": original_end,
                "FTE": row["FTE"],
                "RoleAtStore": row["RoleAtStore"],
                "IsPrimary": np.int8(1),
                "TransferReason": str(reason),
                "Status": "Active",
            })
            transfer_count += 1

        # Append this year's new rows into df so they're visible in subsequent years
        if year_new_rows:
            df = pd.concat([df, pd.DataFrame(year_new_rows)], ignore_index=True)

    if transfer_count == 0:
        info("Transfers: 0 transfers generated (no eligible employees or destinations)")
        return assignments

    # Sort and reassign keys/sequences
    df = df.sort_values(["EmployeeKey", "StartDate"]).reset_index(drop=True)
    df["AssignmentKey"] = np.arange(1, len(df) + 1, dtype=np.int32)
    df["AssignmentSequence"] = df.groupby("EmployeeKey").cumcount().astype(np.int16) + 1

    # Normalize dates
    df["StartDate"] = pd.to_datetime(df["StartDate"]).dt.normalize()
    df["EndDate"] = pd.to_datetime(df["EndDate"]).dt.normalize()

    info(
        f"Transfers: {transfer_count} transfers across "
        f"{df['EmployeeKey'].nunique()} employees, "
        f"{df['StoreKey'].nunique()} stores"
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
    region_stores: Dict[str, List[int]],
    all_physical_keys: List[int],
    same_region_pref: float,
    transfer_date: pd.Timestamp,
    open_dates: Dict[int, pd.Timestamp],
    close_dates: Dict[int, pd.Timestamp],
) -> Optional[int]:
    """Pick a destination store for a transfer."""
    # Decide: same region or cross-region?
    use_same_region = rng.random() < same_region_pref

    if use_same_region and source_region and source_region in region_stores:
        candidates = [
            sk for sk in region_stores[source_region]
            if sk != source_sk
            and _store_is_open(sk, transfer_date, open_dates, close_dates)
        ]
    else:
        candidates = []

    # Fallback to all physical stores if no same-region candidates
    if not candidates:
        candidates = [
            sk for sk in all_physical_keys
            if sk != source_sk
            and _store_is_open(sk, transfer_date, open_dates, close_dates)
        ]

    if not candidates:
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
    sp = assignments[assignments["RoleAtStore"].isin(salesperson_roles)].copy()
    if sp.empty:
        return []

    physical_stores = stores[stores["StoreKey"] <= ONLINE_STORE_KEY_BASE]
    violations: List[Tuple[int, str]] = []

    # Check monthly: first day of each month in the range
    month_starts = pd.date_range(global_start, global_end, freq="MS")

    for check_date in month_starts:
        for _, store in physical_stores.iterrows():
            sk = int(store["StoreKey"])
            if not _store_is_open(sk, check_date, open_dates, close_dates):
                continue

            count = (
                (sp["StoreKey"] == sk)
                & (sp["StartDate"] <= check_date)
                & (sp["EndDate"] >= check_date)
            ).sum()

            if count == 0:
                violations.append((sk, check_date.strftime("%Y-%m")))
                if len(violations) >= 10:
                    return violations  # early exit, enough to report

    return violations
