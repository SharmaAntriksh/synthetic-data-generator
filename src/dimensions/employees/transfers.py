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
from dataclasses import dataclass
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
    reno_start_dates: Optional[Dict[int, pd.Timestamp]] = None,
    reno_end_dates: Optional[Dict[int, pd.Timestamp]] = None,
) -> bool:
    """True if *store_key* is open on *on_date*.

    A store is closed if it hasn't opened yet, has permanently closed,
    or is within its renovation window [reno_start, reno_end).
    """
    opened = open_dates.get(store_key)
    if opened is not None and on_date < opened:
        return False
    closed = close_dates.get(store_key)
    if closed is not None and on_date >= closed:
        return False
    # Renovation window: store is closed during [start, end)
    if reno_start_dates and reno_end_dates:
        rs = reno_start_dates.get(store_key)
        re = reno_end_dates.get(store_key)
        if rs is not None and re is not None and rs <= on_date < re:
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


def _build_open_store_candidates(
    region_stores: Dict[str, List[int]],
    all_physical_keys: List[int],
    year_start: pd.Timestamp,
    year_end: pd.Timestamp,
    open_dates: Dict[int, pd.Timestamp],
    close_dates: Dict[int, pd.Timestamp],
    reno_start_dates: Optional[Dict[int, pd.Timestamp]] = None,
    reno_end_dates: Optional[Dict[int, pd.Timestamp]] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Pre-compute open store arrays for the year.

    Returns (region_open, all_open).  A store is included only if open on
    both year_start and year_end — stores that close or are renovating
    mid-year are excluded entirely (they will never appear as transfer
    destinations).
    """
    all_open = np.array([
        sk for sk in all_physical_keys
        if _store_is_open(sk, year_start, open_dates, close_dates, reno_start_dates, reno_end_dates)
        and _store_is_open(sk, year_end, open_dates, close_dates, reno_start_dates, reno_end_dates)
    ], dtype=np.int32)
    all_open_set = set(all_open)

    region_open: Dict[str, np.ndarray] = {}
    for region, sks in region_stores.items():
        arr = np.array([sk for sk in sks if sk in all_open_set], dtype=np.int32)
        if len(arr) > 0:
            region_open[region] = arr

    return region_open, all_open


# ---------------------------------------------------------------------------
# Coverage budget — replaces the per-candidate guard with constraint-aware
# selection. The budget tracks per-(physical_store, month) salesperson coverage
# and per-cell constrained mask. Each candidate transfer is rejected if it
# would drop a constrained source-month below ``min_coverage``.
# ---------------------------------------------------------------------------

@dataclass
class CoverageBudget:
    """Per-(store, month) salesperson coverage matrix used to gate transfers.

    A salesperson assignment counts toward ``cov[s, m]`` only when it covers
    the *full* month — ``StartDate <= month_start AND EndDate >= month_end``,
    with the last month_end clamped to ``global_end``. This single condition
    enforces both month-start and month-end staffing in one matrix.

    A cell is *constrained* when the store is open through that month and is
    not within a renovation window. Renovation, pre-open, and post-close
    cells are unconstrained — coverage may be 0 there without violation.
    """
    cov: np.ndarray
    constrained: np.ndarray
    store_idx_to_key: np.ndarray
    key_to_store_idx: Dict[int, int]
    month_starts: np.ndarray
    month_ends: np.ndarray
    min_coverage: int = 1


def _build_coverage_budget(
    assignments: pd.DataFrame,
    stores: pd.DataFrame,
    salesperson_roles: List[str],
    global_start: pd.Timestamp,
    global_end: pd.Timestamp,
    open_dates: Dict[int, pd.Timestamp],
    close_dates: Dict[int, pd.Timestamp],
    reno_start_dates: Optional[Dict[int, pd.Timestamp]] = None,
    reno_end_dates: Optional[Dict[int, pd.Timestamp]] = None,
    min_coverage: int = 1,
) -> CoverageBudget:
    """Build the initial coverage matrix from the pre-transfer assignments."""
    month_starts_ts = pd.date_range(global_start, global_end, freq="MS")
    if len(month_starts_ts) == 0:
        month_starts_ts = pd.DatetimeIndex([pd.Timestamp(global_start).normalize()])

    # last day of each month, clamping the final month to global_end
    month_ends_ts_raw = month_starts_ts + pd.offsets.MonthEnd(0)
    global_end_ts = pd.Timestamp(global_end).normalize()
    if month_ends_ts_raw[-1] > global_end_ts:
        month_ends_ts = pd.DatetimeIndex(
            list(month_ends_ts_raw[:-1]) + [global_end_ts]
        )
    else:
        month_ends_ts = month_ends_ts_raw

    month_starts = month_starts_ts.values.astype("datetime64[ns]")
    month_ends = month_ends_ts.values.astype("datetime64[ns]")
    n_months = len(month_starts)

    # Physical store axis (online stores excluded from coverage matrix)
    physical_mask = stores["StoreKey"].to_numpy() <= ONLINE_STORE_KEY_BASE
    physical_keys = stores.loc[physical_mask, "StoreKey"].astype(np.int32).to_numpy()
    n_stores = len(physical_keys)
    key_to_store_idx: Dict[int, int] = {int(k): i for i, k in enumerate(physical_keys)}

    cov = np.zeros((n_stores, n_months), dtype=np.int32)

    # Vectorised cov build: for each salesperson row, the run of months it
    # fully covers is contiguous — its first month_idx is the smallest m where
    # m_start >= start, last is the largest m where m_end <= end. Use
    # searchsorted on both axes and accumulate via np.add.at.
    sp_rows = assignments[assignments["RoleAtStore"].isin(salesperson_roles)]
    if not sp_rows.empty and n_stores > 0 and n_months > 0:
        sp_start = sp_rows["StartDate"].values.astype("datetime64[ns]")
        sp_end = sp_rows["EndDate"].values.astype("datetime64[ns]")
        sp_sk = sp_rows["StoreKey"].astype(int).to_numpy()
        sidx_arr = np.fromiter(
            (key_to_store_idx.get(int(k), -1) for k in sp_sk),
            dtype=np.int64, count=len(sp_sk),
        )
        first_m = np.searchsorted(month_starts, sp_start, side="left")
        last_m = np.searchsorted(month_ends, sp_end, side="right") - 1
        valid = (sidx_arr >= 0) & (first_m <= last_m) & (last_m < n_months) & (first_m >= 0)
        for i in np.where(valid)[0]:
            cov[sidx_arr[i], first_m[i]:last_m[i] + 1] += 1

    # Vectorised constrained mask: build per-store dates as 1-D arrays then
    # broadcast against the month_starts / month_ends 1-D arrays. Sentinels
    # must stay inside the ns range (~1677-2262) to avoid overflow wraparound.
    far_past = np.datetime64("1700-01-01", "ns")
    far_future = np.datetime64("2200-01-01", "ns")
    open_arr = np.array(
        [np.datetime64(open_dates[int(k)], "ns") if int(k) in open_dates else far_past
         for k in physical_keys],
        dtype="datetime64[ns]",
    )
    close_arr = np.array(
        [np.datetime64(close_dates[int(k)], "ns") if int(k) in close_dates else far_future
         for k in physical_keys],
        dtype="datetime64[ns]",
    )
    reno_s_arr = np.full(n_stores, far_future, dtype="datetime64[ns]")
    reno_e_arr = np.full(n_stores, far_past, dtype="datetime64[ns]")
    if reno_start_dates and reno_end_dates:
        for i, k in enumerate(physical_keys):
            ki = int(k)
            if ki in reno_start_dates and ki in reno_end_dates:
                reno_s_arr[i] = np.datetime64(reno_start_dates[ki], "ns")
                reno_e_arr[i] = np.datetime64(reno_end_dates[ki], "ns")

    opened = month_starts[None, :] >= open_arr[:, None]
    not_closed = month_ends[None, :] < close_arr[:, None]
    renovating = (month_starts[None, :] < reno_e_arr[:, None]) & (month_ends[None, :] >= reno_s_arr[:, None])
    constrained = opened & not_closed & ~renovating

    return CoverageBudget(
        cov=cov,
        constrained=constrained,
        store_idx_to_key=physical_keys,
        key_to_store_idx=key_to_store_idx,
        month_starts=month_starts,
        month_ends=month_ends,
        min_coverage=min_coverage,
    )


def _affected_source_months(
    budget: CoverageBudget,
    employee_start: np.datetime64,
    transfer_date_ns: np.datetime64,
    original_end_ns: np.datetime64,
) -> np.ndarray:
    """Boolean mask of months at the source store that the OLD assignment fully
    covered but the truncated (post-transfer) assignment no longer covers.

    Old full coverage: ``start <= m_start AND old_end >= m_end``.
    New coverage stops at ``transfer_date - 1``, so the assignment still fully
    covers a month iff ``m_end < transfer_date``. Loss = old AND NOT new =
    ``start <= m_start AND m_end <= old_end AND m_end >= transfer_date``.
    """
    return (
        (budget.month_starts >= employee_start)
        & (budget.month_ends <= original_end_ns)
        & (budget.month_ends >= transfer_date_ns)
    )


def _affected_dest_months(
    budget: CoverageBudget,
    transfer_date_ns: np.datetime64,
    dst_end_ns: np.datetime64,
) -> np.ndarray:
    """Boolean mask of months at the destination store that the new assignment
    fully covers (``m_start >= transfer_date AND m_end <= dst_end``)."""
    return (
        (budget.month_starts >= transfer_date_ns)
        & (budget.month_ends <= dst_end_ns)
    )


def _is_transfer_feasible(
    budget: CoverageBudget,
    src_key: int,
    employee_start: pd.Timestamp,
    transfer_date: pd.Timestamp,
    original_end_date: pd.Timestamp,
) -> Tuple[bool, List[Tuple[int, int]]]:
    """True iff removing one salesperson from ``src_key`` for the months whose
    coverage is lost would keep every constrained source-month at >= min_coverage.

    Returns ``(feasible, violators)`` where violators is a list of
    ``(store_idx, month_idx)`` cells that would drop below threshold.
    """
    src_idx = budget.key_to_store_idx.get(int(src_key))
    if src_idx is None:
        return True, []

    start_ns = np.datetime64(employee_start, "ns")
    td_ns = np.datetime64(transfer_date, "ns")
    end_ns = np.datetime64(original_end_date, "ns")

    loss_mask = _affected_source_months(budget, start_ns, td_ns, end_ns)
    if not loss_mask.any():
        return True, []

    cov_slice = budget.cov[src_idx, loss_mask]
    constrained_slice = budget.constrained[src_idx, loss_mask]
    would_drop_below = (cov_slice - 1 < budget.min_coverage) & constrained_slice
    if not would_drop_below.any():
        return True, []

    affected_indices = np.where(loss_mask)[0]
    violators = [
        (src_idx, int(affected_indices[i]))
        for i in range(len(affected_indices))
        if would_drop_below[i]
    ]
    return False, violators


def _clamp_dst_end(
    dest_sk: int,
    transfer_date: pd.Timestamp,
    original_end: pd.Timestamp,
    close_dates: Dict[int, pd.Timestamp],
    reno_start_dates: Optional[Dict[int, pd.Timestamp]],
    reno_end_dates: Optional[Dict[int, pd.Timestamp]],
) -> Tuple[pd.Timestamp, str, bool]:
    """Clamp the new destination assignment's EndDate to the destination
    store's last open day before any close/renovation. Matches the convention
    used by ``generate_employee_store_assignments``.

    Returns ``(new_end, status, skip)`` — when ``skip`` is True the caller must
    abandon this candidate (transfer date falls inside the renovation window).
    """
    new_end = original_end
    new_status = "Active"

    dest_close = close_dates.get(dest_sk)
    if dest_close is not None:
        last_open_day = dest_close - pd.Timedelta(days=1)
        if last_open_day < new_end:
            new_end = last_open_day
            new_status = "Completed"

    if reno_start_dates is None or reno_end_dates is None:
        return new_end, new_status, False
    dest_reno_start = reno_start_dates.get(dest_sk)
    dest_reno_end = reno_end_dates.get(dest_sk)
    if dest_reno_start is None or dest_reno_end is None:
        return new_end, new_status, False
    if transfer_date >= dest_reno_end:
        return new_end, new_status, False

    last_pre_reno = dest_reno_start - pd.Timedelta(days=1)
    if last_pre_reno < transfer_date:
        return new_end, new_status, True
    if last_pre_reno < new_end:
        new_end = last_pre_reno
        new_status = "Completed"
    return new_end, new_status, False


def _adjust_budget(
    budget: CoverageBudget,
    src_key: int,
    dst_key: int,
    employee_start: pd.Timestamp,
    transfer_date: pd.Timestamp,
    original_end_date: pd.Timestamp,
    dst_end_date: pd.Timestamp,
    direction: int,
) -> None:
    """Apply (``direction=1``) or revert (``direction=-1``) one transfer's
    effect on the coverage matrix."""
    start_ns = np.datetime64(employee_start, "ns")
    td_ns = np.datetime64(transfer_date, "ns")
    src_end_ns = np.datetime64(original_end_date, "ns")
    dst_end_ns = np.datetime64(dst_end_date, "ns")

    src_idx = budget.key_to_store_idx.get(int(src_key))
    if src_idx is not None:
        loss_mask = _affected_source_months(budget, start_ns, td_ns, src_end_ns)
        if loss_mask.any():
            budget.cov[src_idx, loss_mask] -= direction

    dst_idx = budget.key_to_store_idx.get(int(dst_key))
    if dst_idx is not None:
        gain_mask = _affected_dest_months(budget, td_ns, dst_end_ns)
        if gain_mask.any():
            budget.cov[dst_idx, gain_mask] += direction


def _budget_violations(budget: CoverageBudget) -> List[Tuple[int, int]]:
    """Return list of constrained (store_idx, month_idx) cells with cov < min."""
    bad = budget.constrained & (budget.cov < budget.min_coverage)
    if not bad.any():
        return []
    s_idx, m_idx = np.where(bad)
    return list(zip(s_idx.tolist(), m_idx.tolist()))


@dataclass
class _TransferRecord:
    """Per-transfer book-keeping used for surgical rollback."""
    employee_key: int
    src_key: int
    dst_key: int
    employee_start: pd.Timestamp
    transfer_date: pd.Timestamp
    original_end: pd.Timestamp
    dst_end: pd.Timestamp
    rolled_back: bool = False


def _select_rollback_indices(
    transfer_records: List[_TransferRecord],
    budget: CoverageBudget,
    violations: List[Tuple[int, int]],
) -> List[int]:
    """Pick the smallest set of transfers whose rollback covers all violations.

    Greedy: walk transfers most-recent-first and pick any whose source-loss
    months intersect a remaining violation. Transfers whose source store has
    no violations are skipped without computing the loss mask.
    """
    if not violations:
        return []
    violation_set = set(violations)
    violated_src_idx = {s for s, _ in violation_set}
    rollback: set[int] = set()
    for ti in range(len(transfer_records) - 1, -1, -1):
        rec = transfer_records[ti]
        if rec.rolled_back:
            continue
        src_idx = budget.key_to_store_idx.get(int(rec.src_key))
        if src_idx is None or src_idx not in violated_src_idx:
            continue
        loss_mask = _affected_source_months(
            budget,
            np.datetime64(rec.employee_start, "ns"),
            np.datetime64(rec.transfer_date, "ns"),
            np.datetime64(rec.original_end, "ns"),
        )
        loss_months = np.where(loss_mask)[0]
        hit = any((src_idx, int(m)) in violation_set for m in loss_months)
        if not hit:
            continue
        rollback.add(ti)
        for m in loss_months:
            violation_set.discard((src_idx, int(m)))
        if not violation_set:
            break
    return sorted(rollback)


def _format_rejection_summary(rejections: Dict[str, int], fallback: str) -> str:
    if not rejections:
        return fallback
    return ", ".join(f"{count} {reason}" for reason, count in sorted(rejections.items()))


def _surgical_rollback(
    df: pd.DataFrame,
    budget: CoverageBudget,
    transfer_records: List[_TransferRecord],
    max_attempts: int = 3,
) -> int:
    """Roll back the smallest set of transfers needed to clear violations.

    Mutates ``df`` in place and updates ``budget``/``transfer_records``.
    Returns the number of transfers that were rolled back.
    """
    rolled_back_count = 0
    for _ in range(max_attempts):
        violations = _budget_violations(budget)
        if not violations:
            break
        rb_indices = _select_rollback_indices(transfer_records, budget, violations)
        if not rb_indices:
            break

        # Build a once-per-attempt index for O(1) source-row and dest-row lookup
        # instead of repeated full-DataFrame boolean scans.
        sig_to_idx: Dict[Tuple[int, int, pd.Timestamp], int] = {}
        for idx, row in zip(df.index, df.itertuples(index=False)):
            sig_to_idx[(int(row.EmployeeKey), int(row.StoreKey), row.StartDate)] = idx

        drop_idx: List[int] = []
        for ti in rb_indices:
            rec = transfer_records[ti]
            if rec.rolled_back:
                continue
            _adjust_budget(
                budget, rec.src_key, rec.dst_key,
                rec.employee_start, rec.transfer_date, rec.original_end, rec.dst_end,
                direction=-1,
            )
            src_row = sig_to_idx.get((rec.employee_key, rec.src_key, rec.employee_start))
            if src_row is not None:
                df.at[src_row, "EndDate"] = rec.original_end
                df.at[src_row, "IsPrimary"] = True
                df.at[src_row, "Status"] = "Active"
            dst_row = sig_to_idx.get((rec.employee_key, rec.dst_key, rec.transfer_date))
            if dst_row is not None:
                drop_idx.append(dst_row)
            rec.rolled_back = True
            rolled_back_count += 1

        if drop_idx:
            df.drop(index=drop_idx, inplace=True)
            df.reset_index(drop=True, inplace=True)
    return rolled_back_count


def _log_violation_details(
    violations: List[Tuple[int, int]],
    budget: CoverageBudget,
    transfer_records: List[_TransferRecord],
) -> None:
    """Per-violation warn line listing up to 3 related transfers."""
    for s_idx, m_idx in violations[:10]:
        store_key = int(budget.store_idx_to_key[s_idx])
        month_iso = pd.Timestamp(budget.month_starts[m_idx]).strftime("%Y-%m")
        related = [
            r for r in transfer_records
            if not r.rolled_back and r.src_key == store_key
            and r.transfer_date.strftime("%Y-%m") == month_iso
        ][:3]
        rel_str = (
            ", ".join(
                f"emp={r.employee_key}->store={r.dst_key}@{r.transfer_date.date()}"
                for r in related
            )
            if related else "no matching transfers"
        )
        warn(
            f"  violation store={store_key} month={month_iso} "
            f"cov={int(budget.cov[s_idx, m_idx])} min={budget.min_coverage} "
            f"related: {rel_str}"
        )


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

    reno_start_dates: Dict[int, pd.Timestamp] = {}
    reno_end_dates: Dict[int, pd.Timestamp] = {}
    if "RenovationStartDate" in stores.columns:
        rs = pd.to_datetime(stores["RenovationStartDate"], errors="coerce")
        re = pd.to_datetime(stores["RenovationEndDate"], errors="coerce")
        reno_mask = rs.notna() & re.notna()
        for sk, rsd, red in zip(sk_arr[reno_mask], rs[reno_mask], re[reno_mask]):
            reno_start_dates[int(sk)] = rsd
            reno_end_dates[int(sk)] = red

    if "StoreRegion" in stores.columns:
        sr = stores["StoreRegion"].astype(str).to_numpy()
        for sk, region in zip(sk_arr, sr):
            store_region[int(sk)] = region

    region_stores = _build_region_store_map(stores)

    # All physical store keys (for cross-region fallback)
    all_physical_keys = [
        int(sk) for sk in sk_arr if sk <= ONLINE_STORE_KEY_BASE
    ]

    df = assignments.copy()
    transfer_count = 0
    rejections: Dict[str, int] = defaultdict(int)
    transfer_records: List[_TransferRecord] = []

    budget = _build_coverage_budget(
        df, stores, salesperson_roles, global_start, global_end,
        open_dates, close_dates, reno_start_dates, reno_end_dates,
        min_coverage=1,
    )

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

        # Process chronologically so each candidate sees the budget after
        # prior accepts in the same year.
        t_earliest = max(year_start, global_start + pd.Timedelta(days=1))
        t_latest = min(
            year_end - pd.Timedelta(days=30),
            global_end - pd.Timedelta(days=30),
        )
        if t_earliest >= t_latest:
            continue
        t_range_days = (t_latest - t_earliest).days

        day_offsets = rng.integers(0, t_range_days + 1, size=len(chosen_idx))
        transfer_dates = [t_earliest + pd.Timedelta(days=int(d)) for d in day_offsets]
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

        region_open, all_open = _build_open_store_candidates(
            region_stores, all_physical_keys, year_start, year_end,
            open_dates, close_dates,
            reno_start_dates, reno_end_dates,
        )

        year_new_rows: list[dict] = []

        # Pre-generate transfer reasons for the year (avoids per-transfer rng.choice)
        _reasons_batch = rng.choice(
            EMPLOYEE_TRANSFER_REASON_LABELS,
            size=len(chronological),
            p=EMPLOYEE_TRANSFER_REASON_PROBS,
        )
        _reason_idx = 0

        for ti, (transfer_date, idx) in enumerate(chronological):
            emp_key = int(col_emp_key[idx])
            source_sk = int(col_store_key[idx])
            source_region = store_region.get(source_sk)
            employee_start = pd.Timestamp(start_ns[idx])
            original_end = pd.Timestamp(end_ns[idx])

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
                rejections["no_destination"] += 1
                continue

            new_end, new_status, dst_skip = _clamp_dst_end(
                dest_sk, transfer_date, original_end,
                close_dates, reno_start_dates, reno_end_dates,
            )
            if dst_skip:
                rejections["dst_renovation_window"] += 1
                continue

            feasible, _violators = _is_transfer_feasible(
                budget, source_sk, employee_start, transfer_date, original_end,
            )
            if not feasible:
                rejections["source_coverage"] += 1
                continue

            _adjust_budget(
                budget, source_sk, dest_sk,
                employee_start, transfer_date, original_end, new_end,
                direction=1,
            )

            new_end_ns = np.datetime64(transfer_date - pd.Timedelta(days=1), "ns")
            end_ns[idx] = new_end_ns
            col_is_primary[idx] = np.int32(0)
            col_status[idx] = "Transferred"

            reason = _reasons_batch[_reason_idx]
            _reason_idx += 1

            new_row = {
                "EmployeeKey": np.int32(emp_key),
                "StoreKey": np.int32(dest_sk),
                "StartDate": transfer_date,
                "EndDate": new_end,
                "FTE": col_fte[idx],
                "RoleAtStore": col_role[idx],
                "IsPrimary": True,
                "TransferReason": str(reason),
                "Status": new_status,
            }
            year_new_rows.append(new_row)
            transfer_records.append(_TransferRecord(
                employee_key=emp_key,
                src_key=source_sk,
                dst_key=int(dest_sk),
                employee_start=employee_start,
                transfer_date=transfer_date,
                original_end=original_end,
                dst_end=new_end,
            ))
            transfer_count += 1

        df["EndDate"] = end_ns
        df["IsPrimary"] = col_is_primary
        df["Status"] = col_status

        if year_new_rows:
            df = pd.concat([df, pd.DataFrame(year_new_rows)], ignore_index=True)

    if transfer_count == 0:
        info(
            "Transfers: 0 transfers generated "
            f"({_format_rejection_summary(rejections, 'no candidates')})"
        )
        return assignments

    # Surgical rollback. Feasibility checks should make this a no-op normally;
    # if it does fire, drop only the transfers that intersect violation cells.
    transfer_count -= _surgical_rollback(df, budget, transfer_records)

    final_violations = _budget_violations(budget)
    if final_violations:
        warn(
            f"Transfers: {len(final_violations)} unresolved violations after "
            f"surgical rollback; reverting all transfers."
        )
        info(f"Transfer rejections: {_format_rejection_summary(rejections, 'none')}")
        _log_violation_details(final_violations, budget, transfer_records)
        return assignments

    if transfer_count == 0:
        info(
            "Transfers: 0 transfers retained after rollback "
            f"({_format_rejection_summary(rejections, 'all rolled back')})"
        )
        return assignments

    # Sort and reassign keys/sequences
    df = df.sort_values(["EmployeeKey", "StartDate"]).reset_index(drop=True)
    df["AssignmentKey"] = np.arange(1, len(df) + 1, dtype=np.int32)
    df["AssignmentSequence"] = (df.groupby("EmployeeKey").cumcount() + 1).astype(np.int32)

    # Normalize dates
    df["StartDate"] = pd.to_datetime(df["StartDate"]).dt.normalize()
    df["EndDate"] = pd.to_datetime(df["EndDate"]).dt.normalize()

    info(
        f"Transfers: {transfer_count} transfers across "
        f"{df['EmployeeKey'].nunique()} employees, "
        f"{df['StoreKey'].nunique()} stores"
    )
    if rejections:
        info(f"Transfer rejections: {_format_rejection_summary(rejections, '')}")

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
    reno_start_dates: Optional[Dict[int, pd.Timestamp]] = None,
    reno_end_dates: Optional[Dict[int, pd.Timestamp]] = None,
    min_coverage: int = 1,
) -> List[Tuple[int, str]]:
    """Check that every open physical store has at least ``min_coverage``
    salespeople active *throughout* every month — i.e. for both first-of-month
    and last-of-month staffing. The last month_end is clamped to ``global_end``.

    Returns a list of ``(StoreKey, 'YYYY-MM')`` violations (capped at 10).
    """
    budget = _build_coverage_budget(
        assignments, stores, salesperson_roles,
        global_start, global_end,
        open_dates, close_dates, reno_start_dates, reno_end_dates,
        min_coverage=min_coverage,
    )
    violations: List[Tuple[int, str]] = []
    bad = budget.constrained & (budget.cov < budget.min_coverage)
    if not bad.any():
        return violations
    s_idx, m_idx = np.where(bad)
    for i in range(min(len(s_idx), 10)):
        sk = int(budget.store_idx_to_key[s_idx[i]])
        month_iso = pd.Timestamp(budget.month_starts[m_idx[i]]).strftime("%Y-%m")
        violations.append((sk, month_iso))
    return violations
