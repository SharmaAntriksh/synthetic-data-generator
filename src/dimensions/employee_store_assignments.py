"""Employee Store Assignment bridge table generator.

Produces a non-overlapping, contiguous timeline of store assignments for each
employee.  Every assignment is **permanent** — employees move forward through a
chain of stores and never return to a previous one.

Model:
  1. Every employee starts at their home store (decoded from EmployeeKey).
  2. A fraction (`transfer_rate × role_multiplier`) permanently transfer to
     another store in their district (or globally if pool_scope="all").
  3. Transfers can chain (up to `max_transfers`), with geometric decay.
  4. Store closures force a transfer to the nearest open store.
  5. Online employees never transfer — single assignment, full tenure.

Output columns:
  EmployeeKey, StoreKey, StartDate, EndDate, FTE, RoleAtStore,
  IsPrimary, TransferReason, AssignmentSequence, Status, MaxAssignments
"""
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
# Constants
# ---------------------------------------------------------------------------

_ONE_DAY = pd.Timedelta(days=1)
_SA_ROLE = "Sales Associate"

# Default role profiles: multiplier on base transfer_rate
_DEFAULT_ROLE_PROFILES: Dict[str, Dict[str, float]] = {
    "default":               {"mult": 0.0},
    _SA_ROLE:                {"mult": 2.50},
    "Cashier":               {"mult": 0.0},
    "Fulfillment Associate": {"mult": 0.0},
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _store_assignments_cfg(cfg) -> Dict[str, Any]:
    """Extract store_assignments config from nested or legacy path."""
    cfg = cfg or {}
    emp_cfg = cfg.employees
    nested = dict(emp_cfg.store_assignments) if emp_cfg.store_assignments is not None else {}
    legacy = as_dict(getattr(cfg, "employee_store_assignments", None))

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
    """``DistrictId -> [StoreKey, ...]`` from store managers or staff."""
    emp2 = emp[emp["DistrictId"].notna() & emp["HomeStoreKey"].notna()].copy()
    if emp2.empty:
        return {}
    is_mgr = emp2["Title"].astype(str) == "Store Manager"
    mgrs = emp2[is_mgr]
    src = mgrs if not mgrs.empty else emp2
    out: Dict[Any, List[int]] = {}
    for did, grp in src.groupby("DistrictId"):
        sks = grp["HomeStoreKey"].dropna().astype(int).unique().tolist()
        if sks:
            out[did] = sks
    return out


# ---------------------------------------------------------------------------
# Core: sequential chain builder
# ---------------------------------------------------------------------------

def _build_transfer_chain(
    rng: np.random.Generator,
    home_store: int,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    fte: float,
    role: str,
    n_transfers: int,
    candidate_stores: List[int],
    store_opening_dates: Optional[Dict[int, pd.Timestamp]],
    store_closing_dates: Optional[Dict[int, pd.Timestamp]],
) -> List[dict]:
    """Build a forward-only chain of permanent store assignments.

    Returns list of dicts with keys:
      StoreKey, StartDate, EndDate, FTE, RoleAtStore, IsPrimary, TransferReason
    """
    if start_date > end_date:
        return []

    chain: List[dict] = []
    cursor = start_date
    current_store = home_store
    reason: Any = pd.NA

    # Remove home store from candidates to avoid self-transfer
    pool = [s for s in candidate_stores if s != home_store]

    for i in range(n_transfers):
        if not pool or cursor >= end_date:
            break

        # Transfer date: random point in remaining tenure (leave at least 30 days at new store)
        remaining_days = (end_date - cursor).days
        if remaining_days < 60:
            break  # not enough time for a meaningful transfer

        # Transfer happens between 20% and 80% of remaining tenure
        earliest = cursor + pd.Timedelta(days=max(30, int(remaining_days * 0.20)))
        latest = cursor + pd.Timedelta(days=int(remaining_days * 0.80))
        if earliest >= latest or earliest >= end_date:
            break

        transfer_date = rand_single_date(rng, earliest, latest)

        # Emit segment at current store
        chain.append({
            "StoreKey": current_store,
            "StartDate": cursor,
            "EndDate": (transfer_date - _ONE_DAY).normalize(),
            "FTE": fte,
            "RoleAtStore": role,
            "IsPrimary": True,
            "TransferReason": pd.NA,
        })

        # Pick destination — prefer stores that are open at transfer time
        open_pool = []
        for s in pool:
            s_open = store_opening_dates.get(s) if store_opening_dates else None
            s_close = store_closing_dates.get(s) if store_closing_dates else None
            if s_open and transfer_date < s_open:
                continue
            if s_close and transfer_date >= s_close:
                continue
            open_pool.append(s)

        if not open_pool:
            # No valid destination — stay at current store, abort chain
            # Remove the segment we just appended (we'll emit full remaining below)
            chain.pop()
            break

        dest = int(rng.choice(open_pool))
        reason = str(rng.choice(EMPLOYEE_TRANSFER_REASON_LABELS, p=EMPLOYEE_TRANSFER_REASON_PROBS))

        # Update state for next iteration
        cursor = transfer_date
        pool = [s for s in pool if s != dest]  # don't revisit
        current_store = dest

    # Final segment: current store until end
    if cursor <= end_date:
        chain.append({
            "StoreKey": current_store,
            "StartDate": cursor,
            "EndDate": end_date,
            "FTE": fte,
            "RoleAtStore": role,
            "IsPrimary": True,
            "TransferReason": reason if chain else pd.NA,
        })

    return chain


def _handle_store_closure(
    chain: List[dict],
    close_date: pd.Timestamp,
    dest_store: int,
    global_end: pd.Timestamp,
    term_date: Optional[pd.Timestamp],
) -> List[dict]:
    """Truncate chain at store closure and append transfer destination.

    Modifies the chain in-place and returns it.
    """
    if not chain:
        return chain

    end_limit = term_date if term_date and term_date < global_end else global_end

    # Find the assignment that spans the close date and truncate
    new_chain: List[dict] = []
    closure_handled = False

    def _closure_seg(seg: dict) -> dict:
        return {
            "StoreKey": dest_store, "StartDate": close_date, "EndDate": end_limit,
            "FTE": seg["FTE"], "RoleAtStore": seg["RoleAtStore"],
            "IsPrimary": True, "TransferReason": "Store Closure",
        }

    for seg in chain:
        if closure_handled:
            continue

        seg_start, seg_end = seg["StartDate"], seg["EndDate"]

        if seg_start >= close_date:
            closure_handled = True
            if close_date <= end_limit:
                new_chain.append(_closure_seg(seg))
            continue

        if seg_end >= close_date:
            new_chain.append({**seg, "EndDate": (close_date - _ONE_DAY).normalize()})
            closure_handled = True
            if close_date <= end_limit:
                new_chain.append(_closure_seg(seg))
            continue

        new_chain.append(seg)

    # If closure wasn't triggered (close_date after all segments), just return original
    if not closure_handled:
        return chain

    return new_chain


# ---------------------------------------------------------------------------
# Transfer destination selection (for store closures)
# ---------------------------------------------------------------------------

def _open_physical_stores(
    store_keys: List[int],
    at_date: pd.Timestamp,
    store_closing_dates: Optional[Dict[int, pd.Timestamp]],
    exclude: int = -1,
) -> List[int]:
    """Filter store keys to those that are physical and open at *at_date*."""
    out: List[int] = []
    for sk in store_keys:
        if sk == exclude or sk > ONLINE_STORE_KEY_BASE:
            continue
        if store_closing_dates and sk in store_closing_dates:
            if store_closing_dates[sk] <= at_date:
                continue
        out.append(sk)
    return out


def _pick_closure_destination(
    rng: np.random.Generator,
    closing_store: int,
    district_id: Optional[Any],
    store_by_district: Dict[Any, List[int]],
    store_closing_dates: Optional[Dict[int, pd.Timestamp]],
    close_date: pd.Timestamp,
    prefer_same_district: bool = True,
) -> Optional[int]:
    """Pick a destination store for an employee whose store is closing."""
    candidates: List[int] = []

    if prefer_same_district and district_id and district_id in store_by_district:
        candidates = _open_physical_stores(
            store_by_district[district_id], close_date, store_closing_dates, exclude=closing_store,
        )

    # Fallback to all districts if same-district is empty
    if not candidates:
        all_stores: List[int] = []
        for sks in store_by_district.values():
            all_stores.extend(sks)
        candidates = list(set(_open_physical_stores(
            all_stores, close_date, store_closing_dates, exclude=closing_store,
        )))

    if not candidates:
        return None

    return int(rng.choice(candidates))


# ---------------------------------------------------------------------------
# SA coverage guarantee
# ---------------------------------------------------------------------------

def _cascade_delay(
    df: pd.DataFrame, ek: int, from_idx: int, new_end: pd.Timestamp,
    global_end: pd.Timestamp,
) -> pd.DataFrame:
    """Extend assignment at from_idx to new_end, ripple the shift through the chain.

    When an SA stays longer at their current store, every subsequent assignment
    in their chain must shift forward by the same amount. Assignments that
    become zero-length or negative are dropped. EndDate is clamped to global_end.
    """
    old_end = df.loc[from_idx, "EndDate"]
    shift = new_end - old_end  # positive timedelta
    if shift.days <= 0:
        return df

    df.loc[from_idx, "EndDate"] = min(new_end, global_end)

    # Get all subsequent assignments for this employee, sorted
    rest = df[(df["EmployeeKey"] == ek) & (df["StartDate"] > old_end)].sort_values("StartDate")
    drop_idxs = []
    for idx in rest.index:
        df.loc[idx, "StartDate"] = df.loc[idx, "StartDate"] + shift
        df.loc[idx, "EndDate"] = min(df.loc[idx, "EndDate"] + shift, global_end)
        # Drop if pushed past the original end (absorbed entirely)
        if df.loc[idx, "StartDate"] > df.loc[idx, "EndDate"]:
            drop_idxs.append(idx)

    if drop_idxs:
        df = df.drop(drop_idxs)

    return df


def _ensure_sa_coverage(
    df: pd.DataFrame,
    global_start: pd.Timestamp,
    global_end: pd.Timestamp,
    store_opening_dates: Optional[Dict[int, pd.Timestamp]],
    store_closing_dates: Optional[Dict[int, pd.Timestamp]],
) -> pd.DataFrame:
    """Guarantee continuous Sales Associate coverage for every physical store.

    Strategy: when an SA transfers out and leaves a gap, delay their departure
    (extend EndDate at the current store, cascade the shift through their chain)
    until a replacement SA arrives or the store closes.

    This is realistic: in retail, an employee's transfer is delayed until their
    replacement is onboarded.
    """
    sa_mask = (df["RoleAtStore"] == _SA_ROLE) & (df["StoreKey"] <= ONLINE_STORE_KEY_BASE)
    if not sa_mask.any():
        return df

    physical_stores = df.loc[sa_mask, "StoreKey"].unique()
    fixes = 0
    df = df.copy()

    for sk in physical_stores:
        s_open = store_opening_dates.get(int(sk), global_start) if store_opening_dates else global_start
        s_close = store_closing_dates.get(int(sk), global_end) if store_closing_dates else global_end
        window_start = max(s_open, global_start)
        window_end = min(s_close - _ONE_DAY, global_end) if int(sk) in (store_closing_dates or {}) else global_end

        if window_start > window_end:
            continue

        # Re-query SA assignments after each fix (indices may have shifted)
        store_sa = df[(df["StoreKey"] == sk) & (df["RoleAtStore"] == _SA_ROLE)].sort_values("StartDate")
        if store_sa.empty:
            continue

        intervals = list(zip(store_sa["StartDate"], store_sa["EndDate"], store_sa.index))
        intervals.sort(key=lambda x: x[0])

        cursor = window_start
        for seg_start, seg_end, idx in intervals:
            if seg_start > cursor:
                # Gap found — find the departing SA
                departing = df[
                    (df["StoreKey"] == sk)
                    & (df["RoleAtStore"] == _SA_ROLE)
                    & (df["EndDate"] < seg_start)
                    & (df["EndDate"] >= cursor - _ONE_DAY)
                ]
                if not departing.empty:
                    dep_idx = departing.sort_values("EndDate", ascending=False).index[0]
                    dep_ek = int(df.loc[dep_idx, "EmployeeKey"])
                    new_end = (seg_start - _ONE_DAY).normalize()
                    df = _cascade_delay(df, dep_ek, dep_idx, new_end, global_end)
                    fixes += 1

            cursor = max(cursor, seg_end + _ONE_DAY)

        # Trailing gap
        if cursor <= window_end:
            last_sa = store_sa.iloc[-1]
            last_idx = last_sa.name
            if last_idx in df.index and df.loc[last_idx, "EndDate"] < window_end:
                dep_ek = int(df.loc[last_idx, "EmployeeKey"])
                df = _cascade_delay(df, dep_ek, last_idx, window_end, global_end)
                fixes += 1

    if fixes > 0:
        info(f"SA coverage: patched {fixes} gap(s) by delaying transfers")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_employee_store_assignments(
    employees: pd.DataFrame,
    seed: int,
    global_start: pd.Timestamp,
    global_end: pd.Timestamp,
    *,
    transfer_rate: float = 0.15,
    max_transfers: int = 3,
    pool_scope: str = "district",
    role_profiles: Optional[Dict[str, Any]] = None,
    store_opening_dates: Optional[Dict[int, pd.Timestamp]] = None,
    store_closing_dates: Optional[Dict[int, pd.Timestamp]] = None,
    transfers_df: Optional[pd.DataFrame] = None,
    prefer_same_district: bool = True,
    enabled: bool = True,
) -> pd.DataFrame:
    """Generate the employee store assignments bridge table.

    Sequential chain model: every transfer is permanent, no boomerangs.
    """
    out_cols = [
        "EmployeeKey", "StoreKey", "StartDate", "EndDate",
        "FTE", "RoleAtStore", "IsPrimary", "TransferReason",
    ]

    if not enabled or employees.empty:
        return pd.DataFrame(columns=out_cols + [
            "AssignmentSequence", "Status", "MaxAssignments",
        ])

    rng = np.random.default_rng(seed)

    # Merge role profiles
    # Config normalizer renames "mult" → "role_multiplier"; accept both keys
    profiles = dict(_DEFAULT_ROLE_PROFILES)
    if role_profiles:
        for role_name, prof in role_profiles.items():
            if hasattr(prof, "model_dump"):
                prof = prof.model_dump()
            elif not isinstance(prof, dict):
                prof = {"mult": float(prof)}
            m = prof.get("mult", prof.get("role_multiplier", 0.0))
            profiles[role_name] = {"mult": float(m)}

    # Decode home stores
    emp = employees.copy()
    emp["HomeStoreKey"] = _infer_home_store_key(emp)
    emp["HireDate"] = pd.to_datetime(emp["HireDate"], errors="coerce").dt.normalize()
    emp["TerminationDate"] = pd.to_datetime(emp["TerminationDate"], errors="coerce").dt.normalize()

    # Build district -> stores map
    store_by_district = _build_store_by_district(emp)

    # Build transfer map from employee_transfers sidecar (store closures)
    transfer_map: Dict[int, Tuple[pd.Timestamp, int, Optional[pd.Timestamp]]] = {}
    if transfers_df is not None and not transfers_df.empty:
        for _, row in transfers_df.iterrows():
            ek = int(row["EmployeeKey"])
            close_date = pd.to_datetime(row["TransferDate"]).normalize()
            orig_sk = int(row["OriginalStoreKey"])
            did = row.get("DistrictId")

            dest = _pick_closure_destination(
                rng, orig_sk, did, store_by_district,
                store_closing_dates, close_date, prefer_same_district,
            )
            if dest is not None:
                transfer_map[ek] = (close_date, dest, None)

    # Pre-compute global candidate pool (used when pool_scope != "district")
    _global_candidates: Optional[List[int]] = None
    if pool_scope != "district":
        _all = []
        for sks in store_by_district.values():
            _all.extend(s for s in sks if s <= ONLINE_STORE_KEY_BASE)
        _global_candidates = list(set(_all))

    # Process each employee
    all_rows: List[dict] = []

    for row in emp.to_dict("records"):
        ek = int(row["EmployeeKey"])
        home_sk = row["HomeStoreKey"]
        if pd.isna(home_sk):
            continue
        home_sk = int(home_sk)

        hire = row["HireDate"]
        term = row["TerminationDate"]
        if pd.isna(hire):
            continue

        title = str(row["Title"])
        fte = float(row["FTE"]) if pd.notna(row["FTE"]) else 1.0
        district_id = row["DistrictId"] if pd.notna(row["DistrictId"]) else None

        # Effective window
        start = max(hire, global_start)
        end = term if pd.notna(term) and term < global_end else global_end
        if start > end:
            continue

        # Online employees and Store Managers: single assignment, no voluntary transfers
        is_online = ek >= ONLINE_EMP_KEY_BASE
        is_manager = STORE_MGR_KEY_BASE <= ek < STAFF_KEY_BASE
        if is_online or is_manager:
            all_rows.append({
                "EmployeeKey": ek, "StoreKey": home_sk,
                "StartDate": start, "EndDate": end,
                "FTE": fte, "RoleAtStore": title,
                "IsPrimary": True, "TransferReason": pd.NA,
            })
            # Store closures still apply below via transfer_map
            if ek in transfer_map:
                close_date, dest_store, _ = transfer_map[ek]
                chain = _handle_store_closure(
                    [all_rows.pop()], close_date, dest_store, global_end, term,
                )
                for seg in chain:
                    seg["EmployeeKey"] = ek
                    all_rows.append(seg)
            continue

        # Determine number of voluntary transfers
        prof = profiles.get(title, profiles.get("default", {"mult": 0.0}))
        role_mult = float(prof.get("mult", 0.0))
        effective_rate = min(transfer_rate * role_mult, 0.95)

        # Roll for transfers: geometric decay
        n_transfers = 0
        if rng.random() < effective_rate:
            n_transfers = 1
            for _ in range(max_transfers - 1):
                if rng.random() < 0.30:  # 30% chance of additional transfer
                    n_transfers += 1
                else:
                    break

        # Build candidate store pool
        if pool_scope == "district" and district_id and district_id in store_by_district:
            candidates = [s for s in store_by_district[district_id]
                          if s != home_sk and s <= ONLINE_STORE_KEY_BASE]
        else:
            candidates = [s for s in _global_candidates if s != home_sk] if _global_candidates else []

        # Build the chain
        chain = _build_transfer_chain(
            rng, home_sk, start, end, fte, title,
            n_transfers, candidates,
            store_opening_dates, store_closing_dates,
        )

        # If chain is empty (shouldn't happen, but safety), emit single home assignment
        if not chain:
            chain = [{
                "StoreKey": home_sk, "StartDate": start, "EndDate": end,
                "FTE": fte, "RoleAtStore": title,
                "IsPrimary": True, "TransferReason": pd.NA,
            }]

        # Apply store closure transfer if applicable
        if ek in transfer_map:
            close_date, dest_store, _ = transfer_map[ek]
            chain = _handle_store_closure(chain, close_date, dest_store, global_end, term)

        # Add EmployeeKey and collect
        for seg in chain:
            seg["EmployeeKey"] = ek
            all_rows.append(seg)

    if not all_rows:
        return pd.DataFrame(columns=out_cols + [
            "AssignmentSequence", "Status", "MaxAssignments",
        ])

    # Build output DataFrame
    out = pd.DataFrame(all_rows, columns=out_cols)
    out["EmployeeKey"] = out["EmployeeKey"].astype(np.int32)
    out["StoreKey"] = out["StoreKey"].astype(np.int32)
    out["FTE"] = out["FTE"].astype(np.float64)
    out["IsPrimary"] = out["IsPrimary"].astype(bool)
    out["RoleAtStore"] = out["RoleAtStore"].astype(str)
    out["StartDate"] = pd.to_datetime(out["StartDate"]).dt.normalize()
    out["EndDate"] = pd.to_datetime(out["EndDate"]).dt.normalize()

    # Ensure every physical store has SA coverage for its full operational window
    out = _ensure_sa_coverage(
        out, global_start, global_end,
        store_opening_dates, store_closing_dates,
    )

    # Sort and assign sequence
    out = out.sort_values(["EmployeeKey", "StartDate"]).reset_index(drop=True)
    out["AssignmentSequence"] = out.groupby("EmployeeKey").cumcount() + 1
    out["AssignmentSequence"] = out["AssignmentSequence"].astype(np.int32)

    # Status: last assignment per employee = Active (if no termination), others = Transferred
    max_seq = out.groupby("EmployeeKey")["AssignmentSequence"].transform("max")
    is_last = out["AssignmentSequence"] == max_seq
    out["Status"] = np.where(is_last, "Active", "Transferred")

    # If employee is terminated, last assignment status = Completed
    term_eks = set(
        emp.loc[emp["TerminationDate"].notna(), "EmployeeKey"].astype(int).tolist()
    )
    term_mask = is_last & out["EmployeeKey"].isin(term_eks)
    out.loc[term_mask, "Status"] = "Completed"

    # MaxAssignments must come after Status to match SQL schema column order
    out["MaxAssignments"] = max_seq.astype(np.int32)

    info(
        f"Bridge table: {len(out)} rows, "
        f"{out['EmployeeKey'].nunique()} employees, "
        f"{out['StoreKey'].nunique()} stores, "
        f"max chain length {out['MaxAssignments'].max()}"
    )

    return out


# ---------------------------------------------------------------------------
# Runner entry point (called by dimensions_runner)
# ---------------------------------------------------------------------------

def run_employee_store_assignments(cfg, parquet_folder: Path, out_path: Path = None) -> None:
    """Generate and write the EmployeeStoreAssignments bridge table."""
    if out_path is None:
        out_path = parquet_folder / "employee_store_assignments.parquet"

    a_cfg = _store_assignments_cfg(cfg)

    employees_path = parquet_folder / "employees.parquet"
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
    version_cfg["schema_version"] = 13  # v13: sequential chain model
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
            _stores_df = pd.read_parquet(
                stores_path,
                columns=["StoreKey", "OpeningDate", "ClosingDate"],
            )
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

    _closing_cfg = as_dict(cfg.stores.closing) if hasattr(cfg.stores, "closing") and cfg.stores.closing is not None else {}
    _prefer_same_district = bool(_closing_cfg.get("prefer_same_district", True))

    with stage("Generating Employee Store Assignments"):
        df = generate_employee_store_assignments(
            employees=employees,
            seed=seed,
            global_start=global_start,
            global_end=global_end,
            transfer_rate=float_or(a_cfg.get("transfer_rate", a_cfg.get("mover_share", 0.15)), 0.15),
            max_transfers=int_or(a_cfg.get("max_transfers", 3), 3),
            pool_scope=str_or(a_cfg.get("pool_scope"), "district"),
            role_profiles=role_profiles,
            store_opening_dates=store_opening_dates,
            store_closing_dates=store_closing_dates,
            transfers_df=transfers_df,
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
                    stores_full["EmployeeCount"] = (
                        stores_full["StoreKey"].astype(int).map(actual_counts).fillna(0).astype(np.int64)
                    )

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
    """Extract {EmployeeKey: destination StoreKey} for transferred employees."""
    if transfers_df is None or transfers_df.empty:
        return {}

    transferred_eks = set(transfers_df["EmployeeKey"].astype(int).tolist())
    if not transferred_eks:
        return {}

    sub = df[df["EmployeeKey"].isin(transferred_eks)].copy()
    if sub.empty:
        return {}

    sub = sub.sort_values(["EmployeeKey", "AssignmentSequence"])
    last = sub.groupby("EmployeeKey").last()
    return {int(ek): int(sk) for ek, sk in zip(last.index, last["StoreKey"].values)}
