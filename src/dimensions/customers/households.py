"""Household assignment for customer dimension."""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from src.defaults import (
    CUSTOMER_HOUSEHOLD_HEAD_MIN_AGE as _HEAD_MIN_AGE,
    CUSTOMER_HOUSEHOLD_SPOUSE_MAX_AGE_GAP as _SPOUSE_MAX_AGE_GAP,
    CUSTOMER_HOUSEHOLD_SPOUSE_MIN_AGE as _SPOUSE_MIN_AGE,
    CUSTOMER_HOUSEHOLD_DEPENDENT_MIN_AGE_GAP as _DEPENDENT_MIN_AGE_GAP,
    CUSTOMER_HOUSEHOLD_DEPENDENT_MAX_AGE as _DEPENDENT_MAX_AGE,
)


def assign_households(
    rng: np.random.Generator,
    N: int,
    is_org: np.ndarray,
    last_name: np.ndarray,
    geography_key: np.ndarray,
    gender: np.ndarray,
    ages_years: np.ndarray,
    marital_status: np.ndarray,
    children_raw: np.ndarray,
    home_ownership: np.ndarray,
    household_pct: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign HouseholdKey and HouseholdRole for all customers.

    Strategy — proactively *form* households rather than relying on
    coincidental name/geography collisions:

      1. Select married individuals as household "Head" candidates.
         Pick *household_pct* of them (capped by available married pool).
      2. For each head, find a spouse: opposite gender, age gap within
         threshold, preferring same HomeOwnership.  Copy the head's
         LastName and GeographyKey onto the spouse so the family shares
         a surname and location.
      3. If the head's TotalChildren > 0, recruit up to that many young
         individuals (age gap >= DEPENDENT_MIN_AGE_GAP) as "Dependent",
         again copying LastName and GeographyKey.
      4. Everyone not placed into a multi-person household gets their own
         single-person household with role "Head" (individuals) or
         role=None (orgs).

    Mutates *last_name* and *geography_key* in place so downstream
    columns (CustomerName, Email, address) reflect the shared surname.

    Returns (HouseholdKey, HouseholdRole) arrays of length N.
    """
    household_key = np.zeros(N, dtype="int64")
    household_role = np.empty(N, dtype=object)
    household_role[:] = None

    person_mask = ~is_org
    person_idx = np.where(person_mask)[0]
    n_person = len(person_idx)

    if n_person == 0:
        household_key[:] = np.arange(1, N + 1)
        return household_key, household_role

    # --- Identify head candidates: married individuals old enough ---
    married_mask = (marital_status[person_idx] == "Married") & (ages_years[person_idx] >= _HEAD_MIN_AGE)
    married_idx = person_idx[married_mask]

    n_target_heads = max(1, int(round(n_person * household_pct)))
    n_heads = min(n_target_heads, len(married_idx))

    if n_heads == 0:
        # No married individuals — everyone gets a solo household
        household_key[:] = np.arange(1, N + 1)
        household_role[person_mask] = "Head"
        return household_key, household_role

    head_indices = rng.choice(married_idx, size=n_heads, replace=False)

    # -----------------------------------------------------------------
    # Build sorted-by-age pools per gender for O(log n) spouse search
    # -----------------------------------------------------------------
    head_set = set(head_indices.tolist())
    avail_mask = np.zeros(N, dtype=bool)          # global availability tracker
    avail_mask[person_idx] = True
    avail_mask[head_indices] = False               # heads are not in the pool

    ages_f64 = ages_years.astype(np.float64)       # ensure float for arithmetic
    ho_arr = home_ownership                        # alias for readability

    # Per-gender sorted pools: (sorted_ages, sorted_indices, alive_mask)
    _pools: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for g in ("Male", "Female"):
        g_avail = np.where(avail_mask & (gender == g))[0]
        if len(g_avail) == 0:
            _pools[g] = (np.empty(0), np.empty(0, dtype="int64"), np.empty(0, dtype=bool))
            continue
        order = np.argsort(ages_f64[g_avail])
        sorted_idx = g_avail[order]
        sorted_ages = ages_f64[sorted_idx]
        alive = np.ones(len(sorted_idx), dtype=bool)
        _pools[g] = (sorted_ages, sorted_idx, alive)

    # Dependent pool: all genders, age <= DEPENDENT_MAX_AGE, sorted ascending
    dep_avail = np.where(avail_mask & (ages_f64 <= _DEPENDENT_MAX_AGE))[0]
    dep_order = np.argsort(ages_f64[dep_avail])
    dep_sorted_idx = dep_avail[dep_order]
    dep_sorted_ages = ages_f64[dep_sorted_idx]
    dep_alive = np.ones(len(dep_sorted_idx), dtype=bool)

    hh_id = 0

    for head in head_indices:
        hh_id += 1
        household_key[head] = hh_id
        household_role[head] = "Head"

        head_age = ages_f64[head]
        head_gender = gender[head]
        head_ln = last_name[head]
        head_geo = geography_key[head]
        head_ho = ho_arr[head]

        # --- Find a spouse: opposite gender, within age gap ---
        spouse_gender = "Female" if head_gender == "Male" else "Male"
        s_ages, s_idx, s_alive = _pools.get(spouse_gender, (np.empty(0), np.empty(0, dtype="int64"), np.empty(0, dtype=bool)))

        best_pool_pos = -1

        if len(s_ages) > 0:
            lo_age = max(head_age - _SPOUSE_MAX_AGE_GAP, _SPOUSE_MIN_AGE)
            hi_age = head_age + _SPOUSE_MAX_AGE_GAP
            lo_pos = int(np.searchsorted(s_ages, lo_age, side="left"))
            hi_pos = int(np.searchsorted(s_ages, hi_age, side="right"))

            if lo_pos < hi_pos:
                window_alive = s_alive[lo_pos:hi_pos]
                alive_positions = np.flatnonzero(window_alive)
                if len(alive_positions) > 0:
                    w_ages = s_ages[lo_pos:hi_pos][alive_positions]
                    w_idx = s_idx[lo_pos:hi_pos][alive_positions]
                    scores = np.abs(head_age - w_ages)
                    ho_match = (ho_arr[w_idx] == head_ho)
                    scores = np.where(ho_match, scores - 2.0, scores)
                    best_w = int(np.argmin(scores))
                    best_pool_pos = lo_pos + int(alive_positions[best_w])

        if best_pool_pos >= 0:
            matched = int(s_idx[best_pool_pos])
            household_key[matched] = hh_id
            household_role[matched] = "Spouse"
            s_alive[best_pool_pos] = False
            avail_mask[matched] = False
            last_name[matched] = head_ln
            geography_key[matched] = head_geo

        # --- Recruit dependents based on TotalChildren ---
        n_children_wanted = int(children_raw[head])
        if n_children_wanted > 0:
            max_dep_age = head_age - _DEPENDENT_MIN_AGE_GAP
            if max_dep_age > 0:
                # dep_sorted_ages is ascending; valid range is [0, max_dep_age]
                hi_dep = int(np.searchsorted(dep_sorted_ages, max_dep_age, side="right"))
                # Vectorized: find alive candidates in the eligible window, take oldest first
                _dep_window_alive = np.flatnonzero(dep_alive[:hi_dep])
                if _dep_window_alive.size > 0:
                    # Also check global avail_mask for candidates
                    _dep_cands = dep_sorted_idx[_dep_window_alive]
                    _dep_avail = avail_mask[_dep_cands]
                    _dep_valid = _dep_window_alive[_dep_avail]
                    # Take up to n_children_wanted from the oldest end (reversed)
                    _take = _dep_valid[-n_children_wanted:][::-1] if len(_dep_valid) > n_children_wanted else _dep_valid[::-1]
                    for dp in _take:
                        cand = int(dep_sorted_idx[dp])
                        household_key[cand] = hh_id
                        household_role[cand] = "Dependent"
                        dep_alive[dp] = False
                        avail_mask[cand] = False
                        last_name[cand] = head_ln
                        geography_key[cand] = head_geo

    # --- Assign solo households to everyone not yet assigned ---
    unassigned = np.where(household_key == 0)[0]
    household_key[unassigned] = np.arange(hh_id + 1, hh_id + 1 + len(unassigned))
    # Solo persons get "Head" role; orgs stay None
    solo_persons = unassigned[person_mask[unassigned]]
    household_role[solo_persons] = "Head"

    return household_key, household_role
