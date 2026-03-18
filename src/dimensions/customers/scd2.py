"""SCD Type 2 — Life Event Engine for customer dimension."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import info
from src.defaults import (
    SCD2_END_OF_TIME,
    CUSTOMER_INCOME_MIN as INCOME_MIN,
    CUSTOMER_INCOME_MAX as INCOME_MAX,
    CUSTOMER_MAX_CHILDREN as MAX_CHILDREN,
    CUSTOMER_INCOME_GROUP_EDGES as INCOME_GROUP_EDGES,
    CUSTOMER_INCOME_GROUP_LABELS as INCOME_GROUP_LABELS,
    CUSTOMER_REGION_LAT_LON_CENTER as _REGION_LAT_LON_CENTER,
    CUSTOMER_LAT_LON_JITTER as _LAT_LON_JITTER,
    CUSTOMER_POSTCODE_FMT as _POSTCODE_FMT,
    CUSTOMER_STREET_NAMES as _STREET_NAMES,
    CUSTOMER_STREET_TYPES as _STREET_TYPES,
)


def _income_to_group(income: float) -> str:
    """Map a yearly income value to its income group label."""
    idx = int(np.searchsorted(INCOME_GROUP_EDGES, income))
    return str(INCOME_GROUP_LABELS[min(idx, len(INCOME_GROUP_LABELS) - 1)])


def _available_events(state: dict, tier_keys: np.ndarray) -> list:
    """Return list of (event_name, weight) tuples based on current state."""
    events = []
    ms = state.get("MaritalStatus")
    ho = state.get("HomeOwnership")
    nc = int(state.get("NumberOfChildren") or 0)
    tier = int(state.get("LoyaltyTierKey") or 0)

    # Career growth — always available
    events.append(("career_growth", 0.30))

    # Marriage — if single or divorced
    if ms in ("Single", "Divorced"):
        events.append(("marriage", 0.20))

    # Family growth — if married and room for more children
    if ms == "Married" and nc < MAX_CHILDREN:
        events.append(("family_growth", 0.15))

    # Home purchase — if not already owning
    if ho in ("Rent", "Other"):
        events.append(("home_purchase", 0.10))

    # Relocation — always available
    events.append(("relocation", 0.10))

    # Divorce — if married
    if ms == "Married":
        events.append(("divorce", 0.05))

    # Tier upgrade — if not at max tier
    if len(tier_keys) > 0 and tier < int(tier_keys[-1]):
        events.append(("tier_upgrade", 0.15))

    return events


def _relocate(rng: np.random.Generator, state: dict,
              geo_keys: np.ndarray, geo_lookup, _geo_cache: dict = None) -> None:
    """Change GeographyKey and regenerate address columns to match.

    HomeAddress always changes on relocation.  WorkAddress only changes
    when the customer moves to a different country (cross-country
    relocation implies a job change; within the same country, the old
    workplace is still reachable).

    geo_lookup can be a pd.DataFrame (legacy) or a dict (fast path).
    _geo_cache is an optional pre-built dict {GeographyKey: (Country, City, State)}.
    """
    old_gk = state.get("GeographyKey")
    new_gk = int(rng.choice(geo_keys))
    state["GeographyKey"] = new_gk

    # Fast path: dict lookup instead of DataFrame.loc
    if _geo_cache is not None:
        old_country = _geo_cache.get(old_gk, ("", "", ""))[0]
        new_entry = _geo_cache.get(new_gk, ("", "Unknown", "Unknown"))
        new_country, city, st = new_entry
    elif isinstance(geo_lookup, dict):
        old_country = geo_lookup.get(old_gk, ("", "", ""))[0]
        new_entry = geo_lookup.get(new_gk, ("", "Unknown", "Unknown"))
        new_country, city, st = new_entry
    else:
        # Legacy DataFrame path
        old_country = (
            str(geo_lookup.loc[old_gk, "Country"])
            if old_gk in geo_lookup.index else ""
        )
        if new_gk in geo_lookup.index:
            row = geo_lookup.loc[new_gk]
            new_country = str(row["Country"])
            city = str(row["City"])
            st = str(row["State"])
        else:
            new_country, city, st = "", "Unknown", "Unknown"
    cross_country = old_country != new_country

    # --- Helper: generate a single address string ---
    def _make_address(c, s):
        sn = str(rng.integers(1, 9999))
        sname = str(rng.choice(_STREET_NAMES))
        stype = str(rng.choice(_STREET_TYPES))
        ulabel = str(rng.choice(np.array(["Apt", "Suite", "Unit", "Fl", "#"])))
        unum = str(state.get("CustomerID", 0))
        return f"{sn} {sname} {stype}, {ulabel} {unum}, {c}, {s}"

    # Regenerate home address (always)
    state["HomeAddress"] = _make_address(city, st)

    # Regenerate work address only on cross-country moves
    if cross_country:
        state["WorkAddress"] = _make_address(city, st)

    # Map country to region code for lat/lon and postal code generation
    _country_to_region = {
        "United States": "US", "India": "IN",
        "United Kingdom": "EU", "Germany": "EU", "France": "EU",
        "Japan": "AS", "China": "AS", "Australia": "AS",
    }
    rc = _country_to_region.get(new_country, "US")

    # Regenerate lat/lon
    clat, clon = _REGION_LAT_LON_CENTER.get(rc, (39.8, -98.5))
    jlat, jlon = _LAT_LON_JITTER.get(rc, (5.0, 5.0))
    state["Latitude"] = round(float(clat + rng.uniform(-jlat, jlat)), 4)
    state["Longitude"] = round(float(clon + rng.uniform(-jlon, jlon)), 4)

    # Regenerate postal code
    fmt = _POSTCODE_FMT.get(rc, "5digit")
    if fmt == "5digit":
        state["PostalCode"] = f"{rng.integers(10001, 99999):05d}"
    elif fmt == "6digit":
        state["PostalCode"] = f"{rng.integers(100001, 999999):06d}"
    elif fmt == "uk":
        pfx = str(rng.choice(np.array(["SW", "EC", "W", "SE", "N", "NW", "E", "WC"])))
        state["PostalCode"] = f"{pfx}{rng.integers(1, 19)} {rng.integers(1, 9)}{chr(rng.integers(65, 91))}{chr(rng.integers(65, 91))}"
    elif fmt == "jp":
        v = int(rng.integers(1000000, 9999999))
        state["PostalCode"] = f"{v // 10000:03d}-{v % 10000:04d}"


def _apply_life_event(
    rng: np.random.Generator,
    state: dict,
    event: str,
    geo_keys: np.ndarray,
    tier_keys: np.ndarray,
    geo_lookup,
    *,
    _geo_cache: dict = None,
) -> None:
    """Apply a life event to the customer state dict (mutates in place)."""

    if event == "career_growth":
        bump = rng.uniform(0.12, 0.35)
        income = int(state.get("YearlyIncome") or 0)
        new_income = int(np.clip(income * (1 + bump), INCOME_MIN, INCOME_MAX))
        state["YearlyIncome"] = new_income
        state["IncomeGroup"] = _income_to_group(new_income)
        # 30% chance of tier upgrade alongside career growth
        if rng.random() < 0.30 and len(tier_keys) > 0:
            current_idx = int(np.searchsorted(tier_keys, state["LoyaltyTierKey"]))
            if current_idx < len(tier_keys) - 1:
                state["LoyaltyTierKey"] = int(tier_keys[current_idx + 1])

    elif event == "marriage":
        state["MaritalStatus"] = "Married"
        # 30% chance of relocation with marriage
        if rng.random() < 0.30:
            _relocate(rng, state, geo_keys, geo_lookup, _geo_cache=_geo_cache)

    elif event == "family_growth":
        nc = int(state.get("NumberOfChildren") or 0)
        state["NumberOfChildren"] = nc + 1
        # 40% chance of buying home when having kids
        if state.get("HomeOwnership") in ("Rent", "Other") and rng.random() < 0.40:
            state["HomeOwnership"] = "Own"
        # 20% chance of relocation
        if rng.random() < 0.20:
            _relocate(rng, state, geo_keys, geo_lookup, _geo_cache=_geo_cache)

    elif event == "home_purchase":
        state["HomeOwnership"] = "Own"

    elif event == "relocation":
        _relocate(rng, state, geo_keys, geo_lookup, _geo_cache=_geo_cache)

    elif event == "divorce":
        state["MaritalStatus"] = "Divorced"
        # 50% chance income decreases
        if rng.random() < 0.50:
            income = int(state.get("YearlyIncome") or 0)
            new_income = int(np.clip(income * rng.uniform(0.75, 0.90), INCOME_MIN, INCOME_MAX))
            state["YearlyIncome"] = new_income
            state["IncomeGroup"] = _income_to_group(new_income)
        # 60% chance of relocation
        if rng.random() < 0.60:
            _relocate(rng, state, geo_keys, geo_lookup, _geo_cache=_geo_cache)
        # 35% chance of losing home
        if state.get("HomeOwnership") == "Own" and rng.random() < 0.35:
            state["HomeOwnership"] = "Rent"

    elif event == "tier_upgrade":
        if len(tier_keys) > 0:
            current_idx = int(np.searchsorted(tier_keys, state["LoyaltyTierKey"]))
            if current_idx < len(tier_keys) - 1:
                state["LoyaltyTierKey"] = int(tier_keys[current_idx + 1])


def _build_geo_cache(geo_lookup) -> dict:
    """Build {GeographyKey: (Country, City, State)} cache for fast lookups."""
    cache: dict = {}
    if isinstance(geo_lookup, pd.DataFrame) and not geo_lookup.empty:
        for gk in geo_lookup.index:
            row = geo_lookup.loc[gk]
            cache[gk] = (str(row["Country"]), str(row["City"]), str(row["State"]))
    elif isinstance(geo_lookup, dict):
        cache = geo_lookup
    return cache


def expand_changed_customers(
    rng: np.random.Generator,
    changed_df: pd.DataFrame,
    max_versions: int,
    geo_keys: np.ndarray,
    tier_keys: np.ndarray,
    end_date: pd.Timestamp,
    geo_lookup,
) -> pd.DataFrame:
    """Expand pre-selected changed customers with SCD2 version rows.

    Unlike generate_scd2_versions, this does NOT perform random selection —
    all rows in changed_df are expanded.  The caller is responsible for
    selecting which customers to change.

    Returns a DataFrame containing all version rows (including originals).
    """
    _geo_cache = _build_geo_cache(geo_lookup)

    # Pre-allocate with estimated capacity
    new_rows = []
    _est_rows = int(len(changed_df) * 2.5)
    if _est_rows > 0:
        new_rows = [None] * _est_rows
    _row_count = 0

    _col_names = changed_df.columns.tolist()
    _col_arrays = {col: changed_df[col].to_numpy() for col in _col_names}
    _n_changed = len(changed_df)

    _eff_start_raw = _col_arrays.get("EffectiveStartDate")
    if _eff_start_raw is not None:
        _eff_start_ts = pd.to_datetime(pd.Series(_eff_start_raw))
        _eff_start_days = _eff_start_ts.values.astype("datetime64[D]").astype("int64")
        _eff_start_np = _eff_start_ts.to_numpy()
    else:
        _eff_start_days = None
        _eff_start_np = None
    _end_date_days = np.datetime64(end_date, "D").astype("int64")

    for _ri in range(_n_changed):
        row_dict = {col: _col_arrays[col][_ri] for col in _col_names}

        n_events = int(rng.integers(1, max_versions))

        if _eff_start_days is not None:
            total_days = int(_end_date_days - _eff_start_days[_ri])
            cust_start = pd.Timestamp(_eff_start_np[_ri])
        else:
            cust_start = pd.Timestamp(row_dict["EffectiveStartDate"])
            total_days = (end_date - cust_start).days

        if total_days <= 90:
            if _row_count < len(new_rows):
                new_rows[_row_count] = row_dict
            else:
                new_rows.append(row_dict)
            _row_count += 1
            continue

        n_events = min(n_events, (total_days - 90) // 90)
        if n_events <= 0:
            if _row_count < len(new_rows):
                new_rows[_row_count] = row_dict
            else:
                new_rows.append(row_dict)
            _row_count += 1
            continue

        max_offset = max(91, total_days - 90)
        offsets = np.sort(rng.integers(90, max_offset, size=n_events))
        for i in range(1, len(offsets)):
            if offsets[i] - offsets[i - 1] < 60:
                offsets[i] = min(offsets[i - 1] + 60, max_offset - 1)
        _offset_td = offsets.astype("timedelta64[D]")
        event_dates = np.datetime64(cust_start, "D") + _offset_td

        current_state = dict(row_dict)
        _one_day = np.timedelta64(1, "D")

        for i in range(len(event_dates)):
            event_date_np = event_dates[i]
            event_date = pd.Timestamp(event_date_np)
            current_state["EffectiveEndDate"] = event_date_np - _one_day
            current_state["IsCurrent"] = 0
            if _row_count < len(new_rows):
                new_rows[_row_count] = dict(current_state)
            else:
                new_rows.append(dict(current_state))
            _row_count += 1

            new_state = dict(current_state)
            new_state["VersionNumber"] = i + 2
            new_state["EffectiveStartDate"] = event_date
            new_state["EffectiveEndDate"] = SCD2_END_OF_TIME
            new_state["IsCurrent"] = 1

            available = _available_events(new_state, tier_keys)
            if not available:
                break
            event_names, weights = zip(*available)
            weights_arr = np.asarray(weights, dtype="float64")
            weights_arr = weights_arr / weights_arr.sum()
            chosen = event_names[int(rng.choice(len(event_names), p=weights_arr))]
            _apply_life_event(rng, new_state, chosen, geo_keys, tier_keys,
                              geo_lookup, _geo_cache=_geo_cache)

            current_state = new_state

        if _row_count < len(new_rows):
            new_rows[_row_count] = dict(current_state)
        else:
            new_rows.append(dict(current_state))
        _row_count += 1

    new_rows = new_rows[:_row_count]
    return pd.DataFrame(new_rows)


def generate_scd2_versions(
    rng: np.random.Generator,
    base_df: pd.DataFrame,
    cust_cfg,
    geo_keys: np.ndarray,
    tier_keys: np.ndarray,
    end_date: pd.Timestamp,
    geo_lookup: pd.DataFrame,
) -> pd.DataFrame:
    """
    Expand customer rows with SCD Type 2 version history.

    Selects a fraction of individual customers (change_rate), then delegates
    to expand_changed_customers() for the per-row life event simulation.
    """
    change_rate = float(getattr(cust_cfg, "change_rate", 0.15))
    max_versions = int(getattr(cust_cfg, "max_versions", 4))

    person_mask = base_df["CustomerType"] == "Individual"
    person_ids = base_df.loc[person_mask, "CustomerID"].to_numpy()

    if len(person_ids) == 0:
        return base_df

    n_change = max(1, int(len(person_ids) * change_rate))
    n_change = min(n_change, len(person_ids))
    change_id_set = set(
        rng.choice(person_ids, size=n_change, replace=False).tolist()
    )

    _change_mask = base_df["CustomerID"].isin(change_id_set)
    unchanged_df = base_df[~_change_mask]
    changed_df = base_df[_change_mask]

    expanded_rows_df = expand_changed_customers(
        rng=rng,
        changed_df=changed_df,
        max_versions=max_versions,
        geo_keys=geo_keys,
        tier_keys=tier_keys,
        end_date=end_date,
        geo_lookup=geo_lookup,
    )

    expanded_df = pd.concat(
        [unchanged_df, expanded_rows_df],
        ignore_index=True,
    )

    expanded_df["CustomerKey"] = np.arange(1, len(expanded_df) + 1, dtype="int64")

    n_total = len(expanded_df)
    n_base = len(base_df)
    n_versions = n_total - n_base
    info(f"SCD2: {n_change} customers expanded, {n_versions} version rows added ({n_total} total)")

    return expanded_df
