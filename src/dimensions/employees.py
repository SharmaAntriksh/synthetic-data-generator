from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.utils.logging_utils import info, skip, stage, warn
from src.utils.output_utils import write_parquet_with_date32
from src.versioning import should_regenerate, save_version
from src.utils.name_pools import (
    assign_person_names,
    load_people_pools,
    resolve_people_folder,
    hash_u64,
)
from src.utils.config_helpers import (
    as_dict,
    int_or,
    float_or,
    bool_or,
    parse_global_dates,
    rand_dates_between,
    region_from_iso_code,
)
from src.utils.config_precedence import resolve_seed
from src.defaults import (
    EMPLOYEE_PART_TIME_RATE_BY_ROLE,
    EMPLOYEE_PART_TIME_FTE_VALUES,
    EMPLOYEE_TERMINATION_REASON_LABELS,
    EMPLOYEE_TERMINATION_REASON_PROBS,
)

# EmployeeKey encoding scheme (shared with employee_store_assignments)
STORE_MGR_KEY_BASE: int = 30_000_000
STAFF_KEY_BASE: int = 40_000_000
STAFF_KEY_STORE_MULT: int = 1_000


_STAFF_TITLES = np.array(
    ["Sales Associate", "Cashier", "Stock Associate", "Customer Support", "Fulfillment Associate"],
    dtype=object,
)
_STAFF_TITLES_P = np.array([0.35, 0.25, 0.20, 0.10, 0.10], dtype=float)

# Validate at import time (per CLAUDE.md gotcha #10)
assert abs(float(_STAFF_TITLES_P.sum()) - 1.0) < 1e-9, (
    f"_STAFF_TITLES_P must sum to 1.0, got {_STAFF_TITLES_P.sum()}"
)


# ---------------------------------------------------------
# SA natural attrition constants (no config knobs)
# ---------------------------------------------------------
_SA_ANNUAL_ATTRITION_RATE: float = 0.12    # 12 % annual voluntary turnover
_SA_MIN_TENURE_DAYS: int = 180             # minimum days before an SA can leave
_SA_MAX_TENURE_DAYS: int = 2555            # ~7-year cap
_SA_REPLACEMENT_LEAD_DAYS: int = 14        # new hire overlaps departing employee


# ---------------------------------------------------------
# Internals
# ---------------------------------------------------------

def _stores_signature(stores: pd.DataFrame) -> Dict[str, Any]:
    """Version signature for stores — excludes EmployeeCount to avoid churn
    when run_employees updates stores.parquet with actual counts."""
    if stores.empty:
        return {"rows": 0, "min_store": None, "max_store": None}
    sk = stores["StoreKey"].to_numpy()
    return {
        "rows": int(len(stores)),
        "min_store": int(np.min(sk)),
        "max_store": int(np.max(sk)),
    }


def _parse_employee_dates(
    cfg: Dict[str, Any], emp_cfg: Dict[str, Any]
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Resolve the dataset-wide employee window.

    Uses ``defaults.dates.{start,end}`` exclusively.
    Legacy ``employees.start_date / end_date`` keys are ignored with a warning.
    """
    if emp_cfg.get("start_date", None) is not None or emp_cfg.get("end_date", None) is not None:
        warn(
            "employees.start_date / employees.end_date are IGNORED. "
            "Employee dates now follow defaults.dates exclusively. "
            "Remove these keys from config.yaml to silence this warning."
        )
    return parse_global_dates(
        cfg, emp_cfg,
        allow_override=False,
        dimension_name="employees",
    )


def _apply_deterministic_names(
    df: pd.DataFrame,
    seed: int,
    *,
    people_pools,
    iso_by_geo: dict[int, str] | None = None,
    default_region: str = "US",
) -> None:
    """
    Stable names per EmployeeKey using shared name pools.

    Assigns deterministic Gender (M/F/O) and region-aware first/last/middle names.
    """
    if people_pools is None:
        raise ValueError(
            "people_pools is required for employee name generation. "
            "Ensure name pool CSV files exist under the configured people folder."
        )

    ek = df["EmployeeKey"].astype(np.int32).to_numpy()
    ek_u64 = ek.astype(np.uint64)

    # Deterministic Gender distribution based on hash
    # Thresholds from defaults.py: EMPLOYEE_GENDER_PROBS
    from src.defaults import EMPLOYEE_GENDER_PROBS
    p_other, p_female = EMPLOYEE_GENDER_PROBS["other"], EMPLOYEE_GENDER_PROBS["female"]
    h = hash_u64(ek_u64, int(seed), 9101)
    u = (h % np.uint64(10_000)).astype(np.float64) / 10_000.0
    gender_code = np.where(u < p_other, "O", np.where(u < p_other + p_female, "F", "M")).astype(object)
    df["Gender"] = gender_code

    # Region per row from GeographyKey → ISOCode → region code
    if "GeographyKey" in df.columns and iso_by_geo:
        gk = pd.to_numeric(df["GeographyKey"], errors="coerce").fillna(-1).astype(np.int32).to_numpy()
        iso = np.array(
            [iso_by_geo.get(int(k), "") if k >= 0 else "" for k in gk],
            dtype=object,
        )
        region = np.array(
            [region_from_iso_code(x, default_region) if x else default_region for x in iso],
            dtype=object,
        )
    else:
        region = np.full(len(df), default_region, dtype=object)

    gender_label = np.where(
        gender_code == "M", "Male",
        np.where(gender_code == "F", "Female", "Other"),
    ).astype(object)

    first, last, mid = assign_person_names(
        keys=ek,
        region=region,
        gender=gender_label,
        is_org=np.zeros(len(df), dtype=bool),
        pools=people_pools,
        seed=int(seed),
        include_middle=True,
        default_region=default_region,
    )

    df["FirstName"] = pd.Series(first, dtype="object").astype(str)
    df["LastName"] = pd.Series(last, dtype="object").astype(str)
    df["MiddleName"] = pd.Series(mid, dtype="object").astype(str)
    df["EmployeeName"] = df["FirstName"] + " " + df["LastName"]


def _enrich_employee_hr_columns(
    df: pd.DataFrame,
    rng: np.random.Generator,
    global_end: pd.Timestamp,
    email_domain: str = "contoso.com",
) -> pd.DataFrame:
    """
    Adds Contoso-like HR columns.

    Assumes *df* already has: EmployeeKey, Title, OrgLevel, HireDate,
    TerminationDate, IsActive, and deterministic name columns.
    """
    n = len(df)
    if n == 0:
        return df

    hire = pd.to_datetime(df["HireDate"]).dt.normalize()
    org_level = df["OrgLevel"].astype(int).to_numpy()
    title = df["Title"].astype(str)

    # BirthDate: age-at-hire varies by level (staff younger, management older)
    age_mean = np.where(org_level >= 6, 27, np.where(org_level >= 5, 34, 42))
    ages = np.clip(rng.normal(loc=age_mean, scale=6.0, size=n), 18, 62).astype(int)
    birth_year = hire.dt.year.to_numpy() - ages
    birth_month = rng.integers(1, 13, size=n)
    dim = (
        pd.to_datetime({"year": birth_year, "month": birth_month, "day": np.ones(n, dtype=int)})
        .dt.days_in_month.to_numpy()
    )
    birth_day = np.minimum((rng.random(n) * dim).astype(int) + 1, dim)
    df["BirthDate"] = pd.to_datetime(
        {"year": birth_year, "month": birth_month, "day": birth_day}
    ).dt.normalize()

    is_married = rng.random(n) < np.clip((ages - 22) / 25.0, 0.05, 0.75)
    df["MaritalStatus"] = np.where(is_married, "M", "S").astype(object)

    # Email / Phone
    email_local = (
        df["FirstName"].str.lower().str.replace(" ", "", regex=False)
        + "."
        + df["LastName"].str.lower().str.replace(" ", "", regex=False)
        + "."
        + df["EmployeeKey"].astype("Int32").astype(str)
    )
    df["EmailAddress"] = (email_local + "@" + str(email_domain)).astype(str)

    phone_raw = rng.integers(0, 10, size=n * 10, dtype=np.uint8) + np.uint8(48)
    df["Phone"] = pd.Series(phone_raw.view("S10").astype("U10"), dtype="object")

    # Emergency contacts: pick a plausible name from the employee population
    if n > 1:
        pick = rng.integers(0, n - 1, size=n)
        self_idx = np.arange(n)
        # Shift picks >= self to avoid self-selection (uniform over n-1 others)
        pick = np.where(pick >= self_idx, pick + 1, pick)
        df["EmergencyContactName"] = (
            df["FirstName"].iloc[pick].to_numpy(dtype=object)
            + " "
            + df["LastName"].iloc[pick].to_numpy(dtype=object)
        )
    else:
        self_name = df["FirstName"].iloc[0] + " " + df["LastName"].iloc[0] + " (Self)"
        df["EmergencyContactName"] = pd.Series([self_name], dtype="object")
    ec_raw = rng.integers(0, 10, size=n * 10, dtype=np.uint8) + np.uint8(48)
    df["EmergencyContactPhone"] = pd.Series(ec_raw.view("S10").astype("U10"), dtype="object")

    # Compensation
    salaried = (df["OrgLevel"].astype(int) <= 5).to_numpy()
    df["SalariedFlag"] = salaried.astype(np.int8)
    df["PayFrequency"] = np.where(salaried, 1, 2).astype(np.int16)

    hourly_staff = np.clip(rng.normal(loc=18.0, scale=4.0, size=n), 10.0, 40.0)
    annual_salary = np.clip(rng.normal(loc=70000.0, scale=18000.0, size=n), 38000.0, 160000.0)
    hourly_equiv = annual_salary / 2080.0
    df["BaseRate"] = np.where(salaried, hourly_equiv, hourly_staff).round(2).astype(np.float64)

    # VacationHours: tenure-based
    tenure_days = (global_end.normalize() - hire).dt.days.clip(lower=0)
    base_vac = np.where(salaried, 80, 40) + (tenure_days / 365.0 * np.where(salaried, 6.0, 3.0))
    df["VacationHours"] = np.clip(
        base_vac + rng.normal(0, 10, size=n), 0, 240,
    ).round(0).astype(np.int16)

    # CurrentFlag / Status
    df["CurrentFlag"] = df["IsActive"].astype(np.int8)
    df["Status"] = np.where(
        df["IsActive"].astype(int) == 1, "Active", "Terminated",
    ).astype(object)

    # TerminationReason (only for terminated employees)
    term_mask = df["TerminationDate"].notna() & (df["IsActive"].astype(int) == 0)
    n_term = int(term_mask.sum())
    df["TerminationReason"] = pd.array([pd.NA] * len(df), dtype="object")
    if n_term > 0:
        reasons = rng.choice(
            EMPLOYEE_TERMINATION_REASON_LABELS,
            size=n_term,
            p=EMPLOYEE_TERMINATION_REASON_PROBS,
        )
        df.loc[term_mask, "TerminationReason"] = reasons

    # SalesPersonFlag
    df["SalesPersonFlag"] = title.isin(["Sales Associate"]).astype(np.int8)

    # DepartmentName
    dept = np.where(
        title.isin(["Sales Associate", "Store Manager"]),
        "Sales",
        np.where(
            title.isin(["Cashier"]),
            "Store Operations",
            np.where(
                title.isin(["Stock Associate"]),
                "Inventory",
                np.where(title.isin(["Fulfillment Associate"]), "Fulfillment", "Corporate"),
            ),
        ),
    )
    df["DepartmentName"] = pd.Series(dept, dtype="object")

    return df


def _finalize_employee_integer_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Force specific columns to integer types in parquet output.

    Power BI / Power Query sometimes infers decimal types when a column
    contains nulls.  ParentEmployeeKey stays nullable for DAX ``PATH()``
    semantics; other columns use 0 for corporate-level rows.
    """
    if df.empty:
        return df

    def _to_int(col: str, dtype) -> None:
        if col not in df.columns:
            return
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            s = s.astype(np.int8)
        s = pd.to_numeric(s, errors="coerce")
        df[col] = s.fillna(0).astype(dtype)

    _to_int("EmployeeKey", np.int32)
    if "ParentEmployeeKey" in df.columns:
        df["ParentEmployeeKey"] = pd.to_numeric(
            df["ParentEmployeeKey"], errors="coerce",
        ).astype("Int32")
    _to_int("OrgLevel", np.int16)
    _to_int("SalesPersonFlag", np.int8)
    _to_int("SalariedFlag", np.int8)
    _to_int("CurrentFlag", np.int8)
    _to_int("IsActive", np.int8)
    _to_int("RegionId", np.int16)
    _to_int("DistrictId", np.int16)
    _to_int("StoreKey", np.int32)
    _to_int("GeographyKey", np.int32)
    _to_int("PayFrequency", np.int16)

    return df


# ---------------------------------------------------------
# SA attrition chain generator
# ---------------------------------------------------------

def _generate_attrition_replacements(
    df: pd.DataFrame,
    *,
    rng: np.random.Generator,
    global_start: pd.Timestamp,
    global_end: pd.Timestamp,
    primary_sales_role: str = "Sales Associate",
    store_closing_dates: dict[int, pd.Timestamp] | None = None,
) -> pd.DataFrame:
    """
    For each Sales Associate, simulate a tenure chain: the original SA
    serves for a random tenure drawn from an exponential distribution,
    then departs and is replaced by a new hire (with a
    ``_SA_REPLACEMENT_LEAD_DAYS`` overlap).  The chain continues until
    the last SA's tenure reaches ``global_end``.

    Returns *df* with original SA rows mutated (TerminationDate set)
    plus appended replacement rows.
    """
    ps_role = str(primary_sales_role or "Sales Associate")
    ek_col = pd.to_numeric(df["EmployeeKey"], errors="coerce").fillna(0).astype(np.int64)
    sa_mask = (ek_col >= STAFF_KEY_BASE) & (df["Title"].astype(str) == ps_role)
    sa_idx = np.where(sa_mask.to_numpy())[0]

    if sa_idx.size == 0:
        return df

    # Mean tenure in days from annual attrition rate
    mean_tenure = 365.0 / _SA_ANNUAL_ATTRITION_RATE

    # Track highest within-store seq per store for key allocation
    # Current keys: STAFF_KEY_BASE + store * STAFF_KEY_STORE_MULT + seq
    # Vectorised store_max_seq extraction
    _ek_vals = ek_col.to_numpy()
    _staff_mask = _ek_vals >= STAFF_KEY_BASE
    _staff_keys = _ek_vals[_staff_mask]
    _sk_arr = (_staff_keys - STAFF_KEY_BASE) // STAFF_KEY_STORE_MULT
    _seq_arr = (_staff_keys - STAFF_KEY_BASE) % STAFF_KEY_STORE_MULT
    store_max_seq: dict[int, int] = {}
    _uniq_sk = np.unique(_sk_arr)
    for sk_val in _uniq_sk:
        sk_mask = _sk_arr == sk_val
        store_max_seq[int(sk_val)] = int(_seq_arr[sk_mask].max())

    replacement_rows: list[dict] = []
    term_date_col = df.columns.get_loc("TerminationDate")
    is_active_col = df.columns.get_loc("IsActive")

    # Per-store RNG spawn for determinism regardless of iteration order
    store_keys_sorted = sorted(set(
        ((_ek_vals[sa_idx] - STAFF_KEY_BASE) // STAFF_KEY_STORE_MULT).astype(int).tolist()
    ))
    store_rng_map: dict[int, np.random.Generator] = {}
    spawned = rng.spawn(len(store_keys_sorted))
    for sk, child_rng in zip(store_keys_sorted, spawned):
        store_rng_map[sk] = child_rng

    for i in sa_idx:
        ek_val = int(ek_col.iloc[i])
        store_key = int((ek_val - STAFF_KEY_BASE) // STAFF_KEY_STORE_MULT)
        hire_date = pd.to_datetime(df.iat[i, df.columns.get_loc("HireDate")]).normalize()
        store_rng = store_rng_map[store_key]

        # Determine the effective end for this store (global_end or closing date)
        store_end = global_end
        if store_closing_dates and store_key in store_closing_dates:
            cd = pd.to_datetime(store_closing_dates[store_key]).normalize()
            if global_start <= cd <= global_end:
                store_end = cd

        # Draw tenure for original SA
        tenure_days = int(np.clip(
            store_rng.exponential(mean_tenure),
            _SA_MIN_TENURE_DAYS,
            _SA_MAX_TENURE_DAYS,
        ))
        departure = (hire_date + pd.Timedelta(days=tenure_days)).normalize()

        if departure >= store_end:
            continue  # this SA lasts the whole window — no attrition

        # Set termination on original SA
        df.iat[i, term_date_col] = departure
        df.iat[i, is_active_col] = np.int8(0)
        # Attrition departures are voluntary by default
        if "TerminationReason" in df.columns:
            df.iat[i, df.columns.get_loc("TerminationReason")] = "Voluntary"

        # Build replacement chain
        current_departure = departure
        # Carry forward columns from the original SA row
        template = df.iloc[i].to_dict()

        while current_departure < store_end:
            next_seq = store_max_seq.get(store_key, 0) + 1
            if next_seq >= STAFF_KEY_STORE_MULT:
                raise OverflowError(
                    f"Store {store_key}: replacement employee key overflow "
                    f"(seq={next_seq} >= {STAFF_KEY_STORE_MULT}). "
                    f"Too many attrition replacements for this store."
                )
            store_max_seq[store_key] = next_seq

            new_ek = STAFF_KEY_BASE + store_key * STAFF_KEY_STORE_MULT + next_seq
            new_hire = (current_departure - pd.Timedelta(days=_SA_REPLACEMENT_LEAD_DAYS)).normalize()
            # Clamp hire date to not be before global_start - 5y (consistent with general logic)
            earliest_hire = global_start - pd.Timedelta(days=365 * 5)
            if new_hire < earliest_hire:
                new_hire = earliest_hire

            # Draw tenure for replacement
            rep_tenure = int(np.clip(
                store_rng.exponential(mean_tenure),
                _SA_MIN_TENURE_DAYS,
                _SA_MAX_TENURE_DAYS,
            ))
            rep_departure = (new_hire + pd.Timedelta(days=rep_tenure)).normalize()

            # Determine if this is the last in chain
            is_last = rep_departure >= store_end

            row = dict(template)
            row["EmployeeKey"] = np.int64(new_ek)
            row["ParentEmployeeKey"] = template["ParentEmployeeKey"]
            row["EmployeeName"] = ""  # will be assigned by name generator
            row["HireDate"] = new_hire
            row["TerminationDate"] = pd.NaT if is_last else rep_departure
            row["IsActive"] = np.int8(1) if is_last else np.int8(0)
            row["TerminationReason"] = pd.NA if is_last else "Voluntary"
            # Draw fresh EmploymentType/FTE for replacement
            _pt_rate = EMPLOYEE_PART_TIME_RATE_BY_ROLE.get(
                str(template.get("Title", "")), 0.10,
            )
            _is_pt = bool(store_rng.random() < _pt_rate)
            row["EmploymentType"] = "Part-Time" if _is_pt else "Full-Time"
            row["FTE"] = float(store_rng.choice(EMPLOYEE_PART_TIME_FTE_VALUES)) if _is_pt else 1.0

            replacement_rows.append(row)
            current_departure = rep_departure

            if is_last:
                break

    if not replacement_rows:
        return df

    rep_df = pd.DataFrame(replacement_rows)
    df = pd.concat([df, rep_df], ignore_index=True)
    info(f"SA attrition: generated {len(replacement_rows)} replacement employees")
    return df


# ---------------------------------------------------------
# Generator
# ---------------------------------------------------------

def generate_employee_dimension(
    *,
    stores: pd.DataFrame,
    seed: int,
    global_start: pd.Timestamp,
    global_end: pd.Timestamp,
    district_size: int = 10,
    districts_per_region: int = 8,
    max_staff_per_store: int = 5,
    termination_rate: float = 0.08,
    use_store_employee_count: bool = False,
    min_staff_per_store: int = 3,
    staff_scale: float = 0.25,
    people_pools=None,
    iso_by_geo: dict[int, str] | None = None,
    default_region: str = "US",
    primary_sales_role: str = "Sales Associate",
    min_primary_sales_per_store: int = 1,
    store_manager_names: dict[int, str] | None = None,
    store_opening_dates: dict[int, pd.Timestamp] | None = None,
    store_closing_dates: dict[int, pd.Timestamp] | None = None,
    store_close_reasons: dict[int, str] | None = None,
    transfer_share: float = 0.60,
    notice_days: int = 30,
) -> pd.DataFrame:
    """
    Build a parent-child employee hierarchy with stable keys.

    Sales Associate lifecycle:
      - ALL Sales Associates are hired at or before *global_start*.
      - SAs are subject to natural attrition: after a random tenure they
        depart and are replaced by a new hire (with overlap for training).
        The last SA in each chain serves until ``global_end`` (NaT).
      - This ensures the bridge table always has at least one SA per store
        covering any date in ``[global_start, global_end]``.
    """
    if stores.empty:
        raise ValueError("stores dataframe is empty; cannot generate employees")

    required_cols = {"StoreKey", "GeographyKey", "EmployeeCount", "StoreType"}
    missing = [c for c in required_cols if c not in stores.columns]
    if missing:
        raise ValueError(f"stores.parquet missing required columns: {missing}")

    stores = stores.copy()
    stores["StoreKey"] = stores["StoreKey"].astype(np.int32)

    max_staff_per_store = max(0, int_or(max_staff_per_store, 5))
    min_staff_per_store = max(0, int_or(min_staff_per_store, 3))
    if max_staff_per_store > 0:
        min_staff_per_store = min(min_staff_per_store, max_staff_per_store)
    else:
        min_staff_per_store = 0
    staff_scale = float(np.clip(float_or(staff_scale, 0.25), 0.0, 1.0))
    termination_rate = float(np.clip(float_or(termination_rate, 0.08), 0.0, 1.0))
    rng = np.random.default_rng(int(seed))

    n_stores = len(stores)

    # ----- Hierarchy: prefer stores.parquet columns (single source of truth) -----
    has_store_hierarchy = (
        "StoreDistrict" in stores.columns and "StoreRegion" in stores.columns
    )

    if has_store_hierarchy:
        district_id = (
            stores["StoreDistrict"].astype(str)
            .str.extract(r"(\d+)", expand=False)
            .astype(np.int16)
            .to_numpy()
        )
        region_id = (
            stores["StoreRegion"].astype(str)
            .str.extract(r"(\d+)", expand=False)
            .astype(np.int16)
            .to_numpy()
        )
        stores = stores.drop(
            columns=["StoreDistrict", "StoreRegion", "StoreZone"],
            errors="ignore",
        )
    else:
        # Legacy fallback: compute hierarchy when stores.parquet lacks columns
        warn(
            "stores.parquet missing StoreDistrict/StoreRegion columns; "
            "computing employee hierarchy independently. "
            "This may produce inconsistent hierarchies. "
            "Run --regen-dimensions all to fix."
        )
        sort_cols = []
        has_continent = "Continent" in stores.columns
        if has_continent:
            sort_cols.append("Continent")
        if "Country" in stores.columns:
            sort_cols.append("Country")
        sort_cols.append("StoreKey")
        stores = stores.sort_values(sort_cols).reset_index(drop=True)

        district_size = max(1, int_or(district_size, 10))
        districts_per_region = max(1, int_or(districts_per_region, 8))

        if has_continent:
            district_id = np.zeros(n_stores, dtype=np.int16)
            next_did = 1
            for _, grp_idx in stores.groupby("Continent", sort=False):
                idx = grp_idx.index.to_numpy()
                n_grp = len(idx)
                local_did = np.arange(n_grp) // district_size
                district_id[idx] = (local_did + next_did).astype(np.int16)
                next_did += int(local_did.max()) + 1
        else:
            district_id = (np.arange(n_stores) // district_size + 1).astype(np.int16)

        region_id = ((district_id - 1) // districts_per_region + 1).astype(np.int16)
        stores = stores.drop(columns=["Continent", "Country"], errors="ignore")

    stores["DistrictId"] = district_id
    stores["RegionId"] = region_id

    # --- Key-encoding helpers using module constants ---
    CEO_KEY = np.int32(1)
    VP_OPS_KEY = np.int32(2)

    def _region_mgr_key(rid: int) -> np.int32:
        return np.int32(10_000 + int(rid))

    def _district_mgr_key(did: int) -> np.int32:
        return np.int32(20_000 + int(did))

    # ---------------------------------------------------------------
    # Build corporate / region / district tiers (small — loop is fine)
    # ---------------------------------------------------------------
    rows: list[dict] = []

    rows.append(dict(
        EmployeeKey=CEO_KEY,
        ParentEmployeeKey=pd.NA,
        EmployeeName="",
        Title="Chief Executive Officer",
        OrgLevel=np.int16(1),
        OrgUnitType="Corporate",
        RegionId=pd.NA, DistrictId=pd.NA,
        StoreKey=pd.NA, GeographyKey=pd.NA,
    ))
    rows.append(dict(
        EmployeeKey=VP_OPS_KEY,
        ParentEmployeeKey=CEO_KEY,
        EmployeeName="",
        Title="VP Operations",
        OrgLevel=np.int16(2),
        OrgUnitType="Corporate",
        RegionId=pd.NA, DistrictId=pd.NA,
        StoreKey=pd.NA, GeographyKey=pd.NA,
    ))

    unique_regions = np.unique(region_id)
    for rid in unique_regions:
        rows.append(dict(
            EmployeeKey=_region_mgr_key(int(rid)),
            ParentEmployeeKey=VP_OPS_KEY,
            EmployeeName="",
            Title="Regional Manager",
            OrgLevel=np.int16(3),
            OrgUnitType="Region",
            RegionId=np.int16(rid),
            DistrictId=pd.NA,
            StoreKey=pd.NA, GeographyKey=pd.NA,
        ))

    unique_districts = np.unique(district_id)
    for did in unique_districts:
        rid = int(((did - 1) // districts_per_region) + 1)
        rows.append(dict(
            EmployeeKey=_district_mgr_key(int(did)),
            ParentEmployeeKey=_region_mgr_key(rid),
            EmployeeName="",
            Title="District Manager",
            OrgLevel=np.int16(4),
            OrgUnitType="District",
            RegionId=np.int16(rid),
            DistrictId=np.int16(did),
            StoreKey=pd.NA, GeographyKey=pd.NA,
        ))

    corporate_df = pd.DataFrame(rows)

    # ---------------------------------------------------------------
    # Store managers — vectorized
    # ---------------------------------------------------------------
    sk_arr = stores["StoreKey"].to_numpy(dtype=np.int32)
    did_arr = stores["DistrictId"].to_numpy(dtype=np.int16)
    rid_arr = stores["RegionId"].to_numpy(dtype=np.int16)
    gk_arr = stores["GeographyKey"].to_numpy(dtype=np.int32)

    mgr_parent_keys = np.array(
        [_district_mgr_key(int(d)) for d in did_arr], dtype=np.int32,
    )

    mgr_df = pd.DataFrame({
        "EmployeeKey": (STORE_MGR_KEY_BASE + sk_arr).astype(np.int64),
        "ParentEmployeeKey": mgr_parent_keys,
        "EmployeeName": "",
        "Title": "Store Manager",
        "OrgLevel": np.int16(5),
        "OrgUnitType": "Store",
        "RegionId": rid_arr,
        "DistrictId": did_arr,
        "StoreKey": sk_arr,
        "GeographyKey": gk_arr,
    })

    # ---------------------------------------------------------------
    # Staff counts
    # ---------------------------------------------------------------
    if max_staff_per_store <= 0:
        staff_counts = np.zeros(n_stores, dtype=np.int64)
    elif use_store_employee_count:
        emp_counts = stores["EmployeeCount"].fillna(0).astype(np.int64).to_numpy()
        base = np.maximum(0, emp_counts - 1)
        scaled = np.rint(base.astype(np.float64) * staff_scale).astype(np.int64)
        staff_counts = np.clip(scaled, 0, max_staff_per_store).astype(np.int64)
        has_any = base > 0
        staff_counts = np.where(
            has_any, np.maximum(staff_counts, min_staff_per_store), 0,
        ).astype(np.int64)
    else:
        staff_counts = rng.integers(
            min_staff_per_store, max_staff_per_store + 1,
            size=n_stores, dtype=np.int64,
        )

    # ---------------------------------------------------------------
    # Staff rows — vectorized via np.repeat
    # ---------------------------------------------------------------
    total_staff = int(staff_counts.sum())

    if total_staff > 0:
        store_indices = np.repeat(np.arange(n_stores), staff_counts)
        staff_sk = sk_arr[store_indices]
        staff_did = did_arr[store_indices]
        staff_rid = rid_arr[store_indices]
        staff_gk = gk_arr[store_indices]

        # Per-employee index within each store (1-based)
        # Uses cumsum trick: start with 1s, subtract (prev_count) at each
        # store boundary so cumsum resets to 1 for the next store.
        within_store_idx = np.ones(total_staff, dtype=np.int32)
        offsets = np.cumsum(staff_counts)[:-1]
        if offsets.size > 0:
            np.subtract.at(within_store_idx, offsets, staff_counts[:-1])
        within_store_idx = np.cumsum(within_store_idx)

        max_ek = int(STAFF_KEY_BASE) + int(staff_sk.max()) * int(STAFF_KEY_STORE_MULT) + int(within_store_idx.max())
        if max_ek > np.iinfo(np.int64).max:
            raise OverflowError(f"EmployeeKey would overflow int64: {max_ek}")
        staff_ek = (STAFF_KEY_BASE + staff_sk * STAFF_KEY_STORE_MULT + within_store_idx).astype(np.int64)
        staff_parent = (STORE_MGR_KEY_BASE + staff_sk).astype(np.int64)

        # Sample titles in bulk, then overwrite first k per store with primary sales role
        all_titles = rng.choice(_STAFF_TITLES, size=total_staff, p=_STAFF_TITLES_P).astype(object)
        ps_role = str(primary_sales_role or "Sales Associate")
        k_ps = max(1, int_or(min_primary_sales_per_store, 1))

        # Mark the first k_ps employees of each store as the primary sales role
        k_per_store = np.minimum(staff_counts, k_ps)
        shortfall_stores = int((staff_counts < k_ps).sum())
        if shortfall_stores > 0:
            warn(
                f"{shortfall_stores} store(s) have fewer staff than "
                f"min_primary_sales_per_store={k_ps}; they will have "
                f"fewer '{ps_role}' employees than requested."
            )
        k_total = int(k_per_store.sum())
        if k_total > 0:
            # Build mask of positions that should be primary sales role
            ps_mask = np.zeros(total_staff, dtype=bool)
            pos = 0
            for i in range(n_stores):
                sc = int(staff_counts[i])
                kk = int(k_per_store[i])
                if kk > 0:
                    ps_mask[pos:pos + kk] = True
                pos += sc
            all_titles[ps_mask] = ps_role

        staff_df = pd.DataFrame({
            "EmployeeKey": staff_ek,
            "ParentEmployeeKey": staff_parent,
            "EmployeeName": "",
            "Title": pd.Series(all_titles, dtype="object"),
            "OrgLevel": np.int16(6),
            "OrgUnitType": "Store",
            "RegionId": staff_rid,
            "DistrictId": staff_did,
            "StoreKey": staff_sk,
            "GeographyKey": staff_gk,
        })
    else:
        staff_df = pd.DataFrame(
            columns=corporate_df.columns,
        ).iloc[:0]

    df = pd.concat([corporate_df, mgr_df, staff_df], ignore_index=True)

    # ------------------------------------------------------------------
    # EmploymentType & FTE — determined at hire based on role
    # ------------------------------------------------------------------
    n_all = len(df)
    titles_np = df["Title"].astype(str).to_numpy()
    pt_prob = np.array(
        [EMPLOYEE_PART_TIME_RATE_BY_ROLE.get(t, 0.10) for t in titles_np],
        dtype=np.float64,
    )
    is_part_time = rng.random(n_all) < pt_prob
    df["EmploymentType"] = np.where(is_part_time, "Part-Time", "Full-Time").astype(object)
    df["FTE"] = np.where(
        is_part_time,
        rng.choice(EMPLOYEE_PART_TIME_FTE_VALUES, size=n_all),
        1.0,
    ).astype(np.float64)

    # ------------------------------------------------------------------
    # Dates — with Sales Associate full-window guarantee
    # ------------------------------------------------------------------
    n = len(df)
    ps_role_str = str(primary_sales_role or "Sales Associate")

    ek_all = pd.to_numeric(df["EmployeeKey"], errors="coerce").fillna(0).astype(np.int32)
    is_sales_associate = (ek_all >= STAFF_KEY_BASE) & (df["Title"].astype(str) == ps_role_str)
    sa_mask_np = is_sales_associate.to_numpy()

    # Hire dates: SAs hired before their store opens (or dataset start);
    # everyone else random within the general window.
    hire_start_general = global_start - pd.Timedelta(days=365 * 5)
    hire_dates = rand_dates_between(rng, hire_start_general, global_end, n)

    n_sa = int(sa_mask_np.sum())
    if n_sa > 0:
        if store_opening_dates:
            # Per-SA upper bound: min(global_start, store_opening_date)
            sa_store_keys = pd.to_numeric(
                df.loc[sa_mask_np, "StoreKey"], errors="coerce"
            ).fillna(0).astype(np.int32).to_numpy()
            sa_upper = np.array([
                min(global_start, store_opening_dates.get(int(sk), global_start))
                for sk in sa_store_keys
            ], dtype="datetime64[ns]")
            # Clamp: upper must be >= hire_start_general
            sa_lower = np.full(n_sa, hire_start_general, dtype="datetime64[ns]")
            sa_upper = np.maximum(sa_upper, sa_lower + np.timedelta64(1, "D"))
            # Vectorized random hire dates per-SA
            lo_i = sa_lower.astype("int64")
            hi_i = sa_upper.astype("int64")
            sa_hire_i = rng.integers(lo_i, hi_i + 1, dtype=np.int64)
            hire_dates.iloc[sa_mask_np] = pd.to_datetime(
                sa_hire_i, unit="ns"
            ).normalize().to_numpy()
        else:
            sa_hire = rand_dates_between(rng, hire_start_general, global_start, n_sa)
            hire_dates.iloc[sa_mask_np] = sa_hire.to_numpy()

    # Store managers: hire before their store opens
    if store_opening_dates:
        mgr_mask_np = (df["Title"].astype(str) == "Store Manager").to_numpy()
        n_mgr = int(mgr_mask_np.sum())
        if n_mgr > 0:
            mgr_store_keys = pd.to_numeric(
                df.loc[mgr_mask_np, "StoreKey"], errors="coerce"
            ).fillna(0).astype(np.int32).to_numpy()
            mgr_upper = np.array([
                min(global_end, store_opening_dates.get(int(sk), global_end))
                for sk in mgr_store_keys
            ], dtype="datetime64[ns]")
            mgr_lower = np.full(n_mgr, hire_start_general, dtype="datetime64[ns]")
            mgr_upper = np.maximum(mgr_upper, mgr_lower + np.timedelta64(1, "D"))
            lo_i = mgr_lower.astype("int64")
            hi_i = mgr_upper.astype("int64")
            mgr_hire_i = rng.integers(lo_i, hi_i + 1, dtype=np.int64)
            hire_dates.iloc[mgr_mask_np] = pd.to_datetime(
                mgr_hire_i, unit="ns"
            ).normalize().to_numpy()

    df["HireDate"] = hire_dates

    # Terminations: SAs skip probabilistic termination (attrition handles them);
    # others probabilistic (reduced for senior levels)
    base_p = termination_rate
    level = df["OrgLevel"].astype(np.int16).to_numpy()
    p = np.where(level <= 4, base_p * 0.25, base_p)
    p[sa_mask_np] = 0.0
    term_mask = rng.random(n) < p

    term_dates = pd.Series([pd.NaT] * n, dtype="datetime64[ns]")
    idx = np.where(term_mask)[0]
    if idx.size > 0:
        hire_i = pd.to_datetime(df.loc[idx, "HireDate"]).dt.normalize()
        max_days = (pd.to_datetime(global_end).normalize() - hire_i).dt.days.to_numpy()
        max_days = np.clip(max_days, 0, None)
        # Enforce minimum 30-day tenure to avoid zero-day terminations
        min_tenure_days = 30
        offs = np.clip(
            (rng.random(idx.size) * (max_days + 1)).astype(np.int64),
            np.minimum(min_tenure_days, max_days),
            max_days,
        )
        term_dates.iloc[idx] = (hire_i + pd.to_timedelta(offs, unit="D")).to_numpy(dtype="datetime64[ns]")

    df["TerminationDate"] = term_dates
    df["IsActive"] = (
        df["TerminationDate"].isna() | (df["TerminationDate"] > global_end)
    ).astype(np.int8)

    # ------------------------------------------------------------------
    # SA natural attrition: tenure chains with replacements
    # ------------------------------------------------------------------
    df = _generate_attrition_replacements(
        df,
        rng=np.random.default_rng(int(seed) ^ 0xA7721710),
        global_start=global_start,
        global_end=global_end,
        primary_sales_role=ps_role_str,
        store_closing_dates=store_closing_dates,
    )
    n = len(df)  # update after attrition may have added rows

    # ------------------------------------------------------------------
    # Store-closure: terminate or mark for transfer
    # ------------------------------------------------------------------
    # TransferStatus: None = normal, "Transferred" = relocating, "Terminated_StoreClose" = let go
    df["TransferStatus"] = None
    df["TransferDate"] = pd.NaT
    df["OriginalStoreKey"] = pd.array([pd.NA] * n, dtype="Int32")

    if store_closing_dates:
        from src.defaults import STORE_CLOSE_TRANSFER_SHARE_BY_REASON

        ek_all_np = pd.to_numeric(df["EmployeeKey"], errors="coerce").fillna(0).astype(np.int32).to_numpy()
        sk_all_np = pd.to_numeric(df["StoreKey"], errors="coerce").fillna(0).astype(np.int32).to_numpy()

        for close_sk, close_date in store_closing_dates.items():
            close_date = pd.to_datetime(close_date).normalize()
            if close_date < global_start or close_date > global_end:
                continue

            # Find store employees (managers + staff)
            store_mask = sk_all_np == int(close_sk)
            # Skip already-terminated employees
            already_terminated = df["TerminationDate"].notna() & (df["TerminationDate"] <= close_date)
            eligible = store_mask & ~already_terminated.to_numpy()
            eligible_idx = np.where(eligible)[0]

            if eligible_idx.size == 0:
                continue

            # Determine per-store transfer share (varies by close reason)
            reason = (store_close_reasons or {}).get(int(close_sk), "")
            effective_share = STORE_CLOSE_TRANSFER_SHARE_BY_REASON.get(reason, transfer_share)

            # Decide who transfers vs who gets terminated
            n_eligible = len(eligible_idx)
            n_transfer = max(0, int(round(n_eligible * effective_share)))

            # Sales Associates always transfer (never terminated)
            titles_eligible = df.iloc[eligible_idx]["Title"].astype(str).to_numpy()
            sa_mask_eligible = titles_eligible == ps_role_str
            sa_idx = eligible_idx[sa_mask_eligible]
            non_sa_idx = eligible_idx[~sa_mask_eligible]

            # All SAs transfer; fill remaining transfer slots from non-SAs
            n_transfer_non_sa = max(0, n_transfer - len(sa_idx))
            if n_transfer_non_sa > 0 and len(non_sa_idx) > 0:
                transfer_non_sa = rng.choice(
                    non_sa_idx,
                    size=min(n_transfer_non_sa, len(non_sa_idx)),
                    replace=False,
                )
            else:
                transfer_non_sa = np.array([], dtype=np.intp)

            all_transfer_idx = np.concatenate([sa_idx, transfer_non_sa]).astype(np.intp)
            all_terminate_idx = np.setdiff1d(eligible_idx, all_transfer_idx)

            # Staggered transfer: spread over notice period
            # Non-SAs transfer first, SAs transfer last (ensures sales coverage until closing)
            if all_transfer_idx.size > 0:
                # Split into non-SA and SA groups, sort each by EmployeeKey
                titles_transfer = df.iloc[all_transfer_idx]["Title"].astype(str).to_numpy()
                is_sa = titles_transfer == ps_role_str
                non_sa_transfer = all_transfer_idx[~is_sa]
                sa_transfer = all_transfer_idx[is_sa]
                # Sort each group by EmployeeKey for determinism
                non_sa_transfer = non_sa_transfer[np.argsort(ek_all_np[non_sa_transfer])]
                sa_transfer = sa_transfer[np.argsort(ek_all_np[sa_transfer])]
                # Non-SAs go first, SAs go last — last SA leaves on closing date
                sorted_transfer = np.concatenate([non_sa_transfer, sa_transfer])

                n_t = len(sorted_transfer)
                # Spread transfer dates from (close_date - notice_days) to close_date
                notice_start = close_date - pd.Timedelta(days=max(1, notice_days))
                offsets = np.linspace(0, notice_days, n_t, dtype=np.int64)
                for j, idx in enumerate(sorted_transfer):
                    t_date = (notice_start + pd.Timedelta(days=int(offsets[j]))).normalize()
                    df.iat[idx, df.columns.get_loc("TransferStatus")] = "Transferred"
                    df.iat[idx, df.columns.get_loc("TransferDate")] = t_date
                    df.iat[idx, df.columns.get_loc("OriginalStoreKey")] = int(close_sk)

            # Staggered termination: spread over notice period too
            if all_terminate_idx.size > 0:
                n_term = len(all_terminate_idx)
                notice_start = close_date - pd.Timedelta(days=max(1, notice_days))
                term_offsets = rng.integers(0, notice_days + 1, size=n_term)
                for j, idx in enumerate(all_terminate_idx):
                    t_date = (notice_start + pd.Timedelta(days=int(term_offsets[j]))).normalize()
                    df.iat[idx, df.columns.get_loc("TerminationDate")] = t_date
                    df.iat[idx, df.columns.get_loc("IsActive")] = np.int8(0)
                    df.iat[idx, df.columns.get_loc("TransferStatus")] = "Terminated_StoreClose"
                    df.iat[idx, df.columns.get_loc("OriginalStoreKey")] = int(close_sk)
                    if "TerminationReason" in df.columns:
                        df.iat[idx, df.columns.get_loc("TerminationReason")] = "Involuntary"

    # Names
    _apply_deterministic_names(
        df,
        seed=int(seed),
        people_pools=people_pools,
        iso_by_geo=iso_by_geo,
        default_region=default_region,
    )

    # Override Store Manager names to match stores.parquet (single source of truth)
    if store_manager_names:
        mgr_mask = df["Title"].astype(str) == "Store Manager"
        if mgr_mask.any():
            mgr_ek = df.loc[mgr_mask, "EmployeeKey"]
            mgr_ek_i32 = pd.to_numeric(mgr_ek, errors="coerce").fillna(0).astype(np.int32)
            mgr_sk = (mgr_ek_i32 - STORE_MGR_KEY_BASE).astype(np.int32)
            names = mgr_sk.map(
                lambda sk: store_manager_names.get(int(sk), "")
            )
            valid = (names != "") & names.notna()
            if valid.any():
                vi = names.index[valid]
                vn = names[valid].to_numpy(dtype=object)
                # Parse "First Middle... Last" — last token is last name,
                # first token is first name, everything in between is middle.
                first_arr = np.empty(len(vn), dtype=object)
                middle_arr = np.empty(len(vn), dtype=object)
                last_arr = np.empty(len(vn), dtype=object)
                for j, full_name in enumerate(vn):
                    parts = str(full_name).split()
                    if len(parts) >= 3:
                        first_arr[j] = parts[0]
                        last_arr[j] = parts[-1]
                        middle_arr[j] = " ".join(parts[1:-1])
                    elif len(parts) == 2:
                        first_arr[j] = parts[0]
                        last_arr[j] = parts[1]
                        middle_arr[j] = ""
                    else:
                        first_arr[j] = str(full_name)
                        last_arr[j] = ""
                        middle_arr[j] = ""
                df.loc[vi, "FirstName"] = first_arr
                df.loc[vi, "LastName"] = last_arr
                df.loc[vi, "MiddleName"] = middle_arr
                df.loc[vi, "EmployeeName"] = vn

    # Final integer casts (single consolidated pass)
    df["EmployeeKey"] = pd.to_numeric(df["EmployeeKey"], errors="coerce").fillna(0).astype(np.int32)
    df["ParentEmployeeKey"] = pd.to_numeric(df["ParentEmployeeKey"], errors="coerce").astype("Int32")
    df["OrgLevel"] = pd.to_numeric(df["OrgLevel"], errors="coerce").fillna(0).astype(np.int16)
    df["RegionId"] = pd.to_numeric(df["RegionId"], errors="coerce").fillna(0).astype(np.int16)
    df["DistrictId"] = pd.to_numeric(df["DistrictId"], errors="coerce").fillna(0).astype(np.int16)
    df["StoreKey"] = pd.to_numeric(df["StoreKey"], errors="coerce").fillna(0).astype(np.int32)
    df["GeographyKey"] = pd.to_numeric(df["GeographyKey"], errors="coerce").fillna(0).astype(np.int32)

    return df


# ---------------------------------------------------------
# EmployeeCount sync — update stores.parquet after generation
# ---------------------------------------------------------

def _count_employees_per_store(df: pd.DataFrame) -> dict[int, int]:
    """Count employees per store from EmployeeKey encoding."""
    ek = pd.to_numeric(df["EmployeeKey"], errors="coerce").fillna(0).astype(np.int32).to_numpy()
    store_keys = np.full(len(ek), -1, dtype=np.int32)

    mgr_mask = (ek >= STORE_MGR_KEY_BASE) & (ek < STAFF_KEY_BASE)
    store_keys[mgr_mask] = ek[mgr_mask] - STORE_MGR_KEY_BASE

    staff_mask = ek >= STAFF_KEY_BASE
    store_keys[staff_mask] = (ek[staff_mask] - STAFF_KEY_BASE) // STAFF_KEY_STORE_MULT

    valid = store_keys >= 0
    if not valid.any():
        return {}
    unique, counts = np.unique(store_keys[valid], return_counts=True)
    return dict(zip(unique.astype(int).tolist(), counts.astype(int).tolist()))


def _sync_stores_employee_count(emp_df: pd.DataFrame, stores_path: Path) -> None:
    """Update stores.parquet EmployeeCount with actual employee counts."""
    import re as _re

    actual = _count_employees_per_store(emp_df)
    if not actual:
        return

    stores_full = pd.read_parquet(stores_path)
    new_counts = stores_full["StoreKey"].map(
        lambda sk: actual.get(int(sk), 0)
    ).astype(np.int64)

    if stores_full["EmployeeCount"].astype(np.int64).equals(new_counts):
        return  # already accurate

    stores_full["EmployeeCount"] = new_counts

    # Patch "headcount <N>" in StoreDescription to match actual counts
    if "StoreDescription" in stores_full.columns:
        desc = stores_full["StoreDescription"].astype(str).to_numpy(dtype=object)
        sk_arr = stores_full["StoreKey"].to_numpy()
        headcount_map = {int(sk): str(cnt) for sk, cnt in actual.items()}
        for i in range(len(stores_full)):
            sk_val = int(sk_arr[i])
            cnt = headcount_map.get(sk_val, "0")
            desc[i] = _re.sub(r"(headcount )\d+", rf"\g<1>{cnt}", str(desc[i]))
        stores_full["StoreDescription"] = desc

    write_parquet_with_date32(
        stores_full,
        stores_path,
        date_cols=["OpeningDate", "ClosingDate"],
        cast_all_datetime=False,
        compression="snappy",
        compression_level=None,
        force_date32=True,
    )
    info("Updated stores.parquet EmployeeCount with actual employee counts.")


# ---------------------------------------------------------
# Pipeline entrypoint
# ---------------------------------------------------------

def run_employees(cfg: Dict[str, Any], parquet_folder: Path) -> None:
    cfg = cfg or {}
    emp_cfg = cfg.employees

    parquet_folder = Path(parquet_folder)
    parquet_folder.mkdir(parents=True, exist_ok=True)

    stores_path = parquet_folder / "stores.parquet"
    out_path = parquet_folder / "employees.parquet"

    if not stores_path.exists():
        raise FileNotFoundError(f"Missing stores parquet: {stores_path}")

    seed = resolve_seed(cfg, dict(emp_cfg), fallback=42)
    global_start, global_end = _parse_employee_dates(cfg, dict(emp_cfg))

    _STORES_READ_COLS = [
        "StoreKey", "GeographyKey", "EmployeeCount", "StoreType",
        "StoreDistrict", "StoreRegion", "StoreManager", "OpeningDate",
        "ClosingDate", "CloseReason",
    ]
    try:
        stores = pd.read_parquet(stores_path, columns=_STORES_READ_COLS)
    except (KeyError, ValueError):
        # Legacy stores.parquet may lack hierarchy/manager columns
        stores = pd.read_parquet(
            stores_path,
            columns=["StoreKey", "GeographyKey", "EmployeeCount", "StoreType"],
        )

    version_cfg = as_dict(emp_cfg)
    version_cfg["schema_version"] = 7
    version_cfg["_stores_sig"] = _stores_signature(stores)
    version_cfg["_stores_cfg"] = as_dict(cfg.stores)
    version_cfg["_global_dates"] = {
        "start": str(global_start.date()),
        "end": str(global_end.date()),
    }

    if not should_regenerate("employees", version_cfg, out_path):
        skip("Employees up-to-date")
        return

    people_folder = resolve_people_folder(cfg)
    pf = Path(people_folder)

    enable_asia = (
        (pf / "asia_male_first.csv").exists()
        and (pf / "asia_female_first.csv").exists()
        and (pf / "asia_last.csv").exists()
    )
    people_pools = load_people_pools(
        people_folder, enable_asia=enable_asia, legacy_support=False,
    )

    iso_by_geo: dict[int, str] = {}
    has_store_hierarchy = (
        "StoreDistrict" in stores.columns and "StoreRegion" in stores.columns
    )
    geo_path = parquet_folder / "geography.parquet"
    if geo_path.exists():
        geo_df = pd.read_parquet(geo_path)
        gk = pd.to_numeric(geo_df["GeographyKey"], errors="coerce").dropna().astype(np.int32).to_numpy()
        iso = geo_df.loc[geo_df["GeographyKey"].notna(), "ISOCode"].astype(str).to_numpy()
        iso_by_geo = dict(zip(gk, iso))

        # Merge Continent/Country only when stores lacks hierarchy columns (legacy fallback)
        if not has_store_hierarchy:
            if "Continent" in geo_df.columns and "Country" in geo_df.columns:
                geo_sort = geo_df[["GeographyKey", "Continent", "Country"]].drop_duplicates("GeographyKey").copy()
                geo_sort["GeographyKey"] = pd.to_numeric(geo_sort["GeographyKey"], errors="coerce").astype(np.int32)
                stores = stores.merge(
                    geo_sort, on="GeographyKey", how="left",
                )

    # Build StoreKey → StoreManager name mapping (source of truth for manager names)
    store_manager_names: dict[int, str] | None = None
    if "StoreManager" in stores.columns:
        _sk = stores["StoreKey"].astype(np.int32).to_numpy()
        _nm = stores["StoreManager"].astype(str).to_numpy()
        store_manager_names = dict(zip(_sk.tolist(), _nm.tolist()))
        stores = stores.drop(columns=["StoreManager"], errors="ignore")

    # Build StoreKey → OpeningDate mapping for hire date clamping
    store_opening_dates: dict[int, pd.Timestamp] | None = None
    if "OpeningDate" in stores.columns:
        _od = pd.to_datetime(stores["OpeningDate"], errors="coerce").dt.normalize()
        _sk_od = stores["StoreKey"].astype(np.int32).to_numpy()
        store_opening_dates = {
            int(sk): ts for sk, ts in zip(_sk_od, _od) if pd.notna(ts)
        }
        stores = stores.drop(columns=["OpeningDate"], errors="ignore")

    # Build StoreKey → ClosingDate and CloseReason mappings for store-closure logic
    store_closing_dates: dict[int, pd.Timestamp] | None = None
    store_close_reasons: dict[int, str] | None = None
    if "ClosingDate" in stores.columns:
        _cd = pd.to_datetime(stores["ClosingDate"], errors="coerce").dt.normalize()
        _sk_cd = stores["StoreKey"].astype(np.int32).to_numpy()
        store_closing_dates = {
            int(sk): ts for sk, ts in zip(_sk_cd, _cd) if pd.notna(ts)
        }
        if "CloseReason" in stores.columns:
            _cr = stores["CloseReason"].astype(str).to_numpy()
            store_close_reasons = {
                int(sk): str(cr) for sk, cr, cd in zip(_sk_cd, _cr, _cd) if pd.notna(cd)
            }
        stores = stores.drop(columns=["ClosingDate", "CloseReason"], errors="ignore")

    with stage("Generating Employees"):
        sa_cfg = emp_cfg.store_assignments
        primary_sales_role = str(sa_cfg.primary_sales_role or "Sales Associate")
        min_primary_sales_per_store = sa_cfg.min_primary_sales_per_store
        ensure_store_sales_coverage = sa_cfg.ensure_store_sales_coverage

        # Resolve store closing config for employee fate
        closing_cfg = as_dict(cfg.stores.closing) if hasattr(cfg.stores, "closing") and cfg.stores.closing is not None else {}
        _transfer_share = float_or(closing_cfg.get("transfer_share"), 0.60)
        _notice_days = int_or(closing_cfg.get("notice_days"), 30)

        df = generate_employee_dimension(
            stores=stores,
            seed=seed,
            global_start=global_start,
            global_end=global_end,
            district_size=emp_cfg.district_size,
            districts_per_region=emp_cfg.districts_per_region,
            max_staff_per_store=emp_cfg.max_staff_per_store,
            termination_rate=emp_cfg.termination_rate,
            use_store_employee_count=emp_cfg.use_store_employee_count,
            min_staff_per_store=emp_cfg.min_staff_per_store,
            staff_scale=emp_cfg.staff_scale,
            people_pools=people_pools,
            iso_by_geo=iso_by_geo,
            default_region="US",
            primary_sales_role=primary_sales_role,
            min_primary_sales_per_store=min_primary_sales_per_store,
            store_manager_names=store_manager_names,
            store_opening_dates=store_opening_dates,
            store_closing_dates=store_closing_dates,
            store_close_reasons=store_close_reasons,
            transfer_share=_transfer_share,
            notice_days=_notice_days,
        )

        hr_cfg = emp_cfg.hr
        email_domain = hr_cfg.email_domain

        df = _enrich_employee_hr_columns(
            df,
            rng=np.random.default_rng(int(seed) ^ 0x9E3779B1),
            global_end=global_end,
            email_domain=str(email_domain),
        )

        df = _finalize_employee_integer_cols(df)

        # Write employee_transfers sidecar for the assignment generator
        transfers = df[df["TransferStatus"] == "Transferred"][
            ["EmployeeKey", "OriginalStoreKey", "TransferDate", "Title", "DistrictId", "FTE"]
        ].copy()
        transfers_path = parquet_folder / "employee_transfers.parquet"
        if not transfers.empty:
            transfers["EmployeeKey"] = transfers["EmployeeKey"].astype(np.int32)
            transfers["OriginalStoreKey"] = transfers["OriginalStoreKey"].astype(np.int32)
            write_parquet_with_date32(
                transfers,
                transfers_path,
                date_cols=["TransferDate"],
                cast_all_datetime=False,
                compression="snappy",
                compression_level=None,
                force_date32=True,
            )
            info(f"Employee transfers written: {transfers_path.name} ({len(transfers)} employees)")
        elif transfers_path.exists():
            transfers_path.unlink()

        # Drop internal transfer columns before writing main employees parquet
        df = df.drop(columns=["TransferStatus", "TransferDate", "OriginalStoreKey"], errors="ignore")

        # Reorder columns to match the static schema (CREATE TABLE column order).
        _SCHEMA_ORDER = [
            "EmployeeKey", "ParentEmployeeKey", "EmployeeName", "Title",
            "OrgLevel", "OrgUnitType", "RegionId", "DistrictId",
            "StoreKey", "GeographyKey",
            "HireDate", "TerminationDate", "TerminationReason", "IsActive",
            "EmploymentType", "FTE",
            "Gender", "FirstName", "LastName", "MiddleName",
            "BirthDate", "MaritalStatus", "EmailAddress", "Phone",
            "EmergencyContactName", "EmergencyContactPhone",
            "SalariedFlag", "PayFrequency", "BaseRate", "VacationHours",
            "CurrentFlag", "Status",
            "SalesPersonFlag", "DepartmentName",
        ]
        df = df[_SCHEMA_ORDER]

        compression = emp_cfg.parquet_compression
        compression_level = emp_cfg.parquet_compression_level

        date_cols = ["HireDate", "TerminationDate", "BirthDate"]
        write_parquet_with_date32(
            df,
            out_path,
            date_cols=date_cols,
            cast_all_datetime=False,
            compression=str(compression),
            compression_level=(int(compression_level) if compression_level is not None else None),
            force_date32=True,
        )

    save_version("employees", version_cfg, out_path)
    info(f"Employees dimension written: {out_path.name}")

    # --- Sync stores.parquet EmployeeCount with actual generated counts ---
    _sync_stores_employee_count(df, stores_path)
