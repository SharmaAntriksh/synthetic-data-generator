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
    pick_seed_nested,
    parse_global_dates,
    rand_dates_between,
    region_from_iso_code,
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
    if emp_cfg.get("start_date") is not None or emp_cfg.get("end_date") is not None:
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

    # Deterministic Gender distribution ~ 49/49/2 (M/F/O) based on hash
    h = hash_u64(ek_u64, int(seed), 9101)
    u = (h % np.uint64(10_000)).astype(np.float64) / 10_000.0
    gender_code = np.where(u < 0.02, "O", np.where(u < 0.51, "F", "M")).astype(object)
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
    birth_day = (rng.random(n) * dim).astype(int) + 1
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
        pick = rng.integers(0, n, size=n)
        self_idx = np.arange(n)
        pick = np.where(pick == self_idx, (pick + 1) % n, pick)
        df["EmergencyContactName"] = (
            df["FirstName"].iloc[pick].to_numpy(dtype=object)
            + " "
            + df["LastName"].iloc[pick].to_numpy(dtype=object)
        )
    else:
        df["EmergencyContactName"] = pd.Series(["Jane Doe"], dtype="object")
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

    # CurrentFlag / Status / StartDate / EndDate
    df["CurrentFlag"] = df["IsActive"].astype(np.int8)
    df["StartDate"] = hire
    df["EndDate"] = pd.to_datetime(df["TerminationDate"]).dt.normalize()
    df["Status"] = np.where(
        df["IsActive"].astype(int) == 1, "Active", "Terminated",
    ).astype(object)

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
    ensure_store_sales_coverage: bool = False,
    store_manager_names: dict[int, str] | None = None,
) -> pd.DataFrame:
    """
    Build a parent-child employee hierarchy with stable keys.

    Sales Associate lifecycle guarantee:
      - ALL Sales Associates are hired at or before *global_start*.
      - ALL Sales Associates have no termination (NaT).
      - This ensures the bridge table can assign every Sales Associate for
        the full ``[global_start, global_end]`` window without date conflicts.
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
            "Regenerate stores to enable unified hierarchy."
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
        "EmployeeKey": (STORE_MGR_KEY_BASE + sk_arr).astype(np.int32),
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
        within_store_idx = np.ones(total_staff, dtype=np.int32)
        offsets = np.cumsum(staff_counts)[:-1]
        if offsets.size > 0:
            np.subtract.at(within_store_idx, offsets, staff_counts[:-1] - 1)
        within_store_idx = np.cumsum(within_store_idx)

        staff_ek = (STAFF_KEY_BASE + staff_sk * STAFF_KEY_STORE_MULT + within_store_idx).astype(np.int32)
        staff_parent = (STORE_MGR_KEY_BASE + staff_sk).astype(np.int32)

        # Sample titles in bulk, then overwrite first k per store with primary sales role
        all_titles = rng.choice(_STAFF_TITLES, size=total_staff, p=_STAFF_TITLES_P).astype(object)
        ps_role = str(primary_sales_role or "Sales Associate")
        k_ps = max(1, int_or(min_primary_sales_per_store, 1))

        # Mark the first k_ps employees of each store as the primary sales role
        k_per_store = np.minimum(staff_counts, k_ps)
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
    # Dates — with Sales Associate full-window guarantee
    # ------------------------------------------------------------------
    n = len(df)
    ps_role_str = str(primary_sales_role or "Sales Associate")

    ek_all = pd.to_numeric(df["EmployeeKey"], errors="coerce").fillna(0).astype(np.int32)
    is_sales_associate = (ek_all >= STAFF_KEY_BASE) & (df["Title"].astype(str) == ps_role_str)
    sa_mask_np = is_sales_associate.to_numpy()

    # Hire dates: SAs hired before dataset start; everyone else random
    hire_start_general = global_start - pd.Timedelta(days=365 * 5)
    hire_dates = rand_dates_between(rng, hire_start_general, global_end, n)

    n_sa = int(sa_mask_np.sum())
    if n_sa > 0:
        sa_hire = rand_dates_between(rng, hire_start_general, global_start, n_sa)
        hire_dates.iloc[sa_mask_np] = sa_hire.to_numpy()

    df["HireDate"] = hire_dates

    # Terminations: SAs never terminated; others probabilistic (reduced for senior levels)
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
        offs = (rng.random(idx.size) * (max_days + 1)).astype(np.int64)
        term_dates.iloc[idx] = (hire_i + pd.to_timedelta(offs, unit="D")).to_numpy(dtype="datetime64[ns]")

    df["TerminationDate"] = term_dates
    df["IsActive"] = (
        df["TerminationDate"].isna() | (df["TerminationDate"] > global_end)
    ).astype(np.int8)

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
                first = np.array([str(n).split(" ", 1)[0] for n in vn], dtype=object)
                last = np.array(
                    [str(n).split(" ", 1)[1] if " " in str(n) else "" for n in vn],
                    dtype=object,
                )
                df.loc[vi, "FirstName"] = first
                df.loc[vi, "LastName"] = last
                df.loc[vi, "MiddleName"] = ""
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
        for i in range(len(stores_full)):
            cnt = actual.get(int(sk_arr[i]), 0)
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
    emp_cfg = as_dict(cfg.get("employees"))

    parquet_folder = Path(parquet_folder)
    parquet_folder.mkdir(parents=True, exist_ok=True)

    stores_path = parquet_folder / "stores.parquet"
    out_path = parquet_folder / "employees.parquet"

    if not stores_path.exists():
        raise FileNotFoundError(f"Missing stores parquet: {stores_path}")

    seed = pick_seed_nested(cfg, emp_cfg, fallback=42)
    global_start, global_end = _parse_employee_dates(cfg, emp_cfg)

    _STORES_READ_COLS = [
        "StoreKey", "GeographyKey", "EmployeeCount", "StoreType",
        "StoreDistrict", "StoreRegion", "StoreManager",
    ]
    try:
        stores = pd.read_parquet(stores_path, columns=_STORES_READ_COLS)
    except (KeyError, ValueError):
        # Legacy stores.parquet may lack hierarchy/manager columns
        stores = pd.read_parquet(
            stores_path,
            columns=["StoreKey", "GeographyKey", "EmployeeCount", "StoreType"],
        )

    version_cfg = dict(emp_cfg)
    version_cfg["schema_version"] = 6
    version_cfg["_stores_sig"] = _stores_signature(stores)
    version_cfg["_stores_cfg"] = dict(as_dict(cfg.get("stores")))
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

    with stage("Generating Employees"):
        sa_cfg = as_dict(emp_cfg.get("store_assignments"))
        primary_sales_role = str(sa_cfg.get("primary_sales_role") or "Sales Associate")
        min_primary_sales_per_store = int_or(sa_cfg.get("min_primary_sales_per_store"), 1)
        ensure_store_sales_coverage = bool_or(sa_cfg.get("ensure_store_sales_coverage"), False)

        df = generate_employee_dimension(
            stores=stores,
            seed=seed,
            global_start=global_start,
            global_end=global_end,
            district_size=int_or(emp_cfg.get("district_size"), 10),
            districts_per_region=int_or(emp_cfg.get("districts_per_region"), 8),
            max_staff_per_store=int_or(emp_cfg.get("max_staff_per_store"), 5),
            termination_rate=float_or(emp_cfg.get("termination_rate"), 0.08),
            use_store_employee_count=bool_or(emp_cfg.get("use_store_employee_count"), False),
            min_staff_per_store=int_or(emp_cfg.get("min_staff_per_store"), 3),
            staff_scale=float_or(emp_cfg.get("staff_scale"), 0.25),
            people_pools=people_pools,
            iso_by_geo=iso_by_geo,
            default_region="US",
            primary_sales_role=primary_sales_role,
            min_primary_sales_per_store=min_primary_sales_per_store,
            ensure_store_sales_coverage=ensure_store_sales_coverage,
            store_manager_names=store_manager_names,
        )

        hr_cfg = as_dict(emp_cfg.get("hr"))
        email_domain = hr_cfg.get("email_domain", "contoso.com")

        df = _enrich_employee_hr_columns(
            df,
            rng=np.random.default_rng(int(seed) ^ 0x9E3779B1),
            global_end=global_end,
            email_domain=str(email_domain),
        )

        df = _finalize_employee_integer_cols(df)

        # Reorder columns to match the static schema (CREATE TABLE column order).
        _SCHEMA_ORDER = [
            "EmployeeKey", "ParentEmployeeKey", "EmployeeName", "Title",
            "OrgLevel", "OrgUnitType", "RegionId", "DistrictId",
            "StoreKey", "GeographyKey",
            "HireDate", "TerminationDate", "IsActive",
            "Gender", "FirstName", "LastName", "MiddleName",
            "BirthDate", "MaritalStatus", "EmailAddress", "Phone",
            "EmergencyContactName", "EmergencyContactPhone",
            "SalariedFlag", "PayFrequency", "BaseRate", "VacationHours",
            "CurrentFlag", "StartDate", "EndDate", "Status",
            "SalesPersonFlag", "DepartmentName",
        ]
        df = df[_SCHEMA_ORDER]

        compression = emp_cfg.get("parquet_compression", "snappy")
        compression_level = emp_cfg.get("parquet_compression_level", None)

        date_cols = ["HireDate", "TerminationDate", "BirthDate", "StartDate", "EndDate"]
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
    info(f"Employees dimension written: {out_path}")

    # --- Sync stores.parquet EmployeeCount with actual generated counts ---
    _sync_stores_employee_count(df, stores_path)
