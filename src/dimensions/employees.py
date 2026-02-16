# src/dimensions/employees.py
# ---------------------------------------------------------
#  EMPLOYEES DIMENSION (PARENT-CHILD HIERARCHY)
#  - Reads stores.parquet
#  - Generates org chart: CEO -> VP -> Region -> District -> Store -> Staff
#  - Produces EmployeeKey + ParentEmployeeKey for DAX PATH/PATHITEM
#  - Enriches with Contoso-like HR columns
#  - Default staffing is intentionally small (demo-friendly)
# ---------------------------------------------------------

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.utils.logging_utils import info, skip, stage
from src.utils.output_utils import write_parquet_with_date32
from src.versioning.version_store import should_regenerate, save_version


# ---------------------------------------------------------
# Internals
# ---------------------------------------------------------

def _as_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _int_or(value: Any, default: int) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _float_or(value: Any, default: float) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _bool_or(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "y", "on"}:
        return True
    if s in {"false", "0", "no", "n", "off"}:
        return False
    return bool(default)


def _pick_seed(cfg: Dict[str, Any], emp_cfg: Dict[str, Any], fallback: int = 42) -> int:
    override = _as_dict(emp_cfg.get("override"))
    seed = override.get("seed")
    if seed is None:
        seed = emp_cfg.get("seed")
    if seed is None:
        seed = _as_dict(cfg.get("defaults")).get("seed")
    return _int_or(seed, fallback)


def _stores_signature(stores: pd.DataFrame) -> Dict[str, Any]:
    if stores.empty:
        return {"rows": 0, "min_store": None, "max_store": None, "emp_sum": 0}
    sk = stores["StoreKey"].to_numpy()
    emp_sum = int(stores["EmployeeCount"].fillna(0).astype(np.int64).sum()) if "EmployeeCount" in stores.columns else 0
    return {
        "rows": int(len(stores)),
        "min_store": int(np.min(sk)),
        "max_store": int(np.max(sk)),
        "emp_sum": emp_sum,
    }


_FIRST = np.array(
    [
        # Common US/UK
        "James","John","Robert","Michael","William","David","Richard","Joseph","Thomas","Charles",
        "Christopher","Daniel","Matthew","Anthony","Mark","Donald","Steven","Paul","Andrew","Joshua",
        "Kevin","Brian","George","Edward","Ronald","Timothy","Jason","Jeffrey","Ryan","Jacob",
        "Gary","Nicholas","Eric","Stephen","Jonathan","Larry","Justin","Scott","Brandon","Benjamin",
        "Samuel","Gregory","Frank","Alexander","Raymond","Patrick","Jack","Dennis","Jerry","Tyler",

        # Common female US/UK
        "Mary","Patricia","Jennifer","Linda","Elizabeth","Barbara","Susan","Jessica","Sarah","Karen",
        "Nancy","Lisa","Margaret","Betty","Sandra","Ashley","Kimberly","Emily","Donna","Michelle",
        "Dorothy","Carol","Amanda","Melissa","Deborah","Stephanie","Rebecca","Laura","Sharon","Cynthia",
        "Kathleen","Amy","Shirley","Angela","Helen","Anna","Brenda","Pamela","Nicole","Emma",
        "Samantha","Katherine","Christine","Debra","Rachel","Catherine","Carolyn","Janet","Ruth","Maria",

        # Modern / neutral
        "Alex","Jordan","Taylor","Casey","Morgan","Riley","Jamie","Avery","Cameron","Quinn",
        "Peyton","Reese","Rowan","Skyler","Dakota","Finley","Harper","Hayden","Charlie","Emerson",

        # European mix
        "Luca","Matteo","Marco","Giovanni","Francesco","Alessandro","Sofia","Giulia","Francesca","Chiara",
        "Mateo","Diego","Carlos","Javier","Miguel","Lucia","Carmen","Elena","Isabella","Valentina",
        "Louis","Hugo","Arthur","Camille","Chloe","Julien","Nicolas","Amelie","Lea","Manon",
        "Noah","Leon","Elias","Felix","Mia","Lina","Hannah","Anna","Leonie","Marie",

        # Nordics / misc
        "Oskar","Emil","Oliver","Lucas","Ella","Freya","Ida","Astrid","Signe","Ingrid",
        "Ethan","Logan","Mason","Liam","Noah","Aiden","Sebastian","Nathan","Dylan","Caleb",
    ],
    dtype=object,
)

_LAST = np.array(
    [
        # US/UK common
        "Smith","Johnson","Williams","Brown","Jones","Garcia","Miller","Davis","Rodriguez","Martinez",
        "Hernandez","Lopez","Gonzalez","Wilson","Anderson","Thomas","Taylor","Moore","Jackson","Martin",
        "Lee","Perez","Thompson","White","Harris","Sanchez","Clark","Ramirez","Lewis","Robinson",
        "Walker","Young","Allen","King","Wright","Scott","Torres","Nguyen","Hill","Flores",
        "Green","Adams","Nelson","Baker","Hall","Rivera","Campbell","Mitchell","Carter","Roberts",

        # UK/IE
        "Murphy","Kelly","OBrien","ONeill","Walsh","Ryan","Byrne","Doyle","McCarthy","Gallagher",
        "Campbell","Stewart","Murray","Reid","Graham","Watson","Hunter","Ross","Ferguson","Gibson",

        # Spanish/Portuguese
        "Silva","Santos","Ferreira","Pereira","Oliveira","Costa","Ribeiro","Alves","Carvalho","Gomes",
        "Fernandez","Ruiz","Diaz","Vazquez","Castro","Romero","Mendez","Ramos","Morales","Ortega",

        # French
        "Dubois","Durand","Moreau","Laurent","Simon","Michel","Lefevre","Garner","Petit","Roux",
        "Fournier","Girard","Lambert","Bonnet","Francois","Martins","Henry","Chevalier","Gauthier","Perrin",

        # Italian
        "Rossi","Russo","Ferrari","Esposito","Bianchi","Romano","Colombo","Ricci","Marino","Greco",
        "Bruno","Gallo","Conti","DeLuca","Mancini","Costa","Giordano","Rizzo","Lombardi","Moretti",

        # German/Dutch/Scandi
        "Schmidt","Mueller","Schneider","Fischer","Weber","Meyer","Wagner","Becker","Hoffmann","Schaefer",
        "Jansen","DeVries","VanDijk","Bakker","Visser","Andersen","Nielsen","Larsen","Johansson","Lindberg",

        # Eastern Europe / misc
        "Nowak","Kowalski","Wisniewski","Wojcik","Kovac","Novak","Horvat","Ilic","Popescu","Ionescu",
        "Ivanov","Petrov","Smirnov","Sokolov","Volkov","Kuznetsov","Orlov","Morozov","Pavlov","Fedorov",
    ],
    dtype=object,
)


_STAFF_TITLES = np.array(
    ["Sales Associate", "Cashier", "Stock Associate", "Customer Support", "Fulfillment Associate"],
    dtype=object,
)
_STAFF_TITLES_P = np.array([0.40, 0.18, 0.18, 0.12, 0.12], dtype=float)


def _parse_global_dates(cfg: Dict[str, Any], emp_cfg: Dict[str, Any]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    gd = _as_dict(emp_cfg.get("global_dates"))
    if gd:
        start = pd.to_datetime(gd.get("start")).normalize()
        end = pd.to_datetime(gd.get("end")).normalize()
        if pd.isna(start) or pd.isna(end):
            raise ValueError(f"Invalid employees.global_dates: {gd}")
        return start, end

    defaults_dates = _as_dict((_as_dict(cfg.get("defaults")) or _as_dict(cfg.get("_defaults"))).get("dates"))
    if defaults_dates:
        start = pd.to_datetime(defaults_dates.get("start")).normalize()
        end = pd.to_datetime(defaults_dates.get("end")).normalize()
        if not pd.isna(start) and not pd.isna(end):
            return start, end

    return pd.Timestamp("2021-01-01"), pd.Timestamp("2026-12-31")


def _rand_dates_between(
    rng: np.random.Generator,
    start: pd.Timestamp,
    end: pd.Timestamp,
    n: int,
) -> pd.Series:
    start_i = start.value // 10**9
    end_i = end.value // 10**9
    if end_i < start_i:
        start_i, end_i = end_i, start_i

    secs = rng.integers(start_i, end_i + 1, size=n, dtype=np.int64)
    dt = pd.to_datetime(secs, unit="s")

    # ndarray -> DatetimeIndex
    if isinstance(dt, pd.DatetimeIndex):
        return pd.Series(dt.normalize())

    return dt.dt.normalize()


def _apply_deterministic_names(df: pd.DataFrame, seed: int) -> None:
    """
    Stable names per EmployeeKey (no dependence on row order).
    Ensures EmployeeName is always a real person name (not role labels).
    """
    ek = df["EmployeeKey"].astype(np.int64).to_numpy()
    first = _FIRST[(ek + seed) % len(_FIRST)]
    last = _LAST[((ek // 97) + seed * 3) % len(_LAST)]

    # ~30% have middle initial, deterministic
    mid_mask = ((ek + seed) % 10) < 3
    mid_initial = np.where(
        mid_mask,
        np.array(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), dtype=object)[(ek + seed) % 26],
        pd.NA,
    )
    middle = np.where(pd.isna(mid_initial), pd.NA, (pd.Series(mid_initial).astype(str) + "."))

    df["FirstName"] = pd.Series(first, dtype="object").astype(str)
    df["LastName"] = pd.Series(last, dtype="object").astype(str)
    df["MiddleName"] = pd.Series(middle, dtype="object")

    df["EmployeeName"] = df["FirstName"] + " " + df["LastName"]


def _enrich_employee_hr_columns(
    df: pd.DataFrame,
    rng: np.random.Generator,
    global_end: pd.Timestamp,
    email_domain: str = "contoso.com",
) -> pd.DataFrame:
    """
    Adds Contoso-like HR columns.
    Assumes df already has: EmployeeKey, Title, OrgLevel, HireDate, TerminationDate, IsActive,
    and also has deterministic names: FirstName/LastName/MiddleName/EmployeeName.
    """
    n = len(df)
    if n == 0:
        return df

    hire = pd.to_datetime(df["HireDate"]).dt.normalize()
    org_level = df["OrgLevel"].astype(int).to_numpy()
    title = df["Title"].astype(str)

    # Gender / MaritalStatus
    df["Gender"] = rng.choice(["M", "F", "O"], size=n, p=[0.49, 0.49, 0.02]).astype(object)

    # BirthDate: infer age-at-hire by level (staff younger, management older)
    age_mean = np.where(org_level >= 6, 27, np.where(org_level >= 5, 34, 42))
    ages = np.clip(rng.normal(loc=age_mean, scale=6.0, size=n), 18, 62).astype(int)
    birth_year = hire.dt.year.to_numpy() - ages
    birth_month = rng.integers(1, 13, size=n)
    birth_day = rng.integers(1, 29, size=n)
    df["BirthDate"] = pd.to_datetime({"year": birth_year, "month": birth_month, "day": birth_day}).dt.normalize()

    is_married = rng.random(n) < np.clip((ages - 22) / 25.0, 0.05, 0.75)
    df["MaritalStatus"] = np.where(is_married, "M", "S").astype(object)

    # Email / Phone
    email_local = (
        df["FirstName"].str.lower().str.replace(" ", "", regex=False)
        + "."
        + df["LastName"].str.lower().str.replace(" ", "", regex=False)
        + "."
        + df["EmployeeKey"].astype("Int64").astype(str)
    )
    df["EmailAddress"] = (email_local + "@" + str(email_domain)).astype(str)

    phone_digits = rng.integers(0, 10, size=(n, 10))
    df["Phone"] = pd.Series(["".join(map(str, row)) for row in phone_digits], dtype="object")

    # Emergency contacts
    df["EmergencyContactName"] = np.where(
        is_married,
        "EC " + df["LastName"].astype(str),
        "EC " + df["FirstName"].astype(str),
    ).astype(object)
    ec_digits = rng.integers(0, 10, size=(n, 10))
    df["EmergencyContactPhone"] = pd.Series(["".join(map(str, row)) for row in ec_digits], dtype="object")

    # Compensation-ish fields
    salaried = (df["OrgLevel"].astype(int) <= 5).to_numpy()  # store mgr and above salaried
    df["SalariedFlag"] = salaried.astype(np.int8)
    df["PayFrequency"] = np.where(salaried, 1, 2).astype(np.int16)  # 1=monthly, 2=biweekly/hourly style

    hourly_staff = np.clip(rng.normal(loc=18.0, scale=4.0, size=n), 10.0, 40.0)
    annual_salary = np.clip(rng.normal(loc=70000.0, scale=18000.0, size=n), 38000.0, 160000.0)
    hourly_equiv = annual_salary / 2080.0
    df["BaseRate"] = np.where(salaried, hourly_equiv, hourly_staff).round(2).astype(np.float64)

    # VacationHours: tenure-based
    tenure_days = (global_end.normalize() - hire).dt.days.clip(lower=0)
    base_vac = np.where(salaried, 80, 40) + (tenure_days / 365.0 * np.where(salaried, 6.0, 3.0))
    df["VacationHours"] = np.clip(base_vac + rng.normal(0, 10, size=n), 0, 240).round(0).astype(np.int16)

    # CurrentFlag / Status / StartDate / EndDate
    df["CurrentFlag"] = df["IsActive"].astype(np.int8)
    df["StartDate"] = hire
    df["EndDate"] = pd.to_datetime(df["TerminationDate"]).dt.normalize()
    df["Status"] = np.where(df["IsActive"].astype(int) == 1, "Active", "Terminated").astype(object)

    # SalesPersonFlag
    df["SalesPersonFlag"] = title.isin(["Sales Associate", "Store Manager"]).astype(np.int8)

    # DepartmentName
    dept = np.where(
        title.isin(["Sales Associate", "Store Manager"]),
        "Sales",
        np.where(title.isin(["Cashier"]), "Store Operations",
                 np.where(title.isin(["Stock Associate"]), "Inventory",
                          np.where(title.isin(["Fulfillment Associate"]), "Fulfillment", "Corporate"))),
    )
    df["DepartmentName"] = pd.Series(dept, dtype="object")

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
    district_size: int = 15,
    districts_per_region: int = 8,
    max_staff_per_store: int = 10,
    termination_rate: float = 0.08,
    # new knobs (safe defaults)
    use_store_employee_count: bool = False,
    min_staff_per_store: int = 3,
    staff_scale: float = 0.25,
    include_store_cols: bool = True,
) -> pd.DataFrame:
    """
    Build a parent-child employee hierarchy with stable keys.

    Defaults are intentionally small for demo-friendly datasets:
      - by default, ignores Store.EmployeeCount and samples staff_count ~ U[min_staff, max_staff]
    """
    if stores.empty:
        raise ValueError("stores dataframe is empty; cannot generate employees")

    required_cols = {"StoreKey", "GeographyKey", "EmployeeCount", "StoreType"}
    missing = [c for c in required_cols if c not in stores.columns]
    if missing:
        raise ValueError(f"stores.parquet missing required columns: {missing}")

    stores = stores.copy()
    stores["StoreKey"] = stores["StoreKey"].astype(np.int64)
    stores = stores.sort_values("StoreKey").reset_index(drop=True)

    district_size = max(1, _int_or(district_size, 15))
    districts_per_region = max(1, _int_or(districts_per_region, 8))
    max_staff_per_store = max(0, _int_or(max_staff_per_store, 10))
    min_staff_per_store = max(0, _int_or(min_staff_per_store, 3))
    min_staff_per_store = min(min_staff_per_store, max_staff_per_store) if max_staff_per_store > 0 else 0
    staff_scale = float(max(0.0, min(1.0, _float_or(staff_scale, 0.25))))
    termination_rate = float(max(0.0, min(1.0, _float_or(termination_rate, 0.08))))
    include_store_cols = bool(include_store_cols)

    rng = np.random.default_rng(int(seed))

    # --- Partition stores into districts and regions (deterministic by StoreKey order)
    n_stores = len(stores)
    district_id = (np.arange(n_stores) // district_size + 1).astype(np.int16)
    region_id = ((district_id - 1) // districts_per_region + 1).astype(np.int16)

    stores["DistrictId"] = district_id
    stores["RegionId"] = region_id

    # --- Key scheme (stable, non-overlapping)
    CEO_KEY = np.int64(1)
    VP_OPS_KEY = np.int64(2)

    def _region_mgr_key(rid: int) -> np.int64:
        return np.int64(10_000 + int(rid))

    def _district_mgr_key(did: int) -> np.int64:
        return np.int64(20_000 + int(did))

    def _store_mgr_key(store_key: int) -> np.int64:
        return np.int64(30_000_000 + int(store_key))

    def _staff_key(store_key: int, idx: int) -> np.int64:
        # idx starts at 1
        return np.int64(40_000_000 + int(store_key) * 1_000 + int(idx))

    rows = []

    # --- Corporate root
    rows.append(
        dict(
            EmployeeKey=CEO_KEY,
            ParentEmployeeKey=pd.NA,
            EmployeeName="",  # will be replaced with person name
            Title="Chief Executive Officer",
            OrgLevel=np.int16(1),
            OrgUnitType="Corporate",
            RegionId=pd.NA,
            DistrictId=pd.NA,
            StoreKey=pd.NA,
            GeographyKey=pd.NA,
        )
    )
    rows.append(
        dict(
            EmployeeKey=VP_OPS_KEY,
            ParentEmployeeKey=CEO_KEY,
            EmployeeName="",
            Title="VP Operations",
            OrgLevel=np.int16(2),
            OrgUnitType="Corporate",
            RegionId=pd.NA,
            DistrictId=pd.NA,
            StoreKey=pd.NA,
            GeographyKey=pd.NA,
        )
    )

    # --- Region managers
    unique_regions = np.unique(region_id)
    for rid in unique_regions:
        rows.append(
            dict(
                EmployeeKey=_region_mgr_key(int(rid)),
                ParentEmployeeKey=VP_OPS_KEY,
                EmployeeName="",
                Title="Regional Manager",
                OrgLevel=np.int16(3),
                OrgUnitType="Region",
                RegionId=np.int16(rid),
                DistrictId=pd.NA,
                StoreKey=pd.NA,
                GeographyKey=pd.NA,
            )
        )

    # --- District managers
    unique_districts = np.unique(district_id)
    for did in unique_districts:
        rid = int(((did - 1) // districts_per_region) + 1)
        rows.append(
            dict(
                EmployeeKey=_district_mgr_key(int(did)),
                ParentEmployeeKey=_region_mgr_key(rid),
                EmployeeName="",
                Title="District Manager",
                OrgLevel=np.int16(4),
                OrgUnitType="District",
                RegionId=np.int16(rid),
                DistrictId=np.int16(did),
                StoreKey=pd.NA,
                GeographyKey=pd.NA,
            )
        )

    # --- Store managers
    for i in range(n_stores):
        sk = int(stores.at[i, "StoreKey"])
        did = int(stores.at[i, "DistrictId"])
        rid = int(stores.at[i, "RegionId"])
        gk = int(stores.at[i, "GeographyKey"])

        rows.append(
            dict(
                EmployeeKey=_store_mgr_key(sk),
                ParentEmployeeKey=_district_mgr_key(did),
                EmployeeName="",
                Title="Store Manager",
                OrgLevel=np.int16(5),
                OrgUnitType="Store",
                RegionId=np.int16(rid),
                DistrictId=np.int16(did),
                StoreKey=np.int64(sk),
                GeographyKey=np.int64(gk),
            )
        )

    # --- Staff counts (reduced by default)
    if max_staff_per_store <= 0:
        staff_counts = np.zeros(n_stores, dtype=np.int64)
    elif use_store_employee_count:
        # Scale down Store.EmployeeCount (often large) into a smaller staff count
        emp_counts = stores["EmployeeCount"].fillna(0).astype(np.int64).to_numpy()
        base = np.maximum(0, emp_counts - 1)  # after manager
        scaled = np.rint(base.astype(np.float64) * staff_scale).astype(np.int64)
        staff_counts = np.clip(scaled, 0, max_staff_per_store).astype(np.int64)
        # optional floor (if store has employees, ensure at least min_staff)
        has_any = base > 0
        staff_counts = np.where(has_any, np.maximum(staff_counts, min_staff_per_store), 0).astype(np.int64)
    else:
        # demo-friendly: sample small staff per store regardless of Store.EmployeeCount
        staff_counts = rng.integers(min_staff_per_store, max_staff_per_store + 1, size=n_stores, dtype=np.int64)

    # --- Staff rows (titles sampled; names set later deterministically)
    for i in range(n_stores):
        sk = int(stores.at[i, "StoreKey"])
        did = int(stores.at[i, "DistrictId"])
        rid = int(stores.at[i, "RegionId"])
        gk = int(stores.at[i, "GeographyKey"])
        n_staff = int(staff_counts[i])

        if n_staff <= 0:
            continue

        mgr_key = _store_mgr_key(sk)
        titles = rng.choice(_STAFF_TITLES, size=n_staff, p=_STAFF_TITLES_P)
        for j in range(1, n_staff + 1):
            rows.append(
                dict(
                    EmployeeKey=_staff_key(sk, j),
                    ParentEmployeeKey=mgr_key,
                    EmployeeName="",
                    Title=str(titles[j - 1]),
                    OrgLevel=np.int16(6),
                    OrgUnitType="Store",
                    RegionId=np.int16(rid),
                    DistrictId=np.int16(did),
                    StoreKey=np.int64(sk),
                    GeographyKey=np.int64(gk),
                )
            )

    df = pd.DataFrame(rows)

    # --- Dates (date-only)
    n = len(df)
    hire_start = global_start - pd.Timedelta(days=365 * 5)
    hire_end = global_end
    hire_dates = _rand_dates_between(rng, hire_start, hire_end, n)
    df["HireDate"] = hire_dates

    # Terminations: reduced for senior levels
    base_p = termination_rate
    level = df["OrgLevel"].astype(np.int16).to_numpy()
    p = np.where(level <= 4, base_p * 0.25, base_p)
    term_mask = rng.random(n) < p

    term_dates = pd.Series([pd.NaT] * n)
    idx = np.where(term_mask)[0]
    if idx.size > 0:
        term_dates.iloc[idx] = _rand_dates_between(rng, df.loc[idx, "HireDate"].min(), global_end, idx.size)
        bad = term_dates.notna() & (term_dates < df["HireDate"])
        if bad.any():
            term_dates.loc[bad] = df.loc[bad, "HireDate"]

    df["TerminationDate"] = term_dates
    df["IsActive"] = (df["TerminationDate"].isna() | (df["TerminationDate"] > global_end)).astype(np.int8)

    # --- Names (always person names)
    _apply_deterministic_names(df, seed=int(seed))

    # --- Types
    df["EmployeeKey"] = df["EmployeeKey"].astype(np.int64)
    df["ParentEmployeeKey"] = df["ParentEmployeeKey"].astype("Int64")
    df["OrgLevel"] = df["OrgLevel"].astype(np.int16)
    df["RegionId"] = df["RegionId"].astype("Int16")
    df["DistrictId"] = df["DistrictId"].astype("Int16")
    df["StoreKey"] = df["StoreKey"].astype("Int64")
    df["GeographyKey"] = df["GeographyKey"].astype("Int64")

    # Optional: drop StoreKey/GeographyKey from DimEmployee output
    if not include_store_cols:
        df = df.drop(columns=["StoreKey", "GeographyKey"], errors="ignore")

    return df


# ---------------------------------------------------------
# Pipeline entrypoint
# ---------------------------------------------------------

def run_employees(cfg: Dict[str, Any], parquet_folder: Path) -> None:
    cfg = cfg or {}
    emp_cfg = _as_dict(cfg.get("employees"))

    parquet_folder = Path(parquet_folder)
    parquet_folder.mkdir(parents=True, exist_ok=True)

    stores_path = parquet_folder / "stores.parquet"
    out_path = parquet_folder / "employees.parquet"

    if not stores_path.exists():
        raise FileNotFoundError(f"Missing stores parquet: {stores_path}")

    force = bool(emp_cfg.get("_force_regenerate", False))
    seed = _pick_seed(cfg, emp_cfg, fallback=42)
    global_start, global_end = _parse_global_dates(cfg, emp_cfg)

    stores = pd.read_parquet(
        stores_path,
        columns=["StoreKey", "GeographyKey", "EmployeeCount", "StoreType"],
    )

    version_cfg = dict(emp_cfg)
    version_cfg.pop("_force_regenerate", None)
    # schema changed (names + sizing) => bump
    version_cfg["schema_version"] = 2
    version_cfg["_stores_sig"] = _stores_signature(stores)
    version_cfg["_global_dates"] = {"start": str(global_start.date()), "end": str(global_end.date())}

    if not force and not should_regenerate("employees", version_cfg, out_path):
        skip("Employees up-to-date; skipping.")
        return

    with stage("Generating Employees"):
        df = generate_employee_dimension(
            stores=stores,
            seed=seed,
            global_start=global_start,
            global_end=global_end,
            district_size=_int_or(emp_cfg.get("district_size"), 15),
            districts_per_region=_int_or(emp_cfg.get("districts_per_region"), 8),
            # reduced default if not set
            max_staff_per_store=_int_or(emp_cfg.get("max_staff_per_store"), 10),
            termination_rate=_float_or(emp_cfg.get("termination_rate"), 0.08),
            # new knobs
            use_store_employee_count=_bool_or(emp_cfg.get("use_store_employee_count"), False),
            min_staff_per_store=_int_or(emp_cfg.get("min_staff_per_store"), 3),
            staff_scale=_float_or(emp_cfg.get("staff_scale"), 0.25),
            include_store_cols=_bool_or(emp_cfg.get("include_store_cols"), True),
        )

        hr_cfg = _as_dict(emp_cfg.get("hr"))
        email_domain = hr_cfg.get("email_domain", "contoso.com")

        # deterministic enrichment (still uses rng but seeded)
        df = _enrich_employee_hr_columns(
            df,
            rng=np.random.default_rng(int(seed)),
            global_end=global_end,
            email_domain=str(email_domain),
        )

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
