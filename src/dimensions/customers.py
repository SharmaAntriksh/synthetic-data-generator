# ---------------------------------------------------------
#  CUSTOMERS DIMENSION (REALISTIC CONTOSO VERSION)
# ---------------------------------------------------------

import os
import pandas as pd
import numpy as np
from pathlib import Path

from src.utils import info, skip, stage
from src.versioning import should_regenerate, save_version
from src.engine.dimension_loader import load_dimension


# ---------------------------------------------------------
# Helper: Load CSV lists
# ---------------------------------------------------------
def load_list(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")

    s = pd.read_csv(path, header=None, dtype=str)[0].str.strip()
    s = s[s.str.match(r"^[A-Za-z\-\'. ]+$")]
    return s.str.title().unique()


# ---------------------------------------------------------
# Main generator
# ---------------------------------------------------------
def generate_synthetic_customers(cfg, parquet_dims_folder):

    cust_cfg = cfg["customers"]
    total_customers = cust_cfg["total_customers"]

    active_ratio = cust_cfg.get("active_ratio", 1.0)

    if not isinstance(active_ratio, (int, float)) or not (0 < active_ratio <= 1):
        raise ValueError("customers.active_ratio must be a number in the range (0, 1]")

    pct_india = cust_cfg["pct_india"]
    pct_us = cust_cfg["pct_us"]
    pct_eu = cust_cfg["pct_eu"]
    pct_org = cust_cfg["pct_org"]

    override_seed = cust_cfg.get("override", {}).get("seed")
    seed = override_seed if override_seed is not None else 42
    rng = np.random.default_rng(seed)

    # -------------------------------------------------
    # Email domain pools (simple, realistic)
    # -------------------------------------------------
    PERSONAL_EMAIL_DOMAINS = np.array([
        "gmail.com",
        "yahoo.com",
        "outlook.com",
        "hotmail.com",
    ])

    names_folder = cust_cfg["names_folder"]

    # -----------------------------------------------------
    # Load names
    # -----------------------------------------------------
    paths = {
        "us_male":   os.path.join(names_folder, "us_male_first.csv"),
        "us_female": os.path.join(names_folder, "us_female_first.csv"),
        "us_last":   os.path.join(names_folder, "us_surnames.csv"),
        "in_first":  os.path.join(names_folder, "india_first.csv"),
        "in_last":   os.path.join(names_folder, "india_last.csv"),
        "eu_first":  os.path.join(names_folder, "eu_first.csv"),
        "eu_last":   os.path.join(names_folder, "eu_last.csv"),
    }

    us_male   = load_list(paths["us_male"])
    us_female = load_list(paths["us_female"])
    us_last   = load_list(paths["us_last"])
    in_first  = load_list(paths["in_first"])
    in_last   = load_list(paths["in_last"])
    eu_first  = load_list(paths["eu_first"])
    eu_last   = load_list(paths["eu_last"])

    # -----------------------------------------------------
    # Load Geography
    # -----------------------------------------------------
    geography, _ = load_dimension(
        "geography",
        parquet_dims_folder,
        cfg["geography"]
    )

    geo_keys = geography["GeographyKey"].to_numpy()

    # -----------------------------------------------------
    # Allocate arrays
    # -----------------------------------------------------
    N = total_customers

    CustomerKey = np.arange(1, N + 1)

    active_count = int(np.floor(N * active_ratio))
    if active_count == 0:
        raise ValueError(
            "customers.active_ratio results in zero active customers; "
            "increase active_ratio or total_customers"
        )

    active_customer_keys = (
        rng.choice(CustomerKey, size=active_count, replace=False)
        if active_count < N
        else CustomerKey
    )

    active_customer_set = set(active_customer_keys)

    Region = rng.choice(
        ["IN", "US", "EU"],
        size=N,
        p=[pct_india / 100, pct_us / 100, pct_eu / 100],
    )

    IsOrg = rng.random(N) < (pct_org / 100)

    Gender = np.empty(N, dtype=object)
    Gender[~IsOrg] = rng.choice(["Male", "Female"], size=(~IsOrg).sum())
    Gender[IsOrg] = "Org"

    GeographyKey = rng.choice(geo_keys, size=N, replace=True)

    # -----------------------------------------------------
    # Names
    # -----------------------------------------------------
    FirstName = np.empty(N, dtype=object)
    LastName = np.empty(N, dtype=object)

    mask = (Region == "IN") & (~IsOrg)
    LastName[mask] = rng.choice(in_last, size=mask.sum())

    mask = (Region == "US") & (~IsOrg)
    LastName[mask] = rng.choice(us_last, size=mask.sum())

    mask = (Region == "EU") & (~IsOrg)
    LastName[mask] = rng.choice(eu_last, size=mask.sum())

    LastName[IsOrg] = None

    mask = (Region == "IN") & (~IsOrg)
    FirstName[mask] = rng.choice(in_first, size=mask.sum())

    mask = (Region == "US") & (~IsOrg) & (Gender == "Male")
    FirstName[mask] = rng.choice(us_male, size=mask.sum())

    mask = (Region == "US") & (~IsOrg) & (Gender == "Female")
    FirstName[mask] = rng.choice(us_female, size=mask.sum())

    mask = (Region == "EU") & (~IsOrg)
    FirstName[mask] = rng.choice(eu_first, size=mask.sum())

    FirstName[IsOrg] = None

    safe_first = np.where(FirstName == None, "", FirstName.astype(str))
    safe_last  = np.where(LastName  == None, "", LastName.astype(str))

    # -----------------------------------------------------
    # Organization handling
    # -----------------------------------------------------
    company_pool = np.array([
        "TechNova", "BrightWave", "ZenithSystems", "PrimeSource",
        "ApexCorp", "GlobalWorks", "VertexInnovations",
        "OmniSoft", "NimbusSolutions", "SilverlineTech"
    ])

    CompanyName = np.empty(N, dtype=object)
    CompanyName[IsOrg] = company_pool[rng.integers(0, len(company_pool), size=IsOrg.sum())]
    CompanyName[~IsOrg] = None

    safe_company = np.where(CompanyName == None, "", CompanyName.astype(str))
    OrgDomain = np.where(IsOrg, np.char.lower(safe_company) + ".com", None)

    # -----------------------------------------------------
    # Emails
    # -----------------------------------------------------
    Email = np.empty(N, dtype=object)
    person_mask = ~IsOrg

    email_domain = rng.choice(PERSONAL_EMAIL_DOMAINS, size=person_mask.sum())

    Email[person_mask] = (
        np.char.lower(safe_first[person_mask]) + "." +
        np.char.lower(safe_last[person_mask]) +
        rng.integers(10, 99999, size=person_mask.sum()).astype(str) +
        "@" + email_domain
    )

    Email[IsOrg] = "info@" + OrgDomain[IsOrg]

    # -----------------------------------------------------
    # CustomerName
    # -----------------------------------------------------
    CustomerName = np.where(
        IsOrg,
        "Organization " + CustomerKey.astype(str),
        safe_first + " " + safe_last,
    )

    # -----------------------------------------------------
    # Demographics
    # -----------------------------------------------------
    BirthDate = np.empty(N, dtype=object)
    if person_mask.sum():
        ages = rng.integers(18 * 365, 70 * 365, size=person_mask.sum())
        dates = pd.Timestamp("today").normalize() - pd.to_timedelta(ages, unit="D")
        BirthDate[person_mask] = pd.to_datetime(dates).date

    BirthDate[IsOrg] = None

    MaritalStatus = np.empty(N, dtype=object)
    MaritalStatus[~IsOrg] = rng.choice(
        ["Married", "Single"],
        size=(~IsOrg).sum(),
        p=[0.55, 0.45],
    )
    MaritalStatus[IsOrg] = None

    YearlyIncome = np.where(
        IsOrg,
        None,
        rng.integers(20000, 200000, size=N)
    )

    TotalChildren = pd.Series(
        np.where(IsOrg, pd.NA, rng.integers(0, 5, size=N)),
        dtype="Int64",
    )

    Education = np.where(
        IsOrg,
        None,
        rng.choice(
            ["High School", "Bachelors", "Masters", "PhD"],
            size=N,
            p=[0.2, 0.5, 0.25, 0.05],
        ),
    )

    Occupation = np.where(
        IsOrg,
        None,
        rng.choice(
            ["Professional", "Clerical", "Skilled", "Service", "Executive"],
            size=N,
            p=[0.5, 0.2, 0.15, 0.1, 0.05],
        ),
    )

    # -----------------------------------------------------
    # Final DataFrame
    # -----------------------------------------------------
    df = pd.DataFrame({
        "CustomerKey": CustomerKey,
        "CustomerName": CustomerName,
        "DOB": BirthDate,
        "MaritalStatus": MaritalStatus,
        "Gender": Gender,
        "EmailAddress": Email,
        "YearlyIncome": YearlyIncome,
        "TotalChildren": TotalChildren,
        "Education": Education,
        "Occupation": Occupation,
        "CustomerType": np.where(IsOrg, "Organization", "Person"),
        "CompanyName": CompanyName,
        "GeographyKey": GeographyKey,
        "IsActiveInSales": np.isin(CustomerKey, list(active_customer_set)).astype("int8"),
    })

    return df, active_customer_set


# ---------------------------------------------------------
# Pipeline entry
# ---------------------------------------------------------
def run_customers(cfg, parquet_folder: Path):
    out_path = parquet_folder / "customers.parquet"

    cust_cfg = cfg["customers"]
    force = cust_cfg.get("_force_regenerate", False)

    if not force and not should_regenerate("customers", cust_cfg, out_path):
        skip("Customers up-to-date; skipping.")
        return

    with stage("Generating Customers"):
        df, active_customer_keys = generate_synthetic_customers(cfg, parquet_folder)
        df.to_parquet(out_path, index=False)

    save_version("customers", cust_cfg, out_path)
    info(f"Customers dimension written: {out_path}")
