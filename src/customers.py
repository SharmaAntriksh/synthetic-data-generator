import os
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime
import csv


# ============================================================
# Utility to load name list files
# ============================================================
def load_list(path):
    """Load name lists from a CSV/txt file (one name per line)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing file: {path}")
    s = pd.read_csv(path, header=None, dtype=str)[0].str.strip()
    s = s[s.str.match(r"^[A-Za-z\-\'. ]+$")]  # keep valid names
    return s.str.title().unique()


# ============================================================
# MAIN FUNCTION
# ============================================================
def generate_synthetic_customers(
    total_customers=1000,
    total_geos=40,
    pct_india=1,
    pct_us=60,
    pct_eu=39,
    pct_org=2,
    seed=42,
    names_folder="./data/customer_names",
    save_geography_csv=False,
    save_customer_csv=False,
    out_geo="SyntheticGeography.csv",
    out_cust="SyntheticCustomers.csv"
):

    # ---------------- Setup random seeds ----------------
    rng = np.random.default_rng(seed)
    fake = Faker()
    fake_in = Faker("en_IN")
    fake_us = Faker("en_US")
    fake_eu = Faker("en_GB")

    # ============================================================
    # LOAD NAME DATASETS
    # ============================================================
    print("Loading name datasets...")

    PATHS = {
        "us_male":  os.path.join(names_folder, "us_male_first.csv"),
        "us_female": os.path.join(names_folder, "us_female_first.csv"),
        "us_last": os.path.join(names_folder, "us_surnames.csv"),
        "in_first": os.path.join(names_folder, "india_first.csv"),
        "in_last": os.path.join(names_folder, "india_last.csv"),
        "eu_first": os.path.join(names_folder, "eu_first.csv"),
        "eu_last": os.path.join(names_folder, "eu_last.csv"),
    }

    us_male  = load_list(PATHS["us_male"])
    us_female = load_list(PATHS["us_female"])
    us_last  = load_list(PATHS["us_last"])
    in_first = load_list(PATHS["in_first"])
    in_last  = load_list(PATHS["in_last"])
    eu_first = load_list(PATHS["eu_first"])
    eu_last  = load_list(PATHS["eu_last"])

    print(f"  US male names: {len(us_male):,}")
    print(f"  US female names: {len(us_female):,}")
    print(f"  US surnames: {len(us_last):,}")
    print(f"  IN first names: {len(in_first):,}")
    print(f"  IN last names: {len(in_last):,}")
    print(f"  EU first names: {len(eu_first):,}")
    print(f"  EU last names: {len(eu_last):,}")


    # ============================================================
    # GEOGRAPHY GENERATION
    # ============================================================
    # print("\nGenerating synthetic geography...")

    GEO_City = [fake.city() for _ in range(total_geos)]
    GEO_State = [fake.state() for _ in range(total_geos)]
    GEO_Country = [fake.country() for _ in range(total_geos)]
    GEO_Type = rng.choice(["Urban","Suburban","Rural"], size=total_geos, p=[0.6,0.25,0.15])

    df_geo = pd.DataFrame({
        "GeographyKey": np.arange(1, total_geos+1),
        "City": GEO_City,
        "State": GEO_State,
        "Country": GEO_Country,
        "GeographyType": GEO_Type
    })

    if save_geography_csv:
        df_geo.to_csv(out_geo, index=False)
        print(f"Saved {out_geo}")


    # ============================================================
    # CUSTOMER GENERATION
    # ============================================================
    # print("\nGenerating synthetic customers...")

    N = total_customers
    CustomerKey = np.arange(1, N+1)

    Region = rng.choice(["IN","US","EU"], size=N, p=[pct_india/100, pct_us/100, pct_eu/100])
    IsOrg = rng.random(N) < (pct_org / 100)
    Gender = rng.choice(["M","F"], size=N)
    Gender[IsOrg] = None

    GeographyKey = rng.integers(1, total_geos+1, size=N)

    FirstName = np.empty(N, dtype=object)
    LastName  = np.empty(N, dtype=object)

    # LAST NAMES BY REGION
    mask = (Region=="IN") & (~IsOrg)
    LastName[mask] = rng.choice(in_last, size=mask.sum())
    mask = (Region=="US") & (~IsOrg)
    LastName[mask] = rng.choice(us_last, size=mask.sum())
    mask = (Region=="EU") & (~IsOrg)
    LastName[mask] = rng.choice(eu_last, size=mask.sum())

    LastName[IsOrg] = None

    # FIRST NAMES
    mask = (Region=="IN") & (~IsOrg)
    FirstName[mask] = rng.choice(in_first, size=mask.sum())

    mask = (Region=="US") & (~IsOrg) & (Gender=="M")
    FirstName[mask] = rng.choice(us_male, size=mask.sum())
    mask = (Region=="US") & (~IsOrg) & (Gender=="F")
    FirstName[mask] = rng.choice(us_female, size=mask.sum())

    mask = (Region=="EU") & (~IsOrg)
    FirstName[mask] = rng.choice(eu_first, size=mask.sum())

    FirstName[IsOrg] = None

    safe_first = np.where(FirstName == None, "", FirstName.astype(str))
    safe_last  = np.where(LastName  == None, "", LastName.astype(str))

    # ------------------------------------------------------------
    # ORGANIZATION NAMES
    # ------------------------------------------------------------
    company_pool = np.array([
        "TechNova","BrightWave","ZenithSystems","PrimeSource",
        "ApexCorp","GlobalWorks","VertexInnovations",
        "OmniSoft","NimbusSolutions","SilverlineTech"
    ])

    CompanyName = np.empty(N, dtype=object)
    CompanyName[IsOrg] = company_pool[rng.integers(0, len(company_pool), size=IsOrg.sum())]
    CompanyName[~IsOrg] = None

    safe_company = np.where(CompanyName==None, "", CompanyName.astype(str))
    OrgDomain = np.where(IsOrg, np.char.lower(safe_company) + ".com", None)

    # ------------------------------------------------------------
    # EMAIL
    # ------------------------------------------------------------
    Email = np.empty(N, dtype=object)

    mask = ~IsOrg
    Email[mask] = (
        np.char.lower(safe_first[mask]) + "." +
        np.char.lower(safe_last[mask]) +
        rng.integers(10,99999,size=mask.sum()).astype(str) +
        "@example.com"
    )
    Email[IsOrg] = "info@" + OrgDomain[IsOrg]

    # ------------------------------------------------------------
    # NAMES
    # ------------------------------------------------------------
    CustomerName = np.where(
        IsOrg,
        "Organization " + CustomerKey.astype(str),
        safe_first + " " + safe_last
    )

    # ------------------------------------------------------------
    # DEMOGRAPHICS
    # ------------------------------------------------------------
    BirthDate = np.empty(N, dtype=object)
    person_mask = ~IsOrg

    if person_mask.sum():
        ages = rng.integers(18*365, 70*365, size=person_mask.sum())
        BirthDate_vals = (
            pd.Timestamp("today").normalize() - pd.to_timedelta(ages, unit="D")
        )
        BirthDate_vals = pd.Series(BirthDate_vals).dt.date
        BirthDate[person_mask] = BirthDate_vals

    BirthDate[IsOrg] = None

    MaritalStatus = np.where(IsOrg, None, rng.choice(["M","S"], size=N))
    YearlyIncome = np.where(IsOrg, None, rng.integers(20000,200000,size=N))
    TotalChildren = pd.Series(
        np.where(IsOrg, pd.NA, rng.integers(0,5,size=N)),
        dtype="Int64"    # nullable integer type
    )


    Education = np.where(
        IsOrg,
        None,
        rng.choice(["High School","Bachelors","Masters","PhD"],
                   size=N, p=[0.2,0.5,0.25,0.05])
    )

    Occupation = np.where(
        IsOrg,
        None,
        rng.choice(["Professional","Clerical","Skilled","Service","Executive"],
                   size=N, p=[0.5,0.2,0.15,0.1,0.05])
    )

    # ------------------------------------------------------------
    # FINAL DATAFRAME
    # ------------------------------------------------------------
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
    })

    # df = df.merge(df_geo, on="GeographyKey", how="left")

    # df["ETLLoadID"] = 1
    # df["LoadDate"] = pd.Timestamp("today").normalize()
    # df["UpdateDate"] = df["LoadDate"]

    # Save if required
    if save_customer_csv:
        df.to_csv(out_cust, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)
        print(f"Saved {out_cust}")

    print(f"\nGenerated {len(df):,} synthetic customers.")

    return df, df_geo
