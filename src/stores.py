import pandas as pd
import numpy as np

def generate_store_table(
    geography_parquet_path,
    num_stores=200,
    opening_start="2018-01-01",
    opening_end="2023-01-31",
    closing_end="2025-12-31",
    seed=42
):
    np.random.seed(seed)

    # Load geography
    geo = pd.read_parquet(geography_parquet_path)
    if "GeographyKey" not in geo.columns:
        raise ValueError("Parquet must contain GeographyKey column.")
    geo_keys = geo["GeographyKey"].tolist()

    store_types = ["Supermarket", "Convenience", "Online", "Hypermarket"]
    store_status = ["Open", "Closed", "Renovating"]
    close_reasons = ["Low Sales", "Lease Ended", "Renovation", "Moved Location"]

    df = pd.DataFrame({"StoreKey": range(1, num_stores + 1)})

    df["StoreName"] = df["StoreKey"].apply(lambda x: f"Store #{x:04d}")
    df["StoreType"] = np.random.choice(store_types, num_stores, p=[0.5, 0.3, 0.1, 0.1])
    df["Status"] = np.random.choice(store_status, num_stores, p=[0.85, 0.10, 0.05])

    # GeographyKey
    df["GeographyKey"] = np.random.choice(geo_keys, num_stores)

    # Opening date
    df["OpeningDate"] = pd.to_datetime(
        np.random.randint(
            pd.Timestamp(opening_start).value // 10**9,
            pd.Timestamp(opening_end).value // 10**9,
            num_stores
        ),
        unit="s"
    )

    # Closing date only if closed
    df["ClosingDate"] = df.apply(
        lambda r: pd.to_datetime(
            np.random.randint(
                r["OpeningDate"].value // 10**9,
                pd.Timestamp(closing_end).value // 10**9
            ),
            unit="s"
        ) if r["Status"] == "Closed" else pd.NaT,
        axis=1
    )

    # OpenFlag
    df["OpenFlag"] = df["Status"].apply(lambda s: 1 if s == "Open" else 0)

    # Basic fields
    df["SquareFootage"] = np.random.randint(2000, 10000, num_stores)
    df["EmployeeCount"] = np.random.randint(10, 120, num_stores)
    df["StoreManager"] = df["StoreKey"].apply(lambda x: f"Manager {x:04d}")
    df["Phone"] = df["StoreKey"].apply(lambda x: f"(555) {x%900+100}-{x%10000:04d}")

    # Description
    df["StoreDescription"] = (
        df["StoreType"] + " located in GeographyKey " + df["GeographyKey"].astype(str)
    )

    # Close reason
    df["CloseReason"] = df.apply(
        lambda r: np.random.choice(close_reasons) if r["Status"] == "Closed" else "",
        axis=1
    )

    return df
