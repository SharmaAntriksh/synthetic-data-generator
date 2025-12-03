# ---------------------------------------------------------
#  STORES DIMENSION (PIPELINE READY)
# ---------------------------------------------------------

import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.logging_utils import info, fail, skip, stage
from src.pipeline.versioning import should_regenerate, save_version
from src.pipeline.dimension_loader import load_dimension


# ---------------------------------------------------------
# ORIGINAL GENERATOR  (unchanged) :contentReference[oaicite:0]{index=0}
# ---------------------------------------------------------

def generate_store_table(
    geography_parquet_path="./data/parquet_dims/geography.parquet",
    num_stores=200,
    opening_start="2018-01-01",
    opening_end="2023-01-31",
    closing_end="2025-12-31",
    seed=42
):
    """
    Generate synthetic store dimension table.
    GeographyKey comes from final DimGeography (parquet).
    """
    rng = np.random.default_rng(seed)

    # --------------------------------------------------------
    # Load Geography
    # --------------------------------------------------------
    geo = pd.read_parquet(geography_parquet_path)

    if "GeographyKey" not in geo.columns:
        raise ValueError(
            f"'GeographyKey' missing in geography parquet. "
            f"Found: {list(geo.columns)}"
        )

    geo_keys = geo["GeographyKey"].astype(int).to_numpy()

    # --------------------------------------------------------
    # Base structure
    # --------------------------------------------------------
    df = pd.DataFrame({"StoreKey": np.arange(1, num_stores + 1)})

    df["StoreName"] = df["StoreKey"].map(lambda x: f"Store #{x:04d}")

    store_types = ["Supermarket", "Convenience", "Online", "Hypermarket"]
    store_status = ["Open", "Closed", "Renovating"]
    close_reasons = ["Low Sales", "Lease Ended", "Renovation", "Moved Location"]

    df["StoreType"] = rng.choice(store_types, num_stores, p=[0.5, 0.3, 0.1, 0.1])
    df["Status"] = rng.choice(store_status, num_stores, p=[0.85, 0.10, 0.05])

    # --------------------------------------------------------
    # GeographyKey assignment
    # --------------------------------------------------------
    df["GeographyKey"] = rng.choice(geo_keys, size=num_stores, replace=True)

    # --------------------------------------------------------
    # Opening Date
    # --------------------------------------------------------
    open_start_ts = pd.Timestamp(opening_start).value // 10**9
    open_end_ts   = pd.Timestamp(opening_end).value   // 10**9

    df["OpeningDate"] = pd.to_datetime(
        rng.integers(open_start_ts, open_end_ts, num_stores),
        unit="s"
    )

    # --------------------------------------------------------
    # Closing Date
    # --------------------------------------------------------
    closing_cutoff_ts = pd.Timestamp(closing_end).value // 10**9

    def compute_close_date(row):
        if row["Status"] != "Closed":
            return pd.NaT
        open_ts = row["OpeningDate"].value // 10**9
        return pd.to_datetime(
            rng.integers(open_ts, closing_cutoff_ts),
            unit="s"
        )

    df["ClosingDate"] = df.apply(compute_close_date, axis=1)

    # --------------------------------------------------------
    # Additional attributes
    # --------------------------------------------------------
    df["OpenFlag"] = (df["Status"] == "Open").astype(int)
    df["SquareFootage"] = rng.integers(2000, 10000, num_stores)
    df["EmployeeCount"] = rng.integers(10, 120, num_stores)

    df["StoreManager"] = df["StoreKey"].map(lambda x: f"Manager {x:04d}")
    df["Phone"] = df["StoreKey"].map(
        lambda x: f"(555) {x % 900 + 100}-{x % 10000:04d}"
    )

    df["StoreDescription"] = (
        df["StoreType"] + " located in GeographyKey " + df["GeographyKey"].astype(str)
    )

    df["CloseReason"] = np.where(
        df["Status"] == "Closed",
        rng.choice(close_reasons, size=num_stores),
        ""
    )

    return df


# ---------------------------------------------------------
#  PIPELINE ENTRYPOINT
# ---------------------------------------------------------

def run_stores(cfg, parquet_folder: Path):
    """
    Pipeline wrapper for store dimension generation.
    Handles:
    - version checks
    - logging
    - geography dim loading
    - writing parquet
    - version saving
    """

    out_path = parquet_folder / "stores.parquet"

    # Use only the stores section for versioning
    store_cfg = cfg["stores"]

    if not should_regenerate("stores", store_cfg, out_path):
        skip("Stores up-to-date; skipping.")
        return

    geo_path = parquet_folder / "geography.parquet"

    with stage("Generating Stores"):
        df = generate_store_table(
            geography_parquet_path=geo_path,
            num_stores=store_cfg.get("num_stores", 200),
            opening_start=store_cfg.get("opening", {}).get("start", "1995-01-01"),
            opening_end=store_cfg.get("opening", {}).get("end", "2020-12-31"),
            closing_end=store_cfg.get("closing_end", "2025-12-31"),
            seed=store_cfg.get("override", {}).get("seed", 42),
        )
        df.to_parquet(out_path, index=False)

    save_version("stores", store_cfg, out_path)
    info(f"Stores dimension written → {out_path}")

    info(f"Stores dimension written → {out_path}")
