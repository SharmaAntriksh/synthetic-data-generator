# ---------------------------------------------------------
#  GEOGRAPHY DIMENSION (PIPELINE READY)
# ---------------------------------------------------------

import pandas as pd
import numpy as np
from pathlib import Path

from src.utils.logging_utils import info, fail, skip, stage
from src.versioning.version_store import should_regenerate, save_version


# =============================================================
# CURATED GEOGRAPHY ROWS  (unchanged)
# =============================================================
CURATED_ROWS = [
    ("New York", "NY", "United States", "North America", "USD"),
    ("Los Angeles", "CA", "United States", "North America", "USD"),
    ("Chicago", "IL", "United States", "North America", "USD"),
    ("Houston", "TX", "United States", "North America", "USD"),
    ("Miami", "FL", "United States", "North America", "USD"),

    ("Toronto", "ON", "Canada", "North America", "CAD"),
    ("Vancouver", "BC", "Canada", "North America", "CAD"),
    ("Montreal", "QC", "Canada", "North America", "CAD"),

    ("London", "London", "United Kingdom", "Europe", "GBP"),
    ("Manchester", "Manchester", "United Kingdom", "Europe", "GBP"),

    ("Berlin", "Berlin", "Germany", "Europe", "EUR"),
    ("Munich", "Bavaria", "Germany", "Europe", "EUR"),

    ("Paris", "Île-de-France", "France", "Europe", "EUR"),
    ("Lyon", "Auvergne-Rhône-Alpes", "France", "Europe", "EUR"),

    ("Mumbai", "MH", "India", "Asia", "INR"),
    ("Delhi", "DL", "India", "Asia", "INR"),
    ("Bengaluru", "KA", "India", "Asia", "INR"),

    ("Sydney", "NSW", "Australia", "Oceania", "AUD"),
    ("Melbourne", "VIC", "Australia", "Oceania", "AUD"),
]


# =============================================================
# ORIGINAL GENERATOR (unchanged, except no disk write)
# =============================================================

def build_dim_geography(cfg):
    """
    Build curated + weighted geography dimension.
    Filtering based on allowed currencies from exchange_rates config.
    """

    allowed_iso = set(cfg["exchange_rates"]["currencies"])
    gcfg = cfg["geography"]
    target_rows = gcfg.get("target_rows", 200)

    # ----------------------------------------------
    # Convert curated rows → DataFrame
    # ----------------------------------------------
    df = pd.DataFrame(
        CURATED_ROWS,
        columns=["City", "State", "Country", "Continent", "ISOCode"],
    )

    # ----------------------------------------------
    # Filter by allowed currency codes
    # ----------------------------------------------
    before = len(df)
    df = df[df["ISOCode"].isin(allowed_iso)].reset_index(drop=True)
    after = len(df)

    if after == 0:
        raise ValueError(
            f"No geography rows remain after filtering by allowed currencies: {sorted(allowed_iso)}"
        )

    # ----------------------------------------------
    # Country weights
    # ----------------------------------------------
    country_weights = gcfg.get("country_weights", {})

    total_w = sum(country_weights.values())
    if total_w > 0 and total_w != 1.0:
        for k in country_weights:
            country_weights[k] /= total_w

    df["Weight"] = df["Country"].apply(
        lambda c: country_weights.get(c, country_weights.get("Rest", 0))
    )

    wsum = df["Weight"].sum()
    if wsum == 0:
        raise ValueError("All country weights resolved to zero. Check config.")
    df["Weight"] = df["Weight"] / wsum

    # ----------------------------------------------
    # Weighted sampling
    # ----------------------------------------------
    sampled = df.sample(
        n=target_rows,
        replace=True,
        weights=df["Weight"],
        random_state=42,
    ).reset_index(drop=True)

    # Assign deterministic keys
    sampled.insert(0, "GeographyKey", np.arange(1, target_rows + 1))

    return sampled


# =============================================================
# PIPELINE WRAPPER
# =============================================================

def run_geography(cfg, parquet_folder: Path):
    """
    Pipeline wrapper for geography dimension.
    Handles:
    - version check
    - logging
    - writing parquet
    - saving version
    """
    out_path = parquet_folder / "geography.parquet"

    geo_cfg = cfg["geography"]
    force = geo_cfg.get("_force_regenerate", False)

    if not force and not should_regenerate("geography", geo_cfg, out_path):
        skip("Geography up-to-date; skipping.")
        return

    with stage("Generating Geography"):
        df = build_dim_geography(cfg)

        # Explicit domain column order (no SQL dependency)
        output_cols = [
            "GeographyKey",
            "City",
            "State",
            "Country",
            "Continent",
            "ISOCode",
        ]

        # Drop internal columns like Weight
        df = df[output_cols]
        df.to_parquet(out_path, index=False)

    save_version("geography", geo_cfg, out_path)

    info(f"Geography dimension written: {out_path}")
