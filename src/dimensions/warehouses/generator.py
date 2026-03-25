"""Warehouse dimension generator.

Derives one warehouse per unique (StoreZone, Country) from the stores
dimension, plus a dedicated online fulfillment warehouse.  Each store
is assigned to exactly one warehouse (many-to-one).

Output columns:
  WarehouseKey, WarehouseName, WarehouseType, Zone, Country,
  GeographyKey, Capacity, SquareFootage
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.defaults import (
    ONLINE_STORE_KEY_BASE,
    ONLINE_WAREHOUSE_KEY,
    WAREHOUSE_TYPES,
    WAREHOUSE_TYPES_P,
)
from src.utils.logging_utils import info, skip, stage
from src.utils.output_utils import write_parquet_with_date32
from src.versioning import should_regenerate, save_version
from src.utils.config_helpers import as_dict
from src.utils.config_precedence import resolve_seed


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def _rep_geo_key(series: pd.Series) -> int:
    """Most common GeographyKey in the group, or 0 if empty."""
    if series.empty:
        return 0
    return int(series.astype(np.int64).mode().iloc[0])


def generate_warehouse_table(
    stores_df: pd.DataFrame,
    parquet_dims: Path,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict[int, int]]:
    """Generate warehouse dimension from store distribution.

    Returns:
        (warehouses_df, store_to_warehouse)
        where store_to_warehouse maps StoreKey → WarehouseKey.
    """
    rng = np.random.default_rng(seed)

    sk = stores_df["StoreKey"].astype(np.int64)
    physical = stores_df[sk < ONLINE_STORE_KEY_BASE].copy()
    online = stores_df[sk >= ONLINE_STORE_KEY_BASE].copy()

    if "StoreZone" not in physical.columns:
        raise ValueError("stores.parquet missing StoreZone column")

    zone_col = physical["StoreZone"].astype(str)
    country_col = _resolve_country(physical, parquet_dims)

    groups = physical.groupby([zone_col.rename("Zone"), country_col.rename("Country")])

    rows = []
    store_to_wh: Dict[int, int] = {}
    wk = 1

    for (zone, country), grp in sorted(groups, key=lambda x: x[0]):
        n_stores = len(grp)
        wtype = rng.choice(WAREHOUSE_TYPES, p=WAREHOUSE_TYPES_P)
        rep_geo = _rep_geo_key(grp["GeographyKey"])

        capacity = int(rng.integers(5_000, 15_000) * n_stores)
        sqft = int(rng.integers(50_000, 200_000) + n_stores * 5_000)

        rows.append({
            "WarehouseKey": np.int32(wk),
            "WarehouseName": f"{zone} {country} {wtype}",
            "WarehouseType": wtype,
            "Zone": zone,
            "Country": country,
            "GeographyKey": np.int32(rep_geo),
            "Capacity": np.int32(capacity),
            "SquareFootage": np.int32(sqft),
        })

        for s in grp["StoreKey"].astype(int):
            store_to_wh[s] = wk

        wk += 1

    # --- Online fulfillment warehouse ---
    online_geo = _rep_geo_key(online.get("GeographyKey", pd.Series(dtype="int64")))

    rows.append({
        "WarehouseKey": np.int32(ONLINE_WAREHOUSE_KEY),
        "WarehouseName": f"Online {WAREHOUSE_TYPES[2]}",
        "WarehouseType": WAREHOUSE_TYPES[2],
        "Zone": "Online",
        "Country": "Global",
        "GeographyKey": np.int32(online_geo),
        "Capacity": np.int32(rng.integers(100_000, 500_000)),
        "SquareFootage": np.int32(rng.integers(200_000, 800_000)),
    })

    for s in online["StoreKey"].astype(int):
        store_to_wh[s] = ONLINE_WAREHOUSE_KEY

    warehouses_df = pd.DataFrame(rows)
    n_physical = len(warehouses_df) - 1  # exclude online
    info(
        f"Warehouses: {n_physical} physical + 1 online = "
        f"{len(warehouses_df)} total, serving {len(store_to_wh)} stores"
    )

    return warehouses_df, store_to_wh


def _resolve_country(stores_df: pd.DataFrame, parquet_dims: Path) -> pd.Series:
    """Resolve country for each store by joining to geography.parquet."""
    # If a Country column already exists on stores, use it directly
    if "Country" in stores_df.columns:
        return stores_df["Country"].astype(str).fillna("Unknown")

    # Join to geography dimension to get Country
    geo_path = parquet_dims / "geography.parquet"
    if geo_path.exists() and "GeographyKey" in stores_df.columns:
        geo = pd.read_parquet(str(geo_path), columns=["GeographyKey", "Country"])
        merged = stores_df[["GeographyKey"]].merge(geo, on="GeographyKey", how="left")
        return merged["Country"].astype(str).fillna("Unknown")

    # Last resort: zone as country proxy
    return stores_df["StoreZone"].astype(str).fillna("Unknown")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_warehouses(cfg, parquet_folder: Path) -> None:
    """Generate and write the Warehouses dimension + update Stores with FK."""
    parquet_folder = Path(parquet_folder)
    out_path = parquet_folder / "warehouses.parquet"
    stores_path = parquet_folder / "stores.parquet"

    if not stores_path.exists():
        raise FileNotFoundError(f"Missing stores parquet: {stores_path}")

    stores_df = pd.read_parquet(str(stores_path))

    wh_cfg = as_dict(getattr(cfg, "warehouses", None)) or {}
    seed = resolve_seed(cfg, wh_cfg, fallback=42)

    version_cfg = dict(wh_cfg)
    version_cfg["schema_version"] = 1
    version_cfg["_n_stores"] = len(stores_df)
    if "StoreZone" in stores_df.columns:
        version_cfg["_zones"] = sorted(stores_df["StoreZone"].unique().tolist())

    if not should_regenerate("warehouses", version_cfg, out_path):
        skip("Warehouses up-to-date")
        return

    with stage("Generating Warehouses"):
        warehouses_df, store_to_wh = generate_warehouse_table(stores_df, parquet_folder, seed=seed)

        write_parquet_with_date32(
            warehouses_df, out_path,
            cast_all_datetime=False,
            compression="snappy",
        )

        # Add WarehouseKey FK to stores and overwrite
        stores_df["WarehouseKey"] = (
            stores_df["StoreKey"].astype(int).map(store_to_wh).astype("Int32")
        )
        write_parquet_with_date32(
            stores_df, stores_path,
            cast_all_datetime=True,
            compression="snappy",
        )
        info(f"Updated {stores_path.name} with WarehouseKey column")

    save_version("warehouses", version_cfg, out_path)
    info(f"Warehouses written: {out_path.name}")
