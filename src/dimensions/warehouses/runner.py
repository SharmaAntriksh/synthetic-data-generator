"""Warehouse dimension runner.

Config, versioning, and parquet IO around the warehouse allocator in
:mod:`src.dimensions.warehouses.generator`. Reads ``stores.parquet``, allocates
warehouses, writes ``warehouses.parquet``, and stamps the ``WarehouseKey`` FK
back onto ``stores.parquet``.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from src.dimensions.warehouses.generator import generate_warehouse_table
from src.utils.logging_utils import info, skip, stage
from src.utils.output_utils import write_parquet_with_date32
from src.versioning import should_regenerate, save_version
from src.utils.config_helpers import as_dict
from src.utils.config_precedence import resolve_seed

if TYPE_CHECKING:
    from src.engine.config.config_schema import AppConfig


def run_warehouses(cfg: AppConfig, parquet_folder: Path) -> None:
    """Generate and write the Warehouses dimension + update Stores with FK."""
    parquet_folder = Path(parquet_folder)
    out_path = parquet_folder / "warehouses.parquet"
    stores_path = parquet_folder / "stores.parquet"

    if not stores_path.exists():
        raise FileNotFoundError(f"Missing stores parquet: {stores_path}")

    stores_df = pd.read_parquet(str(stores_path))

    wh_cfg = as_dict(getattr(cfg, "warehouses", None)) or {}
    seed = resolve_seed(cfg, wh_cfg, fallback=42)
    min_spw = int(wh_cfg.get("min_stores_per_warehouse", 15))
    min_sow = int(wh_cfg.get("min_stores_for_own_warehouse", 5))

    version_cfg = dict(wh_cfg)
    version_cfg["schema_version"] = 3  # v3: resolved seed folded into version key
    version_cfg["seed"] = int(seed)
    version_cfg["_n_stores"] = len(stores_df)
    if "StoreZone" in stores_df.columns:
        version_cfg["_zones"] = sorted(stores_df["StoreZone"].unique().tolist())

    if not should_regenerate("warehouses", version_cfg, out_path):
        skip("Warehouses up-to-date")
        return

    with stage("Generating Warehouses"):
        warehouses_df, store_to_wh = generate_warehouse_table(
            stores_df, parquet_folder, seed=seed,
            min_stores_per_warehouse=min_spw,
            min_stores_for_own_warehouse=min_sow,
        )

        write_parquet_with_date32(
            warehouses_df, out_path,
            cast_all_datetime=False,
            compression="snappy",
        )

        # Add WarehouseKey FK to stores and overwrite.
        # NOTE: this rewrite hardcodes cast_all_datetime=True + compression="snappy",
        # ignoring stores' configured parquet_compression / force_date32. Consolidating
        # stores.parquet under a single writer is tracked in docs/EMPLOYEES_REFACTOR_PLAN.md.
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
