"""Inventory snapshot pipeline runner.

Called from sales_runner.py after sales generation completes.
Uses the InventoryAccumulator that was populated during sales generation.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.logging_utils import stage, info, done, short_path

from .accumulator import InventoryAccumulator
from .engine import load_inventory_config, compute_inventory_snapshots


def run_inventory_pipeline(
    *,
    accumulator: InventoryAccumulator,
    parquet_dims: Path,
    fact_out: Path,
    cfg: Dict[str, Any],
    file_format: str = "parquet",
) -> Optional[Dict[str, Any]]:
    """
    Generate InventorySnapshot fact table from streaming-aggregated sales demand.

    Args:
        accumulator:   InventoryAccumulator populated during sales generation
        parquet_dims:  path to generated dimension parquets
        fact_out:      path to write inventory fact outputs
        cfg:           full config dict
        file_format:   "csv" | "parquet" | "deltaparquet"

    Returns:
        summary dict or None if inventory is disabled / no data
    """
    icfg = load_inventory_config(cfg)
    if not icfg.enabled:
        info("Inventory snapshot generation: disabled in config")
        return None

    if not accumulator.has_data:
        info("Inventory snapshot generation: no sales demand accumulated, skipping")
        return None

    t0 = time.time()

    demand = accumulator.finalize()

    n_pairs = demand.groupby(["ProductKey", "StoreKey"]).ngroups
    months_per_pair = demand.groupby(["ProductKey", "StoreKey"]).size()
    qualified_pairs = int((months_per_pair >= icfg.min_demand_months).sum())
    info(
        f"Inventory demand: {len(demand):,} monthly demand rows "
        f"({n_pairs:,} product-store pairs, "
        f"{qualified_pairs:,} with {icfg.min_demand_months}+ months of demand, "
        f"{demand['Year'].nunique()} years)"
    )

    snapshots = compute_inventory_snapshots(
        demand=demand,
        parquet_dims=parquet_dims,
        icfg=icfg,
    )

    inv_out = fact_out / "inventory"
    inv_out.mkdir(parents=True, exist_ok=True)

    _write_inventory(snapshots, inv_out, "inventory_snapshot", file_format)

    elapsed = time.time() - t0
    n_rows = len(snapshots)

    stockout_pct = 0.0
    if n_rows > 0:
        stockout_pct = float(snapshots["StockoutFlag"].sum()) / n_rows * 100

    info(
        f"Inventory snapshot: {n_rows:,} rows, "
        f"{stockout_pct:.1f}% stockout rate, "
        f"{elapsed:.1f}s"
    )

    return {
        "rows": n_rows,
        "product_store_pairs": n_pairs,
        "stockout_pct": round(stockout_pct, 2),
        "elapsed_sec": round(elapsed, 2),
    }


def _write_inventory(
    df: pd.DataFrame,
    out_dir: Path,
    name: str,
    file_format: str,
) -> None:
    """Write an inventory DataFrame in the requested format."""
    table = pa.Table.from_pandas(df, preserve_index=False)

    if file_format == "deltaparquet":
        delta_dir = out_dir / name
        delta_dir.mkdir(parents=True, exist_ok=True)
        try:
            from deltalake import write_deltalake
        except ImportError:
            from deltalake.writer import write_deltalake
        write_deltalake(str(delta_dir), table, mode="overwrite")
        info(f"Wrote {name}: {len(df):,} rows -> {short_path(delta_dir)}/")
        return

    parquet_path = out_dir / f"{name}.parquet"
    pq.write_table(
        table, str(parquet_path),
        compression="snappy",
        row_group_size=500_000,
        use_dictionary=True,
    )

    if file_format == "csv":
        csv_path = out_dir / f"{name}.csv"
        csv_df = _prepare_inventory_csv(df)
        csv_df.to_csv(str(csv_path), index=False)
        info(f"Wrote {name}: {len(df):,} rows -> {short_path(csv_path)}")
    else:
        info(f"Wrote {name}: {len(df):,} rows -> {short_path(parquet_path)}")


_INVENTORY_CSV_COLUMNS = [
    "ProductKey", "StoreKey", "SnapshotDate",
    "QuantityOnHand", "QuantityOnOrder",
    "QuantitySold", "QuantityReceived",
    "ReorderFlag", "StockoutFlag", "DaysOutOfStock",
]

_INVENTORY_CSV_INT_COLS = (
    "ProductKey", "StoreKey",
    "QuantityOnHand", "QuantityOnOrder",
    "QuantitySold", "QuantityReceived",
    "ReorderFlag", "StockoutFlag", "DaysOutOfStock",
)


def _prepare_inventory_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in _INVENTORY_CSV_COLUMNS:
        if col not in out.columns:
            out[col] = None
    out = out[_INVENTORY_CSV_COLUMNS]

    for col in _INVENTORY_CSV_INT_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    return out
