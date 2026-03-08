"""Inventory snapshot pipeline runner.

Called from sales_runner.py after sales generation completes.
Uses the InventoryAccumulator that was populated during sales generation.

For large datasets (many product-store pairs), partitions demand by store
groups and runs the simulation in parallel across multiple processes.
Each worker writes its own chunk files directly (CSV + parquet), avoiding
the need to build a single massive DataFrame in one process.
"""
from __future__ import annotations

import dataclasses
import time
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.logging_utils import info, work, short_path

from .accumulator import InventoryAccumulator
from .engine import load_inventory_config, compute_inventory_snapshots, InventoryConfig
from .worker import _inventory_worker_task

# Below this pair count, run single-process (overhead of spawning isn't worth it)
_PARALLEL_THRESHOLD = 50_000


def run_inventory_pipeline(
    *,
    accumulator: InventoryAccumulator,
    parquet_dims: Path,
    fact_out: Path,
    cfg: Dict[str, Any],
    file_format: str = "parquet",
    workers: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """
    Generate InventorySnapshot fact table from streaming-aggregated sales demand.

    For large datasets, automatically partitions by store groups and uses
    multiprocessing for parallel simulation + chunked writes.
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
    n_stores = demand["StoreKey"].nunique()
    info(
        f"Inventory demand: {len(demand):,} monthly demand rows "
        f"({n_pairs:,} product-store pairs, "
        f"{qualified_pairs:,} with {icfg.min_demand_months}+ months of demand, "
        f"{n_stores} stores, "
        f"{demand['Year'].nunique()} years)"
    )

    inv_out = fact_out / "inventory"
    inv_out.mkdir(parents=True, exist_ok=True)

    if qualified_pairs >= _PARALLEL_THRESHOLD and n_stores >= 2:
        result = _run_parallel(demand, parquet_dims, icfg, inv_out, file_format, n_stores, workers=workers)
    else:
        result = _run_single(demand, parquet_dims, icfg, inv_out, file_format)

    elapsed = time.time() - t0
    n_rows = result["rows"]
    stockout_pct = result["stockout_pct"]

    info(
        f"Inventory snapshot: {n_rows:,} rows, "
        f"{stockout_pct:.1f}% stockout rate, "
        f"{elapsed:.1f}s"
    )

    result["elapsed_sec"] = round(elapsed, 2)
    return result


# ------------------------------------------------------------------
# Single-process path (original behavior, for small datasets)
# ------------------------------------------------------------------

def _run_single(
    demand: pd.DataFrame,
    parquet_dims: Path,
    icfg: InventoryConfig,
    inv_out: Path,
    file_format: str,
) -> Dict[str, Any]:
    """Original monolithic path — used for small datasets."""
    snapshots = compute_inventory_snapshots(
        demand=demand,
        parquet_dims=parquet_dims,
        icfg=icfg,
    )

    _write_inventory(snapshots, inv_out, "inventory_snapshot", file_format)

    n_rows = len(snapshots)
    stockout_pct = 0.0
    if n_rows > 0:
        stockout_pct = float(snapshots["StockoutFlag"].sum()) / n_rows * 100

    n_pairs = demand.groupby(["ProductKey", "StoreKey"]).ngroups
    return {
        "rows": n_rows,
        "product_store_pairs": n_pairs,
        "stockout_pct": round(stockout_pct, 2),
    }


# ------------------------------------------------------------------
# Parallel path (partitioned by store groups)
# ------------------------------------------------------------------

def _partition_demand_by_store(
    demand: pd.DataFrame,
    n_chunks: int,
) -> list[pd.DataFrame]:
    """Split demand into n_chunks groups by StoreKey (round-robin)."""
    unique_stores = np.sort(demand["StoreKey"].unique())
    store_to_chunk = {s: i % n_chunks for i, s in enumerate(unique_stores)}
    chunk_id = demand["StoreKey"].map(store_to_chunk)
    return [group_df for _, group_df in demand.groupby(chunk_id, sort=False)]


def _demand_to_arrays(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Convert demand DataFrame to dict of numpy arrays for pickling."""
    return {
        "ProductKey": df["ProductKey"].to_numpy(copy=True),
        "StoreKey": df["StoreKey"].to_numpy(copy=True),
        "Year": df["Year"].to_numpy(copy=True),
        "Month": df["Month"].to_numpy(copy=True),
        "QuantitySold": df["QuantitySold"].to_numpy(copy=True),
    }


def _run_parallel(
    demand: pd.DataFrame,
    parquet_dims: Path,
    icfg: InventoryConfig,
    inv_out: Path,
    file_format: str,
    n_stores: int,
    workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Partition by store groups and run simulation in parallel."""
    from src.facts.sales.sales_worker.pool import PoolRunSpec, iter_imap_unordered

    n_cpus = max(1, cpu_count() - 1)
    if workers is not None and workers >= 1:
        n_cpus = min(n_cpus, workers)

    n_chunks = min(n_stores, n_cpus * 2)
    n_chunks = max(2, n_chunks)

    partitions = _partition_demand_by_store(demand, n_chunks)
    partitions = [p for p in partitions if len(p) > 0]
    n_chunks = len(partitions)

    n_workers = min(n_chunks, n_cpus)

    info(f"Inventory parallel: {n_chunks} store-group chunks across {n_workers} workers")

    icfg_dict = dataclasses.asdict(icfg)
    parquet_dims_str = str(parquet_dims)

    tasks = []
    for idx, part_df in enumerate(partitions):
        # Base path without extension — worker appends .parquet and/or .csv
        out_base = str(inv_out / f"inventory_chunk_{idx:05d}")
        tasks.append((
            idx,
            _demand_to_arrays(part_df),
            parquet_dims_str,
            icfg_dict,
            out_base,
            file_format,
        ))

    del partitions

    pool_spec = PoolRunSpec(
        processes=n_workers,
        chunksize=1,
        label="inventory",
    )

    total_rows = 0
    total_stockout = 0
    completed = 0

    for result in iter_imap_unordered(
        tasks=tasks,
        task_fn=_inventory_worker_task,
        spec=pool_spec,
    ):
        completed += 1
        total_rows += result["rows"]
        total_stockout += result["stockout_sum"]
        work(
            f"[{completed}/{n_chunks}] inventory_chunk_{result['chunk_idx']:05d}"
            f" -> {result['rows']:,} rows"
        )

    stockout_pct = 0.0
    if total_rows > 0:
        stockout_pct = total_stockout / total_rows * 100

    n_pairs = demand.groupby(["ProductKey", "StoreKey"]).ngroups
    return {
        "rows": total_rows,
        "product_store_pairs": n_pairs,
        "stockout_pct": round(stockout_pct, 2),
        "chunks": completed,
    }


# ------------------------------------------------------------------
# Single-file write helpers (used by single-process path)
# ------------------------------------------------------------------

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
