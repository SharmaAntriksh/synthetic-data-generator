"""Inventory snapshot pipeline runner.

Called from sales_runner.py after sales generation completes.
Uses the InventoryAccumulator that was populated during sales generation.

For large datasets (many product-store pairs), partitions demand by store
groups and runs the simulation in parallel across multiple processes.
Each worker writes its own chunk files directly (CSV + parquet), avoiding
the need to build a single massive DataFrame in one process.

After parallel chunks are written, they are merged into a single Parquet
file (like Sales).  For deltaparquet mode, chunks are consolidated into a
Delta Lake table partitioned by Year + Month.
"""
from __future__ import annotations

import dataclasses
import os
import time
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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

    inv_cfg = cfg.get("inventory", {})
    sales_cfg = cfg.get("sales", {})
    merge_enabled = bool(sales_cfg.get("merge_parquet", True))
    merge_file = "inventory_snapshot.parquet"
    delete_chunks = bool(sales_cfg.get("delete_chunks", True))
    partition_by: List[str] = inv_cfg.get("partition_by") or []

    if qualified_pairs >= _PARALLEL_THRESHOLD and n_stores >= 2:
        result = _run_parallel(
            demand, parquet_dims, icfg, inv_out, file_format, n_stores,
            workers=workers,
            merge_enabled=merge_enabled,
            merge_file=merge_file,
            delete_chunks=delete_chunks,
            partition_by=partition_by,
        )
    else:
        result = _run_single(demand, parquet_dims, icfg, inv_out, file_format, partition_by=partition_by)

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
    partition_by: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Original monolithic path — used for small datasets."""
    snapshots = compute_inventory_snapshots(
        demand=demand,
        parquet_dims=parquet_dims,
        icfg=icfg,
    )

    _write_inventory(snapshots, inv_out, "inventory_snapshot", file_format, partition_by=partition_by)

    # For deltaparquet the delta table is written outside inv_out;
    # remove the empty temporary directory.
    if file_format == "deltaparquet":
        try:
            inv_out.rmdir()
        except OSError:
            pass

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
    merge_enabled: bool = True,
    merge_file: str = "inventory_snapshot.parquet",
    delete_chunks: bool = True,
    partition_by: Optional[List[str]] = None,
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

    work(f"{completed}/{n_chunks} inventory chunks completed ({total_rows:,} total rows)")

    stockout_pct = 0.0
    if total_rows > 0:
        stockout_pct = total_stockout / total_rows * 100

    # ------------------------------------------------------------------
    # Post-processing: merge chunk parquets into a single file, or
    # consolidate into a Delta Lake table with Year+Month partitioning.
    # ------------------------------------------------------------------
    if total_rows > 0:
        chunk_files = sorted(inv_out.glob("inventory_chunk_*.parquet"))

        if file_format == "deltaparquet":
            _merge_chunks_to_delta(
                chunk_files=chunk_files,
                inv_out=inv_out,
                partition_by=partition_by or [],
                delete_chunks=delete_chunks,
            )
            # Remove the now-empty temporary chunk directory
            try:
                inv_out.rmdir()
            except OSError:
                pass
        elif merge_enabled and file_format in ("parquet", "csv"):
            _merge_inventory_chunks(
                chunk_files=chunk_files,
                merged_path=inv_out / merge_file,
                delete_chunks=delete_chunks,
            )

    n_pairs = demand.groupby(["ProductKey", "StoreKey"]).ngroups
    return {
        "rows": total_rows,
        "product_store_pairs": n_pairs,
        "stockout_pct": round(stockout_pct, 2),
        "chunks": completed,
    }


# ------------------------------------------------------------------
# Chunk merge helpers (used by parallel path)
# ------------------------------------------------------------------

def _add_year_month_to_table(table: pa.Table) -> pa.Table:
    """Derive Year and Month int columns from SnapshotDate for partitioning."""
    import pyarrow.compute as pc

    dates = table.column("SnapshotDate")
    # Cast to date32 if timestamp
    if pa.types.is_timestamp(dates.type):
        dates = pc.cast(dates, pa.date32())

    year = pc.year(dates).cast(pa.int16())
    month = pc.month(dates).cast(pa.int8())

    table = table.append_column("Year", year)
    table = table.append_column("Month", month)
    return table


def _merge_inventory_chunks(
    chunk_files: list[Path],
    merged_path: Path,
    delete_chunks: bool = True,
) -> None:
    """Merge parallel inventory chunk parquets into one file (reuses sales merge logic)."""
    from src.facts.sales.sales_writer.parquet_merge import _merge_parquet_files_common

    str_files = [str(f) for f in chunk_files]
    result = _merge_parquet_files_common(
        str_files,
        str(merged_path),
        delete_after=delete_chunks,
        compression="snappy",
        use_dictionary=True,
        log_prefix="",
    )
    if not result:
        info(f"Inventory merge: no output for {short_path(merged_path)}")


def _merge_chunks_to_delta(
    chunk_files: list[Path],
    inv_out: Path,
    partition_by: List[str],
    delete_chunks: bool = True,
) -> None:
    """Consolidate inventory chunk parquets into a partitioned Delta Lake table."""
    try:
        from deltalake import write_deltalake
    except ImportError:
        from deltalake.writer import write_deltalake

    delta_dir = inv_out.parent / "inventory_snapshot"
    delta_dir.mkdir(parents=True, exist_ok=True)

    needs_year_month = any(c in partition_by for c in ("Year", "Month"))

    for i, chunk_path in enumerate(chunk_files):
        table = pq.read_table(str(chunk_path))

        if needs_year_month and "Year" not in table.column_names:
            table = _add_year_month_to_table(table)

        # Validate partition cols against actual schema
        pcols = [c for c in partition_by if c in table.column_names]

        write_deltalake(
            str(delta_dir),
            table,
            mode="overwrite" if i == 0 else "append",
            partition_by=pcols if pcols else None,
        )

    info(f"[DELTA] Writing {len(chunk_files)} parts (Arrow -> Delta) table=InventorySnapshot")

    if delete_chunks:
        for f in chunk_files:
            try:
                os.remove(f)
            except OSError:
                pass


# ------------------------------------------------------------------
# Single-file write helpers (used by single-process path)
# ------------------------------------------------------------------

def _write_inventory(
    df: pd.DataFrame,
    out_dir: Path,
    name: str,
    file_format: str,
    partition_by: Optional[List[str]] = None,
) -> None:
    """Write an inventory DataFrame in the requested format."""
    table = pa.Table.from_pandas(df, preserve_index=False)

    if file_format == "deltaparquet":
        delta_dir = out_dir.parent / name
        delta_dir.mkdir(parents=True, exist_ok=True)
        try:
            from deltalake import write_deltalake
        except ImportError:
            from deltalake.writer import write_deltalake

        pcols = list(partition_by or [])
        needs_year_month = any(c in pcols for c in ("Year", "Month"))
        if needs_year_month and "Year" not in table.column_names:
            table = _add_year_month_to_table(table)
        pcols = [c for c in pcols if c in table.column_names]

        write_deltalake(
            str(delta_dir), table, mode="overwrite",
            partition_by=pcols if pcols else None,
        )
        info(
            f"Wrote {name}: {len(df):,} rows -> {short_path(delta_dir)}/"
            + (f" (partitioned by {pcols})" if pcols else "")
        )
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
