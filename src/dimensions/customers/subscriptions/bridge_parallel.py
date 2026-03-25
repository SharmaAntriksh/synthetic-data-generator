"""Parallel bridge writer — worker task + orchestrator + chunk merger."""
from __future__ import annotations

from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.utils.logging_utils import info

from .helpers import (
    SubscriptionsCfg,
    _NS_PER_DAY,
    bridge_schema,
    build_type_groups,
    compute_customer_windows,
    generate_subscriptions_bulk,
    write_empty_bridge,
)


# ---------------------------------------------------------------------------
# Worker (must be top-level for Windows spawn pickling)
# ---------------------------------------------------------------------------

def _subscription_worker_task(args: Tuple) -> Dict[str, Any]:
    """Generate billing-period rows for a customer chunk and write chunk parquet."""
    (
        chunk_idx, seed, n_chunks,
        eligible_ck, eligible_lo, eligible_hi, eligible_span,
        plan_keys, plan_cycle_prices, plan_cycle_months,
        unique_types, type_members, type_weights,
        g_end_ns, max_subs,
        avg_subscriptions, churn_rate, trial_rate, trial_conversion_rate,
        trial_days,
        out_chunk_path,
    ) = args

    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(n_chunks)
    rng = np.random.default_rng(child_seeds[chunk_idx])

    table = generate_subscriptions_bulk(
        eligible_ck=eligible_ck,
        eligible_lo=eligible_lo,
        eligible_hi=eligible_hi,
        eligible_span=eligible_span,
        plan_keys=plan_keys,
        plan_cycle_prices=plan_cycle_prices,
        plan_cycle_months=plan_cycle_months,
        unique_types=unique_types,
        type_members=type_members,
        type_weights=type_weights,
        g_end_ns=g_end_ns,
        max_subs=max_subs,
        avg_subscriptions=avg_subscriptions,
        churn_rate=churn_rate,
        trial_rate=trial_rate,
        trial_conversion_rate=trial_conversion_rate,
        trial_days=trial_days,
        rng=rng,
    )

    n_rows = len(table)
    if n_rows > 0:
        out_path = Path(out_chunk_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out_chunk_path, compression="snappy", row_group_size=500_000)

    return {"chunk_idx": chunk_idx, "rows": n_rows}


# ---------------------------------------------------------------------------
# Chunk merger
# ---------------------------------------------------------------------------

def _merge_subscription_chunks(
    scratch_dir: Path,
    out_bridge: Path,
    n_chunks: int,
    delete_chunks: bool = True,
) -> int:
    """Read chunk parquets in order, write final merged parquet."""
    schema = bridge_schema()
    chunk_files = sorted(scratch_dir.glob("sub_chunk_*.parquet"))

    if not chunk_files:
        write_empty_bridge(out_bridge)
        return 0

    out_bridge.parent.mkdir(parents=True, exist_ok=True)
    if out_bridge.exists():
        out_bridge.unlink()

    total_rows = 0

    with pq.ParquetWriter(out_bridge, schema=schema, compression="snappy") as writer:
        for chunk_path in chunk_files:
            tbl = pq.read_table(chunk_path)
            n = len(tbl)
            if n == 0:
                continue
            writer.write_table(tbl)
            total_rows += n
            if delete_chunks:
                try:
                    chunk_path.unlink()
                except OSError:
                    pass

    return total_rows


# ---------------------------------------------------------------------------
# Parallel orchestrator
# ---------------------------------------------------------------------------

def write_bridge_parallel(
    customers: pd.DataFrame,
    dim_plans: pd.DataFrame,
    c: SubscriptionsCfg,
    g_start: pd.Timestamp,
    g_end: pd.Timestamp,
    out_bridge: Path,
    workers: Optional[int] = None,
) -> int:
    """Parallel bridge writer for large datasets. Returns total rows written."""
    from src.facts.sales.sales_worker.pool import PoolRunSpec, iter_imap_unordered

    if "CustomerKey" not in customers.columns:
        raise KeyError("customers must include CustomerKey")

    plan_keys = dim_plans["PlanKey"].astype(np.int32).to_numpy()
    plan_types = dim_plans["PlanType"].astype(str).to_numpy()
    plan_cycle_prices = dim_plans["CyclePrice"].to_numpy(dtype=np.float64, na_value=0.0)
    plan_cycle_months = dim_plans["CycleMonths"].astype(np.int32).to_numpy()
    unique_types, type_members, type_weights = build_type_groups(plan_types)

    cust_keys, cust_start_ns, cust_end_ns = compute_customer_windows(customers, g_start, g_end)
    g_end_ns = np.int64(g_end.value)
    max_subs = min(c.max_subscriptions, len(unique_types))

    n_cust = len(cust_keys)
    rng_main = np.random.default_rng(c.seed)
    participate_mask = rng_main.random(n_cust) < c.participation_rate
    _part_idx = np.where(participate_mask)[0]
    _spans = np.maximum(0, (cust_end_ns[_part_idx] - cust_start_ns[_part_idx]) // _NS_PER_DAY)
    _ok = _spans >= 30
    eligible_ck = cust_keys[_part_idx[_ok]]
    eligible_lo = cust_start_ns[_part_idx[_ok]]
    eligible_hi = cust_end_ns[_part_idx[_ok]]
    eligible_span = _spans[_ok]
    n_eligible = len(eligible_ck)

    if n_eligible == 0:
        write_empty_bridge(out_bridge)
        return 0

    n_cpus = max(1, cpu_count() - 1)
    if workers is not None and workers >= 1:
        n_cpus = min(n_cpus, workers)
    n_chunks = max(2, min(n_eligible, n_cpus * 2))
    n_workers = min(n_chunks, n_cpus)

    info(f"Subscriptions parallel: {n_eligible:,} eligible customers, "
         f"{n_chunks} chunks, {n_workers} workers")

    chunk_boundaries = np.array_split(np.arange(n_eligible), n_chunks)

    scratch_dir = out_bridge.parent / "_sub_chunks"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for idx, indices in enumerate(chunk_boundaries):
        if len(indices) == 0:
            continue
        chunk_path = str(scratch_dir / f"sub_chunk_{idx:05d}.parquet")
        tasks.append((
            idx,
            c.seed,
            n_chunks,
            eligible_ck[indices],
            eligible_lo[indices],
            eligible_hi[indices],
            eligible_span[indices],
            plan_keys,
            plan_cycle_prices,
            plan_cycle_months,
            unique_types,
            type_members,
            type_weights,
            int(g_end_ns),
            max_subs,
            c.avg_subscriptions,
            c.churn_rate,
            c.trial_rate,
            c.trial_conversion_rate,
            c.trial_days,
            chunk_path,
        ))

    actual_n_chunks = len(tasks)

    pool_spec = PoolRunSpec(
        processes=n_workers,
        chunksize=1,
        label="subscriptions",
    )

    chunk_rows: List[int] = [0] * actual_n_chunks
    completed = 0
    for result in iter_imap_unordered(
        tasks=tasks,
        task_fn=_subscription_worker_task,
        spec=pool_spec,
    ):
        completed += 1
        chunk_rows[result["chunk_idx"]] = result["rows"]

    info(f"Subscriptions: {completed}/{actual_n_chunks} chunks done")

    total_rows = _merge_subscription_chunks(
        scratch_dir=scratch_dir,
        out_bridge=out_bridge,
        n_chunks=actual_n_chunks,
        delete_chunks=True,
    )

    try:
        scratch_dir.rmdir()
    except OSError:
        pass

    return total_rows
