"""Parallel bridge writer — worker task + orchestrator + chunk merger."""
from __future__ import annotations

from datetime import date
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.logging_utils import info

from .catalog import PAYMENT_METHODS, _PAYMENT_WEIGHTS
from .helpers import (
    SubscriptionsCfg,
    _NS_PER_DAY,
    bridge_schema,
    build_type_groups,
    choose_plans_diverse,
    compute_customer_windows,
    expand_subscription_periods,
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
        payment_weights,
        out_chunk_path,
    ) = args

    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(n_chunks)
    rng = np.random.default_rng(child_seeds[chunk_idx])

    n_eligible = len(eligible_ck)
    schema = bridge_schema()

    all_sk: List[int] = []
    all_ck: List[int] = []
    all_pk: List[int] = []
    all_ps: List[date] = []
    all_pe: List[date] = []
    all_price: List[float] = []
    all_first: List[int] = []
    all_churn: List[int] = []
    all_trial: List[int] = []
    all_cycle: List[int] = []

    local_sub_key = 1

    for i in range(n_eligible):
        ck = int(eligible_ck[i])
        lo_ns = int(eligible_lo[i])
        hi_ns = int(eligible_hi[i])
        span_days = int(eligible_span[i])

        n_subs = max(1, int(rng.poisson(avg_subscriptions)))
        n_subs = min(n_subs, max_subs)

        chosen_idx = choose_plans_diverse(
            rng, n_subs, unique_types, type_members, type_weights,
        )
        n_subs = len(chosen_idx)
        sub_offsets = np.sort(rng.integers(0, max(span_days - 30, 1), size=n_subs))

        for s in range(n_subs):
            pidx = chosen_idx[s]
            pk = int(plan_keys[pidx])

            sub_ns = lo_ns + int(sub_offsets[s]) * _NS_PER_DAY

            cycle_months = int(plan_cycle_months[pidx])
            n_periods = max(1, int(rng.geometric(0.3)))
            base_duration_days = cycle_months * 30 * n_periods

            cprice = float(plan_cycle_prices[pidx])

            has_trial = rng.random() < trial_rate
            trial_end_ns: Optional[int] = int(sub_ns + trial_days * _NS_PER_DAY) if has_trial else None
            converts = rng.random() < trial_conversion_rate if has_trial else True

            is_churned = rng.random() < churn_rate
            end_ns = sub_ns + int(base_duration_days) * _NS_PER_DAY

            if not converts:
                cancel_ns: Optional[int] = int(sub_ns)
            elif is_churned and end_ns <= hi_ns:
                cancel_ns = int(end_ns)
            else:
                cancel_ns = None

            # RNG consumption for payment method (maintain stream compatibility)
            rng.choice(len(PAYMENT_METHODS), p=payment_weights)

            (
                sk, ck_l, pk_l, ps, pe, pr,
                first, churn, trial, cyc,
            ) = expand_subscription_periods(
                sub_key=local_sub_key,
                ck=ck,
                pk=pk,
                sub_ns=sub_ns,
                cancel_ns=cancel_ns,
                trial_end_ns=trial_end_ns,
                cycle_months=cycle_months,
                cycle_price=cprice,
                g_end_ns=int(g_end_ns),
            )

            all_sk.extend(sk)
            all_ck.extend(ck_l)
            all_pk.extend(pk_l)
            all_ps.extend(ps)
            all_pe.extend(pe)
            all_price.extend(pr)
            all_first.extend(first)
            all_churn.extend(churn)
            all_trial.extend(trial)
            all_cycle.extend(cyc)

            local_sub_key += 1

    n_rows = len(all_sk)

    if n_rows > 0:
        out_path = Path(out_chunk_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        table = pa.Table.from_arrays(
            [
                pa.array(all_sk, type=pa.int64()),
                pa.array(all_ck, type=pa.int64()),
                pa.array(all_pk, type=pa.int32()),
                pa.array(all_ps, type=pa.date32()),
                pa.array(all_pe, type=pa.date32()),
                pa.array(all_price, type=pa.float64()),
                pa.array(all_first, type=pa.int8()),
                pa.array(all_churn, type=pa.int8()),
                pa.array(all_trial, type=pa.int8()),
                pa.array(all_cycle, type=pa.int32()),
            ],
            schema=schema,
        )
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
            _PAYMENT_WEIGHTS,
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
