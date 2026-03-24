"""Streaming (serial) bridge writer for small datasets."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .catalog import PAYMENT_METHODS, _PAYMENT_WEIGHTS
from .helpers import (
    SubscriptionsCfg,
    _NS_PER_DAY,
    bridge_schema,
    build_type_groups,
    choose_plans_diverse,
    compute_customer_windows,
    expand_subscription_periods,
)


def write_bridge_streaming(
    customers: pd.DataFrame,
    dim_plans: pd.DataFrame,
    c: SubscriptionsCfg,
    g_start: pd.Timestamp,
    g_end: pd.Timestamp,
    out_bridge: Path,
) -> int:
    """Stream-write customer_subscriptions billing-period fact to parquet.
    Returns number of rows written.
    """
    if "CustomerKey" not in customers.columns:
        raise KeyError("customers must include CustomerKey")

    plan_keys = dim_plans["PlanKey"].astype(np.int32).to_numpy()
    plan_types = dim_plans["PlanType"].astype(str).to_numpy()
    plan_cycle_prices = dim_plans["CyclePrice"].to_numpy(dtype=np.float64, na_value=0.0)
    plan_cycle_months = dim_plans["CycleMonths"].astype(np.int32).to_numpy()
    unique_types, type_members, type_weights = build_type_groups(plan_types)

    cust_keys, cust_start_ns, cust_end_ns = compute_customer_windows(
        customers, g_start, g_end,
    )

    g_end_ns = np.int64(g_end.value)
    write_chunk_rows = max(c.write_chunk_rows, 10_000)
    max_subs = min(c.max_subscriptions, len(unique_types))

    rng = np.random.default_rng(c.seed)
    schema = bridge_schema()

    n_cust = len(cust_keys)
    participate_mask = rng.random(n_cust) < c.participation_rate

    _part_idx = np.where(participate_mask)[0]
    _spans = np.maximum(0, (cust_end_ns[_part_idx] - cust_start_ns[_part_idx]) // _NS_PER_DAY)
    _ok = _spans >= 30
    eligible_idx = _part_idx[_ok]
    eligible_ck = cust_keys[eligible_idx]
    eligible_lo = cust_start_ns[eligible_idx]
    eligible_hi = cust_end_ns[eligible_idx]
    eligible_span = _spans[_ok]
    n_eligible = len(eligible_idx)

    # Accumulators for billing-period rows
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

    total_rows = 0
    next_sub_key = 1

    def flush(writer: pq.ParquetWriter) -> None:
        nonlocal total_rows
        nonlocal all_sk, all_ck, all_pk
        nonlocal all_ps, all_pe, all_price
        nonlocal all_first, all_churn, all_trial, all_cycle
        n = len(all_sk)
        if n == 0:
            return

        arrays: List[pa.Array] = [
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
        ]

        table = pa.Table.from_arrays(arrays, schema=schema)
        writer.write_table(table)
        total_rows += n

        # Reset accumulators
        all_sk = []
        all_ck = []
        all_pk = []
        all_ps = []
        all_pe = []
        all_price = []
        all_first = []
        all_churn = []
        all_trial = []
        all_cycle = []

    out_bridge.parent.mkdir(parents=True, exist_ok=True)
    if out_bridge.exists():
        out_bridge.unlink()

    with pq.ParquetWriter(out_bridge, schema=schema, compression="snappy") as writer:
        for i in range(n_eligible):
            ck = int(eligible_ck[i])
            lo_ns = int(eligible_lo[i])
            hi_ns = int(eligible_hi[i])
            span_days = int(eligible_span[i])

            n_subs = max(1, int(rng.poisson(c.avg_subscriptions)))
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

                has_trial = rng.random() < c.trial_rate
                if has_trial:
                    trial_end_ns = sub_ns + c.trial_days * _NS_PER_DAY
                    converts = rng.random() < c.trial_conversion_rate
                else:
                    trial_end_ns = None
                    converts = True

                is_churned = rng.random() < c.churn_rate
                end_ns = sub_ns + int(base_duration_days) * _NS_PER_DAY

                if not converts:
                    cancel_ns = sub_ns
                elif is_churned and end_ns <= hi_ns:
                    cancel_ns = end_ns
                else:
                    cancel_ns = None

                # RNG consumption for payment method (maintain stream compatibility)
                rng.choice(len(PAYMENT_METHODS), p=_PAYMENT_WEIGHTS)

                (
                    sk, ck_l, pk_l, ps, pe, pr,
                    first, churn, trial, cyc,
                ) = expand_subscription_periods(
                    sub_key=next_sub_key,
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

                next_sub_key += 1

                if len(all_sk) >= write_chunk_rows:
                    flush(writer)

        flush(writer)

    return total_rows
