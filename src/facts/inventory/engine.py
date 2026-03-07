"""Inventory snapshot engine.

Simulates monthly inventory levels for each (ProductKey, StoreKey) pair
using accumulated sales demand and ProductProfile replenishment attributes.

Vectorized: loops only over months (sequential dependency), processes all
product-store pairs simultaneously via numpy. Typical runtime: <2s for
240K pairs × 48 months.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


# ================================================================
# Config
# ================================================================

@dataclass(frozen=True)
class InventoryConfig:
    enabled: bool = False
    seed: int = 42
    grain: str = "monthly"

    initial_stock_multiplier: float = 3.0
    reorder_compliance: float = 0.85
    lead_time_variance: float = 0.20
    overstock_bias: float = 1.15

    shrinkage_enabled: bool = True
    shrinkage_rate: float = 0.02

    min_demand_months: int = 3

    abc_stock_multiplier: Dict[str, float] = field(default_factory=lambda: {
        "A": 1.30,
        "B": 1.00,
        "C": 0.70,
    })


def load_inventory_config(cfg: Dict[str, Any]) -> InventoryConfig:
    raw = cfg.get("inventory", {}) or {}
    if not isinstance(raw, dict):
        return InventoryConfig()

    shrinkage = raw.get("shrinkage", {}) or {}
    abc = raw.get("abc_stock_multiplier", {
        "A": 1.30, "B": 1.00, "C": 0.70,
    })

    return InventoryConfig(
        enabled=bool(raw.get("enabled", False)),
        seed=int(raw.get("seed", 42)),
        grain=str(raw.get("grain", "monthly")).lower(),
        initial_stock_multiplier=float(raw.get("initial_stock_multiplier", 3.0)),
        reorder_compliance=float(raw.get("reorder_compliance", 0.85)),
        lead_time_variance=float(raw.get("lead_time_variance", 0.20)),
        overstock_bias=float(raw.get("overstock_bias", 1.15)),
        shrinkage_enabled=bool(shrinkage.get("enabled", True)),
        shrinkage_rate=float(shrinkage.get("rate", 0.02)),
        abc_stock_multiplier=dict(abc),
        min_demand_months=int(raw.get("min_demand_months", 3)),
    )


# ================================================================
# ProductProfile attribute loading
# ================================================================

_PRODUCT_PROFILE_COLS = [
    "ProductKey",
    "SafetyStockUnits",
    "ReorderPointUnits",
    "LeadTimeDays",
    "ABCClassification",
]

_PROFILE_DEFAULTS = {
    "SafetyStockUnits": 20,
    "ReorderPointUnits": 10,
    "LeadTimeDays": 14,
    "ABCClassification": "B",
}


def _load_product_attrs(parquet_dims: Path) -> pd.DataFrame:
    """Load replenishment-relevant columns from ProductProfile."""
    pp_path = parquet_dims / "product_profile.parquet"
    if not pp_path.exists():
        return pd.DataFrame(columns=_PRODUCT_PROFILE_COLS)

    try:
        df = pd.read_parquet(pp_path, columns=_PRODUCT_PROFILE_COLS)
    except Exception:
        available = pd.read_parquet(pp_path, columns=None).columns.tolist()
        cols_to_load = [c for c in _PRODUCT_PROFILE_COLS if c in available]
        if "ProductKey" not in cols_to_load:
            return pd.DataFrame(columns=_PRODUCT_PROFILE_COLS)
        df = pd.read_parquet(pp_path, columns=cols_to_load)

    for col, default in _PROFILE_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default

    return df


# ================================================================
# Vectorized simulation
# ================================================================

def compute_inventory_snapshots(
    demand: pd.DataFrame,
    parquet_dims: Any,
    icfg: InventoryConfig,
) -> pd.DataFrame:
    """
    Simulate monthly inventory snapshots from accumulated sales demand.

    Fully vectorized across all (ProductKey, StoreKey) pairs. Loops only
    over the month axis (sequential dependency).
    """
    if demand.empty:
        return _empty_snapshot()

    parquet_dims = Path(parquet_dims)
    rng = np.random.default_rng(icfg.seed)

    # ------------------------------------------------------------------
    # 1. Build sorted month timeline
    # ------------------------------------------------------------------
    month_keys = demand[["Year", "Month"]].drop_duplicates().sort_values(["Year", "Month"])
    years_arr = month_keys["Year"].to_numpy(dtype=np.int32)
    months_arr = month_keys["Month"].to_numpy(dtype=np.int32)
    n_months = len(years_arr)

    if n_months == 0:
        return _empty_snapshot()

    # ------------------------------------------------------------------
    # 2. Build unique pairs and demand matrix (N_pairs × N_months)
    # ------------------------------------------------------------------
    pair_df = demand[["ProductKey", "StoreKey"]].drop_duplicates().reset_index(drop=True)
    n_pairs = len(pair_df)

    pair_pk = pair_df["ProductKey"].to_numpy(dtype=np.int32)
    pair_sk = pair_df["StoreKey"].to_numpy(dtype=np.int32)

    max_pk = int(pair_pk.max()) + 1
    max_sk = int(pair_sk.max()) + 1
    pair_lookup = np.full((max_pk, max_sk), -1, dtype=np.int32)
    pair_lookup[pair_pk, pair_sk] = np.arange(n_pairs, dtype=np.int32)

    demand_matrix = np.zeros((n_pairs, n_months), dtype=np.int32)

    d_pk = demand["ProductKey"].to_numpy(dtype=np.int32)
    d_sk = demand["StoreKey"].to_numpy(dtype=np.int32)
    d_yr = demand["Year"].to_numpy(dtype=np.int32)
    d_mo = demand["Month"].to_numpy(dtype=np.int32)
    d_qty = demand["QuantitySold"].to_numpy(dtype=np.int32)

    d_pair_idx = pair_lookup[d_pk, d_sk]
    d_month_idx = np.empty(len(d_yr), dtype=np.int32)
    for i in range(n_months):
        mask = (d_yr == years_arr[i]) & (d_mo == months_arr[i])
        d_month_idx[mask] = i

    valid = d_pair_idx >= 0
    np.add.at(demand_matrix, (d_pair_idx[valid], d_month_idx[valid]), d_qty[valid])

    # Filter to pairs with recurring demand (stocked assortment items)
    months_with_demand = (demand_matrix > 0).sum(axis=1)
    keep_mask = months_with_demand >= icfg.min_demand_months

    if not keep_mask.any():
        return _empty_snapshot()

    demand_matrix = demand_matrix[keep_mask]
    pair_pk = pair_pk[keep_mask]
    pair_sk = pair_sk[keep_mask]
    n_pairs = int(keep_mask.sum())

    # ------------------------------------------------------------------
    # 3. Load product attributes into dense arrays aligned to pairs
    # ------------------------------------------------------------------
    product_attrs = _load_product_attrs(parquet_dims)

    attr_safety = np.full(n_pairs, 20, dtype=np.int32)
    attr_reorder_pt = np.full(n_pairs, 10, dtype=np.int32)
    attr_lead_days = np.full(n_pairs, 14, dtype=np.int32)
    attr_abc_mult = np.ones(n_pairs, dtype=np.float64)

    if not product_attrs.empty:
        pa_pk = product_attrs["ProductKey"].to_numpy(dtype=np.int32)
        pa_ss = product_attrs["SafetyStockUnits"].to_numpy(dtype=np.int32)
        pa_rp = product_attrs["ReorderPointUnits"].to_numpy(dtype=np.int32)
        pa_ld = product_attrs["LeadTimeDays"].to_numpy(dtype=np.int32)
        pa_abc = product_attrs["ABCClassification"].to_numpy().astype(str)

        pa_max = int(pa_pk.max()) + 1
        dense_safety = np.full(pa_max, 20, dtype=np.int32)
        dense_reorder = np.full(pa_max, 10, dtype=np.int32)
        dense_lead = np.full(pa_max, 14, dtype=np.int32)
        dense_abc_mult = np.ones(pa_max, dtype=np.float64)

        dense_safety[pa_pk] = pa_ss
        dense_reorder[pa_pk] = pa_rp
        dense_lead[pa_pk] = pa_ld
        for abc_class, mult in icfg.abc_stock_multiplier.items():
            dense_abc_mult[pa_pk[pa_abc == abc_class]] = mult

        in_range = pair_pk < pa_max
        attr_safety[in_range] = dense_safety[pair_pk[in_range]]
        attr_reorder_pt[in_range] = dense_reorder[pair_pk[in_range]]
        attr_lead_days[in_range] = dense_lead[pair_pk[in_range]]
        attr_abc_mult[in_range] = dense_abc_mult[pair_pk[in_range]]

    attr_lead_months = np.maximum(1, np.round(attr_lead_days / 30.0).astype(np.int32))

    # ------------------------------------------------------------------
    # 4. Compute per-pair initial conditions
    # ------------------------------------------------------------------
    # First demand month per pair — inventory only exists from this month onward.
    has_demand = demand_matrix > 0
    first_demand = np.where(
        has_demand.any(axis=1),
        has_demand.argmax(axis=1),
        n_months,  # sentinel: pair never had demand (shouldn't happen, but safe)
    )

    demand_positive = demand_matrix.copy().astype(np.float64)
    demand_positive[demand_positive <= 0] = np.nan
    avg_demand = np.nanmean(demand_positive, axis=1)
    avg_demand = np.where(np.isnan(avg_demand), 1.0, np.maximum(avg_demand, 1.0))

    initial_stock = (
        avg_demand * icfg.initial_stock_multiplier * attr_abc_mult * icfg.overstock_bias
    ).astype(np.int32)

    reorder_qty = (
        (attr_safety + avg_demand * attr_lead_months) * attr_abc_mult * icfg.overstock_bias
    ).astype(np.int32)
    reorder_qty = np.maximum(reorder_qty, (avg_demand * attr_lead_months * attr_abc_mult).astype(np.int32))

    monthly_shrinkage = icfg.shrinkage_rate / 12.0 if icfg.shrinkage_enabled else 0.0

    # ------------------------------------------------------------------
    # 5. Pre-generate all random draws (avoids per-step RNG calls)
    # ------------------------------------------------------------------
    rand_compliance = rng.random((n_pairs, n_months))
    rand_lt_jitter = rng.uniform(
        -icfg.lead_time_variance, icfg.lead_time_variance,
        size=(n_pairs, n_months),
    )

    # ------------------------------------------------------------------
    # 6. Vectorized month-by-month simulation
    # ------------------------------------------------------------------
    out_qoh = np.zeros((n_pairs, n_months), dtype=np.int32)
    out_on_order = np.zeros((n_pairs, n_months), dtype=np.int32)
    out_sold = demand_matrix.copy()
    out_received = np.zeros((n_pairs, n_months), dtype=np.int32)
    out_reorder = np.zeros((n_pairs, n_months), dtype=np.int8)
    out_stockout = np.zeros((n_pairs, n_months), dtype=np.int8)
    out_days_oos = np.zeros((n_pairs, n_months), dtype=np.int8)

    # active[i] tracks whether pair i has been activated (first sale reached)
    active_mask = np.zeros((n_pairs, n_months), dtype=bool)

    pending = np.zeros((n_pairs, n_months + 60), dtype=np.int32)

    qoh = np.zeros(n_pairs, dtype=np.int32)

    for t in range(n_months):
        # Activate pairs whose first demand month is this month
        newly_active = first_demand == t
        qoh[newly_active] = initial_stock[newly_active]

        is_active = t >= first_demand
        active_mask[:, t] = is_active

        received = pending[:, t]
        qoh += received * is_active
        out_received[:, t] = received * is_active

        sold = demand_matrix[:, t]
        shrink = (qoh.astype(np.float64) * monthly_shrinkage).astype(np.int32) if monthly_shrinkage > 0 else 0
        qoh -= (sold + shrink) * is_active

        stockout_mask = (qoh < 0) & is_active
        daily_demand = np.maximum(1.0, sold.astype(np.float64) / 30.0)
        days_oos = np.where(
            stockout_mask,
            np.minimum(30, (np.abs(qoh).astype(np.float64) / daily_demand).astype(np.int32)),
            0,
        )
        qoh = np.maximum(qoh, 0)

        reorder_mask = (qoh <= attr_reorder_pt) & is_active
        comply_mask = rand_compliance[:, t] < icfg.reorder_compliance
        trigger_mask = reorder_mask & comply_mask

        lt_jittered = (
            attr_lead_months.astype(np.float64)
            * (1.0 + rand_lt_jitter[:, t])
        ).astype(np.int32)
        arrival_offset = np.maximum(1, lt_jittered)
        arrival_t = t + arrival_offset

        triggered_idx = np.flatnonzero(trigger_mask)
        if triggered_idx.size > 0:
            arr_t = arrival_t[triggered_idx]
            rq = reorder_qty[triggered_idx]
            valid = arr_t < pending.shape[1]
            if valid.any():
                valid_idx = triggered_idx[valid]
                valid_arr_t = arr_t[valid]
                valid_rq = rq[valid]
                np.add.at(pending, (valid_idx, valid_arr_t), valid_rq)
            out_on_order[triggered_idx, t] = rq

        out_qoh[:, t] = qoh
        out_reorder[:, t] = reorder_mask.astype(np.int8)
        out_stockout[:, t] = stockout_mask.astype(np.int8)
        out_days_oos[:, t] = days_oos.astype(np.int8)

    # ------------------------------------------------------------------
    # 7. Flatten to DataFrame (only active cells)
    # ------------------------------------------------------------------
    snapshot_dates = pd.to_datetime(
        [f"{years_arr[m]}-{months_arr[m]:02d}-01" for m in range(n_months)]
    )

    row_idx, col_idx = np.nonzero(active_mask)
    if row_idx.size == 0:
        return _empty_snapshot()

    result = pd.DataFrame({
        "ProductKey": pair_pk[row_idx],
        "StoreKey": pair_sk[row_idx],
        "SnapshotDate": snapshot_dates[col_idx],
        "QuantityOnHand": out_qoh[row_idx, col_idx],
        "QuantityOnOrder": out_on_order[row_idx, col_idx],
        "QuantitySold": out_sold[row_idx, col_idx],
        "QuantityReceived": out_received[row_idx, col_idx],
        "ReorderFlag": out_reorder[row_idx, col_idx],
        "StockoutFlag": out_stockout[row_idx, col_idx],
        "DaysOutOfStock": out_days_oos[row_idx, col_idx],
    })

    result["ProductKey"] = result["ProductKey"].astype(np.int32)
    result["StoreKey"] = result["StoreKey"].astype(np.int32)
    result["QuantityOnHand"] = result["QuantityOnHand"].astype(np.int32)
    result["QuantityOnOrder"] = result["QuantityOnOrder"].astype(np.int32)
    result["QuantitySold"] = result["QuantitySold"].astype(np.int32)
    result["QuantityReceived"] = result["QuantityReceived"].astype(np.int32)
    result["ReorderFlag"] = result["ReorderFlag"].astype(np.int8)
    result["StockoutFlag"] = result["StockoutFlag"].astype(np.int8)
    result["DaysOutOfStock"] = result["DaysOutOfStock"].astype(np.int8)

    return result


def _empty_snapshot() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "ProductKey", "StoreKey", "SnapshotDate",
        "QuantityOnHand", "QuantityOnOrder",
        "QuantitySold", "QuantityReceived",
        "ReorderFlag", "StockoutFlag", "DaysOutOfStock",
    ])
