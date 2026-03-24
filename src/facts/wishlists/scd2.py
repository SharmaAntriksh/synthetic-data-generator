"""SCD2 product price resolution for wishlists."""
from __future__ import annotations

from typing import Optional, Tuple

import logging as _logging

import numpy as np
import pandas as pd

_log = _logging.getLogger(__name__)


def build_scd2_price_lookup(
    all_products_df: pd.DataFrame,
    prod_keys_current: np.ndarray,
    prod_prices_current: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Build SCD2 version lookup tables for wishlist price resolution.

    Args:
        all_products_df: full products DataFrame (all versions, not filtered to IsCurrent)
        prod_keys_current: ProductKey array for IsCurrent=1 products
        prod_prices_current: ListPrice array for IsCurrent=1 products

    Returns (starts, prices) or None if SCD2 is not active:
      - starts: shape (n_products, max_ver) int64 — EffectiveStartDate as epoch days
      - prices: shape (n_products, max_ver) float64 — ListPrice per version slot
    """
    if "EffectiveStartDate" not in all_products_df.columns:
        return None

    all_df = all_products_df[["ProductKey", "ListPrice", "EffectiveStartDate"]].copy()

    n_unique = all_df["ProductKey"].nunique()
    if len(all_df) <= n_unique:
        return None  # no SCD2 versions, all products have single row

    all_df["eff_start_days"] = (
        pd.to_datetime(all_df["EffectiveStartDate"])
        .values.astype("datetime64[D]")
        .astype(np.int64)
    )

    n_products = len(prod_keys_current)

    # Map ProductKey → product index (0..n_products-1)
    max_key = max(int(prod_keys_current.max()), int(all_df["ProductKey"].max())) + 1
    key_lookup = np.full(max_key, -1, dtype=np.int32)
    key_lookup[prod_keys_current] = np.arange(n_products, dtype=np.int32)

    pkey_arr = all_df["ProductKey"].to_numpy()
    pidx = key_lookup[pkey_arr]
    mask = pidx >= 0
    pidx = pidx[mask]
    eff_start = all_df["eff_start_days"].to_numpy()[mask]
    lprice = all_df["ListPrice"].to_numpy(dtype=np.float64)[mask]

    # Sort by (product_index, eff_start)
    order = np.lexsort((eff_start, pidx))
    pidx = pidx[order]
    eff_start = eff_start[order]
    lprice = lprice[order]

    # Compute per-product version slot indices
    group_starts = np.concatenate([[0], np.where(pidx[1:] != pidx[:-1])[0] + 1])
    slot = np.arange(len(pidx), dtype=np.int32)
    slot -= np.repeat(group_starts, np.diff(np.append(group_starts, len(pidx))))

    max_ver = int(slot.max()) + 1 if len(slot) > 0 else 1

    # Initialize with current-version defaults
    starts = np.full((n_products, max_ver), np.iinfo(np.int64).max, dtype=np.int64)
    prices = np.tile(prod_prices_current[:, np.newaxis], (1, max_ver))

    # Scatter historical versions
    valid = slot < max_ver
    pi = pidx[valid]
    si = slot[valid]
    starts[pi, si] = eff_start[valid]
    prices[pi, si] = lprice[valid]

    # First version covers all time before second version
    starts[pi, 0] = 0

    _log.debug("Wishlist SCD2 price lookup: %d products, %d max versions", n_products, max_ver)
    return starts, prices


def resolve_scd2_prices(
    prod_idx: np.ndarray,
    date_ns: np.ndarray,
    scd2_starts: np.ndarray,
    scd2_prices: np.ndarray,
) -> np.ndarray:
    """Resolve product prices using SCD2 version lookup based on date.

    Args:
        prod_idx: product indices (into current-product arrays)
        date_ns: dates as int64 nanoseconds
        scd2_starts: (n_products, max_ver) epoch-day boundaries
        scd2_prices: (n_products, max_ver) ListPrice per version

    Returns:
        float64 array of resolved prices
    """
    epoch_days = date_ns // (24 * 3600 * 10**9)  # ns → days
    p_starts = scd2_starts[prod_idx]  # (n, max_ver)
    ver_idx = np.sum(p_starts <= epoch_days[:, np.newaxis], axis=1) - 1
    ver_idx = np.clip(ver_idx, 0, scd2_starts.shape[1] - 1)
    return scd2_prices[prod_idx, ver_idx]
