"""Promotion assignment: date-windowed, optionally weighted promo selection."""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _as_date32(x):
    """
    Normalize incoming date arrays to datetime64[D] for consistent comparisons.
    """
    if x is None:
        return None
    a = np.asarray(x)
    if a.size == 0:
        return a
    return a.astype("datetime64[D]", copy=False)


def _safe_clip_pct(p):
    p = np.asarray(p, dtype=np.float64)
    p = np.where(np.isfinite(p), p, 0.0)
    return np.clip(p, 0.0, 1.0)


def _sanitize_weights(promo_weight_all, promo_valid_glob):
    """
    Returns:
      weights: float64 array aligned with promo_keys_all, or None if unusable
    """
    if promo_weight_all is None:
        return None

    w = np.asarray(promo_weight_all, dtype=np.float64)
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.maximum(w, 0.0)
    w[~promo_valid_glob] = 0.0

    return w if w.sum() > 0.0 else None


# ----------------------------------------------------------------
# Main promotion assignment
# ----------------------------------------------------------------

def apply_promotions(
    rng,
    n,
    order_dates,
    promo_keys_all,
    promo_pct_all,
    promo_start_all,
    promo_end_all,
    no_discount_key=1,
    promo_weight_all=None,
):
    """
    Assign at most one promotion per row.

    Default behavior (matches previous):
      - Uniform random choice among active promotions for that row.

    Optional:
      - promo_weight_all: 1D array aligned with promo_keys_all (non-negative weights)
        enables weighted choice among active promos.

    Returns:
      promo_keys: int64 array (len n)
      promo_pct:  float64 array (len n), clipped to [0, 1]
    """
    n = int(n)
    promo_keys = np.full(n, int(no_discount_key), dtype=np.int64)
    promo_pct = np.zeros(n, dtype=np.float64)

    if n <= 0:
        return promo_keys, promo_pct
    if promo_keys_all is None:
        return promo_keys, promo_pct

    promo_keys_all = np.asarray(promo_keys_all, dtype=np.int64)
    P = int(promo_keys_all.size)
    if P == 0:
        return promo_keys, promo_pct

    # Clip promo pct
    promo_pct_all = _safe_clip_pct(promo_pct_all)

    promo_start_all = _as_date32(promo_start_all)
    promo_end_all = _as_date32(promo_end_all)
    order_dates = _as_date32(order_dates)

    if promo_start_all is None or promo_end_all is None or order_dates is None:
        return promo_keys, promo_pct
    if order_dates.shape[0] != n:
        raise ValueError("order_dates length must match n")
    if promo_start_all.shape[0] != P or promo_end_all.shape[0] != P or promo_pct_all.shape[0] != P:
        raise ValueError("promo_*_all arrays must align with promo_keys_all length")

    # Exclude the no-discount key globally (we will fill default with no_discount_key)
    promo_valid_glob = (promo_keys_all != int(no_discount_key))
    if not promo_valid_glob.any():
        return promo_keys, promo_pct

    weights = _sanitize_weights(promo_weight_all, promo_valid_glob)
    if promo_weight_all is not None and weights is None:
        # weights provided but unusable => treat as uniform
        weights = None

    # ------------------------------------------------------------
    # Group rows by unique date (fast path for month-sliced generation)
    # ------------------------------------------------------------
    unique_dates, inv = np.unique(order_dates, return_inverse=True)
    U = unique_dates.size

    # Precompute active promo indices per unique date
    # U is typically <= 31 when month-sliced; building lists is fast and avoids a U×P boolean matrix
    active_indices_per_date = []
    for d in unique_dates:
        active = promo_valid_glob & (d >= promo_start_all) & (d <= promo_end_all)
        active_indices_per_date.append(np.nonzero(active)[0])

    # Precompute per-group row counts and contiguous offsets for scatter-free assignment
    group_counts = np.bincount(inv, minlength=U).astype(np.int64)
    group_order = np.argsort(inv, kind="mergesort")
    group_starts = np.zeros(U + 1, dtype=np.int64)
    np.cumsum(group_counts, out=group_starts[1:])

    # Assign per unique-date group — one vectorized rng call per group
    for code in range(U):
        idx = active_indices_per_date[code]
        if idx.size == 0:
            continue

        count = int(group_counts[code])
        gs = int(group_starts[code])
        ge = int(group_starts[code + 1])
        rows = group_order[gs:ge]

        if weights is None:
            # uniform among active promos
            chosen = idx[rng.integers(0, idx.size, size=count)]
        else:
            # weighted among active promos via CDF + searchsorted
            w = weights[idx]
            s = float(w.sum())
            if s <= 0.0:
                chosen = idx[rng.integers(0, idx.size, size=count)]
            else:
                cdf = np.cumsum(w, dtype=np.float64)
                cdf /= cdf[-1]
                u = rng.random(count)
                j = np.searchsorted(cdf, u, side="right")
                chosen = idx[np.minimum(j, idx.size - 1)]

        promo_keys[rows] = promo_keys_all[chosen]
        promo_pct[rows] = promo_pct_all[chosen]

    return promo_keys, promo_pct
