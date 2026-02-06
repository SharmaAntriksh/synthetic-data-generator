import numpy as np


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


def _group_by_inverse(inv: np.ndarray):
    """
    Efficient grouping for inv codes:
      inv: int array (len n) with values in [0..U-1]
    Yields tuples (code, row_idx_array)
    """
    order = np.argsort(inv, kind="stable")
    inv_sorted = inv[order]
    if inv_sorted.size == 0:
        return

    # boundaries where code changes
    cuts = np.flatnonzero(inv_sorted[1:] != inv_sorted[:-1]) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, inv_sorted.size]

    for s, e in zip(starts, ends):
        code = int(inv_sorted[int(s)])
        yield code, order[int(s):int(e)]


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

    # Precompute active promos per unique date: U x P comparisons
    # U is typically <= 31 when month-sliced; safe & fast.
    # active_u: list of index arrays of promos active that day
    active_u = []
    for d in unique_dates:
        active = promo_valid_glob & (d >= promo_start_all) & (d <= promo_end_all)
        idx = np.nonzero(active)[0]
        active_u.append(idx)

    # Assign per-group (date)
    for code, rows in _group_by_inverse(inv):
        idx = active_u[code]
        if idx.size == 0:
            continue

        if weights is None:
            # uniform among active promos
            chosen = idx[rng.integers(0, idx.size, size=rows.size)]
        else:
            # weighted among active promos, vectorized via CDF + searchsorted
            w = weights[idx]
            s = float(w.sum())
            if s <= 0.0:
                chosen = idx[rng.integers(0, idx.size, size=rows.size)]
            else:
                cdf = np.cumsum(w, dtype=np.float64)
                cdf /= cdf[-1]
                u = rng.random(rows.size)
                j = np.searchsorted(cdf, u, side="right")
                chosen = idx[np.minimum(j, idx.size - 1)]

        promo_keys[rows] = promo_keys_all[chosen]
        promo_pct[rows] = promo_pct_all[chosen]

    return promo_keys, promo_pct
