"""
Activity thinning model.

Shapes month-to-month order density by keeping/dropping rows based on
seasonal baselines, volatility, and per-row weighting.

Config source: models.yaml -> models.activity
Runtime state:  State.models_cfg  (the inner "models" dict)
"""
from __future__ import annotations

import numpy as np

from src.facts.sales.sales_logic import State


# ---------------------------------------------------------------
# Config loading (process-local cache; safe with multiprocessing)
# ---------------------------------------------------------------
_CFG_VERSION: int = -1
_CFG_CACHE: dict | None = None


def _load_cfg() -> dict:
    """
    Parse and validate the activity config block.

    Defaults are aligned exactly with models.yaml so that missing keys
    produce identical behavior to the shipped config.
    """
    global _CFG_VERSION, _CFG_CACHE

    models = getattr(State, "models_cfg", None) or {}
    version = id(models)
    if version == _CFG_VERSION and _CFG_CACHE is not None:
        return _CFG_CACHE

    cfg = models.get("activity", {}) or {}

    # --- enabled ---
    enabled = bool(cfg.get("enabled", True))

    # --- monthly baseline (must be exactly 12 floats) ---
    baseline = cfg.get("monthly_baseline", None)
    if baseline is None:
        baseline = [0.92, 0.86, 0.96, 1.00, 1.13, 1.25,
                    1.04, 1.02, 0.99, 1.06, 1.28, 1.46]
    if len(baseline) != 12:
        raise ValueError("models.activity.monthly_baseline must have 12 values")

    # --- size scaling ---
    size_threshold = float(cfg.get("size_scale_threshold", 200_000))
    size_threshold = max(size_threshold, 1.0)

    # --- month inertia ---
    inertia = float(np.clip(float(cfg.get("month_inertia", 0.73)), 0.0, 0.98))

    # --- volatility ---
    # Support both flat key and nested dict for backward compat.
    vol_block = cfg.get("volatility", None)
    if isinstance(vol_block, dict):
        vol_sigma = float(vol_block.get("sigma", 0.03))
    else:
        vol_sigma = float(cfg.get("volatility_sigma", 0.03))

    # --- row noise ---
    # Support both flat keys and nested dict.
    noise_block = cfg.get("row_noise", None)
    if isinstance(noise_block, dict):
        noise_small = float(noise_block.get("sigma_small_dataset", 0.10))
        noise_large = float(noise_block.get("sigma_large_dataset", 0.30))
    else:
        noise_small = float(cfg.get("row_noise_sigma", 0.10))
        noise_large = float(cfg.get("row_noise_sigma_large", 0.30))

    # --- bounds ---
    bounds = cfg.get("bounds", {}) or {}
    split_threshold = int(bounds.get("split_threshold",
                          bounds.get("dataset_split_threshold", 300_000)))

    small_bounds = bounds.get("small", bounds.get("small_dataset",
                   {"low": 0.82, "high": 1.22}))
    large_bounds = bounds.get("large", bounds.get("large_dataset",
                   {"low": 0.82, "high": 1.18}))

    # --- preserve row count ---
    preserve = bool(cfg.get("preserve_row_count", True))

    out = {
        "enabled": enabled,
        "monthly_baseline": np.asarray(baseline, dtype=np.float64),
        "size_scale_threshold": size_threshold,
        "month_inertia": inertia,
        "volatility_sigma": vol_sigma,
        "row_noise_small": noise_small,
        "row_noise_large": noise_large,
        "split_threshold": split_threshold,
        "bounds_small_low": float(small_bounds.get("low", 0.82)),
        "bounds_small_high": float(small_bounds.get("high", 1.22)),
        "bounds_large_low": float(large_bounds.get("low", 0.82)),
        "bounds_large_high": float(large_bounds.get("high", 1.18)),
        "preserve_row_count": preserve,
    }

    _CFG_VERSION = version
    _CFG_CACHE = out
    return out


def _reset_cache() -> None:
    """Reset module cache.  Call from State.reset() or tests."""
    global _CFG_VERSION, _CFG_CACHE
    _CFG_VERSION = -1
    _CFG_CACHE = None


# ---------------------------------------------------------------
# Fast month-group iteration
# ---------------------------------------------------------------

def _iter_month_groups(inv: np.ndarray):
    """
    Yield (group_code, row_indices) for month codes in ascending order.

    Uses argsort + boundary detection instead of repeated np.where.
    """
    if inv.size == 0:
        return

    order = np.argsort(inv, kind="stable")
    sorted_inv = inv[order]

    cuts = np.flatnonzero(sorted_inv[1:] != sorted_inv[:-1]) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, sorted_inv.size]

    for s, e in zip(starts, ends):
        yield int(sorted_inv[s]), order[s:e]


# ---------------------------------------------------------------
# Weighted row selection
# ---------------------------------------------------------------

def _select_keep(rng, row_indices: np.ndarray, target: int, row_sigma: float) -> np.ndarray:
    """
    Return up to `target` indices chosen from `row_indices` via lognormal-weighted
    sampling without replacement.  Returns the full array when target >= size.
    """
    if row_indices.size <= target:
        return row_indices

    weights = rng.lognormal(mean=0.0, sigma=row_sigma, size=row_indices.size)
    total = weights.sum()

    if total > 0.0 and np.isfinite(total):
        weights /= total
        return rng.choice(row_indices, size=target, replace=False, p=weights)

    return rng.choice(row_indices, size=target, replace=False)


# ---------------------------------------------------------------
# Public API
# ---------------------------------------------------------------

def apply_activity_thinning(rng, order_dates):
    """
    Produce a boolean keep_mask for the provided rows.

    - preserve_row_count = True  → returns exactly len(order_dates) True values
    - preserve_row_count = False → actually thins rows for realism

    Works for single-month input (common) and multi-month input (legacy).

    Parameters
    ----------
    rng : numpy.random.Generator
    order_dates : array-like of datetime64

    Returns
    -------
    np.ndarray[bool] of shape (n,)
    """
    cfg = _load_cfg()
    n = len(order_dates)

    if not cfg["enabled"] or n == 0:
        return np.ones(n, dtype=bool)

    # ---- Dataset-size scaling for volatility ----
    size_scale = min(1.0, float(n) / cfg["size_scale_threshold"])

    # ---- Small vs large dataset parameters ----
    is_small = n < cfg["split_threshold"]
    low = cfg["bounds_small_low"] if is_small else cfg["bounds_large_low"]
    high = cfg["bounds_small_high"] if is_small else cfg["bounds_large_high"]
    row_sigma = cfg["row_noise_small"] if is_small else cfg["row_noise_large"]

    vol_sigma_eff = cfg["volatility_sigma"] * size_scale
    inertia = cfg["month_inertia"]
    baseline = cfg["monthly_baseline"]

    months = np.asarray(order_dates).astype("datetime64[M]", copy=False)

    # ---- Fast path: single-month chunk (most common call pattern) ----
    # Avoids np.unique (O(n log n)) and the grouping argsort entirely.
    m0 = months[0]
    if (months == m0).all():
        mn = int(m0.astype("int64") % 12)
        volatility = float(rng.lognormal(mean=0.0, sigma=vol_sigma_eff)) if vol_sigma_eff > 0.0 else 1.0
        raw_target = n * float(baseline[mn]) * volatility
        target = int(np.clip(raw_target, n * low, n * high))

        keep_mask = np.zeros(n, dtype=bool)
        chosen = _select_keep(rng, np.arange(n), target, row_sigma)
        keep_mask[chosen] = True

        if cfg["preserve_row_count"] and chosen.size < n:
            off = np.flatnonzero(~keep_mask)
            add_count = n - chosen.size
            if add_count > 0:
                keep_mask[rng.choice(off, size=add_count, replace=False)] = True

        return keep_mask

    # ---- Multi-month path ----
    unique_months, inv = np.unique(months, return_inverse=True)
    n_months = unique_months.size
    avg_per_month = max(1, n // n_months)

    # Month-of-year lookup (0–11) for each unique month
    month_of_year = (unique_months.astype("int64") % 12).astype(np.int64)

    keep_mask = np.zeros(n, dtype=bool)
    # Track the pre-clamp smoothed value so the EMA state is not anchored to
    # clamp boundaries when seasonal bounds are hit.
    ema_state: float | None = None

    for m_idx, row_indices in _iter_month_groups(inv):
        if row_indices.size == 0:
            continue

        mn = int(month_of_year[m_idx])

        # Month-level volatility (lognormal centered at 1.0)
        volatility = float(rng.lognormal(mean=0.0, sigma=vol_sigma_eff)) if vol_sigma_eff > 0.0 else 1.0
        raw_target = avg_per_month * float(baseline[mn]) * volatility

        # Exponential smoothing over unclamped signal
        if ema_state is None or inertia <= 0.0:
            smoothed = raw_target
        else:
            smoothed = inertia * ema_state + (1.0 - inertia) * raw_target

        ema_state = smoothed  # preserve pre-clamp value for next iteration
        target = int(np.clip(smoothed, avg_per_month * low, avg_per_month * high))

        chosen = _select_keep(rng, row_indices, target, row_sigma)
        keep_mask[chosen] = True

    # ---- Balancing: restore full row count when preserve_row_count is set ----
    if cfg["preserve_row_count"]:
        kept = int(keep_mask.sum())
        if kept < n:
            off_indices = np.flatnonzero(~keep_mask)
            add_count = min(n - kept, off_indices.size)
            if add_count > 0:
                keep_mask[rng.choice(off_indices, size=add_count, replace=False)] = True

    return keep_mask
