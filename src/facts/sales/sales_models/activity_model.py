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
# Defaults (aligned exactly with models.yaml)
# ---------------------------------------------------------------

_DEFAULT_BASELINE = [0.92, 0.86, 0.96, 1.00, 1.13, 1.25,
                     1.04, 1.02, 0.99, 1.06, 1.28, 1.46]

_DEFAULT_BOUNDS_SMALL = {"low": 0.82, "high": 1.22}
_DEFAULT_BOUNDS_LARGE = {"low": 0.82, "high": 1.18}


# ---------------------------------------------------------------
# Config loading (process-local cache; safe with multiprocessing)
# ---------------------------------------------------------------
_CFG_VERSION: int = -1
_CFG_CACHE: dict | None = None


def _get_nested_or_flat(cfg: dict, block_key: str, nested_key: str,
                        flat_key: str, default: float) -> float:
    """
    Resolve a value from either a nested dict or a flat key.

    Checks cfg[block_key][nested_key] first, then cfg[flat_key],
    falling back to *default*.
    """
    block = cfg.get(block_key)
    if isinstance(block, dict):
        return float(block.get(nested_key, default))
    return float(cfg.get(flat_key, default))


def _get_bounds(bounds: dict, key: str, legacy_key: str,
                defaults: dict) -> tuple[float, float]:
    """Extract (low, high) from a bounds sub-dict with legacy fallback."""
    sub = bounds.get(key, bounds.get(legacy_key, defaults))
    return float(sub.get("low", defaults["low"])), float(sub.get("high", defaults["high"]))


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

    enabled = bool(cfg.get("enabled", True))

    # --- monthly baseline (must be exactly 12 floats) ---
    baseline = cfg.get("monthly_baseline")
    if baseline is None:
        baseline = list(_DEFAULT_BASELINE)
    if len(baseline) != 12:
        raise ValueError("models.activity.monthly_baseline must have 12 values")

    # --- size scaling ---
    size_threshold = max(1.0, float(cfg.get("size_scale_threshold", 200_000)))

    # --- month inertia ---
    inertia = float(np.clip(cfg.get("month_inertia", 0.73), 0.0, 0.98))

    # --- volatility (nested dict or flat key) ---
    vol_sigma = _get_nested_or_flat(cfg, "volatility", "sigma",
                                    "volatility_sigma", 0.03)

    # --- row noise (nested dict or flat keys) ---
    noise_block = cfg.get("row_noise")
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

    sm_lo, sm_hi = _get_bounds(bounds, "small", "small_dataset", _DEFAULT_BOUNDS_SMALL)
    lg_lo, lg_hi = _get_bounds(bounds, "large", "large_dataset", _DEFAULT_BOUNDS_LARGE)

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
        "bounds_small_low": sm_lo,
        "bounds_small_high": sm_hi,
        "bounds_large_low": lg_lo,
        "bounds_large_high": lg_hi,
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
    starts = np.empty(cuts.size + 1, dtype=np.intp)
    starts[0] = 0
    starts[1:] = cuts

    ends = np.empty(cuts.size + 1, dtype=np.intp)
    ends[:-1] = cuts
    ends[-1] = sorted_inv.size

    for s, e in zip(starts, ends):
        yield int(sorted_inv[s]), order[s:e]


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

    # ---- Bucket rows by calendar month ----
    months = np.asarray(order_dates).astype("datetime64[M]", copy=False)
    unique_months, inv = np.unique(months, return_inverse=True)
    n_months = unique_months.size

    if n_months == 0:
        return np.ones(n, dtype=bool)

    avg_per_month = max(1, n // max(1, n_months))

    # ---- Small vs large dataset parameters ----
    is_small = (n < cfg["split_threshold"])
    low = cfg["bounds_small_low"] if is_small else cfg["bounds_large_low"]
    high = cfg["bounds_small_high"] if is_small else cfg["bounds_large_high"]
    row_sigma = cfg["row_noise_small"] if is_small else cfg["row_noise_large"]

    vol_sigma_eff = cfg["volatility_sigma"] * size_scale
    inertia = cfg["month_inertia"]
    baseline = cfg["monthly_baseline"]

    # ---- Month-of-year lookup (0–11) for each unique month ----
    month_of_year = (unique_months.astype("int64") % 12).astype(np.int64)

    keep_mask = np.zeros(n, dtype=bool)
    prev_target: float | None = None

    for m_idx, row_indices in _iter_month_groups(inv):
        if row_indices.size == 0:
            continue

        mn = int(month_of_year[m_idx])

        # Month-level volatility (lognormal centered at 1.0)
        volatility = (float(rng.lognormal(mean=0.0, sigma=vol_sigma_eff))
                      if vol_sigma_eff > 0.0 else 1.0)

        raw_target = avg_per_month * float(baseline[mn]) * volatility

        # Exponential smoothing
        if prev_target is None or inertia <= 0.0:
            smoothed = raw_target
        else:
            smoothed = inertia * prev_target + (1.0 - inertia) * raw_target

        # Clamp to bounds and convert to int
        target = int(np.clip(smoothed, avg_per_month * low, avg_per_month * high))
        prev_target = float(target)

        # If month has fewer rows than target, keep all
        if row_indices.size <= target:
            keep_mask[row_indices] = True
            continue

        # Weighted sampling without replacement
        weights = rng.lognormal(mean=0.0, sigma=row_sigma, size=row_indices.size)
        total = weights.sum()

        if total > 0.0 and np.isfinite(total):
            weights /= total
            chosen = rng.choice(row_indices, size=target, replace=False, p=weights)
        else:
            chosen = rng.choice(row_indices, size=target, replace=False)

        keep_mask[chosen] = True

    # ---- Legacy balancing: force exactly n rows kept ----
    if cfg["preserve_row_count"]:
        kept = int(keep_mask.sum())
        if kept < n:
            off_indices = np.flatnonzero(~keep_mask)
            add_count = min(n - kept, off_indices.size)
            if add_count > 0:
                add = rng.choice(off_indices, size=add_count, replace=False)
                keep_mask[add] = True
        elif kept > n:
            on_indices = np.flatnonzero(keep_mask)
            drop_count = min(kept - n, on_indices.size)
            if drop_count > 0:
                drop = rng.choice(on_indices, size=drop_count, replace=False)
                keep_mask[drop] = False

    return keep_mask
