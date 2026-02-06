import numpy as np
from src.facts.sales.sales_logic.globals import State


# ---------------------------------------------------------------------
# Config cache (process-local; safe with multiprocessing)
# ---------------------------------------------------------------------
_CFG_CACHE_KEY = None
_CFG_CACHE_VAL = None


def _get_cfg():
    """
    Robustly load activity config.

    Expected structure (recommended):
      models:
        activity:
          enabled: true
          monthly_baseline: [..12 floats..]
          size_scale_threshold: 200000
          bounds:
            dataset_split_threshold: 100000
            small_dataset: {low: 0.60, high: 1.60}
            large_dataset: {low: 0.80, high: 1.30}
          volatility: {sigma: 0.12}
          row_noise:
            sigma_small_dataset: 0.8
            sigma_large_dataset: 0.4
          month_inertia: 0.35
          preserve_row_count: false   # recommended for month-sliced generation
    """
    global _CFG_CACHE_KEY, _CFG_CACHE_VAL

    models = getattr(State, "models_cfg", None) or {}
    key = id(models)
    if _CFG_CACHE_KEY == key and _CFG_CACHE_VAL is not None:
        return _CFG_CACHE_VAL

    cfg = models.get("activity", {}) or {}

    enabled = bool(cfg.get("enabled", True))

    monthly_baseline = cfg.get("monthly_baseline", None)
    if monthly_baseline is None:
        monthly_baseline = [1.0] * 12
    if len(monthly_baseline) != 12:
        raise ValueError("models.activity.monthly_baseline must have 12 values")

    size_scale_threshold = float(cfg.get("size_scale_threshold", 200_000.0))
    size_scale_threshold = max(size_scale_threshold, 1.0)

    bounds = cfg.get("bounds", None) or {
        "dataset_split_threshold": 100_000,
        "small_dataset": {"low": 0.60, "high": 1.60},
        "large_dataset": {"low": 0.80, "high": 1.30},
    }

    volatility = cfg.get("volatility", None) or {"sigma": 0.12}
    row_noise = cfg.get("row_noise", None) or {"sigma_small_dataset": 0.80, "sigma_large_dataset": 0.40}

    inertia = float(cfg.get("month_inertia", 0.35))
    inertia = float(np.clip(inertia, 0.0, 0.98))

    preserve_row_count = bool(cfg.get("preserve_row_count", False))

    out = {
        "enabled": enabled,
        "monthly_baseline": np.asarray(monthly_baseline, dtype=np.float64),

        "size_scale_threshold": size_scale_threshold,
        "bounds": bounds,

        "volatility_sigma": float(volatility.get("sigma", 0.12)),
        "row_noise": row_noise,
        "month_inertia": inertia,

        "preserve_row_count": preserve_row_count,
    }

    _CFG_CACHE_KEY = key
    _CFG_CACHE_VAL = out
    return out


def _iter_groups_from_inv(inv: np.ndarray):
    """
    Yield (group_code, row_indices) for inv codes [0..U-1] in ascending group_code order,
    without doing np.where(inv == m) repeatedly.
    """
    order = np.argsort(inv, kind="stable")
    inv_sorted = inv[order]
    if inv_sorted.size == 0:
        return

    cuts = np.flatnonzero(inv_sorted[1:] != inv_sorted[:-1]) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, inv_sorted.size]

    for s, e in zip(starts, ends):
        code = int(inv_sorted[int(s)])
        yield code, order[int(s):int(e)]


def apply_activity_thinning(rng, order_dates):
    """
    Produces a boolean keep_mask for the provided rows.

    Recommended behavior in the new pipeline (month-sliced generation):
      - preserve_row_count = False  (actually thins rows)
    Legacy-compatible behavior:
      - preserve_row_count = True   (returns exactly n True values)

    Notes:
    - Works for single-month input (common now) and multi-month input (legacy).
    """
    cfg = _get_cfg()
    n = len(order_dates)

    if not cfg["enabled"]:
        return np.ones(n, dtype=bool)
    if n == 0:
        return np.zeros(0, dtype=bool)

    # Dataset size scaling (kept intent)
    threshold = cfg["size_scale_threshold"]
    size_scale = min(1.0, float(n) / float(threshold))

    # Month buckets
    months = np.asarray(order_dates).astype("datetime64[M]", copy=False)
    unique_months, inv = np.unique(months, return_inverse=True)
    U = int(unique_months.size)
    if U == 0:
        return np.zeros(n, dtype=bool)

    base_activity = cfg["monthly_baseline"]

    # Avg rows per month (avoid div by zero)
    avg_per_month = max(1, n // max(1, U))

    # Small vs large dataset behavior
    bounds = cfg["bounds"]
    split = int(bounds.get("dataset_split_threshold", 100_000))
    is_small = n < split

    small_bounds = bounds.get("small_dataset", {"low": 0.60, "high": 1.60})
    large_bounds = bounds.get("large_dataset", {"low": 0.80, "high": 1.30})

    low = float(small_bounds.get("low", 0.60) if is_small else large_bounds.get("low", 0.80))
    high = float(small_bounds.get("high", 1.60) if is_small else large_bounds.get("high", 1.30))

    row_noise = cfg["row_noise"]
    row_sigma = float(
        row_noise.get("sigma_small_dataset", 0.80) if is_small
        else row_noise.get("sigma_large_dataset", 0.40)
    )

    vol_sigma = float(cfg["volatility_sigma"])
    vol_sigma_eff = float(vol_sigma * size_scale)

    inertia = float(cfg["month_inertia"])

    keep_mask = np.zeros(n, dtype=bool)

    # Month-of-year per unique month (0–11)
    month_num = (unique_months.astype("int64") % 12).astype(np.int64, copy=False)

    prev_target = None

    # Iterate month groups efficiently
    for m, month_rows in _iter_groups_from_inv(inv):
        if month_rows.size == 0:
            continue

        mn = int(month_num[m])

        # Month-level volatility
        if vol_sigma_eff > 0.0:
            volatility = float(rng.lognormal(mean=0.0, sigma=vol_sigma_eff))
        else:
            volatility = 1.0

        raw_target = float(avg_per_month) * float(base_activity[mn]) * volatility

        if prev_target is None or inertia <= 0.0:
            target_f = raw_target
        else:
            target_f = inertia * float(prev_target) + (1.0 - inertia) * raw_target

        # Clamp AFTER smoothing
        target = int(np.clip(target_f, avg_per_month * low, avg_per_month * high))
        prev_target = target

        if month_rows.size <= target:
            keep_mask[month_rows] = True
            continue

        # Row weights (lognormal) + weighted sample without replacement
        # (same behavior as original; just avoids repeated np.where and minor overhead)
        row_weights = rng.lognormal(mean=0.0, sigma=row_sigma, size=month_rows.size).astype(np.float64, copy=False)
        s = float(row_weights.sum())
        if s <= 0.0 or not np.isfinite(s):
            # fallback uniform
            chosen = rng.choice(month_rows, size=target, replace=False)
        else:
            row_weights /= s
            chosen = rng.choice(month_rows, size=target, replace=False, p=row_weights)

        keep_mask[chosen] = True

    # ------------------------------------------------------------
    # Optional legacy balancing (kept exactly as before)
    # ------------------------------------------------------------
    if cfg["preserve_row_count"]:
        kept = int(keep_mask.sum())

        if kept > n:
            extra = rng.choice(np.where(keep_mask)[0], size=kept - n, replace=False)
            keep_mask[extra] = False
        elif kept < n:
            missing = rng.choice(np.where(~keep_mask)[0], size=n - kept, replace=False)
            keep_mask[missing] = True

    return keep_mask
