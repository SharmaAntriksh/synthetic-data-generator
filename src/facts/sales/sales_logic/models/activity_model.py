import numpy as np
from src.facts.sales.sales_logic.globals import State


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
    models = getattr(State, "models_cfg", None) or {}
    cfg = models.get("activity", {}) or {}

    # Allow disabling without breaking pipeline
    enabled = bool(cfg.get("enabled", True))

    monthly_baseline = cfg.get("monthly_baseline", None)
    if monthly_baseline is None:
        # neutral baseline: no month-of-year effect
        monthly_baseline = [1.0] * 12

    if len(monthly_baseline) != 12:
        raise ValueError("models.activity.monthly_baseline must have 12 values")

    out = {
        "enabled": enabled,
        "monthly_baseline": np.array(monthly_baseline, dtype=np.float64),

        "size_scale_threshold": float(cfg.get("size_scale_threshold", 200_000.0)),

        "bounds": cfg.get("bounds", {
            "dataset_split_threshold": 100_000,
            "small_dataset": {"low": 0.60, "high": 1.60},
            "large_dataset": {"low": 0.80, "high": 1.30},
        }),

        "volatility": cfg.get("volatility", {"sigma": 0.12}),
        "row_noise": cfg.get("row_noise", {"sigma_small_dataset": 0.80, "sigma_large_dataset": 0.40}),
        "month_inertia": float(cfg.get("month_inertia", 0.35)),

        # New behavior toggle
        "preserve_row_count": bool(cfg.get("preserve_row_count", False)),
    }

    return out


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
    if not cfg["enabled"]:
        return np.ones(len(order_dates), dtype=bool)

    n = len(order_dates)
    if n == 0:
        return np.zeros(0, dtype=bool)

    # Dataset size scaling (kept from original intent, but safe)
    threshold = max(cfg["size_scale_threshold"], 1.0)
    size_scale = min(1.0, n / threshold)

    # Month buckets
    months = order_dates.astype("datetime64[M]")
    unique_months, inv = np.unique(months, return_inverse=True)

    base_activity = cfg["monthly_baseline"]

    keep_mask = np.zeros(n, dtype=bool)

    # Average rows per month (avoid div by zero)
    avg_per_month = max(1, n // max(1, len(unique_months)))

    # Small vs large dataset behavior
    bounds = cfg["bounds"]
    split = int(bounds.get("dataset_split_threshold", 100_000))

    is_small = n < split

    small_bounds = bounds.get("small_dataset", {"low": 0.60, "high": 1.60})
    large_bounds = bounds.get("large_dataset", {"low": 0.80, "high": 1.30})

    low = float(small_bounds.get("low", 0.60) if is_small else large_bounds.get("low", 0.80))
    high = float(small_bounds.get("high", 1.60) if is_small else large_bounds.get("high", 1.30))

    row_noise = cfg["row_noise"]
    row_sigma = float(row_noise.get("sigma_small_dataset", 0.80) if is_small else row_noise.get("sigma_large_dataset", 0.40))

    vol_sigma = float(cfg.get("volatility", {}).get("sigma", 0.12))

    # Month-to-month smoothing (still relevant if input spans multiple months)
    inertia = float(cfg.get("month_inertia", 0.35))
    inertia = min(max(inertia, 0.0), 0.98)

    prev_target = None

    for m in range(len(unique_months)):
        month_rows = np.where(inv == m)[0]
        if month_rows.size == 0:
            continue

        # Month-of-year (0–11)
        month_num = int(unique_months[m].astype("int64") % 12)

        # Month-level volatility
        volatility = rng.lognormal(mean=0.0, sigma=vol_sigma * size_scale)

        raw_target = avg_per_month * base_activity[month_num] * volatility

        if prev_target is None or inertia <= 0.0:
            target = raw_target
        else:
            target = inertia * prev_target + (1.0 - inertia) * raw_target

        # Clamp AFTER smoothing
        target = int(np.clip(target, avg_per_month * low, avg_per_month * high))
        prev_target = target

        if month_rows.size <= target:
            keep_mask[month_rows] = True
        else:
            row_weights = rng.lognormal(mean=0.0, sigma=row_sigma, size=month_rows.size).astype(np.float64)
            row_weights /= row_weights.sum()

            chosen = rng.choice(month_rows, size=target, replace=False, p=row_weights)
            keep_mask[chosen] = True

    # ------------------------------------------------------------
    # Optional legacy balancing
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
