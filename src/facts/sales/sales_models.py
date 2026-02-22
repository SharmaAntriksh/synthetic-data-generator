"""sales_models.py

Combined (behavior-preserving) from:
- sales_logic/models/activity_model.py
- sales_logic/models/customer_lifecycle.py
- sales_logic/models/pricing_pipeline.py
- sales_logic/models/quantity_model.py

Mechanical adjustments:
- Updated State import path to `from src.facts.sales.sales_logic import State` (sales_logic is now a module).
- Namespaced colliding private identifiers from the original separate modules to avoid cross-module cache/helper collisions.
  Public function names and logic are unchanged.
"""

from __future__ import annotations

# =============================================================================
# activity_model.py (combined)
# =============================================================================
import numpy as np
from src.facts.sales.sales_logic import State


# ---------------------------------------------------------------------
# Config cache (process-local; safe with multiprocessing)
# ---------------------------------------------------------------------
_ACT_CFG_CACHE_KEY = None
_ACT_CFG_CACHE_VAL = None


def _act_get_cfg():
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
    global _ACT_CFG_CACHE_KEY, _ACT_CFG_CACHE_VAL

    models = getattr(State, "models_cfg", None) or {}
    key = id(models)
    if _ACT_CFG_CACHE_KEY == key and _ACT_CFG_CACHE_VAL is not None:
        return _ACT_CFG_CACHE_VAL

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

    _ACT_CFG_CACHE_KEY = key
    _ACT_CFG_CACHE_VAL = out
    return out


def _act_iter_groups_from_inv(inv: np.ndarray):
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
    cfg = _act_get_cfg()
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
    for m, month_rows in _act_iter_groups_from_inv(inv):
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


# =============================================================================
# customer_lifecycle.py (combined)
# =============================================================================
"""
NEW ROLE (post-overhaul):
- Customer existence over time is encoded in customers.parquet:
    IsActiveInSales, CustomerStartMonth, CustomerEndMonth
- This module optionally applies a *monthly activity overlay* on top of that:
    temporary inactivity (dormancy), optional reactivation, mild seasonality/noise

IMPORTANT:
- Do NOT use this module to "grow the customer base" anymore.
  That conflicts with CustomerStartMonth/CustomerEndMonth generated in customers.py.
"""
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from src.facts.sales.sales_logic import State


# -------------------------------------------------------------------
# Config model (safe defaults; all optional)
# -------------------------------------------------------------------
@dataclass(frozen=True)
class ActivityOverlayCfg:
    enabled: bool = False

    # Base probability that an eligible customer is "active" in a given month.
    # If enabled=False, activity mask == eligibility mask.
    base_monthly_activity_rate: float = 0.85

    # Noise applied to base activity rate each month (multiplicative)
    monthly_noise: float = 0.08

    # Seasonality (multiplicative): activity_rate *= 1 + seasonal_amplitude*sin(2πm/period)
    seasonal_amplitude: float = 0.15
    seasonal_period_months: int = 24

    # Reactivation: fraction of inactive customers that can return each month
    monthly_reactivation_rate: float = 0.10

    # Optional: if customer_temperature array exists, scale volatility
    # Effective noise = monthly_noise * (1 + temperature_noise_scale*(temp-1))
    temperature_noise_scale: float = 0.50


def _load_overlay_cfg() -> ActivityOverlayCfg:
    """
    Reads config from State.models_cfg if present.

    Recommended config location (doesn't break old configs):
      customer_activity_overlay:
        enabled: true
        base_monthly_activity_rate: 0.85
        monthly_noise: 0.08
        seasonal_amplitude: 0.15
        seasonal_period_months: 24
        monthly_reactivation_rate: 0.10
        temperature_noise_scale: 0.50

    If absent, returns disabled overlay (no behavior change).
    """
    cfg = (getattr(State, "models_cfg", None) or {})
    block = cfg.get("customer_activity_overlay", {}) or {}
    try:
        return ActivityOverlayCfg(
            enabled=bool(block.get("enabled", False)),
            base_monthly_activity_rate=float(block.get("base_monthly_activity_rate", 0.85)),
            monthly_noise=float(block.get("monthly_noise", 0.08)),
            seasonal_amplitude=float(block.get("seasonal_amplitude", 0.15)),
            seasonal_period_months=int(block.get("seasonal_period_months", 24)),
            monthly_reactivation_rate=float(block.get("monthly_reactivation_rate", 0.10)),
            temperature_noise_scale=float(block.get("temperature_noise_scale", 0.50)),
        )
    except Exception as e:
        raise ValueError(f"Invalid customer_activity_overlay config: {e}") from e


def _normalize_end_month(end_month_arr, n: int) -> np.ndarray:
    """
    Convert nullable end-month representations into an int64 array with -1 meaning "no end".
    """
    n = int(n)
    if end_month_arr is None:
        return np.full(n, -1, dtype=np.int64)

    a = np.asarray(end_month_arr)

    # ints
    if np.issubdtype(a.dtype, np.integer):
        out = a.astype(np.int64, copy=False)
        out = np.where(out < 0, -1, out)
        return out

    # floats (NaN -> -1)
    if np.issubdtype(a.dtype, np.floating):
        out = np.where(np.isnan(a), -1, a).astype(np.int64)
        out[out < 0] = -1
        return out

    # objects / nullable ints (pd.NA/None)
    if a.dtype == object:
        try:
            import pandas as pd

            s = pd.Series(a, copy=False)
            num = pd.to_numeric(s, errors="coerce")
            out = num.fillna(-1).astype("int64").to_numpy()
            out[out < 0] = -1
            return out
        except Exception:
            out = np.full(n, -1, dtype=np.int64)
            lim = min(n, a.shape[0])
            for i in range(lim):
                v = a[i]
                if v is None:
                    continue
                try:
                    if v is np.nan:  # noqa: E721
                        continue
                except Exception:
                    pass
                try:
                    iv = int(v)
                    out[i] = iv if iv >= 0 else -1
                except Exception:
                    pass
            return out

    # last resort
    try:
        out = a.astype(np.int64, copy=False)
        out[out < 0] = -1
        return out
    except Exception:
        return np.full(n, -1, dtype=np.int64)


# -------------------------------------------------------------------
# Core: Eligibility mask from Customers dimension (authoritative)
# -------------------------------------------------------------------
def build_eligibility_by_month(
    start_month_arr: np.ndarray,
    end_month_arr: Optional[np.ndarray],
    is_active_in_sales_arr: Optional[np.ndarray],
    start_month: int,
    end_month: int,
) -> Dict[int, np.ndarray]:
    """
    Build month -> eligibility mask over customers (aligned by customer row index).

    Eligibility rule:
      - IsActiveInSales == 1 (if provided, else assume all eligible globally)
      - CustomerStartMonth <= m
      - CustomerEndMonth < 0 or missing => no end
        else m <= CustomerEndMonth
    """
    start_month_arr = np.asarray(start_month_arr, dtype=np.int64)
    n = int(start_month_arr.shape[0])

    start_m = int(start_month)
    end_m = int(end_month)
    if end_m < start_m:
        return {}

    if is_active_in_sales_arr is None:
        active_gate = np.ones(n, dtype=bool)
    else:
        active_gate = (np.asarray(is_active_in_sales_arr, dtype=np.int64) == 1)

    end_month_norm = _normalize_end_month(end_month_arr, n)

    # Initial mask for start_m (single pass comparisons)
    mask = active_gate & (start_month_arr <= start_m) & ((end_month_norm < 0) | (start_m <= end_month_norm))

    out: Dict[int, np.ndarray] = {start_m: mask.copy()}

    # Precompute start-events and end-events to update incrementally
    # Starts: indices sorted by start_month
    start_sorted = np.argsort(start_month_arr, kind="mergesort")
    start_vals = start_month_arr[start_sorted]
    ptr_start = int(np.searchsorted(start_vals, start_m + 1, side="left"))

    # Ends: consider only finite ends (>=0); remove after month m if end == m
    finite_end_idx = np.flatnonzero(end_month_norm >= 0)
    if finite_end_idx.size:
        end_vals_f = end_month_norm[finite_end_idx]
        end_sorted = finite_end_idx[np.argsort(end_vals_f, kind="mergesort")]
        end_vals_sorted = end_month_norm[end_sorted]
        ptr_end = int(np.searchsorted(end_vals_sorted, start_m, side="left"))
    else:
        end_sorted = np.empty(0, dtype=np.int64)
        end_vals_sorted = np.empty(0, dtype=np.int64)
        ptr_end = 0

    # Increment month by month
    for m in range(start_m, end_m):
        next_m = m + 1

        # Remove customers whose lifecycle ends at month m
        while ptr_end < end_sorted.size and int(end_vals_sorted[ptr_end]) == m:
            i = int(end_sorted[ptr_end])
            mask[i] = False
            ptr_end += 1

        # Add customers whose lifecycle starts at month next_m
        while ptr_start < start_sorted.size and int(start_vals[ptr_start]) == next_m:
            i = int(start_sorted[ptr_start])
            if active_gate[i] and (end_month_norm[i] < 0 or end_month_norm[i] >= next_m):
                mask[i] = True
            ptr_start += 1

        out[next_m] = mask.copy()

    return out


# -------------------------------------------------------------------
# Optional: Activity overlay (temporary inactivity)
# -------------------------------------------------------------------
def apply_activity_overlay_by_month(
    eligibility_by_month: Dict[int, np.ndarray],
    seed: int,
    customer_temperature: Optional[np.ndarray] = None,
) -> Dict[int, np.ndarray]:
    """
    Applies an optional activity overlay on top of eligibility masks.
    Returns month -> active_mask where active_mask <= eligibility_mask.

    If overlay is disabled, returns eligibility_by_month unchanged.
    """
    cfg = _load_overlay_cfg()
    if not cfg.enabled:
        return eligibility_by_month

    # Validate rates (fail fast)
    if not (0.0 <= cfg.base_monthly_activity_rate <= 1.0):
        raise ValueError("base_monthly_activity_rate must be in [0, 1]")
    if cfg.monthly_noise < 0.0:
        raise ValueError("monthly_noise must be >= 0")
    if cfg.seasonal_period_months <= 0:
        raise ValueError("seasonal_period_months must be > 0")
    if not (0.0 <= cfg.monthly_reactivation_rate <= 1.0):
        raise ValueError("monthly_reactivation_rate must be in [0, 1]")

    rng_global = np.random.default_rng(int(seed) + 9100)

    # Optional temperature (aligned with customers)
    temp = None
    if customer_temperature is not None:
        temp = np.asarray(customer_temperature, dtype=np.float64)
        temp = np.where(np.isfinite(temp), temp, 1.0)
        temp = np.clip(temp, 0.2, 3.0)

    out: Dict[int, np.ndarray] = {}
    prev_active: Optional[np.ndarray] = None

    months = sorted(eligibility_by_month.keys())
    if not months:
        return out

    for m in months:
        elig = np.asarray(eligibility_by_month[m], dtype=bool)
        n = int(elig.shape[0])

        # Month-scoped RNG for month-level jitter (keeps original determinism pattern)
        rng_m = np.random.default_rng(int(seed) + 10000 + int(m) * 7919)

        # Seasonality multiplier
        cycle = np.sin(2.0 * np.pi * (float(m) / float(cfg.seasonal_period_months)))
        seasonal_mult = 1.0 + cfg.seasonal_amplitude * cycle

        # Base activity probability for this month (scalar)
        p = cfg.base_monthly_activity_rate * seasonal_mult
        if cfg.monthly_noise > 0:
            p *= float(rng_m.uniform(1.0 - cfg.monthly_noise, 1.0 + cfg.monthly_noise))
        p = float(np.clip(p, 0.0, 1.0))

        # Per-customer probability (optional temp-driven volatility)
        if temp is not None:
            noise_scale = cfg.monthly_noise * (1.0 + cfg.temperature_noise_scale * (temp - 1.0))
            noise_scale = np.clip(noise_scale, 0.0, 0.50)
            per_cust_p = p * rng_m.uniform(1.0 - noise_scale, 1.0 + noise_scale, size=n)
            per_cust_p = np.clip(per_cust_p, 0.0, 1.0)
        else:
            per_cust_p = p  # scalar

        if prev_active is None:
            u = rng_global.random(n)
            active = elig & (u < per_cust_p)
            out[m] = active
            prev_active = active
            continue

        # Stay active
        u1 = rng_global.random(n)
        active = (prev_active & elig) & (u1 < per_cust_p)

        # Reactivate
        if cfg.monthly_reactivation_rate > 0.0:
            u2 = rng_global.random(n)
            active |= ((~prev_active) & elig) & (u2 < cfg.monthly_reactivation_rate)

        out[m] = active
        prev_active = active

    return out


# -------------------------------------------------------------------
# Backward-compatible API (kept because old code imported it)
# -------------------------------------------------------------------
def build_active_customer_pool(
    all_customers,
    start_month: int,
    end_month: int,
    seed: int,
):
    """
    BACKWARD-COMPAT SHIM.

    New behavior:
    - If State has lifecycle arrays, compute eligibility from them and
      optionally apply an activity overlay.
    - If lifecycle arrays are missing, fall back to "all active" masks
      across months (closest old behavior for compatibility).

    Returns:
        dict[int, np.ndarray]: month_idx -> boolean mask over customers (row-aligned)
    """
    cust_keys = getattr(State, "customer_keys", None)
    start_arr = getattr(State, "customer_start_month", None)
    end_arr = getattr(State, "customer_end_month", None)
    act_arr = getattr(State, "customer_is_active_in_sales", None)
    temp_arr = getattr(State, "customer_temperature", None)  # optional future

    if cust_keys is not None and start_arr is not None:
        elig = build_eligibility_by_month(
            start_month_arr=start_arr,
            end_month_arr=end_arr,
            is_active_in_sales_arr=act_arr,
            start_month=start_month,
            end_month=end_month,
        )
        return apply_activity_overlay_by_month(
            eligibility_by_month=elig,
            seed=seed,
            customer_temperature=temp_arr,
        )

    # Fallback: treat all customers as eligible+active
    all_customers = np.asarray(all_customers, dtype=np.int64)
    n = int(all_customers.shape[0])
    mask = np.ones(n, dtype=bool)
    return {m: mask.copy() for m in range(int(start_month), int(end_month) + 1)}


# =============================================================================
# pricing_pipeline.py (combined)
# =============================================================================
import numpy as np
from src.facts.sales.sales_logic import State


# ---------------------------------------------------------------------
# Module caches (process-local; safe with multiprocessing)
# ---------------------------------------------------------------------
_APPEAR_CACHE_KEY = None
_APPEAR_CACHE_VAL = None

_MONTH_NOISE_CACHE = {}  # (vol_seed, sigma, month_int) -> multiplier


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _as_f64(x):
    a = np.asarray(x, dtype=np.float64)
    return np.where(np.isfinite(a), a, 0.0)


def _parse_bands_to_arrays(bands, default):
    """
    bands: list[dict] -> arrays (maxs, steps) sorted by max
    """
    out = []
    if isinstance(bands, list):
        for b in bands:
            if not isinstance(b, dict):
                continue
            mx = b.get("max", None)
            st = b.get("step", None)
            try:
                mx = float(mx)
                st = float(st)
            except Exception:
                continue
            if mx <= 0 or st <= 0:
                continue
            out.append((mx, st))

    if not out:
        out = list(default)

    out.sort(key=lambda t: t[0])
    maxs = np.asarray([m for m, _ in out], dtype=np.float64)
    steps = np.asarray([s for _, s in out], dtype=np.float64)
    if maxs.size == 0:
        maxs = np.asarray([1e18], dtype=np.float64)
        steps = np.asarray([0.01], dtype=np.float64)
    return maxs, steps


def _choose_step_by_magnitude(x: np.ndarray, band_max: np.ndarray, band_step: np.ndarray) -> np.ndarray:
    """
    Vectorized: choose first band where x <= band_max. Uses searchsorted.
    """
    x = np.asarray(x, dtype=np.float64)
    idx = np.searchsorted(band_max, x, side="left")
    idx = np.minimum(idx, band_step.size - 1)
    step = band_step[idx]
    return np.where(step > 0.0, step, 0.01)


def _quantize(x: np.ndarray, step: np.ndarray, rounding: str) -> np.ndarray:
    if rounding == "floor":
        return np.floor(x / step) * step
    # nearest
    return np.round(x / step) * step


def _safe_prob(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= 0.0:
        return np.full(w.shape[0], 1.0 / max(1, w.shape[0]), dtype=np.float64)
    return w / s


# ---------------------------------------------------------------------
# Appearance config parsing (cached)
# ---------------------------------------------------------------------
def _parse_endings(endings, *, default_if_missing: bool):
    """
    Returns (vals, probs) or (None, None) if missing and default_if_missing=False.
    vals are cents endings in [0.0, 0.99].
    """
    if not isinstance(endings, list) or len(endings) == 0:
        if not default_if_missing:
            return None, None
        endings = [
            {"value": 0.99, "weight": 0.70},
            {"value": 0.50, "weight": 0.25},
            {"value": 0.00, "weight": 0.05},
        ]

    vals = []
    wts = []
    for e in endings:
        if not isinstance(e, dict):
            continue
        try:
            v = float(e.get("value", 0.0))
            w = float(e.get("weight", 0.0))
        except Exception:
            continue
        if w <= 0:
            continue
        v = float(np.clip(v, 0.0, 0.99))
        vals.append(v)
        wts.append(w)

    if not vals:
        if not default_if_missing:
            return None, None
        vals = [0.99]
        wts = [1.0]

    w = _safe_prob(np.asarray(wts, dtype=np.float64))
    return np.asarray(vals, dtype=np.float64), w


def _appearance_cfg():
    """
    Cache parse of models_cfg.pricing.appearance.
    """
    global _APPEAR_CACHE_KEY, _APPEAR_CACHE_VAL

    models = getattr(State, "models_cfg", None) or {}
    key = id(models)
    if _APPEAR_CACHE_KEY == key and _APPEAR_CACHE_VAL is not None:
        return _APPEAR_CACHE_VAL

    p = models.get("pricing", {}) or {}
    appearance = p.get("appearance", {}) or {}

    enabled = bool(appearance.get("enabled", False))

    # Unit price snapping
    unit_cfg = appearance.get("unit_price", {}) or {}
    up_round = str(unit_cfg.get("rounding", "nearest")).strip().lower()
    if up_round not in ("nearest", "floor"):
        up_round = "nearest"

    up_max, up_step = _parse_bands_to_arrays(
        unit_cfg.get("bands", None),
        default=[(200.0, 1.0), (1000.0, 5.0), (10000.0, 5.0), (1e18, 10.0)],
    )
    up_end_vals, up_end_w = _parse_endings(unit_cfg.get("endings", None), default_if_missing=True)

    # Unit cost snapping
    cost_cfg = appearance.get("unit_cost", {}) or {}
    uc_round = str(cost_cfg.get("rounding", "nearest")).strip().lower()
    if uc_round not in ("nearest", "floor"):
        uc_round = "nearest"

    uc_max, uc_step = _parse_bands_to_arrays(
        cost_cfg.get("bands", None),
        default=[(200.0, 0.05), (1000.0, 0.10), (10000.0, 1.0), (1e18, 5.0)],
    )
    # IMPORTANT: unit_cost endings are OPTIONAL; default is None (preserves old behavior)
    uc_end_vals, uc_end_w = _parse_endings(cost_cfg.get("endings", None), default_if_missing=False)

    # Discount snapping
    disc_cfg = appearance.get("discount", {}) or {}
    d_round = str(disc_cfg.get("rounding", "floor")).strip().lower()
    if d_round not in ("nearest", "floor"):
        d_round = "floor"

    d_max, d_step = _parse_bands_to_arrays(
        disc_cfg.get("bands", None),
        default=[(50.0, 0.50), (200.0, 1.0), (1000.0, 5.0), (5000.0, 10.0), (1e18, 25.0)],
    )

    out = {
        "enabled": enabled,
        "up_round": up_round,
        "up_band_max": up_max,
        "up_band_step": up_step,
        "up_end_vals": up_end_vals,
        "up_end_w": up_end_w,
        "uc_round": uc_round,
        "uc_band_max": uc_max,
        "uc_band_step": uc_step,
        "uc_end_vals": uc_end_vals,
        "uc_end_w": uc_end_w,
        "d_round": d_round,
        "d_band_max": d_max,
        "d_band_step": d_step,
    }

    _APPEAR_CACHE_KEY = key
    _APPEAR_CACHE_VAL = out
    return out


# ---------------------------------------------------------------------
# Snap / appearance
# ---------------------------------------------------------------------
def _snap_unit_price(rng, up: np.ndarray, appearance_cfg: dict) -> np.ndarray:
    if not appearance_cfg.get("enabled", False):
        return up

    up = _as_f64(up)
    up = np.maximum(up, 0.0)

    step = _choose_step_by_magnitude(up, appearance_cfg["up_band_max"], appearance_cfg["up_band_step"])
    anchor = _quantize(up, step, rounding=appearance_cfg["up_round"])

    end_vals = appearance_cfg["up_end_vals"]
    end_w = appearance_cfg["up_end_w"]

    idx = rng.choice(end_vals.size, size=up.shape[0], p=end_w)
    ending = end_vals[idx]

    # Price ending as cents: anchor is typically integer-dollar already due to steps
    snapped = anchor + ending
    return np.maximum(snapped, 0.01)


def _snap_cost(rng, uc: np.ndarray, appearance_cfg: dict) -> np.ndarray:
    if not appearance_cfg.get("enabled", False):
        return uc

    uc = _as_f64(uc)
    uc = np.maximum(uc, 0.0)

    step = _choose_step_by_magnitude(uc, appearance_cfg["uc_band_max"], appearance_cfg["uc_band_step"])
    anchor = _quantize(uc, step, rounding=appearance_cfg["uc_round"])

    # Optional endings for cost: force cents via floor(anchor) + ending
    end_vals = appearance_cfg.get("uc_end_vals", None)
    end_w = appearance_cfg.get("uc_end_w", None)
    if end_vals is not None and end_w is not None and end_vals.size > 0:
        idx = rng.choice(end_vals.size, size=uc.shape[0], p=end_w)
        ending = end_vals[idx]
        snapped = np.floor(anchor) + ending
    else:
        snapped = anchor

    return np.maximum(snapped, 0.0)


def _snap_discount(disc: np.ndarray, up: np.ndarray, appearance_cfg: dict) -> np.ndarray:
    if not appearance_cfg.get("enabled", False):
        return disc

    disc = _as_f64(disc)
    disc = np.maximum(disc, 0.0)

    # discount step determined by UnitPrice magnitude
    up = _as_f64(up)
    step = _choose_step_by_magnitude(np.maximum(up, 0.0), appearance_cfg["d_band_max"], appearance_cfg["d_band_step"])
    snapped = _quantize(disc, step, rounding=appearance_cfg["d_round"])
    return np.maximum(snapped, 0.0)


# ---------------------------------------------------------------------
# Inflation / drift config (kept same keys)
# ---------------------------------------------------------------------
def _cfg():
    """
    models:
      pricing:
        inflation:
          annual_rate: 0.04
          month_volatility_sigma: 0.01
          factor_clip: [0.90, 1.15]
          scale_discount: true
          volatility_seed: 123
        appearance:
          enabled: true
          ...
    """
    models = getattr(State, "models_cfg", None) or {}
    p = models.get("pricing", {}) or {}
    infl = p.get("inflation", {}) or {}

    annual_rate = float(infl.get("annual_rate", 0.0))
    month_sigma = float(infl.get("month_volatility_sigma", 0.0))
    scale_discount = bool(infl.get("scale_discount", True))

    clip = infl.get("factor_clip", [0.0, 10.0])
    if not (isinstance(clip, (list, tuple)) and len(clip) == 2):
        clip = [0.0, 10.0]
    lo, hi = float(clip[0]), float(clip[1])
    if hi < lo:
        lo, hi = hi, lo

    vol_seed = int(infl.get("volatility_seed", 0))

    annual_rate = float(np.clip(annual_rate, -0.50, 2.0))
    month_sigma = float(np.clip(month_sigma, 0.0, 0.25))
    lo = float(max(lo, 0.0))
    hi = float(max(hi, lo))

    return annual_rate, month_sigma, lo, hi, scale_discount, vol_seed


def _global_start_month_int(order_dates: np.ndarray) -> int:
    dp = getattr(State, "date_pool", None)
    if dp is not None:
        try:
            if len(dp) > 0:
                d0 = np.min(np.asarray(dp).astype("datetime64[D]"))
                return int(d0.astype("datetime64[M]").astype("int64"))
        except Exception:
            pass

    d0 = np.min(np.asarray(order_dates).astype("datetime64[D]"))
    return int(d0.astype("datetime64[M]").astype("int64"))


def _month_noise(month_int: int, seed: int, sigma: float) -> float:
    """
    Deterministic per-month multiplicative noise using lognormal with mean adjusted
    so E[multiplier] = 1.

    Cached because uniq months repeat across chunks.
    """
    if sigma <= 0.0:
        return 1.0

    key = (int(seed), float(sigma), int(month_int))
    v = _MONTH_NOISE_CACHE.get(key)
    if v is not None:
        return float(v)

    s = (int(seed) ^ (int(month_int) * 1000003)) & 0xFFFFFFFF
    rng = np.random.default_rng(s)

    mu = -0.5 * (float(sigma) ** 2)
    v = float(rng.lognormal(mean=mu, sigma=float(sigma), size=1)[0])

    _MONTH_NOISE_CACHE[key] = v
    return v


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
def build_prices(rng, order_dates, qty, price):
    """
    Apply only mild time drift to the prices computed from Products,
    then snap to nice retail-looking price points.

    Expects `price` dict from compute_prices():
      final_unit_price, final_unit_cost, discount_amt, final_net_price

    qty is accepted for signature compatibility (not used here).
    """
    _ = qty  # intentionally unused

    annual_rate, month_sigma, clip_lo, clip_hi, scale_discount, vol_seed = _cfg()

    order_dates = np.asarray(order_dates)
    n = int(order_dates.shape[0])
    if n <= 0:
        return price

    # Month index per row
    order_month_i = order_dates.astype("datetime64[M]").astype("int64")
    uniq_months, inv = np.unique(order_month_i, return_inverse=True)

    start_m = _global_start_month_int(order_dates)
    months_since = (uniq_months.astype(np.int64) - int(start_m)).astype(np.float64)

    # Deterministic inflation curve
    infl = (1.0 + annual_rate) ** (months_since / 12.0)
    infl = np.where(np.isfinite(infl), infl, 1.0)

    # Deterministic per-month noise
    if month_sigma > 0.0:
        noises = np.fromiter(
            (_month_noise(int(m), vol_seed, month_sigma) for m in uniq_months),
            dtype=np.float64,
            count=uniq_months.size,
        )
    else:
        noises = np.ones_like(infl, dtype=np.float64)

    factor_u = np.clip(infl * noises, clip_lo, clip_hi)
    factor = factor_u[inv]

    up = _as_f64(price["final_unit_price"]) * factor
    uc = _as_f64(price["final_unit_cost"]) * factor

    disc = _as_f64(price["discount_amt"])
    if scale_discount:
        disc = disc * factor

    # Invariants pre-snap
    up = np.maximum(up, 0.0)
    uc = np.maximum(uc, 0.0)
    disc = np.maximum(disc, 0.0)

    disc = np.minimum(disc, up)
    net = np.clip(up - disc, 0.0, up)

    # keep cost <= net (avoid negative margin after scaling)
    bad = uc > net
    if bad.any():
        uc[bad] = net[bad]

    # ---------------------------------------------------------
    # SNAP / APPEARANCE
    # ---------------------------------------------------------
    appearance = _appearance_cfg()

    up = _snap_unit_price(rng, up, appearance)

    # Discount should be snapped after unit price snap
    disc = np.minimum(disc, up)
    disc = _snap_discount(disc, up, appearance)

    disc = np.minimum(disc, up)
    net = np.clip(up - disc, 0.0, up)

    uc = _snap_cost(rng, uc, appearance)
    uc = np.minimum(uc, net)

    # cents + post-round safety
    up = np.round(up, 2)
    uc = np.round(uc, 2)
    disc = np.round(np.minimum(np.maximum(disc, 0.0), up), 2)
    net = np.round(np.maximum(up - disc, 0.0), 2)
    uc = np.round(np.minimum(uc, net), 2)

    price["final_unit_price"] = up
    price["final_unit_cost"] = uc
    price["discount_amt"] = disc
    price["final_net_price"] = net
    return price


# =============================================================================
# quantity_model.py (combined)
# =============================================================================
import numpy as np
from src.facts.sales.sales_logic import State


# ---------------------------------------------------------------------
# Config cache (process-local)
# ---------------------------------------------------------------------
_QTY_CFG_CACHE_KEY = None
_QTY_CFG_CACHE_VAL = None


def _qty_default_cfg():
    """
    Safe defaults so quantity generation works even if models.quantity is missing.
    Defaults are intentionally conservative.
    """
    return {
        "base_poisson_lambda": 1.2,         # average base basket size ~2.2 after +1
        "monthly_factors": [1.0] * 12,      # neutral seasonality by default
        "month_inertia": 0.25,              # mild smoothing across months
        "noise_sd": 0.60,                   # additive row noise
        "min_qty": 1,
        "max_qty": 12,
        # Optional: allow heavier tails
        "tail_boost": {
            "enabled": False,
            "p": 0.03,
            "multiplier_min": 1.5,
            "multiplier_max": 3.0,
        },
    }


def _qty_merge_cfg(user_cfg: dict) -> dict:
    """
    Shallow merge with defaults, plus nested merge for tail_boost.
    """
    d = _qty_default_cfg()
    for k, v in user_cfg.items():
        d[k] = v

    if "tail_boost" in user_cfg and isinstance(user_cfg["tail_boost"], dict):
        tb = dict(d.get("tail_boost", {}))
        tb.update(user_cfg["tail_boost"])
        d["tail_boost"] = tb

    # Validate monthly factors
    mf = d.get("monthly_factors", None)
    if mf is None or len(mf) != 12:
        raise ValueError("models.quantity.monthly_factors must be a list of 12 floats")

    # Validate lambda
    lam = float(d.get("base_poisson_lambda", 0.0))
    if lam < 0:
        raise ValueError("models.quantity.base_poisson_lambda must be >= 0")

    # Validate clamps
    min_qty = int(d.get("min_qty", 1))
    max_qty = int(d.get("max_qty", 12))
    if max_qty < min_qty:
        min_qty, max_qty = max_qty, min_qty
    d["min_qty"] = min_qty
    d["max_qty"] = max_qty

    # Validate inertia, noise
    inertia = float(d.get("month_inertia", 0.0))
    d["month_inertia"] = float(np.clip(inertia, 0.0, 0.98))

    noise_sd = float(d.get("noise_sd", 0.0))
    d["noise_sd"] = float(max(0.0, noise_sd))

    # Tail boost validation (soft)
    tb = d.get("tail_boost", {}) or {}
    tb_enabled = bool(tb.get("enabled", False))
    if tb_enabled:
        p = float(tb.get("p", 0.03))
        tb["p"] = float(np.clip(p, 0.0, 0.50))

        mn = float(tb.get("multiplier_min", 1.5))
        mx = float(tb.get("multiplier_max", 3.0))
        if mx < mn:
            mn, mx = mx, mn
        tb["multiplier_min"] = float(max(1.0, mn))
        tb["multiplier_max"] = float(max(tb["multiplier_min"], mx))
        d["tail_boost"] = tb

    return d


def _qty_get_cfg():
    """
    Cached config access for hot-path calls from workers.
    """
    global _QTY_CFG_CACHE_KEY, _QTY_CFG_CACHE_VAL

    models = getattr(State, "models_cfg", None) or {}
    key = id(models)
    if _QTY_CFG_CACHE_KEY == key and _QTY_CFG_CACHE_VAL is not None:
        return _QTY_CFG_CACHE_VAL

    q = models.get("quantity", None)
    if q is None:
        cfg = _qty_default_cfg()
        # still normalize/validate defaults
        cfg = _qty_merge_cfg({})
    else:
        if not isinstance(q, dict):
            raise ValueError("models.quantity must be a mapping")
        cfg = _qty_merge_cfg(q)

    _QTY_CFG_CACHE_KEY = key
    _QTY_CFG_CACHE_VAL = cfg
    return cfg


def build_quantity(rng, order_dates):
    """
    Generate order line quantities with smooth month-to-month transitions.

    Month-level inertia is applied to the expected quantity so that
    basket sizes evolve gradually over time instead of jumping independently
    each month. Works with both single-month and multi-month batches.
    """
    cfg = _qty_get_cfg()
    n = int(len(order_dates))
    if n <= 0:
        return np.zeros(0, dtype=np.int64)

    lam = float(cfg["base_poisson_lambda"])

    # ------------------------------------------------------------
    # BASE QUANTITY (ROW-LEVEL)
    # ------------------------------------------------------------
    # poisson returns int; keep float for later noise/multipliers
    qty = rng.poisson(lam, n).astype(np.float64, copy=False) + 1.0

    # ------------------------------------------------------------
    # MONTH-LEVEL FACTOR (WITH INERTIA)
    # ------------------------------------------------------------
    order_months = np.asarray(order_dates).astype("datetime64[M]", copy=False)
    unique_months, inv = np.unique(order_months, return_inverse=True)

    monthly_factors = np.asarray(cfg["monthly_factors"], dtype=np.float64)
    monthly_factors = np.where(np.isfinite(monthly_factors), monthly_factors, 1.0)

    inertia = float(cfg["month_inertia"])

    # month-of-year for each unique month (vectorized)
    month_num = (unique_months.astype("int64") % 12).astype(np.int64, copy=False)
    raw = monthly_factors[month_num]

    if unique_months.size == 1 or inertia <= 0.0:
        smoothed = raw
    else:
        smoothed = np.empty_like(raw, dtype=np.float64)
        prev = float(raw[0])
        smoothed[0] = prev
        # tight loop over unique months (U is small; keep loop but no heavy work inside)
        for i in range(1, raw.size):
            prev = inertia * prev + (1.0 - inertia) * float(raw[i])
            smoothed[i] = prev

    qty *= smoothed[inv]

    # ------------------------------------------------------------
    # OPTIONAL HEAVY TAIL BOOST
    # ------------------------------------------------------------
    tb = cfg.get("tail_boost", {}) or {}
    if bool(tb.get("enabled", False)):
        p = float(tb.get("p", 0.03))
        if p > 0.0:
            mask = rng.random(n) < p
            if mask.any():
                mn = float(tb.get("multiplier_min", 1.5))
                mx = float(tb.get("multiplier_max", 3.0))
                if mx < mn:
                    mn, mx = mx, mn
                qty[mask] *= rng.uniform(mn, mx, size=int(mask.sum()))

    # ------------------------------------------------------------
    # ADDITIVE NOISE (ROW-LEVEL)
    # ------------------------------------------------------------
    noise_sd = float(cfg["noise_sd"])
    if noise_sd > 0.0:
        qty = rng.normal(loc=qty, scale=noise_sd)
        qty = np.where(np.isfinite(qty), qty, 1.0)

    # ------------------------------------------------------------
    # FINALIZE: integer + clamp
    # ------------------------------------------------------------
    qty = np.rint(qty).astype(np.int64, copy=False)
    return np.clip(qty, int(cfg["min_qty"]), int(cfg["max_qty"]))
