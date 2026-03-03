"""
Quantity / basket-size model.

Generates per-order item counts using Poisson + monthly seasonality +
optional tail boost for rare large orders.

Config source: models.yaml -> models.quantity
Runtime state:  State.models_cfg  (the inner "models" dict)
"""
from __future__ import annotations

import numpy as np

from src.facts.sales.sales_logic import State


# ---------------------------------------------------------------
# Defaults (aligned exactly with models.yaml)
# ---------------------------------------------------------------

_DEFAULTS = {
    "base_poisson_lambda": 1.8,
    "monthly_factors": [0.99, 0.98, 1.00, 1.00, 1.01, 1.02,
                        1.02, 1.01, 1.00, 1.03, 1.06, 1.05],
    "month_inertia": 0.82,
    "noise_sigma": 0.12,
    "min_qty": 1,
    "max_qty": 4,
    "tail_boost": {
        "enabled": False,
        "probability": 0.03,
        "multiplier_range": [1.5, 3.0],
    },
}


# ---------------------------------------------------------------
# Config loading (process-local cache)
# ---------------------------------------------------------------
_CFG_VERSION: int = -1
_CFG_CACHE: dict | None = None


def _resolve_range(source: dict | None, default_lo: float, default_hi: float):
    """
    Extract (lo, hi) multiplier bounds from a config dict.

    Supports two formats:
      - multiplier_range: [lo, hi]          (preferred)
      - multiplier_min / multiplier_max     (legacy flat keys)

    Returns a sorted, clamped pair with lo >= 1.0.
    """
    if source is None:
        return default_lo, default_hi

    rng = source.get("multiplier_range")
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        lo, hi = float(rng[0]), float(rng[1])
    else:
        lo = float(source.get("multiplier_min", default_lo))
        hi = float(source.get("multiplier_max", default_hi))

    if hi < lo:
        lo, hi = hi, lo
    return max(1.0, lo), max(lo, hi)


def _load_cfg() -> dict:
    """
    Load, validate, and cache the quantity config.

    Supports both the current simplified keys and legacy nested keys
    for backward compatibility.
    """
    global _CFG_VERSION, _CFG_CACHE

    models = getattr(State, "models_cfg", None) or {}
    version = id(models)
    if version == _CFG_VERSION and _CFG_CACHE is not None:
        return _CFG_CACHE

    raw = models.get("quantity", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError("models.quantity must be a mapping")

    # --- scalar parameters ---
    lam = float(raw.get("base_poisson_lambda", _DEFAULTS["base_poisson_lambda"]))
    if lam < 0:
        raise ValueError("models.quantity.base_poisson_lambda must be >= 0")

    inertia = float(np.clip(
        raw.get("month_inertia", _DEFAULTS["month_inertia"]),
        0.0, 0.98,
    ))

    # Support both "noise_sigma" (current) and "noise_sd" (legacy)
    noise = max(0.0, float(raw.get("noise_sigma", raw.get("noise_sd", _DEFAULTS["noise_sigma"]))))

    min_qty = int(raw.get("min_qty", _DEFAULTS["min_qty"]))
    max_qty = int(raw.get("max_qty", _DEFAULTS["max_qty"]))
    if max_qty < min_qty:
        min_qty, max_qty = max_qty, min_qty

    # --- monthly factors (must be 12 floats) ---
    factors = raw.get("monthly_factors")
    if factors is None:
        factors = list(_DEFAULTS["monthly_factors"])
    if len(factors) != 12:
        raise ValueError("models.quantity.monthly_factors must be a list of 12 floats")
    factors_arr = np.asarray(factors, dtype=np.float64)
    factors_arr = np.where(np.isfinite(factors_arr), factors_arr, 1.0)

    # --- tail boost ---
    tb_defaults = _DEFAULTS["tail_boost"]
    user_tb = raw.get("tail_boost")
    if not isinstance(user_tb, dict):
        user_tb = None

    tb_enabled = bool((user_tb or tb_defaults).get("enabled", False))

    # Probability: support "probability" and legacy "p" alias
    tb_prob_raw = None
    if user_tb is not None:
        tb_prob_raw = user_tb.get("probability", user_tb.get("p"))
    if tb_prob_raw is None:
        tb_prob_raw = tb_defaults.get("probability", 0.03)
    tb_prob = float(np.clip(float(tb_prob_raw), 0.0, 0.50))

    # Multiplier range: prefer user overrides, fall back to defaults
    default_range = tb_defaults.get("multiplier_range", [1.5, 3.0])
    tb_min, tb_max = _resolve_range(user_tb, default_range[0], default_range[1])

    out = {
        "base_poisson_lambda": lam,
        "monthly_factors": factors_arr,
        "month_inertia": inertia,
        "noise_sigma": noise,
        "min_qty": min_qty,
        "max_qty": max_qty,
        "tb_enabled": tb_enabled,
        "tb_prob": tb_prob,
        "tb_min": tb_min,
        "tb_max": tb_max,
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
# Public API
# ---------------------------------------------------------------

def build_quantity(rng, order_dates):
    """
    Generate per-order item quantities.

    Pipeline:
      1. Draw from Poisson(λ) + 1  (minimum 1 item)
      2. Multiply by smoothed monthly seasonal factors
      3. Optionally boost a small fraction of rows (tail boost)
      4. Add Gaussian noise
      5. Round and clamp to [min_qty, max_qty]

    Parameters
    ----------
    rng : numpy.random.Generator
    order_dates : array-like of datetime64

    Returns
    -------
    np.ndarray[int64] of shape (n,)
    """
    cfg = _load_cfg()
    n = int(len(order_dates))
    if n <= 0:
        return np.zeros(0, dtype=np.int64)

    # 1. Base Poisson draw (+1 ensures minimum of 1)
    qty = rng.poisson(cfg["base_poisson_lambda"], n).astype(np.float64) + 1.0

    # 2. Monthly seasonal factors with inertia smoothing
    order_months = np.asarray(order_dates).astype("datetime64[M]", copy=False)
    unique_months, inv = np.unique(order_months, return_inverse=True)
    month_of_year = (unique_months.astype("int64") % 12).astype(np.int64)

    raw_factors = cfg["monthly_factors"][month_of_year]
    inertia = cfg["month_inertia"]

    if unique_months.size <= 1 or inertia <= 0.0:
        smoothed = raw_factors
    else:
        smoothed = np.empty_like(raw_factors, dtype=np.float64)
        prev = float(raw_factors[0])
        smoothed[0] = prev
        for i in range(1, raw_factors.size):
            prev = inertia * prev + (1.0 - inertia) * float(raw_factors[i])
            smoothed[i] = prev

    qty *= smoothed[inv]

    # 3. Tail boost (rare large orders)
    if cfg["tb_enabled"] and cfg["tb_prob"] > 0.0:
        mask = rng.random(n) < cfg["tb_prob"]
        count = int(mask.sum())
        if count > 0:
            qty[mask] *= rng.uniform(cfg["tb_min"], cfg["tb_max"], size=count)

    # 4. Gaussian noise
    # Use lognormal-style multiplicative noise to avoid negative values.
    # This prevents the truncation bias that additive Gaussian noise causes
    # when qty values are small relative to sigma.
    sigma = cfg["noise_sigma"]
    if sigma > 0.0:
        qty *= rng.lognormal(mean=0.0, sigma=sigma, size=n)

    # 5. Round and clamp
    qty = np.rint(qty).astype(np.int64)
    return np.clip(qty, cfg["min_qty"], cfg["max_qty"])
