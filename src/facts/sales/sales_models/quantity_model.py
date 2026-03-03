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


def _resolve_multiplier_range(user_tb: dict | None, merged: dict) -> tuple[float, float]:
    """
    Resolve tail-boost multiplier bounds from user overrides and merged defaults.

    Precedence:
      1. user_tb["multiplier_range"]  (list/tuple of 2)
      2. user_tb["multiplier_min"] / user_tb["multiplier_max"]
      3. merged["multiplier_range"]
      4. merged["multiplier_min"] / merged["multiplier_max"]
    """
    if user_tb is not None and isinstance(user_tb, dict):
        user_range = user_tb.get("multiplier_range", None)
        if isinstance(user_range, (list, tuple)) and len(user_range) == 2:
            return float(user_range[0]), float(user_range[1])
        if "multiplier_min" in user_tb or "multiplier_max" in user_tb:
            return (
                float(user_tb.get("multiplier_min", merged.get("multiplier_min", 1.5))),
                float(user_tb.get("multiplier_max", merged.get("multiplier_max", 3.0))),
            )

    merged_range = merged.get("multiplier_range", None)
    if isinstance(merged_range, (list, tuple)) and len(merged_range) == 2:
        return float(merged_range[0]), float(merged_range[1])

    return (
        float(merged.get("multiplier_min", 1.5)),
        float(merged.get("multiplier_max", 3.0)),
    )


def _load_cfg() -> dict:
    """
    Load, validate, and cache the quantity config.

    Supports both the new simplified keys and legacy nested keys
    for backward compatibility.
    """
    global _CFG_VERSION, _CFG_CACHE

    # Use the object identity of models_cfg itself as the cache key.
    # Avoid `or {}` here: that would create a new dict on every call when
    # models_cfg is None/empty, making id() unstable and the cache useless.
    models_cfg = getattr(State, "models_cfg", None)
    version = id(models_cfg)
    if version == _CFG_VERSION and _CFG_CACHE is not None:
        return _CFG_CACHE

    models = models_cfg or {}
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

    # Support both "noise_sigma" (new) and "noise_sd" (legacy)
    noise = float(raw.get("noise_sigma", raw.get("noise_sd", _DEFAULTS["noise_sigma"])))
    noise = max(0.0, noise)

    min_qty = int(raw.get("min_qty", _DEFAULTS["min_qty"]))
    max_qty = int(raw.get("max_qty", _DEFAULTS["max_qty"]))
    if max_qty < min_qty:
        min_qty, max_qty = max_qty, min_qty

    # --- monthly factors (must be 12 floats) ---
    factors = raw.get("monthly_factors", None)
    if factors is None:
        factors = list(_DEFAULTS["monthly_factors"])
    if not isinstance(factors, (list, tuple, np.ndarray)):
        raise ValueError(
            "models.quantity.monthly_factors must be a list of 12 floats, "
            f"got {type(factors).__name__}"
        )
    if len(factors) != 12:
        raise ValueError("models.quantity.monthly_factors must be a list of 12 floats")
    factors_arr = np.asarray(factors, dtype=np.float64)
    factors_arr = np.where(np.isfinite(factors_arr), factors_arr, 1.0)

    # --- tail boost (deep-merge defaults with user overrides) ---
    user_tb = raw.get("tail_boost", None)
    tb_defaults = _DEFAULTS["tail_boost"]
    tb = dict(tb_defaults)
    if isinstance(user_tb, dict):
        tb.update(user_tb)

    tb_enabled = bool(tb.get("enabled", False))

    # Resolve probability: prefer user-supplied key (either name) over merged default.
    if isinstance(user_tb, dict):
        raw_prob = user_tb.get("probability", user_tb.get("p", None))
    else:
        raw_prob = None
    if raw_prob is None:
        raw_prob = tb.get("probability", tb.get("p", 0.03))
    tb_prob = float(np.clip(float(raw_prob), 0.0, 0.50))

    tb_min, tb_max = _resolve_multiplier_range(user_tb if isinstance(user_tb, dict) else None, tb)
    if tb_max < tb_min:
        tb_min, tb_max = tb_max, tb_min
    tb_min = max(1.0, tb_min)
    tb_max = max(tb_min, tb_max)

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
      4. Apply multiplicative lognormal noise (mean-corrected to be unbiased)
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
    n = len(order_dates)
    if n <= 0:
        return np.zeros(0, dtype=np.int64)

    # 1. Base Poisson draw (+1 ensures minimum of 1)
    qty = rng.poisson(cfg["base_poisson_lambda"], n).astype(np.float64) + 1.0

    # 2. Monthly seasonal factors with inertia smoothing
    order_months = np.asarray(order_dates, copy=False).astype("datetime64[M]", copy=False)
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

    # 4. Multiplicative lognormal noise.
    # mean = -σ²/2 centres the expected multiplier at exactly 1.0, avoiding
    # the upward bias that mean=0.0 would introduce (E[lognormal(0,σ)] = e^(σ²/2)).
    sigma = cfg["noise_sigma"]
    if sigma > 0.0:
        noise = rng.lognormal(mean=-0.5 * sigma * sigma, sigma=sigma, size=n)
        qty *= noise

    # 5. Round and clamp
    qty = np.rint(qty).astype(np.int64)
    return np.clip(qty, cfg["min_qty"], cfg["max_qty"])
