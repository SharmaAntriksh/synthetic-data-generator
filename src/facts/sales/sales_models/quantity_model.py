"""
Quantity / basket-size model.

Generates per-order item counts using Poisson + monthly seasonality +
multiplicative noise.

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
    "base_poisson_lambda": 1.7,
    "monthly_factors": [0.99, 0.98, 1.00, 1.00, 1.01, 1.02,
                        1.02, 1.01, 1.00, 1.03, 1.06, 1.05],
    "noise_sigma": 0.12,
    "min_qty": 1,
    "max_qty": 8,
}


# ---------------------------------------------------------------
# Config loading (process-local cache)
# ---------------------------------------------------------------
_CFG_VERSION: int = -1
_CFG_CACHE: dict | None = None


def _cfg_hash(models: dict) -> int:
    """Content-based hash of the quantity-relevant config subset."""
    raw = models.get("quantity", {}) or {}
    items = []
    for k, v in sorted(raw.items()):
        if isinstance(v, (list, tuple)):
            items.append((k, tuple(v)))
        else:
            items.append((k, v))
    return hash(tuple(items))


def _load_cfg() -> dict:
    """
    Load, validate, and cache the quantity config.

    Supports both the current simplified keys and legacy nested keys
    for backward compatibility.
    """
    global _CFG_VERSION, _CFG_CACHE

    models = getattr(State, "models_cfg", None) or {}
    version = _cfg_hash(models)
    if version == _CFG_VERSION and _CFG_CACHE is not None:
        return _CFG_CACHE

    raw = models.get("quantity", {}) or {}
    if not isinstance(raw, dict):
        raise ValueError("models.quantity must be a mapping")

    # --- scalar parameters ---
    lam = float(raw.get("base_poisson_lambda", _DEFAULTS["base_poisson_lambda"]))
    if lam < 0:
        raise ValueError("models.quantity.base_poisson_lambda must be >= 0")

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
    factors_arr = np.clip(factors_arr, 0.01, None)  # Prevent zero-factor months

    out = {
        "base_poisson_lambda": lam,
        "monthly_factors": factors_arr,
        "noise_sigma": noise,
        "min_qty": min_qty,
        "max_qty": max_qty,
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
      2. Multiply by monthly seasonal factors
      3. Add multiplicative noise
      4. Round and clamp to [min_qty, max_qty]

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
        return np.zeros(0, dtype=np.int32)

    # 1. Base Poisson draw (+1 ensures minimum of 1)
    qty = rng.poisson(cfg["base_poisson_lambda"], n).astype(np.float64) + 1.0

    # 2. Monthly seasonal factors (direct lookup, no smoothing)
    order_months = np.asarray(order_dates).astype("datetime64[M]", copy=False)
    unique_months, inv = np.unique(order_months, return_inverse=True)
    month_of_year = (unique_months.astype("int64") % 12).astype(np.int64)

    qty *= cfg["monthly_factors"][month_of_year][inv]

    # 3. Multiplicative noise (lognormal to avoid negative values)
    sigma = cfg["noise_sigma"]
    if sigma > 0.0:
        qty *= rng.lognormal(mean=0.0, sigma=sigma, size=n)

    # 4. Round and clamp
    qty = np.rint(qty).astype(np.int32)
    return np.clip(qty, cfg["min_qty"], cfg["max_qty"])
