from __future__ import annotations

import numpy as np

from src.facts.sales.sales_logic import State


_QTY_CFG_CACHE_KEY = None
_QTY_CFG_CACHE_VAL = None


def _qty_default_cfg():
    return {
        "base_poisson_lambda": 1.2,
        "monthly_factors": [1.0] * 12,
        "month_inertia": 0.25,
        "noise_sd": 0.60,
        "min_qty": 1,
        "max_qty": 12,
        "tail_boost": {
            "enabled": False,
            "p": 0.03,
            "multiplier_min": 1.5,
            "multiplier_max": 3.0,
        },
    }


def _qty_merge_cfg(user_cfg: dict) -> dict:
    d = _qty_default_cfg()
    for k, v in user_cfg.items():
        d[k] = v

    if "tail_boost" in user_cfg and isinstance(user_cfg["tail_boost"], dict):
        tb = dict(d.get("tail_boost", {}))
        tb.update(user_cfg["tail_boost"])
        d["tail_boost"] = tb

    mf = d.get("monthly_factors", None)
    if mf is None or len(mf) != 12:
        raise ValueError("models.quantity.monthly_factors must be a list of 12 floats")

    lam = float(d.get("base_poisson_lambda", 0.0))
    if lam < 0:
        raise ValueError("models.quantity.base_poisson_lambda must be >= 0")

    min_qty = int(d.get("min_qty", 1))
    max_qty = int(d.get("max_qty", 12))
    if max_qty < min_qty:
        min_qty, max_qty = max_qty, min_qty
    d["min_qty"] = min_qty
    d["max_qty"] = max_qty

    inertia = float(d.get("month_inertia", 0.0))
    d["month_inertia"] = float(np.clip(inertia, 0.0, 0.98))

    noise_sd = float(d.get("noise_sd", 0.0))
    d["noise_sd"] = float(max(0.0, noise_sd))

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
    global _QTY_CFG_CACHE_KEY, _QTY_CFG_CACHE_VAL

    models = getattr(State, "models_cfg", None) or {}
    key = id(models)
    if _QTY_CFG_CACHE_KEY == key and _QTY_CFG_CACHE_VAL is not None:
        return _QTY_CFG_CACHE_VAL

    q = models.get("quantity", None)
    if q is None:
        cfg = _qty_default_cfg()
        cfg = _qty_merge_cfg({})
    else:
        if not isinstance(q, dict):
            raise ValueError("models.quantity must be a mapping")
        cfg = _qty_merge_cfg(q)

    _QTY_CFG_CACHE_KEY = key
    _QTY_CFG_CACHE_VAL = cfg
    return cfg


def build_quantity(rng, order_dates):
    cfg = _qty_get_cfg()
    n = int(len(order_dates))
    if n <= 0:
        return np.zeros(0, dtype=np.int64)

    lam = float(cfg["base_poisson_lambda"])

    qty = rng.poisson(lam, n).astype(np.float64, copy=False) + 1.0

    order_months = np.asarray(order_dates).astype("datetime64[M]", copy=False)
    unique_months, inv = np.unique(order_months, return_inverse=True)

    monthly_factors = np.asarray(cfg["monthly_factors"], dtype=np.float64)
    monthly_factors = np.where(np.isfinite(monthly_factors), monthly_factors, 1.0)

    inertia = float(cfg["month_inertia"])

    month_num = (unique_months.astype("int64") % 12).astype(np.int64, copy=False)
    raw = monthly_factors[month_num]

    if unique_months.size == 1 or inertia <= 0.0:
        smoothed = raw
    else:
        smoothed = np.empty_like(raw, dtype=np.float64)
        prev = float(raw[0])
        smoothed[0] = prev
        for i in range(1, raw.size):
            prev = inertia * prev + (1.0 - inertia) * float(raw[i])
            smoothed[i] = prev

    qty *= smoothed[inv]

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

    noise_sd = float(cfg["noise_sd"])
    if noise_sd > 0.0:
        qty = rng.normal(loc=qty, scale=noise_sd)
        qty = np.where(np.isfinite(qty), qty, 1.0)

    qty = np.rint(qty).astype(np.int64, copy=False)
    return np.clip(qty, int(cfg["min_qty"]), int(cfg["max_qty"]))