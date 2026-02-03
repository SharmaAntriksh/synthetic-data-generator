import numpy as np
from src.facts.sales.sales_logic.globals import State


def _default_cfg():
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


def _get_cfg():
    models = getattr(State, "models_cfg", None) or {}
    q = models.get("quantity", None)
    if q is None:
        return _default_cfg()

    # Shallow merge with defaults
    d = _default_cfg()
    for k, v in q.items():
        d[k] = v

    # Merge nested tail_boost if present
    if "tail_boost" in q and isinstance(q["tail_boost"], dict):
        tb = d.get("tail_boost", {}).copy()
        tb.update(q["tail_boost"])
        d["tail_boost"] = tb

    # Validate monthly factors
    mf = d.get("monthly_factors", None)
    if mf is None or len(mf) != 12:
        raise ValueError("models.quantity.monthly_factors must be a list of 12 floats")

    return d


def build_quantity(rng, order_dates):
    """
    Generate order line quantities with smooth month-to-month transitions.

    Month-level inertia is applied to the expected quantity so that
    basket sizes evolve gradually over time instead of jumping independently
    each month. Works with both single-month and multi-month batches.
    """
    cfg = _get_cfg()
    n = len(order_dates)
    if n == 0:
        return np.zeros(0, dtype=np.int64)

    lam = float(cfg["base_poisson_lambda"])
    if lam < 0:
        raise ValueError("models.quantity.base_poisson_lambda must be >= 0")

    # ------------------------------------------------------------
    # BASE QUANTITY (ROW-LEVEL)
    # ------------------------------------------------------------
    base_qty = rng.poisson(lam, n).astype(np.float64) + 1.0

    # ------------------------------------------------------------
    # MONTH-LEVEL FACTOR (WITH INERTIA)
    # ------------------------------------------------------------
    order_months = order_dates.astype("datetime64[M]")
    unique_months, inv = np.unique(order_months, return_inverse=True)

    monthly_factors = np.array(cfg["monthly_factors"], dtype=np.float64)

    inertia = float(cfg.get("month_inertia", 0.0))
    inertia = float(np.clip(inertia, 0.0, 0.98))

    smoothed_factor_by_month = np.empty(len(unique_months), dtype=np.float64)
    prev = None

    for i in range(len(unique_months)):
        # month-of-year: 0..11
        month_num = int(unique_months[i].astype("int64") % 12)
        raw = float(monthly_factors[month_num])

        # smooth over months if multi-month; if single-month, this is just raw
        if prev is None or inertia <= 0.0:
            f = raw
        else:
            f = inertia * prev + (1.0 - inertia) * raw

        prev = f
        smoothed_factor_by_month[i] = f

    qty = base_qty * smoothed_factor_by_month[inv]

    # ------------------------------------------------------------
    # OPTIONAL HEAVY TAIL BOOST (future-proof; off by default)
    # ------------------------------------------------------------
    tb = cfg.get("tail_boost", {}) or {}
    if bool(tb.get("enabled", False)):
        p = float(tb.get("p", 0.03))
        p = float(np.clip(p, 0.0, 0.50))
        mask = rng.random(n) < p
        if mask.any():
            mult_min = float(tb.get("multiplier_min", 1.5))
            mult_max = float(tb.get("multiplier_max", 3.0))
            if mult_max < mult_min:
                mult_min, mult_max = mult_max, mult_min
            qty[mask] *= rng.uniform(mult_min, mult_max, size=mask.sum())

    # ------------------------------------------------------------
    # ADDITIVE NOISE (ROW-LEVEL VARIATION)
    # ------------------------------------------------------------
    noise_sd = float(cfg["noise_sd"])
    if noise_sd > 0:
        qty = rng.normal(qty, noise_sd)

    # ------------------------------------------------------------
    # FINALIZE: integer + clamp
    # ------------------------------------------------------------
    qty = np.rint(qty).astype(np.int64)

    min_qty = int(cfg["min_qty"])
    max_qty = int(cfg["max_qty"])
    if max_qty < min_qty:
        min_qty, max_qty = max_qty, min_qty

    return np.clip(qty, min_qty, max_qty)
