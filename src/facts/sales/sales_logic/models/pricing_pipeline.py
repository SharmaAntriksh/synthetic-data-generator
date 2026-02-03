import numpy as np
from src.facts.sales.sales_logic.globals import State


def _default_cfg():
    """
    Safe defaults so pipeline can run even if models.pricing block is missing.
    These defaults are mild to avoid extreme values.
    """
    return {
        "month_inertia": 0.0,
        "apply_base_factor_to_discount": True,  # keep legacy behavior by default :contentReference[oaicite:1]{index=1}
        "inflation": {
            "annual_rate": 0.03,
            "noise_sigma": 0.02,
        },
        "seasonality": {
            "enabled": True,
            "amplitude": 0.06,
            "sharpness": 1.6,
            "phase_shift_months": 0,
        },
        "ramp": {
            "months": 18,
            "min_multiplier": 0.85,
            "max_multiplier": 1.05,
            "discount_scale": 1.0,
            "discount_noise_sigma": 0.15,
        },
        "discount": {
            "max_pct_of_price": 0.85,
        },
        "floors": {
            "min_net_price": 0.10,
        },
        "cost": {
            "cost_ratio_min": 0.55,
            "cost_ratio_max": 0.85,
            "min_margin_pct": 0.05,
        },
    }


def _get_cfg():
    models = getattr(State, "models_cfg", None) or {}
    cfg = models.get("pricing", None)
    if cfg is None:
        return _default_cfg()

    # Shallow merge with defaults so missing keys don't crash
    d = _default_cfg()
    for k, v in cfg.items():
        d[k] = v
    # merge nested dicts if present
    for k in ("inflation", "seasonality", "ramp", "discount", "floors", "cost"):
        if k in cfg and isinstance(cfg[k], dict):
            dd = d.get(k, {}).copy()
            dd.update(cfg[k])
            d[k] = dd
    return d


def build_prices(rng, order_dates, qty, price):
    """
    Applies:
      - inflation (macro trend)
      - mild seasonality
      - month-level inertia smoothing over the *base* price signal
      - early ramp (business maturity)
      - discount noise
      - cost anchoring with margin safety
      - rounding and invariants

    Compatible with month-sliced generation: works with single-month batches.
    """
    cfg = _get_cfg()

    n = len(order_dates)
    if n == 0:
        return price

    # ------------------------------------------------------------
    # MONTH INDEXING
    # ------------------------------------------------------------
    order_months = order_dates.astype("datetime64[M]")
    unique_months, inv = np.unique(order_months, return_inverse=True)

    month_inertia = float(cfg.get("month_inertia", 0.0))
    month_inertia = float(np.clip(month_inertia, 0.0, 0.98))

    infl_cfg = cfg["inflation"]
    seas_cfg = cfg["seasonality"]

    # base_year index for inflation
    base_year = order_dates.astype("datetime64[Y]").min().astype("int64")
    month_year_idx = unique_months.astype("datetime64[Y]").astype("int64") - base_year

    month_numbers = (unique_months.astype("int64") % 12).astype("int64")

    prev_price_factor = None
    month_price_factor = np.empty(len(unique_months), dtype=np.float64)

    # ------------------------------------------------------------
    # MONTH-LEVEL BASE PRICE FACTOR (INFLATION + SEASONALITY + INERTIA)
    # ------------------------------------------------------------
    annual_rate = float(infl_cfg.get("annual_rate", 0.03))
    noise_sigma = float(infl_cfg.get("noise_sigma", 0.02))

    seas_enabled = bool(seas_cfg.get("enabled", True))
    seas_amp = float(seas_cfg.get("amplitude", 0.06))
    seas_sharp = float(seas_cfg.get("sharpness", 1.6))
    seas_phase = int(seas_cfg.get("phase_shift_months", 0))

    for i, m in enumerate(month_numbers):
        # inflation
        inflation = (1.0 + annual_rate) ** float(month_year_idx[i])
        if noise_sigma > 0:
            inflation *= rng.lognormal(mean=0.0, sigma=noise_sigma)

        # seasonality
        if seas_enabled:
            angle = 2 * np.pi * (float(m + seas_phase) / 12.0)
            seasonal = 1.0 + seas_amp * np.tanh(seas_sharp * np.sin(angle))
        else:
            seasonal = 1.0

        raw_factor = float(inflation * seasonal)

        # inertia smoothing
        if prev_price_factor is None or month_inertia <= 0.0:
            factor = raw_factor
        else:
            factor = month_inertia * prev_price_factor + (1.0 - month_inertia) * raw_factor

        prev_price_factor = factor
        month_price_factor[i] = factor

    base_price_factor = month_price_factor[inv]

    # ------------------------------------------------------------
    # APPLY BASE PRICE FACTOR
    # ------------------------------------------------------------
    # NOTE: legacy behavior multiplied discount_amt too :contentReference[oaicite:2]{index=2}
    apply_discount = bool(cfg.get("apply_base_factor_to_discount", True))

    price["final_unit_price"] *= base_price_factor
    price["final_unit_cost"] *= base_price_factor
    if apply_discount:
        price["discount_amt"] *= base_price_factor

    # ------------------------------------------------------------
    # EARLY RAMP (BUSINESS MATURITY)
    # ------------------------------------------------------------
    ramp_cfg = cfg["ramp"]
    ramp_months = max(1.0, float(ramp_cfg.get("months", 18)))

    months_since_start = (order_months.astype("int64") - order_months.min().astype("int64")).astype("float64")

    ramp = np.clip(
        months_since_start / ramp_months,
        float(ramp_cfg.get("min_multiplier", 0.85)),
        float(ramp_cfg.get("max_multiplier", 1.05)),
    )

    price["final_unit_price"] *= ramp
    price["final_unit_cost"] *= ramp

    discount_scale = float(ramp_cfg.get("discount_scale", 1.0))
    disc_noise_sigma = float(ramp_cfg.get("discount_noise_sigma", 0.15))

    if disc_noise_sigma > 0:
        disc_noise = rng.lognormal(mean=0.0, sigma=disc_noise_sigma, size=n)
    else:
        disc_noise = 1.0

    price["discount_amt"] *= (ramp * discount_scale * disc_noise)

    # Guard: discount cannot exceed price
    max_discount_pct = float(cfg["discount"].get("max_pct_of_price", 0.85))
    max_discount_pct = float(np.clip(max_discount_pct, 0.0, 1.0))

    price["discount_amt"] = np.clip(
        price["discount_amt"],
        0.0,
        price["final_unit_price"] * max_discount_pct,
    )

    price["final_net_price"] = price["final_unit_price"] - price["discount_amt"]

    # ------------------------------------------------------------
    # ABSOLUTE PRICE FLOOR
    # ------------------------------------------------------------
    min_net = float(cfg["floors"].get("min_net_price", 0.10))
    price["final_net_price"] = np.maximum(price["final_net_price"], min_net)

    # ------------------------------------------------------------
    # FINAL COST ANCHOR + MARGIN FLOOR (NON-CIRCULAR)
    # ------------------------------------------------------------
    cost_cfg = cfg["cost"]

    cost_ratio_min = float(cost_cfg.get("cost_ratio_min", 0.55))
    cost_ratio_max = float(cost_cfg.get("cost_ratio_max", 0.85))
    if cost_ratio_max < cost_ratio_min:
        cost_ratio_min, cost_ratio_max = cost_ratio_max, cost_ratio_min

    min_margin_pct = float(cost_cfg.get("min_margin_pct", 0.05))
    min_margin_pct = float(np.clip(min_margin_pct, 0.0, 0.95))

    cost_ratio = rng.uniform(cost_ratio_min, cost_ratio_max, size=n)

    price["final_unit_cost"] = price["final_net_price"] * cost_ratio

    max_allowed_cost = price["final_net_price"] * (1.0 - min_margin_pct)
    price["final_unit_cost"] = np.minimum(price["final_unit_cost"], max_allowed_cost)

    # ------------------------------------------------------------
    # ROUNDING + CONSISTENCY
    # ------------------------------------------------------------
    price["final_unit_price"] = np.round(price["final_unit_price"], 2)
    price["discount_amt"] = np.round(price["discount_amt"], 2)
    price["final_net_price"] = np.round(price["final_unit_price"] - price["discount_amt"], 2)
    price["final_unit_cost"] = np.round(price["final_unit_cost"], 2)

    # Invariant: cost <= net
    # Avoid hard assert in production runs; use a guard clamp.
    bad = price["final_unit_cost"] > price["final_net_price"]
    if np.any(bad):
        price["final_unit_cost"][bad] = price["final_net_price"][bad]

    return price
