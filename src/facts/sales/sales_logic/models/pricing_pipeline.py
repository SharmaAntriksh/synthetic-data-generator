import numpy as np
from src.facts.sales.sales_logic.globals import State


def _default_cfg():
    """
    Safe defaults so the pipeline can run even if models.pricing block is missing.
    Defaults are intentionally mild to avoid extreme values.
    """
    return {
        "month_inertia": 0.0,
        # Legacy behavior: historically we scaled discount_amt by macro factor too.
        "apply_base_factor_to_discount": True,
        "inflation": {
            "annual_rate": 0.03,     # compounded
            "noise_sigma": 0.02,     # lognormal sigma applied per-month factor
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
            # NOTE: cost_ratio_* are legacy and intentionally UNUSED here.
            # Cost is anchored to Products.UnitCost (from compute_prices input),
            # then transformed by macro factors and ramp.
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

    # Merge nested dicts if present
    for k in ("inflation", "seasonality", "ramp", "discount", "floors", "cost"):
        if k in cfg and isinstance(cfg[k], dict):
            dd = d.get(k, {}).copy()
            dd.update(cfg[k])
            d[k] = dd

    return d


def _get_global_start_month_int(order_dates: np.ndarray) -> int:
    """
    Returns the global start month index (datetime64[M] as int64) used to anchor
    inflation/ramp in month-sliced generation.

    Preference order:
      1) State.date_pool min (authoritative timeline)
      2) State.start_date / State.sales_start_date (if present)
      3) min(order_dates) (fallback)
    """
    # 1) date_pool (best)
    dp = getattr(State, "date_pool", None)
    if dp is not None:
        try:
            if len(dp) > 0:
                d0 = np.min(dp.astype("datetime64[D]"))
                return d0.astype("datetime64[M]").astype("int64")
        except Exception:
            pass

    # 2) known start dates (optional)
    for attr in ("sales_start_date", "start_date"):
        d = getattr(State, attr, None)
        if d is not None:
            try:
                return np.asarray(d, dtype="datetime64[D]").astype("datetime64[M]").astype("int64")
            except Exception:
                pass

    # 3) fallback: min of provided batch
    d0 = np.min(order_dates.astype("datetime64[D]"))
    return d0.astype("datetime64[M]").astype("int64")


def _enforce_price_invariants(price: dict, min_net_price: float) -> None:
    """
    Enforce invariant relationships:
      - 0 <= unit_cost <= unit_price
      - 0 <= net_price <= unit_price
      - discount = unit_price - net_price and discount >= 0
      - net_price >= min_net_price BUT never above unit_price
    """
    up = np.asarray(price["final_unit_price"], dtype=np.float64)
    uc = np.asarray(price["final_unit_cost"], dtype=np.float64)
    disc = np.asarray(price["discount_amt"], dtype=np.float64)

    # Non-finite -> safe
    up = np.where(np.isfinite(up), up, 0.0)
    uc = np.where(np.isfinite(uc), uc, 0.0)
    disc = np.where(np.isfinite(disc), disc, 0.0)

    # Non-negative
    up = np.maximum(up, 0.0)
    uc = np.maximum(uc, 0.0)
    disc = np.maximum(disc, 0.0)

    # Cost cannot exceed unit price
    uc = np.minimum(uc, up)

    # Net derived and bounded
    net = up - disc
    net = np.where(np.isfinite(net), net, 0.0)
    net = np.minimum(up, np.maximum(net, float(min_net_price)))

    # Recompute discount from bounded net
    disc = up - net
    disc = np.maximum(disc, 0.0)

    price["final_unit_price"] = up
    price["final_unit_cost"] = uc
    price["final_net_price"] = net
    price["discount_amt"] = disc


def build_prices(rng, order_dates, qty, price):
    """
    Apply month-level macro transforms to per-row prices produced by compute_prices():

      - inflation: compounded smoothly per month since global start
      - seasonality: smooth bounded sinusoid transform
      - month inertia: optional smoothing across contiguous months (works in month-sliced mode)
      - ramp: linear interpolation from min_multiplier to max_multiplier over ramp.months since global start
      - discount noise + max discount cap
      - margin floor: enforce via net-price (i.e., reducing discounts), NOT by resampling cost
      - final rounding + invariants

    NOTE: 'qty' is intentionally unused here (kept for signature stability).
    """
    cfg = _get_cfg()

    order_dates = np.asarray(order_dates)
    n = int(order_dates.shape[0])
    if n <= 0:
        return price

    # ---- Anchor to global start month for month-sliced correctness ----
    global_start_month_i = _get_global_start_month_int(order_dates)

    order_months = order_dates.astype("datetime64[M]")
    order_month_i = order_months.astype("int64")

    unique_months, inv = np.unique(order_month_i, return_inverse=True)  # sorted unique month indices

    # ---- Config ----
    month_inertia = float(cfg.get("month_inertia", 0.0))
    month_inertia = float(np.clip(month_inertia, 0.0, 0.98))

    infl_cfg = cfg["inflation"]
    seas_cfg = cfg["seasonality"]
    ramp_cfg = cfg["ramp"]
    disc_cfg = cfg["discount"]
    floor_cfg = cfg["floors"]
    cost_cfg = cfg["cost"]

    annual_rate = float(infl_cfg.get("annual_rate", 0.03))
    # prevent pathological negative rates
    annual_rate = float(np.clip(annual_rate, -0.95, 10.0))

    noise_sigma = float(infl_cfg.get("noise_sigma", 0.02))
    noise_sigma = float(max(0.0, noise_sigma))

    seas_enabled = bool(seas_cfg.get("enabled", True))
    seas_amp = float(seas_cfg.get("amplitude", 0.06))
    seas_sharp = float(seas_cfg.get("sharpness", 1.6))
    seas_phase = int(seas_cfg.get("phase_shift_months", 0))

    ramp_months = max(1.0, float(ramp_cfg.get("months", 18)))
    min_mul = float(ramp_cfg.get("min_multiplier", 0.85))
    max_mul = float(ramp_cfg.get("max_multiplier", 1.05))

    discount_scale = float(ramp_cfg.get("discount_scale", 1.0))
    disc_noise_sigma = float(ramp_cfg.get("discount_noise_sigma", 0.15))
    disc_noise_sigma = float(max(0.0, disc_noise_sigma))

    max_discount_pct = float(disc_cfg.get("max_pct_of_price", 0.85))
    max_discount_pct = float(np.clip(max_discount_pct, 0.0, 1.0))

    min_net = float(floor_cfg.get("min_net_price", 0.10))
    min_net = float(max(0.0, min_net))

    min_margin_pct = float(cost_cfg.get("min_margin_pct", 0.05))
    min_margin_pct = float(np.clip(min_margin_pct, 0.0, 0.95))

    # ------------------------------------------------------------
    # 1) Month-level BASE factor: inflation + seasonality (+ optional inertia)
    # ------------------------------------------------------------
    months_since_start_unique = (unique_months - global_start_month_i).astype(np.float64)

    # Smooth monthly compounding of annual rate
    inflation = (1.0 + annual_rate) ** (months_since_start_unique / 12.0)
    inflation = np.where(np.isfinite(inflation), inflation, 1.0)

    if noise_sigma > 0.0:
        inflation *= rng.lognormal(mean=0.0, sigma=noise_sigma, size=inflation.shape[0])

    if seas_enabled:
        # month-of-year: 0..11
        month_numbers = (unique_months % 12).astype(np.int64)
        angle = 2.0 * np.pi * ((month_numbers.astype(np.float64) + float(seas_phase)) / 12.0)
        seasonal = 1.0 + seas_amp * np.tanh(seas_sharp * np.sin(angle))
    else:
        seasonal = np.ones_like(inflation, dtype=np.float64)

    raw_factor = inflation * seasonal
    raw_factor = np.where(np.isfinite(raw_factor), raw_factor, 1.0)

    # Inertia smoothing across contiguous months; optionally continue from prior call
    month_price_factor = np.empty_like(raw_factor, dtype=np.float64)

    prev_factor = getattr(State, "_pricing_prev_factor", None)
    prev_month_i = getattr(State, "_pricing_prev_month_i", None)

    for i in range(raw_factor.shape[0]):
        m_i = int(unique_months[i])

        if month_inertia > 0.0:
            if i > 0 and int(unique_months[i - 1]) == m_i - 1:
                # contiguous within this batch
                month_price_factor[i] = month_inertia * month_price_factor[i - 1] + (1.0 - month_inertia) * raw_factor[i]
            elif prev_factor is not None and prev_month_i is not None and int(prev_month_i) == m_i - 1:
                # contiguous from prior call (month-sliced mode)
                month_price_factor[i] = month_inertia * float(prev_factor) + (1.0 - month_inertia) * raw_factor[i]
            else:
                month_price_factor[i] = raw_factor[i]
        else:
            month_price_factor[i] = raw_factor[i]

        prev_factor = float(month_price_factor[i])
        prev_month_i = m_i

    # Persist last factor to enable inertia in month-sliced sequential generation
    try:
        setattr(State, "_pricing_prev_factor", prev_factor)
        setattr(State, "_pricing_prev_month_i", prev_month_i)
    except Exception:
        pass

    base_price_factor = month_price_factor[inv]

    # ------------------------------------------------------------
    # 2) Apply BASE factor to unit price / cost (and optionally discount)
    # ------------------------------------------------------------
    apply_discount = bool(cfg.get("apply_base_factor_to_discount", True))

    price["final_unit_price"] = np.asarray(price["final_unit_price"], dtype=np.float64) * base_price_factor
    price["final_unit_cost"] = np.asarray(price["final_unit_cost"], dtype=np.float64) * base_price_factor
    if apply_discount:
        price["discount_amt"] = np.asarray(price["discount_amt"], dtype=np.float64) * base_price_factor
    else:
        price["discount_amt"] = np.asarray(price["discount_amt"], dtype=np.float64)

    # Bring net into sync pre-ramp
    price["final_net_price"] = price["final_unit_price"] - price["discount_amt"]

    # ------------------------------------------------------------
    # 3) Ramp (business maturity) anchored to GLOBAL start
    # ------------------------------------------------------------
    months_since_start_row = (order_month_i.astype(np.float64) - float(global_start_month_i))
    t = np.clip(months_since_start_row / float(ramp_months), 0.0, 1.0)
    ramp = min_mul + t * (max_mul - min_mul)

    price["final_unit_price"] *= ramp
    price["final_unit_cost"] *= ramp

    # Discount noise/scaling (keep discount roughly proportional to price evolution)
    if disc_noise_sigma > 0.0:
        disc_noise = rng.lognormal(mean=0.0, sigma=disc_noise_sigma, size=n)
    else:
        disc_noise = 1.0

    price["discount_amt"] *= (ramp * discount_scale * disc_noise)

    # Cap discount (never exceed pct of price)
    price["discount_amt"] = np.clip(
        price["discount_amt"],
        0.0,
        price["final_unit_price"] * max_discount_pct,
    )

    price["final_net_price"] = price["final_unit_price"] - price["discount_amt"]

    # Floor net but never above unit price
    price["final_net_price"] = np.minimum(price["final_unit_price"], np.maximum(price["final_net_price"], min_net))
    price["discount_amt"] = price["final_unit_price"] - price["final_net_price"]

    # ------------------------------------------------------------
    # 4) Products-based cost anchor + margin floor (via net price)
    # ------------------------------------------------------------
    # Sanitize cost first
    price["final_unit_cost"] = np.asarray(price["final_unit_cost"], dtype=np.float64)
    price["final_unit_cost"] = np.where(np.isfinite(price["final_unit_cost"]), price["final_unit_cost"], 0.0)
    price["final_unit_cost"] = np.maximum(price["final_unit_cost"], 0.0)

    # Ensure cost not above unit price
    price["final_unit_cost"] = np.minimum(price["final_unit_cost"], price["final_unit_price"])

    if min_margin_pct > 0.0:
        required_net = price["final_unit_cost"] / (1.0 - min_margin_pct)
        # Raise net (reduce discount) but never above unit price
        price["final_net_price"] = np.minimum(
            price["final_unit_price"],
            np.maximum(price["final_net_price"], required_net),
        )
        price["discount_amt"] = price["final_unit_price"] - price["final_net_price"]

    # Final safety: cost cannot exceed net
    bad = price["final_unit_cost"] > price["final_net_price"]
    if np.any(bad):
        price["final_unit_cost"][bad] = price["final_net_price"][bad]

    # Re-apply absolute net floor after all adjustments (still capped by unit price)
    price["final_net_price"] = np.minimum(price["final_unit_price"], np.maximum(price["final_net_price"], min_net))
    price["discount_amt"] = price["final_unit_price"] - price["final_net_price"]

    # ------------------------------------------------------------
    # 5) Final rounding + invariants
    # ------------------------------------------------------------
    _enforce_price_invariants(price, min_net_price=min_net)

    price["final_unit_price"] = np.round(price["final_unit_price"], 2)
    price["final_unit_cost"] = np.round(price["final_unit_cost"], 2)

    # Recompute net/discount from rounded unit price for tight consistency
    price["final_net_price"] = np.minimum(
        price["final_unit_price"],
        np.maximum(price["final_net_price"], min_net),
    )
    price["discount_amt"] = price["final_unit_price"] - price["final_net_price"]

    price["discount_amt"] = np.round(np.maximum(price["discount_amt"], 0.0), 2)
    price["final_net_price"] = np.round(price["final_unit_price"] - price["discount_amt"], 2)

    # Final: cost <= net (after rounding)
    bad2 = price["final_unit_cost"] > price["final_net_price"]
    if np.any(bad2):
        price["final_unit_cost"][bad2] = price["final_net_price"][bad2]

    return price
