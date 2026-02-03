import numpy as np
from src.facts.sales.sales_logic.globals import State


# Allow discounts to eat into margin up to this factor
# net_price >= cost * multiplier
MAX_DISCOUNT_COST_MULTIPLIER = 0.90

# -------------------------------------------------
# Discount ladder (intentional, weighted)
# Values are in USD (pre any higher-level scaling)
# -------------------------------------------------
DISCOUNT_LADDER = [
    ("none", 0.00, 60),

    # percentage discounts (most common)
    ("pct",  0.05, 12),
    ("pct",  0.10, 10),
    ("pct",  0.15, 8),
    ("pct",  0.20, 6),
    ("pct",  0.30, 4),

    # absolute USD discounts (rare, high impact)
    ("abs",  5,    6),
    ("abs",  10,   5),
    ("abs",  25,   3),
    ("abs",  50,   2),
    ("abs",  75,   1),
    ("abs",  100,  1),
]


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def _quantize(values, decimals=2):
    return np.round(values.astype(np.float64, copy=False), decimals)


def _get_cfg():
    """
    Optional config block.

    models:
      pricing_logic:
        price_pressure_bounds: [0.90, 1.20]
        cost_pressure_follow: 0.05      # how much cost follows price pressure
        enable_row_jitter: false
        row_jitter_sigma: 0.01          # lognormal sigma
        promo_clip: [0.0, 1.0]
    """
    models = getattr(State, "models_cfg", None) or {}
    cfg = models.get("pricing_logic", {}) or {}

    out = {
        "price_pressure_bounds": cfg.get("price_pressure_bounds", [0.90, 1.20]),
        "cost_pressure_follow": float(cfg.get("cost_pressure_follow", 0.05)),
        "enable_row_jitter": bool(cfg.get("enable_row_jitter", False)),
        "row_jitter_sigma": float(cfg.get("row_jitter_sigma", 0.01)),
        "promo_clip": cfg.get("promo_clip", [0.0, 1.0]),
    }

    # Validate bounds
    pp = out["price_pressure_bounds"]
    if not (isinstance(pp, (list, tuple)) and len(pp) == 2):
        raise ValueError("models.pricing_logic.price_pressure_bounds must be a 2-item list")
    lo, hi = float(pp[0]), float(pp[1])
    if hi < lo:
        lo, hi = hi, lo
    out["price_pressure_bounds"] = [lo, hi]

    pc = out["promo_clip"]
    if not (isinstance(pc, (list, tuple)) and len(pc) == 2):
        raise ValueError("models.pricing_logic.promo_clip must be a 2-item list")
    plo, phi = float(pc[0]), float(pc[1])
    if phi < plo:
        plo, phi = phi, plo
    out["promo_clip"] = [plo, phi]

    out["cost_pressure_follow"] = float(np.clip(out["cost_pressure_follow"], 0.0, 0.50))
    out["row_jitter_sigma"] = float(np.clip(out["row_jitter_sigma"], 0.0, 0.20))
    return out


def compute_prices(
    rng,
    n: int,
    unit_price,
    unit_cost,
    promo_pct=0.0,
    price_pressure=1.0,
):
    """
    Deterministic, vectorized price realization.

    Preserves:
    - discount ladder semantics
    - promo_pct application (multiplicative to net)
    - loss-leader protection (net >= cost * MAX_DISCOUNT_COST_MULTIPLIER)
    - rounding rules

    Notes aligned with the new pipeline:
    - Month-level inflation/seasonality/ramp is now handled in pricing_pipeline.build_prices().
      This function should remain a "micro" realization step per row.
    """
    if n <= 0:
        return {
            "final_unit_price": np.empty(0, dtype=np.float64),
            "final_net_price": np.empty(0, dtype=np.float64),
            "final_unit_cost": np.empty(0, dtype=np.float64),
            "discount_amt": np.empty(0, dtype=np.float64),
        }

    cfg = _get_cfg()

    # -------------------------------------------------
    # 1) Base values + guardrails
    # -------------------------------------------------
    base_price = np.asarray(unit_price, dtype=np.float64).copy()
    cost = np.asarray(unit_cost, dtype=np.float64).copy()

    # Replace non-finite with safe fallbacks
    base_price = np.where(np.isfinite(base_price), base_price, 0.0)
    cost = np.where(np.isfinite(cost), cost, 0.0)

    # Ensure non-negative
    base_price = np.maximum(base_price, 0.0)
    cost = np.maximum(cost, 0.0)

    # price_pressure can be scalar or array
    pp_lo, pp_hi = cfg["price_pressure_bounds"]
    pp = np.asarray(price_pressure, dtype=np.float64)
    pp = np.where(np.isfinite(pp), pp, 1.0)
    pp = np.clip(pp, pp_lo, pp_hi)

    base_price *= pp

    # Cost follows price pressure slightly (not 1:1)
    follow = cfg["cost_pressure_follow"]
    cost *= (1.0 + (pp - 1.0) * follow)

    # Hard sanity: cost cannot exceed base_price
    cost = np.minimum(cost, base_price)

    # Optional row-level jitter (very small realism)
    if cfg["enable_row_jitter"] and cfg["row_jitter_sigma"] > 0:
        jitter = rng.lognormal(mean=0.0, sigma=cfg["row_jitter_sigma"], size=n)
        base_price *= jitter
        cost *= (1.0 + (jitter - 1.0) * 0.25)  # damped jitter for cost

    # -------------------------------------------------
    # 2) Discount ladder (vectorized)
    # -------------------------------------------------
    types, values, weights = zip(*DISCOUNT_LADDER)
    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()

    values = np.asarray(values, dtype=np.float64)
    is_pct = np.array([t == "pct" for t in types], dtype=bool)
    is_abs = np.array([t == "abs" for t in types], dtype=bool)

    choices = rng.choice(len(values), size=n, p=weights)

    discount_amt = np.zeros(n, dtype=np.float64)

    pct_mask = is_pct[choices]
    if pct_mask.any():
        discount_amt[pct_mask] = base_price[pct_mask] * values[choices[pct_mask]]

    abs_mask = is_abs[choices]
    if abs_mask.any():
        discount_amt[abs_mask] = values[choices[abs_mask]]

    # Guard: discount cannot exceed base price
    discount_amt = np.clip(discount_amt, 0.0, base_price)

    # -------------------------------------------------
    # 3) Net price + promo
    # -------------------------------------------------
    net_price = base_price - discount_amt

    # promo_pct can be scalar or vector; clip safely
    promo = np.asarray(promo_pct, dtype=np.float64)
    promo = np.where(np.isfinite(promo), promo, 0.0)
    plo, phi = cfg["promo_clip"]
    promo = np.clip(promo, plo, phi)

    # apply promo as a multiplicative net discount
    net_price *= (1.0 - promo)

    # -------------------------------------------------
    # 4) Loss-leader protection + consistency
    # -------------------------------------------------
    min_allowed = cost * MAX_DISCOUNT_COST_MULTIPLIER
    net_price = np.maximum(net_price, min_allowed)

    # net cannot exceed base
    net_price = np.minimum(net_price, base_price)

    # recompute discount from protected net price
    discount_amt = base_price - net_price

    # final safety: cost cannot exceed net
    cost = np.minimum(cost, net_price)

    # -------------------------------------------------
    # 5) Final rounding (authoritative)
    # -------------------------------------------------
    final_unit_price = _quantize(base_price, decimals=2)
    final_net_price = _quantize(net_price, decimals=2)
    final_unit_cost = _quantize(cost, decimals=2)

    # single source of truth for discount after rounding
    discount_amt = _quantize(final_unit_price - final_net_price, decimals=2)

    # Ensure finiteness post-rounding
    final_unit_price = np.where(np.isfinite(final_unit_price), final_unit_price, 0.0)
    final_net_price = np.where(np.isfinite(final_net_price), final_net_price, 0.0)
    final_unit_cost = np.where(np.isfinite(final_unit_cost), final_unit_cost, 0.0)
    discount_amt = np.where(np.isfinite(discount_amt), discount_amt, 0.0)

    return {
        "final_unit_price": final_unit_price,
        "final_net_price": final_net_price,
        "final_unit_cost": final_unit_cost,
        "discount_amt": discount_amt,
    }
