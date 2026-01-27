import numpy as np
from src.facts.sales.sales_logic.globals import State

# Allow discounts to eat into margin up to this factor
MAX_DISCOUNT_COST_MULTIPLIER = 0.90

# -------------------------------------------------
# Discount ladder (intentional, weighted)
# Values are in USD (pre value_scale)
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
def _quantize(values, decimals=4):
    return np.round(values.astype(np.float64, copy=False), decimals)


def compute_prices(
    rng,
    n,
    unit_price,
    unit_cost,
    promo_pct=0.0,
    price_pressure=1.0,
):
    """
    Deterministic, vectorized price realization.

    Preserves:
    - discount ladder semantics
    - loss-leader protection
    - rounding rules
    """

    # -------------------------------------------------
    # 1. AUTHORITATIVE BASE VALUES
    # -------------------------------------------------
    base_price = unit_price.astype(np.float64, copy=True)
    cost = unit_cost.astype(np.float64, copy=True)

    # Guardrail: prevent extreme pricing behavior
    price_pressure = np.clip(price_pressure, 0.90, 1.20)

    base_price = base_price * price_pressure
    
    # Cost follows price, but less aggressively
    cost = cost * (1.0 + (price_pressure - 1.0) * 0.05)

    # Hard sanity (product bug protection)
    cost = np.clip(cost, 0, base_price)

    # -------------------------------------------------
    # 2. DISCOUNT LADDER (FULLY VECTORIZED)
    # -------------------------------------------------
    types, values, weights = zip(*DISCOUNT_LADDER)

    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()

    values = np.asarray(values, dtype=np.float64)
    is_pct = np.array([t == "pct" for t in types], dtype=bool)
    is_abs = np.array([t == "abs" for t in types], dtype=bool)
    # "none" implicitly handled (zero)

    choices = rng.choice(len(values), size=n, p=weights)

    discount_amt = np.zeros(n, dtype=np.float64)

    # Percentage discounts
    pct_mask = is_pct[choices]
    if pct_mask.any():
        discount_amt[pct_mask] = (
            base_price[pct_mask] * values[choices[pct_mask]]
        )

    # Absolute discounts
    abs_mask = is_abs[choices]
    if abs_mask.any():
        discount_amt[abs_mask] = values[choices[abs_mask]]

    # -------------------------------------------------
    # 3. NET PRICE (PRE-SAFETY)
    # -------------------------------------------------
    net_price = base_price - discount_amt

    # Promotional discount (vectorized)
    net_price = net_price * (1.0 - promo_pct)

    # -------------------------------------------------
    # 4. LOSS-LEADER PROTECTION
    # -------------------------------------------------
    min_allowed = cost * MAX_DISCOUNT_COST_MULTIPLIER
    net_price = np.maximum(net_price, min_allowed)
    net_price = np.minimum(net_price, base_price)

    discount_amt = base_price - net_price

    # -------------------------------------------------
    # 5. FINAL SAFETY
    # -------------------------------------------------
    cost = np.minimum(cost, net_price)

    # -------------------------------------------------
    # 6. FINAL ROUNDING (AUTHORITATIVE)
    # -------------------------------------------------
    final_unit_price = _quantize(base_price, decimals=2)
    final_net_price = _quantize(net_price, decimals=2)
    final_unit_cost = _quantize(cost, decimals=2)

    # 🔒 SINGLE SOURCE OF TRUTH
    discount_amt = _quantize(
        final_unit_price - final_net_price,
        decimals=2,
    )

    return {
        "final_unit_price": final_unit_price,
        "final_net_price": final_net_price,
        "final_unit_cost": final_unit_cost,
        "discount_amt": discount_amt,
    }
