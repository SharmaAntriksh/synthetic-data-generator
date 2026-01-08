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
    return np.round(values.astype(np.float64), decimals)


def compute_prices(
    rng,
    n,
    unit_price,
    unit_cost,
    promo_pct=0.0,
):
    """
    Simple, deterministic price realization.

    Inputs (authoritative):
    - unit_price : from products.parquet
    - unit_cost  : from products.parquet

    Applies:
    - discount ladder
    - loss-leader protection

    Does NOT:
    - rescale base prices
    - clamp catalog prices
    """

    S = State

    # -------------------------------------------------
    # 1. AUTHORITATIVE BASE VALUES
    # -------------------------------------------------
    base_price = unit_price.astype(np.float64, copy=True)
    cost = unit_cost.astype(np.float64, copy=True)

    # Hard sanity (product bug protection)
    cost = np.clip(cost, 0, base_price)

    # -------------------------------------------------
    # 2. DISCOUNT LADDER
    # -------------------------------------------------
    types, values, weights = zip(*DISCOUNT_LADDER)
    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()

    choices = rng.choice(len(DISCOUNT_LADDER), size=n, p=weights)
    discount_amt = np.zeros(n, dtype=np.float64)

    for i, idx in enumerate(choices):
        t = types[idx]
        v = values[idx]

        if t == "pct":
            discount_amt[i] = base_price[i] * v
        elif t == "abs":
            discount_amt[i] = v
        # "none" → 0

    # -------------------------------------------------
    # 3. NET PRICE (PRE-SAFETY)
    # -------------------------------------------------
    net_price = base_price - discount_amt

    # PROMOTIONAL DISCOUNT (VECTORISED)
    net_price = net_price * (1.0 - promo_pct)

    # -------------------------------------------------
    # 4. LOSS-LEADER PROTECTION
    # -------------------------------------------------
    min_allowed = cost * MAX_DISCOUNT_COST_MULTIPLIER
    net_price = np.maximum(net_price, min_allowed)
    net_price = np.minimum(net_price, base_price)

    discount_amt = base_price - net_price

    # -------------------------------------------------
    # 7. FINAL SAFETY
    # -------------------------------------------------
    cost = np.minimum(cost, net_price)

    # -------------------------------------------------
    # 8. FINAL ROUNDING (AUTHORITATIVE)
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
