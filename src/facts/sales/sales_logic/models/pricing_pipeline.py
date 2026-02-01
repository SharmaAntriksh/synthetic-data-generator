# Responsibility
#   Price shaping only
#   NO magnitude control (handled at Product level)
#   NO revenue re-scaling
#   All invariants enforced

import numpy as np


def build_prices(
    rng,
    order_dates,
    qty,
    price,
):
    """
    Applies:
      - ramp (business maturity)
      - inflation (macro trend)
      - mild seasonality
      - discount noise
      - cost anchoring with margin safety

    IMPORTANT:
      - Assumes unit price & cost are ALREADY value-scaled at product level
      - Does NOT re-scale prices based on revenue
    """

    n = len(order_dates)

    # ------------------------------------------------------------
    # EARLY RAMP (business maturity)
    # ------------------------------------------------------------
    months_since_start = (
        order_dates.astype("datetime64[M]").astype(int)
        - order_dates.astype("datetime64[M]").min().astype(int)
    )

    ramp = np.clip(months_since_start / 18.0, 0.85, 1.05)

    price["final_unit_price"] *= ramp
    price["final_unit_cost"] *= ramp

    price["discount_amt"] *= (
        ramp
        * 0.7
        * rng.lognormal(mean=0.0, sigma=0.30, size=n)
    )

    # Guard: discount cannot exceed price
    price["discount_amt"] = np.clip(
        price["discount_amt"],
        0.0,
        price["final_unit_price"] * 0.90,
    )

    price["final_net_price"] = (
        price["final_unit_price"] - price["discount_amt"]
    )

    # ------------------------------------------------------------
    # INFLATION (macro trend + very small noise)
    # ------------------------------------------------------------
    base_year = order_dates.astype("datetime64[Y]").min().astype(int)
    year_idx = (
        order_dates.astype("datetime64[Y]").astype(int) - base_year
    )

    inflation = (1.0 + 0.035) ** year_idx
    inflation *= rng.lognormal(mean=0.0, sigma=0.01, size=n)

    for k in ("final_unit_price", "discount_amt", "final_unit_cost"):
        price[k] *= inflation

    price["discount_amt"] = np.clip(
        price["discount_amt"],
        0.0,
        price["final_unit_price"] * 0.90,
    )

    price["final_net_price"] = (
        price["final_unit_price"] - price["discount_amt"]
    )

    # ------------------------------------------------------------
    # SEASONALITY (PRICE RESPONSE ONLY)
    # ------------------------------------------------------------
    order_months = order_dates.astype("datetime64[M]")
    month_idx = order_months.astype(int) % 12

    raw = np.sin(2 * np.pi * (month_idx - 1) / 12)

    seasonality = 1.0 + 0.025 * np.tanh(1.0 * raw)

    for k in ("final_unit_price", "discount_amt", "final_unit_cost"):
        price[k] *= seasonality

    price["discount_amt"] = np.clip(
        price["discount_amt"],
        0.0,
        price["final_unit_price"] * 0.90,
    )

    price["final_net_price"] = (
        price["final_unit_price"] - price["discount_amt"]
    )

    # Absolute floor
    price["final_net_price"] = np.maximum(
        price["final_net_price"], 0.01
    )

    # ------------------------------------------------------------
    # FINAL COST ANCHOR + MARGIN FLOOR (NON-CIRCULAR)
    # ------------------------------------------------------------
    MIN_MARGIN_PCT = 0.05

    cost_ratio = rng.uniform(0.60, 0.82, size=n)
    price["final_unit_cost"] = price["final_net_price"] * cost_ratio

    max_allowed_cost = price["final_net_price"] * (1.0 - MIN_MARGIN_PCT)
    price["final_unit_cost"] = np.minimum(
        price["final_unit_cost"],
        max_allowed_cost,
    )

    # Final invariant
    assert np.all(price["final_unit_cost"] <= price["final_net_price"])

    return price
