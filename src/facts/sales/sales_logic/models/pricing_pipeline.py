import numpy as np
from src.facts.sales.sales_logic.globals import State


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
    """

    cfg = State.models_cfg["pricing"]

    n = len(order_dates)

    # ------------------------------------------------------------
    # EARLY RAMP (business maturity)
    # ------------------------------------------------------------
    ramp_cfg = cfg["ramp"]

    months_since_start = (
        order_dates.astype("datetime64[M]").astype(int)
        - order_dates.astype("datetime64[M]").min().astype(int)
    )

    ramp = np.clip(
        months_since_start / float(ramp_cfg["months"]),
        ramp_cfg["min_multiplier"],
        ramp_cfg["max_multiplier"],
    )

    price["final_unit_price"] *= ramp
    price["final_unit_cost"] *= ramp

    price["discount_amt"] *= (
        ramp
        * ramp_cfg["discount_scale"]
        * rng.lognormal(
            mean=0.0,
            sigma=ramp_cfg["discount_noise_sigma"],
            size=n,
        )
    )

    # Guard: discount cannot exceed price
    max_discount_pct = cfg["discount"]["max_pct_of_price"]
    price["discount_amt"] = np.clip(
        price["discount_amt"],
        0.0,
        price["final_unit_price"] * max_discount_pct,
    )

    price["final_net_price"] = (
        price["final_unit_price"] - price["discount_amt"]
    )

    # ------------------------------------------------------------
    # INFLATION (macro trend + very small noise)
    # ------------------------------------------------------------
    infl_cfg = cfg["inflation"]

    base_year = order_dates.astype("datetime64[Y]").min().astype(int)
    year_idx = (
        order_dates.astype("datetime64[Y]").astype(int) - base_year
    )

    inflation = (1.0 + infl_cfg["annual_rate"]) ** year_idx
    inflation *= rng.lognormal(
        mean=0.0,
        sigma=infl_cfg["noise_sigma"],
        size=n,
    )

    for k in ("final_unit_price", "discount_amt", "final_unit_cost"):
        price[k] *= inflation

    price["discount_amt"] = np.clip(
        price["discount_amt"],
        0.0,
        price["final_unit_price"] * max_discount_pct,
    )

    price["final_net_price"] = (
        price["final_unit_price"] - price["discount_amt"]
    )

    # ------------------------------------------------------------
    # SEASONALITY (PRICE RESPONSE ONLY)
    # ------------------------------------------------------------
    seas_cfg = cfg["seasonality"]

    if seas_cfg.get("enabled", True):
        order_months = order_dates.astype("datetime64[M]")
        month_idx = order_months.astype(int) % 12

        raw = np.sin(
            2 * np.pi * (month_idx + seas_cfg["phase_shift_months"]) / 12
        )

        seasonality = (
            1.0
            + seas_cfg["amplitude"]
            * np.tanh(seas_cfg["sharpness"] * raw)
        )

        for k in ("final_unit_price", "discount_amt", "final_unit_cost"):
            price[k] *= seasonality

        price["discount_amt"] = np.clip(
            price["discount_amt"],
            0.0,
            price["final_unit_price"] * max_discount_pct,
        )

        price["final_net_price"] = (
            price["final_unit_price"] - price["discount_amt"]
        )

    # Absolute floor
    price["final_net_price"] = np.maximum(
        price["final_net_price"],
        cfg["floors"]["min_net_price"],
    )

    # ------------------------------------------------------------
    # FINAL COST ANCHOR + MARGIN FLOOR (NON-CIRCULAR)
    # ------------------------------------------------------------
    cost_cfg = cfg["cost"]

    cost_ratio = rng.uniform(
        cost_cfg["cost_ratio_min"],
        cost_cfg["cost_ratio_max"],
        size=n,
    )

    price["final_unit_cost"] = price["final_net_price"] * cost_ratio

    max_allowed_cost = (
        price["final_net_price"]
        * (1.0 - cost_cfg["min_margin_pct"])
    )

    price["final_unit_cost"] = np.minimum(
        price["final_unit_cost"],
        max_allowed_cost,
    )

    # Final invariant
    assert np.all(price["final_unit_cost"] <= price["final_net_price"])

    return price
