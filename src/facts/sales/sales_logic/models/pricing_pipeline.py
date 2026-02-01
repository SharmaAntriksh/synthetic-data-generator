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

    Month-level inertia is applied to the *base price signal*
    (inflation + seasonality), not to discounts.
    """

    cfg = State.models_cfg["pricing"]
    n = len(order_dates)

    # ------------------------------------------------------------
    # MONTH INDEXING (FOR INERTIA)
    # ------------------------------------------------------------
    order_months = order_dates.astype("datetime64[M]")
    unique_months, inv = np.unique(order_months, return_inverse=True)
    month_numbers = unique_months.astype(int) % 12

    month_inertia = cfg.get("month_inertia", 0.0)
    prev_price_factor = None
    month_price_factor = np.empty(len(unique_months), dtype=np.float64)

    # ------------------------------------------------------------
    # MONTH-LEVEL BASE PRICE FACTOR (SEASONALITY + INFLATION)
    # ------------------------------------------------------------
    infl_cfg = cfg["inflation"]
    seas_cfg = cfg["seasonality"]

    base_year = order_dates.astype("datetime64[Y]").min().astype(int)
    month_year_idx = (
        unique_months.astype("datetime64[Y]").astype(int) - base_year
    )

    for i, m in enumerate(month_numbers):

        # --- inflation (year-based, smooth by nature) ---
        inflation = (1.0 + infl_cfg["annual_rate"]) ** month_year_idx[i]
        inflation *= rng.lognormal(
            mean=0.0,
            sigma=infl_cfg["noise_sigma"],
        )

        # --- seasonality ---
        if seas_cfg.get("enabled", True):
            angle = 2 * np.pi * (m + seas_cfg["phase_shift_months"]) / 12
            seasonal = (
                1.0
                + seas_cfg["amplitude"]
                * np.tanh(seas_cfg["sharpness"] * np.sin(angle))
            )
        else:
            seasonal = 1.0

        raw_factor = inflation * seasonal

        # --- inertia smoothing ---
        if prev_price_factor is None or month_inertia <= 0.0:
            factor = raw_factor
        else:
            factor = (
                month_inertia * prev_price_factor
                + (1.0 - month_inertia) * raw_factor
            )

        prev_price_factor = factor
        month_price_factor[i] = factor

    # Expand month factor to rows
    base_price_factor = month_price_factor[inv]

    # ------------------------------------------------------------
    # APPLY BASE PRICE FACTOR
    # ------------------------------------------------------------
    for k in ("final_unit_price", "discount_amt", "final_unit_cost"):
        price[k] *= base_price_factor

    # ------------------------------------------------------------
    # EARLY RAMP (BUSINESS MATURITY)
    # ------------------------------------------------------------
    ramp_cfg = cfg["ramp"]

    months_since_start = (
        order_months.astype(int)
        - order_months.min().astype(int)
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
    # ABSOLUTE PRICE FLOOR
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # FINAL ROUNDING + CONSISTENCY (AUTHORITATIVE)
    # ------------------------------------------------------------
    price["final_unit_price"] = np.round(price["final_unit_price"], 2)
    price["discount_amt"] = np.round(price["discount_amt"], 2)

    price["final_net_price"] = np.round(
        price["final_unit_price"] - price["discount_amt"],
        2,
    )

    price["final_unit_cost"] = np.round(price["final_unit_cost"], 2)

    # Final invariant
    assert np.all(price["final_unit_cost"] <= price["final_net_price"])

    return price
