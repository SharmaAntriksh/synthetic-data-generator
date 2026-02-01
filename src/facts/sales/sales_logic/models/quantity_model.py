import numpy as np
from src.facts.sales.sales_logic.globals import State


def build_quantity(rng, order_dates):
    """
    Generate order line quantities with smooth month-to-month transitions.

    Monthly inertia is applied to the expected quantity so that
    basket sizes evolve gradually over time instead of jumping
    independently each month.
    """

    cfg = State.models_cfg["quantity"]

    n = len(order_dates)

    # ------------------------------------------------------------
    # BASE QUANTITY (ROW-LEVEL)
    # ------------------------------------------------------------
    base_qty = rng.poisson(
        cfg["base_poisson_lambda"],
        n,
    ) + 1

    order_months = order_dates.astype("datetime64[M]")
    month_idx = order_months.astype(int) % 12

    monthly_factors = np.array(
        cfg["monthly_factors"],
        dtype=np.float64,
    )

    # ------------------------------------------------------------
    # MONTH-LEVEL EXPECTATION (WITH INERTIA)
    # ------------------------------------------------------------
    inertia = cfg.get("month_inertia", 0.0)
    prev_factor = None

    smoothed_factor = np.empty(n, dtype=np.float64)

    # Process months in order
    unique_months, inv = np.unique(order_months, return_inverse=True)

    for m in range(len(unique_months)):
        rows = np.where(inv == m)[0]
        month_num = int(unique_months[m].astype(int) % 12)

        raw_factor = monthly_factors[month_num]

        if prev_factor is None or inertia <= 0.0:
            factor = raw_factor
        else:
            factor = (
                inertia * prev_factor
                + (1.0 - inertia) * raw_factor
            )

        prev_factor = factor
        smoothed_factor[rows] = factor

    # Apply smooth monthly expectation
    qty = base_qty * smoothed_factor

    # ------------------------------------------------------------
    # ADDITIVE NOISE (ROW-LEVEL VARIATION)
    # ------------------------------------------------------------
    qty = rng.normal(
        qty,
        cfg["noise_sd"],
    )

    qty = np.round(qty).astype(int)

    return np.clip(
        qty,
        cfg["min_qty"],
        cfg["max_qty"],
    )
