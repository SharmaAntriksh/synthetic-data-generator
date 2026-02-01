import numpy as np
from src.facts.sales.sales_logic.globals import State


def build_quantity(rng, order_dates):
    """
    Generate order line quantities with month-level variation.
    """

    cfg = State.models_cfg["quantity"]

    n = len(order_dates)

    base_qty = rng.poisson(
        cfg["base_poisson_lambda"],
        n,
    ) + 1

    order_months = order_dates.astype("datetime64[M]")
    month_idx = order_months.astype(int) % 12

    txn_month_factor = np.array(
        cfg["monthly_factors"],
        dtype=np.float64,
    )

    qty = base_qty * txn_month_factor[month_idx]

    # additive noise (intentional, not multiplicative)
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
