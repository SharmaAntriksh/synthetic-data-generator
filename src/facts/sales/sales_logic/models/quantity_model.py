# Responsibility
#     Basket size
#     Quantity randomness
#     Transaction intensity

import numpy as np


def build_quantity(rng, order_dates):
    """
    Generate order line quantities with month-level variation.
    """
    n = len(order_dates)

    base_qty = rng.poisson(2.2, n) + 1

    order_months = order_dates.astype("datetime64[M]")
    month_idx = order_months.astype(int) % 12

    txn_month_factor = np.array([
        0.97,  # Jan
        0.96,  # Feb
        0.98,  # Mar
        1.00,  # Apr
        1.01,  # May
        1.02,  # Jun
        1.02,  # Jul
        1.01,  # Aug
        1.00,  # Sep
        1.03,  # Oct
        1.05,  # Nov
        1.04,  # Dec
    ])

    qty = base_qty * txn_month_factor[month_idx]
    qty = rng.normal(qty, 0.4)   # additive noise instead of multiplicative
    qty = np.round(qty).astype(int)
    
    return np.clip(qty, 1, 5)

