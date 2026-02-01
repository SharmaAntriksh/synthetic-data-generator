import numpy as np
from src.facts.sales.sales_logic.globals import State


def apply_customer_churn(
    rng,
    customer_keys,
    order_dates,
    all_customers,
    seed,
):
    """
    Applies smooth customer lifecycle dynamics with:
    - gradual monthly churn
    - gradual monthly growth
    - strong month-to-month continuity
    """

    cfg = State.models_cfg["customers"]

    base_active_rate = cfg["base_active_rate"]
    annual_growth_rate = cfg["annual_growth_rate"]
    annual_churn_rate = cfg["annual_churn_rate"]
    monthly_noise = cfg["monthly_noise"]

    # ------------------------------------------------------------
    # Convert dates to year-month index
    # ------------------------------------------------------------
    months = order_dates.astype("datetime64[M]").astype(int)
    min_month = months.min()
    month_idx = months - min_month
    n_months = month_idx.max() + 1

    rng_global = np.random.default_rng(seed + 9000)

    # Initial active customers
    active_customers = set(
        rng_global.choice(
            all_customers,
            size=int(len(all_customers) * base_active_rate),
            replace=False,
        )
    )

    out = customer_keys.copy()

    # Convert annual rates to monthly
    monthly_growth = annual_growth_rate / 12.0
    monthly_churn = annual_churn_rate / 12.0

    for m in range(n_months):
        mask = month_idx == m
        if not mask.any():
            continue

        rng_month = np.random.default_rng(seed + 10000 + m * 7919)

        # --------------------------------------------------------
        # CHURN
        # --------------------------------------------------------
        churn_rate = monthly_churn * rng_month.uniform(
            1 - monthly_noise,
            1 + monthly_noise,
        )
        churn_n = int(len(active_customers) * churn_rate)

        if churn_n > 0:
            churned = rng_month.choice(
                list(active_customers),
                size=min(churn_n, len(active_customers)),
                replace=False,
            )
            active_customers -= set(churned)

        # --------------------------------------------------------
        # GROWTH
        # --------------------------------------------------------
        available = list(set(all_customers) - active_customers)
        growth_rate = monthly_growth * rng_month.uniform(
            1 - monthly_noise,
            1 + monthly_noise,
        )
        grow_n = int(len(active_customers) * growth_rate)

        if available and grow_n > 0:
            added = rng_month.choice(
                available,
                size=min(grow_n, len(available)),
                replace=False,
            )
            active_customers |= set(added)

        # --------------------------------------------------------
        # ASSIGN CUSTOMERS FOR THIS MONTH
        # --------------------------------------------------------
        active_list = np.array(list(active_customers))

        out[mask] = rng_month.choice(
            active_list,
            size=mask.sum(),
            replace=True,
        )

    return out
