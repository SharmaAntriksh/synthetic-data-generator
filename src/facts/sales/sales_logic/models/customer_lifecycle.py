import numpy as np


def apply_customer_churn(
    rng,
    customer_keys,
    order_dates,
    all_customers,
    seed,
    base_active_rate=0.65,
    annual_growth_rate=0.06,
    annual_churn_rate=0.10,
    monthly_noise=0.02,
):
    """
    Applies smooth customer lifecycle dynamics with:
    - gradual monthly churn
    - gradual monthly growth
    - strong month-to-month continuity
    """

    # Convert dates to year-month index
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

        # --- Churn ---
        churn_rate = monthly_churn * rng_month.uniform(
            1 - monthly_noise, 1 + monthly_noise
        )
        churn_n = int(len(active_customers) * churn_rate)

        if churn_n > 0:
            churned = rng_month.choice(
                list(active_customers),
                size=min(churn_n, len(active_customers)),
                replace=False,
            )
            active_customers -= set(churned)

        # --- Growth ---
        available = list(set(all_customers) - active_customers)
        growth_rate = monthly_growth * rng_month.uniform(
            1 - monthly_noise, 1 + monthly_noise
        )
        grow_n = int(len(active_customers) * growth_rate)

        if available and grow_n > 0:
            added = rng_month.choice(
                available,
                size=min(grow_n, len(available)),
                replace=False,
            )
            active_customers |= set(added)

        # --- Assign customers for this month ---
        active_list = np.array(list(active_customers))

        out[mask] = rng_month.choice(
            active_list,
            size=mask.sum(),
            replace=True,
        )

    return out
