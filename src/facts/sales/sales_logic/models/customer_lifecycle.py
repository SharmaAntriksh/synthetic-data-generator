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
    Customer lifecycle model with:
    - gradual churn
    - cyclical acquisition (increase / decrease / increase)
    - acquisition floor to prevent late-stage collapse
    - stability across any simulation length
    """

    cfg = State.models_cfg["customers"]

    base_active_rate = cfg["base_active_rate"]
    annual_growth_rate = cfg["annual_growth_rate"]
    annual_churn_rate = cfg["annual_churn_rate"]
    monthly_noise = cfg["monthly_noise"]

    # Cyclical acquisition knobs (month-based, single system)
    seasonal_amplitude = cfg.get("seasonal_growth_amplitude", 0.0)
    seasonal_period = cfg.get("seasonal_period_months", 24)
    min_monthly_new = cfg.get("min_monthly_new_customers", 0)

    # ------------------------------------------------------------
    # Convert order dates → contiguous month index
    # ------------------------------------------------------------
    months = order_dates.astype("datetime64[M]").astype(int)
    min_month = months.min()
    month_idx = months - min_month
    n_months = int(month_idx.max()) + 1

    rng_global = np.random.default_rng(seed + 9000)

    # ------------------------------------------------------------
    # Initial active customer pool
    # ------------------------------------------------------------
    active_customers = set(
        rng_global.choice(
            all_customers,
            size=int(len(all_customers) * base_active_rate),
            replace=False,
        )
    )

    out = customer_keys.copy()

    monthly_growth_base = annual_growth_rate / 12.0
    monthly_churn_base = annual_churn_rate / 12.0

    # ------------------------------------------------------------
    # Month-by-month lifecycle simulation
    # ------------------------------------------------------------
    for m in range(n_months):
        mask = month_idx == m
        if not mask.any():
            continue

        rng_month = np.random.default_rng(seed + 10000 + m * 7919)

        # --------------------------------------------------------
        # Cyclical growth multiplier (business cycles)
        # --------------------------------------------------------
        cycle = np.sin(2 * np.pi * m / seasonal_period)
        cycle_multiplier = 1.0 + seasonal_amplitude * cycle

        # --------------------------------------------------------
        # CHURN
        # --------------------------------------------------------
        churn_rate = monthly_churn_base * rng_month.uniform(
            1 - monthly_noise,
            1 + monthly_noise,
        )

        churn_n = int(len(active_customers) * churn_rate)

        if churn_n > 0 and active_customers:
            churned = rng_month.choice(
                list(active_customers),
                size=min(churn_n, len(active_customers)),
                replace=False,
            )
            active_customers -= set(churned)

        # --------------------------------------------------------
        # GROWTH (with acquisition floor)
        # --------------------------------------------------------
        available = list(set(all_customers) - active_customers)

        growth_rate = (
            monthly_growth_base
            * cycle_multiplier
            * rng_month.uniform(1 - monthly_noise, 1 + monthly_noise)
        )

        market_base = max(len(all_customers) * 0.01, len(active_customers) * growth_rate)
        grow_n = int(market_base * cycle_multiplier)
        grow_n = max(grow_n, min_monthly_new)

        if available and grow_n > 0:
            added = rng_month.choice(
                available,
                size=min(grow_n, len(available)),
                replace=False,
            )
            active_customers |= set(added)

        # --------------------------------------------------------
        # Assign customers to orders in this month
        # --------------------------------------------------------
        active_list = np.array(list(active_customers))

        if len(active_list) == 0:
            # Safety fallback (should never occur)
            active_list = np.array(all_customers)

    return customer_keys
