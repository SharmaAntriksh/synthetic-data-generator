import numpy as np
from src.facts.sales.sales_logic.globals import State


def build_active_customer_pool(
    all_customers,
    start_month: int,
    end_month: int,
    seed: int,
):
    """
    Build active customer pool per month.

    Returns:
        dict[int, np.ndarray]: month_idx -> active customer keys
    """

    cfg = State.models_cfg["customers"]

    base_active_rate = cfg["base_active_rate"]
    annual_growth_rate = cfg["annual_growth_rate"]
    annual_churn_rate = cfg["annual_churn_rate"]
    monthly_noise = cfg["monthly_noise"]

    seasonal_amplitude = cfg.get("seasonal_growth_amplitude", 0.0)
    seasonal_period = cfg.get("seasonal_period_months", 24)
    min_monthly_new = cfg.get("min_monthly_new_customers", 0)

    rng_global = np.random.default_rng(seed + 9000)

    # Initial active pool
    active_customers = set(
        rng_global.choice(
            all_customers,
            size=int(len(all_customers) * base_active_rate),
            replace=False,
        )
    )

    monthly_growth_base = annual_growth_rate / 12.0
    monthly_churn_base = annual_churn_rate / 12.0

    active_by_month = {}

    for m in range(start_month, end_month + 1):
        rng_month = np.random.default_rng(seed + 10000 + m * 7919)

        # Business cycle
        cycle = np.sin(2 * np.pi * m / seasonal_period)
        cycle_multiplier = 1.0 + seasonal_amplitude * cycle

        # --- CHURN ---
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

        # --- GROWTH ---
        available = list(set(all_customers) - active_customers)

        growth_rate = (
            monthly_growth_base
            * cycle_multiplier
            * rng_month.uniform(1 - monthly_noise, 1 + monthly_noise)
        )

        market_base = max(
            len(all_customers) * 0.01,
            len(active_customers) * growth_rate,
        )

        grow_n = max(int(market_base), min_monthly_new)

        if available and grow_n > 0:
            added = rng_month.choice(
                available,
                size=min(grow_n, len(available)),
                replace=False,
            )
            active_customers |= set(added)

        # Snapshot (IMPORTANT: copy)
        active_by_month[m] = np.array(list(active_customers), dtype=np.int64)

    return active_by_month
