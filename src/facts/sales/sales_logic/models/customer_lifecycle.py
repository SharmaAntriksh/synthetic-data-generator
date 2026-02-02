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
        dict[int, np.ndarray]: month_idx -> boolean mask over all_customers
    """

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    all_customers = np.asarray(all_customers, dtype=np.int64)
    n_cust = len(all_customers)

    cfg = State.models_cfg["customers"]

    base_active_rate = cfg["base_active_rate"]
    annual_growth_rate = cfg["annual_growth_rate"]
    annual_churn_rate = cfg["annual_churn_rate"]
    monthly_noise = cfg["monthly_noise"]

    seasonal_amplitude = cfg.get("seasonal_growth_amplitude", 0.0)
    seasonal_period = cfg.get("seasonal_period_months", 24)
    min_monthly_new = cfg.get("min_monthly_new_customers", 0)

    monthly_growth_base = annual_growth_rate / 12.0
    monthly_churn_base = annual_churn_rate / 12.0

    rng_global = np.random.default_rng(seed + 9000)

    # ------------------------------------------------------------------
    # Authoritative lifecycle state: boolean mask
    # ------------------------------------------------------------------
    mask = np.zeros(n_cust, dtype=bool)

    init_n = int(n_cust * base_active_rate)
    if init_n > 0:
        init_idx = rng_global.choice(n_cust, size=init_n, replace=False)
        mask[init_idx] = True

    active_by_month = {}

    # ------------------------------------------------------------------
    # Month-by-month lifecycle simulation
    # ------------------------------------------------------------------
    for m in range(start_month, end_month + 1):
        rng_month = np.random.default_rng(seed + 10000 + m * 7919)

        # -----------------------------
        # Seasonality
        # -----------------------------
        cycle = np.sin(2 * np.pi * m / seasonal_period)
        cycle_multiplier = 1.0 + seasonal_amplitude * cycle

        # -----------------------------
        # CHURN
        # -----------------------------
        churn_rate = monthly_churn_base * rng_month.uniform(
            1.0 - monthly_noise,
            1.0 + monthly_noise,
        )

        active_idx = np.flatnonzero(mask)
        churn_n = int(len(active_idx) * churn_rate)

        if churn_n > 0 and len(active_idx) > 0:
            churned = rng_month.choice(
                active_idx,
                size=min(churn_n, len(active_idx)),
                replace=False,
            )
            mask[churned] = False

        # -----------------------------
        # GROWTH
        # -----------------------------
        growth_rate = (
            monthly_growth_base
            * cycle_multiplier
            * rng_month.uniform(1.0 - monthly_noise, 1.0 + monthly_noise)
        )

        market_base = max(
            n_cust * 0.01,
            np.count_nonzero(mask) * growth_rate,
        )

        grow_n = max(int(market_base), min_monthly_new)

        inactive_idx = np.flatnonzero(~mask)

        if grow_n > 0 and len(inactive_idx) > 0:
            added = rng_month.choice(
                inactive_idx,
                size=min(grow_n, len(inactive_idx)),
                replace=False,
            )
            mask[added] = True

        # -----------------------------
        # Snapshot (IMPORTANT: copy)
        # -----------------------------
        active_by_month[m] = mask.copy()

    return active_by_month
