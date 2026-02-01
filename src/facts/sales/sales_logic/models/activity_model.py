import numpy as np
from src.facts.sales.sales_logic.globals import State


def apply_activity_thinning(rng, order_dates):
    cfg = State.models_cfg["activity"]

    n = len(order_dates)

    # ------------------------------------------------------------
    # DATASET SIZE SCALING
    # ------------------------------------------------------------
    size_scale = min(
        1.0,
        n / cfg["size_scale_threshold"],
    )

    # Month buckets
    months = order_dates.astype("datetime64[M]")
    unique_months, inv = np.unique(months, return_inverse=True)

    base_activity = np.array(
        cfg["monthly_baseline"],
        dtype=np.float64,
    )

    keep_mask = np.zeros(n, dtype=bool)

    # Global scale: average rows per month
    avg_per_month = n // len(unique_months)

    # Small vs large dataset behavior
    is_small = n < cfg["bounds"]["dataset_split_threshold"]

    low = (
        cfg["bounds"]["small_dataset"]["low"]
        if is_small
        else cfg["bounds"]["large_dataset"]["low"]
    )
    high = (
        cfg["bounds"]["small_dataset"]["high"]
        if is_small
        else cfg["bounds"]["large_dataset"]["high"]
    )

    row_sigma = (
        cfg["row_noise"]["sigma_small_dataset"]
        if is_small
        else cfg["row_noise"]["sigma_large_dataset"]
    )

    # ------------------------------------------------------------
    # MONTH-TO-MONTH INERTIA (KEY SMOOTHING MECHANISM)
    # ------------------------------------------------------------
    inertia = cfg.get("month_inertia", 0.0)
    prev_target = None

    for m in range(len(unique_months)):
        month_rows = np.where(inv == m)[0]

        # Month-of-year (0–11)
        month_num = int(unique_months[m].astype(int) % 12)

        # Month-level volatility
        volatility = rng.lognormal(
            mean=0.0,
            sigma=cfg["volatility"]["sigma"] * size_scale,
        )

        # Raw (unsmoothed) monthly target
        raw_target = (
            avg_per_month
            * base_activity[month_num]
            * volatility
        )

        # Apply temporal smoothing
        if prev_target is None or inertia <= 0.0:
            target = raw_target
        else:
            target = (
                inertia * prev_target
                + (1.0 - inertia) * raw_target
            )

        # Clamp AFTER smoothing (important)
        target = int(np.clip(
            target,
            avg_per_month * low,
            avg_per_month * high,
        ))

        prev_target = target

        # --------------------------------------------------------
        # ROW SELECTION
        # --------------------------------------------------------
        if len(month_rows) <= target:
            keep_mask[month_rows] = True
        else:
            row_weights = rng.lognormal(
                mean=0.0,
                sigma=row_sigma,
                size=len(month_rows),
            )
            row_weights /= row_weights.sum()

            chosen = rng.choice(
                month_rows,
                size=target,
                replace=False,
                p=row_weights,
            )
            keep_mask[chosen] = True

    # ------------------------------------------------------------
    # FINAL BALANCING (DO NOT REMOVE)
    # ------------------------------------------------------------
    kept = keep_mask.sum()

    if kept > n:
        extra = rng.choice(
            np.where(keep_mask)[0],
            size=kept - n,
            replace=False,
        )
        keep_mask[extra] = False

    elif kept < n:
        missing = rng.choice(
            np.where(~keep_mask)[0],
            size=n - kept,
            replace=False,
        )
        keep_mask[missing] = True

    return keep_mask
