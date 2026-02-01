# Responsibility
#     Month-based activity
#     Row thinning (keep_mask)

import numpy as np


def apply_activity_thinning(rng, order_dates):
    n = len(order_dates)

    # Damp volatility for small datasets
    size_scale = min(1.0, n / 50_000)

    # Month buckets
    months = order_dates.astype("datetime64[M]")
    unique_months, inv = np.unique(months, return_inverse=True)

    # Strong seasonality baseline
    base_activity = np.array([
        0.85,  # Jan
        0.82,  # Feb
        0.95,  # Mar
        1.00,  # Apr
        1.05,  # May
        1.10,  # Jun
        1.08,  # Jul
        1.00,  # Aug
        0.95,  # Sep
        1.05,  # Oct
        1.12,  # Nov
        1.10,  # Dec
    ])

    keep_mask = np.zeros(n, dtype=bool)

    # Global scale: average rows per month
    avg_per_month = n // len(unique_months)

    for m in range(len(unique_months)):
        month_rows = np.where(inv == m)[0]

        # Month-of-year
        month_num = int(unique_months[m].astype(int) % 12)

        # BIG volatility (this is the key)
        volatility = rng.lognormal(mean=0.0, sigma=0.35 * size_scale)

        target = int(
            avg_per_month
            * base_activity[month_num]
            * volatility
        )

        # Hard safety bounds
        low = 0.75 if n < 30_000 else 0.65
        high = 1.25 if n < 30_000 else 1.45

        target = np.clip(
            target,
            int(avg_per_month * low),
            int(avg_per_month * high),
        )

        if len(month_rows) <= target:
            keep_mask[month_rows] = True
        else:
            # NEW: row-level noise inside the month
            row_sigma = 0.15 if n < 30_000 else 0.4
            row_weights = rng.lognormal(mean=0.0, sigma=row_sigma, size=len(month_rows))
            row_weights = row_weights / row_weights.sum()

            chosen = rng.choice(
                month_rows,
                size=target,
                replace=False,
                p=row_weights
            )
            keep_mask[chosen] = True

    kept = keep_mask.sum()

    if kept > n:
        # Too many rows kept → drop some
        extra = rng.choice(
            np.where(keep_mask)[0],
            size=kept - n,
            replace=False,
        )
        keep_mask[extra] = False

    elif kept < n:
        # Too few rows kept → revive some
        missing = rng.choice(
            np.where(~keep_mask)[0],
            size=n - kept,
            replace=False,
        )
        keep_mask[missing] = True

    return keep_mask
