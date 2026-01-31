import numpy as np


def build_orders(
    rng,
    n: int,
    skip_cols: bool,
    date_pool,
    date_prob,
    customers,
    product_keys,          # kept for API stability
    _len_date_pool: int,
    _len_customers: int,
    volume_multiplier=None,
):
    """
    Generate order-level structure and expand to line-level rows.

    volume_multiplier:
        Optional array aligned to `n` rows (one per eventual line),
        encoding demand, shocks, and capacity. If None, behavior
        matches legacy logic.
    """

    if skip_cols not in (True, False):
        raise RuntimeError("skip_cols must be a boolean")

    # ------------------------------------------------------------
    # Baseline order count (lines → orders heuristic)
    # ------------------------------------------------------------
    avg_lines = 2.0
    base_order_count = max(1, int(n / avg_lines))

    # ------------------------------------------------------------
    # Build DATE-LEVEL demand weights (critical)
    # ------------------------------------------------------------
    if volume_multiplier is not None:
        # Map row-level multipliers → date_pool grain
        date_volume = np.zeros(_len_date_pool, dtype=np.float64)
        date_counts = np.zeros(_len_date_pool, dtype=np.int64)

        # We assume volume_multiplier corresponds to eventual rows,
        # but we only need a *relative* date signal.
        # Sample indices deterministically.
        sample_idx = rng.integers(0, n, size=_len_date_pool)

        for i in range(_len_date_pool):
            vm = volume_multiplier[sample_idx[i]]
            date_volume[i] = vm
            date_counts[i] = 1

        date_volume /= date_volume.mean()
    else:
        date_volume = None

    # ------------------------------------------------------------
    # Demand-weighted date sampling
    # ------------------------------------------------------------
    if date_volume is not None:
        demand = date_prob * date_volume
    else:
        demand = date_prob

    demand /= demand.sum()

    # ------------------------------------------------------------
    # Elastic order count (IMPORTANT: do NOT average demand away)
    # ------------------------------------------------------------
    if volume_multiplier is not None:
        # Use upper-tail pressure, not mean
        scale = np.percentile(volume_multiplier, 80)
        order_count = max(1, int(base_order_count * scale))
    else:
        order_count = base_order_count

    od_idx = rng.choice(
        _len_date_pool,
        size=order_count,
        p=demand,
    )

    order_dates = date_pool[od_idx]

    # ------------------------------------------------------------
    # Order IDs (deterministic, sortable)
    # ------------------------------------------------------------
    date_int = (
        order_dates.astype("datetime64[D]")
        .astype(str)
    )
    date_int = np.char.replace(date_int, "-", "").astype(np.int64)

    suffix_int = rng.integers(
        0,
        1_000_000_000,
        size=order_count,
        dtype=np.int64,
    )

    order_ids_int = date_int * 1_000_000_000 + suffix_int

    # ------------------------------------------------------------
    # Customers (uniform by design, dimension handles realism)
    # ------------------------------------------------------------
    cust_idx = rng.integers(0, _len_customers, size=order_count)
    order_customers = customers[cust_idx].astype(np.int64, copy=False)

    # ------------------------------------------------------------
    # Lines per order (demand-sensitive, non-cyclical)
    # ------------------------------------------------------------
    if volume_multiplier is not None:
        line_pressure = volume_multiplier[
            rng.integers(0, n, size=order_count)
        ]
    else:
        line_pressure = np.ones(order_count, dtype=np.float64)

    base_p = np.array([0.55, 0.25, 0.10, 0.06, 0.04])
    high_p = np.array([0.40, 0.30, 0.15, 0.10, 0.05])

    probs = np.where(
        line_pressure[:, None] > 1.1,
        high_p,
        base_p,
    )

    lines_per_order = np.array(
        [rng.choice([1, 2, 3, 4, 5], p=p) for p in probs],
        dtype=np.int8,
    )

    expanded_len = int(lines_per_order.sum())

    # ------------------------------------------------------------
    # Expand orders → lines
    # ------------------------------------------------------------
    order_starts = np.empty(order_count, dtype=np.int64)
    np.cumsum(lines_per_order, out=order_starts)
    order_starts -= lines_per_order

    customer_keys = np.repeat(order_customers, lines_per_order)
    order_dates_expanded = np.repeat(order_dates, lines_per_order)
    sales_order_num_int = np.repeat(order_ids_int, lines_per_order)

    line_num = (
        np.arange(expanded_len, dtype=np.int64)
        - np.repeat(order_starts, lines_per_order)
        + 1
    )

    # ------------------------------------------------------------
    # Pad deterministically if needed
    # ------------------------------------------------------------
    if expanded_len < n:
        extra = n - expanded_len
        sl = slice(0, extra)

        customer_keys = np.concatenate((customer_keys, customer_keys[sl]))
        order_dates_expanded = np.concatenate(
            (order_dates_expanded, order_dates_expanded[sl])
        )
        sales_order_num_int = np.concatenate(
            (sales_order_num_int, sales_order_num_int[sl])
        )
        line_num = np.concatenate((line_num, line_num[sl]))

    # ------------------------------------------------------------
    # Trim to exactly n rows
    # ------------------------------------------------------------
    customer_keys = customer_keys[:n]
    order_dates_expanded = order_dates_expanded[:n]
    sales_order_num_int = sales_order_num_int[:n]
    line_num = line_num[:n]

    # ------------------------------------------------------------
    # Output (strict skip semantics)
    # ------------------------------------------------------------
    result = {
        "customer_keys": customer_keys,
        "order_dates": order_dates_expanded.astype("datetime64[D]", copy=False),
    }

    if not skip_cols:
        result["order_ids_int"] = sales_order_num_int
        result["line_num"] = line_num
        result["order_ids_str"] = sales_order_num_int.astype(str)

    return result
