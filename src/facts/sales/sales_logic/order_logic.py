import numpy as np

# ------------------------------------------------------------
# Time-based demand shaping
# ------------------------------------------------------------
MONTH_DEMAND = np.array([
    0.85,  # Jan
    0.90,  # Feb
    1.00,  # Mar
    1.05,  # Apr
    1.10,  # May
    1.00,  # Jun
    0.95,  # Jul
    1.00,  # Aug
    1.05,  # Sep
    1.20,  # Oct
    1.35,  # Nov
    1.50,  # Dec
], dtype=np.float64)

YEAR_DEMAND = {
    2021: 0.95,
    2022: 1.10,
}

def build_orders(
    rng,
    n: int,
    skip_cols: bool,
    date_pool,
    date_prob,
    customers,
    product_keys,          # kept for API stability (not used here)
    _len_date_pool: int,
    _len_customers: int,
):
    """
    Generate order-level structure and expand to line-level rows.

    Returns a dict with:
      - customer_keys
      - order_dates
      - (optionally) order_ids_int, line_num, order_ids_str
    """

    if skip_cols not in (True, False):
        raise RuntimeError("skip_cols must be a boolean")

    # ------------------------------------------------------------
    # Order count heuristic
    # ------------------------------------------------------------
    avg_lines = 2.0
    order_count = max(1, int(n / avg_lines))

    # ------------------------------------------------------------
    # Order-level data
    # ------------------------------------------------------------
    dates = date_pool.astype("datetime64[M]")
    months = dates.astype(int) % 12
    years = (dates.astype("datetime64[Y]").astype(int) + 1970)

    month_factor = MONTH_DEMAND[months]
    year_factor = np.array(
        [YEAR_DEMAND.get(y, 1.0) for y in years],
        dtype=np.float64
    )

    demand = date_prob * month_factor * year_factor
    demand /= demand.sum()

    od_idx = rng.choice(_len_date_pool, size=order_count, p=demand)

    order_dates = date_pool[od_idx]

    # Fast YYYYMMDD integer construction
    date_int = (
        order_dates.astype("datetime64[D]")
        .astype("datetime64[D]")
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

    cust_idx = rng.integers(0, _len_customers, size=order_count)
    order_customers = customers[cust_idx].astype(np.int64, copy=False)

    # ------------------------------------------------------------
    # Lines per order
    # ------------------------------------------------------------
    holiday_boost = month_factor[od_idx] > 1.25

    base_p = np.array([0.55, 0.25, 0.10, 0.06, 0.04])
    holiday_p = np.array([0.40, 0.30, 0.15, 0.10, 0.05])

    p = np.where(
        holiday_boost[:, None],
        holiday_p,
        base_p
    )

    lines_per_order = np.array([
        rng.choice([1,2,3,4,5], p=pi)
        for pi in p
    ], dtype=np.int8)

    expanded_len = int(lines_per_order.sum())

    order_starts = np.empty(order_count, dtype=np.int64)
    np.cumsum(lines_per_order, out=order_starts)
    order_starts -= lines_per_order

    customer_keys = np.repeat(order_customers, lines_per_order)
    order_dates_expanded = np.repeat(order_dates, lines_per_order)

    # Only constructed once; sliced later
    sales_order_num_int = np.repeat(order_ids_int, lines_per_order)

    line_num = (
        np.arange(expanded_len, dtype=np.int64)
        - np.repeat(order_starts, lines_per_order)
        + 1
    )

    # ------------------------------------------------------------
    # Pad if needed (rare but deterministic)
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
        # String version only when needed
        result["order_ids_str"] = sales_order_num_int.astype(str)

    return result
