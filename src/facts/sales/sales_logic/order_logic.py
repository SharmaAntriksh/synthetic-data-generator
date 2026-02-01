import numpy as np

# ------------------------------------------------------------
# Demand model (parameter-driven, deterministic)
# ------------------------------------------------------------
def build_month_demand(
    base=1.0,
    amplitude=0.55,
    q4_boost=0.60,
    phase_shift=-2,
):
    """
    Generate month-level demand multipliers.

    amplitude   : strength of annual seasonality
    q4_boost    : extra holiday uplift (Oct–Dec)
    phase_shift : moves peak earlier/later in year
    """
    months = np.arange(12)

    seasonal = base + amplitude * np.sin(
        2 * np.pi * (months + phase_shift) / 12
    )

    # Q4 uplift
    seasonal[9:12] *= (1.0 + q4_boost)

    return seasonal.astype(np.float64)


def year_demand(year, base_year=2021, growth=0.08):
    """
    Compound year-over-year growth.
    """
    return (1.0 + growth) ** (year - base_year)


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

    month_demand = build_month_demand()
    month_factor = month_demand[months]
    
    demand = date_prob / date_prob.sum()

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
    holiday_boost = month_factor[od_idx] > 1.10

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
