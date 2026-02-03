import numpy as np


# ------------------------------------------------------------
# Line-count model (month-of-year effect only)
# ------------------------------------------------------------
def build_month_demand(
    base=1.0,
    amplitude=0.55,
    q4_boost=0.60,
    phase_shift=-2,
):
    """
    Month-level demand multipliers used ONLY to adjust expected
    lines per order (basket depth), not date selection.

    - amplitude: strength of annual seasonality
    - q4_boost: extra holiday uplift (Oct–Dec)
    - phase_shift: moves peak earlier/later in year
    """
    months = np.arange(12)

    seasonal = base + amplitude * np.sin(2 * np.pi * (months + phase_shift) / 12)

    # Q4 uplift (Oct-Dec => 9,10,11)
    seasonal[9:12] *= (1.0 + q4_boost)

    return seasonal.astype(np.float64)


def _safe_normalized_prob(p):
    """
    Normalize p to sum to 1, handling None, zeros, and NaNs.
    Returns None for uniform sampling.
    """
    if p is None:
        return None
    p = np.asarray(p, dtype=np.float64)
    if p.size == 0:
        return None
    s = p.sum()
    if not np.isfinite(s) or s <= 0:
        return None
    p = p / s
    # Avoid tiny negatives from numeric issues
    p = np.clip(p, 0.0, 1.0)
    s2 = p.sum()
    if s2 <= 0:
        return None
    return p / s2


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

    IMPORTANT NEW ASSUMPTIONS:
    - `date_pool` is typically month-sliced upstream (chunk_builder).
    - `customers` is typically a per-row sampled CustomerKey array produced upstream.
      We do NOT apply lifecycle logic here.
    - `_len_customers` is ignored for sampling range (kept for API compatibility).

    Returns:
      dict with:
        - customer_keys (len n)
        - order_dates (len n, datetime64[D])
        - optionally order_ids_int, line_num, order_ids_str when skip_cols=False
    """
    if skip_cols not in (True, False):
        raise RuntimeError("skip_cols must be a boolean")

    if n <= 0:
        return {
            "customer_keys": np.empty(0, dtype=np.int64),
            "order_dates": np.empty(0, dtype="datetime64[D]"),
        }

    date_pool = np.asarray(date_pool)
    if date_pool.size == 0:
        raise RuntimeError("date_pool is empty")

    customers = np.asarray(customers, dtype=np.int64)
    if customers.size == 0:
        raise RuntimeError("customers array is empty")

    # ------------------------------------------------------------
    # Order count heuristic (avg lines/order)
    # ------------------------------------------------------------
    # This drives order_ids and line structure only. It does not affect total rows.
    avg_lines = 2.0
    order_count = max(1, int(n / avg_lines))

    # ------------------------------------------------------------
    # Order-level date sampling (use provided date_prob if valid)
    # ------------------------------------------------------------
    demand = _safe_normalized_prob(date_prob)
    if demand is None:
        od_idx = rng.integers(0, _len_date_pool, size=order_count)
    else:
        od_idx = rng.choice(_len_date_pool, size=order_count, p=demand)

    order_dates = date_pool[od_idx].astype("datetime64[D]", copy=False)

    # ------------------------------------------------------------
    # Order IDs: YYYYMMDD * 1e9 + random suffix
    # ------------------------------------------------------------
    # Keep stable format + high uniqueness; month-sliced still fine.
    date_str = np.datetime_as_string(order_dates, unit="D")
    date_int = np.char.replace(date_str, "-", "").astype(np.int64)

    suffix_int = rng.integers(0, 1_000_000_000, size=order_count, dtype=np.int64)
    order_ids_int = date_int * 1_000_000_000 + suffix_int

    # ------------------------------------------------------------
    # Assign a customer per order
    # ------------------------------------------------------------
    # Upstream already sampled a per-row customer distribution; we preserve it by:
    # - sampling order customers from that distribution (with replacement),
    #   rather than from a global universe.
    order_customers = rng.choice(customers, size=order_count, replace=True)

    # ------------------------------------------------------------
    # Lines per order (holiday basket depth uplift)
    # ------------------------------------------------------------
    month_demand = build_month_demand()

    # month-of-year (0-11) from order_dates
    months = order_dates.astype("datetime64[M]").astype("int64") % 12
    month_factor = month_demand[months]

    holiday_boost = month_factor > 1.10

    base_p = np.array([0.55, 0.25, 0.10, 0.06, 0.04], dtype=np.float64)
    holiday_p = np.array([0.40, 0.30, 0.15, 0.10, 0.05], dtype=np.float64)

    # choose p per order
    # (vectorized build of per-order categorical distribution is awkward; loop is cheap at order_count scale)
    lines_per_order = np.empty(order_count, dtype=np.int8)
    for i in range(order_count):
        pi = holiday_p if holiday_boost[i] else base_p
        lines_per_order[i] = rng.choice([1, 2, 3, 4, 5], p=pi)

    expanded_len = int(lines_per_order.sum())

    # prefix sums for line numbering
    order_starts = np.empty(order_count, dtype=np.int64)
    np.cumsum(lines_per_order, out=order_starts)
    order_starts -= lines_per_order

    # expand to line level
    customer_keys = np.repeat(order_customers, lines_per_order)
    order_dates_expanded = np.repeat(order_dates, lines_per_order)
    sales_order_num_int = np.repeat(order_ids_int, lines_per_order)

    line_num = (
        np.arange(expanded_len, dtype=np.int64)
        - np.repeat(order_starts, lines_per_order)
        + 1
    )

    # ------------------------------------------------------------
    # Pad or trim to exactly n rows (deterministic)
    # ------------------------------------------------------------
    if expanded_len < n:
        extra = n - expanded_len
        sl = slice(0, extra)

        customer_keys = np.concatenate((customer_keys, customer_keys[sl]))
        order_dates_expanded = np.concatenate((order_dates_expanded, order_dates_expanded[sl]))
        sales_order_num_int = np.concatenate((sales_order_num_int, sales_order_num_int[sl]))
        line_num = np.concatenate((line_num, line_num[sl]))

    customer_keys = customer_keys[:n]
    order_dates_expanded = order_dates_expanded[:n]
    sales_order_num_int = sales_order_num_int[:n]
    line_num = line_num[:n]

    # ------------------------------------------------------------
    # Output (strict skip semantics)
    # ------------------------------------------------------------
    result = {
        "customer_keys": customer_keys.astype(np.int64, copy=False),
        "order_dates": order_dates_expanded.astype("datetime64[D]", copy=False),
    }

    if not skip_cols:
        result["order_ids_int"] = sales_order_num_int
        result["line_num"] = line_num
        result["order_ids_str"] = sales_order_num_int.astype(str)

    return result
