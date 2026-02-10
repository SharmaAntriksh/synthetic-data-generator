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
    months = np.arange(12, dtype=np.float64)
    seasonal = base + amplitude * np.sin(2.0 * np.pi * (months + phase_shift) / 12.0)
    seasonal[9:12] *= (1.0 + q4_boost)  # Oct-Dec uplift
    return seasonal.astype(np.float64, copy=False)


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
    s = float(p.sum())
    if not np.isfinite(s) or s <= 0.0:
        return None
    p = p / s
    p = np.clip(p, 0.0, 1.0)
    s2 = float(p.sum())
    if s2 <= 0.0:
        return None
    return p / s2


def _yyyymmdd_from_days(days: np.ndarray) -> np.ndarray:
    """
    Convert days-since-epoch (int64) to YYYYMMDD (int64) WITHOUT string ops.
    Uses numpy datetime64 conversions (fast, vectorized).
    """
    d = days.astype("datetime64[D]")
    y = d.astype("datetime64[Y]").astype(np.int64) + 1970  # years since 1970
    m = (d.astype("datetime64[M]").astype(np.int64) % 12) + 1
    day = (d - d.astype("datetime64[M]")).astype(np.int64) + 1
    return (y * 10000 + m * 100 + day).astype(np.int64, copy=False)


def build_orders(
    rng,
    n: int,
    skip_cols: bool,
    date_pool,
    date_prob,
    customers,
    product_keys,
    _len_date_pool: int,
    _len_customers: int,
    *,
    order_id_start: int | None = None,  # NEW
):

    """
    Generate order-level structure and expand to line-level rows.

    Assumptions:
    - `date_pool` is typically month-sliced upstream.
    - `customers` is typically a per-row sampled CustomerKey array produced upstream.
      Lifecycle logic is handled upstream.

    Returns dict:
      - customer_keys (len n)
      - order_dates (len n, datetime64[D])
      - if skip_cols=False: order_ids_int, line_num, order_ids_str
    """
    if skip_cols not in (True, False):
        raise RuntimeError("skip_cols must be a boolean")

    n = int(n)
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
    avg_lines = 2.0
    order_count = max(1, int(n / avg_lines))

    # ------------------------------------------------------------
    # Order-level date sampling
    # ------------------------------------------------------------
    demand = _safe_normalized_prob(date_prob)
    if demand is None:
        od_idx = rng.integers(0, _len_date_pool, size=order_count, dtype=np.int64)
    else:
        # rng.choice returns int64 indices
        od_idx = rng.choice(_len_date_pool, size=order_count, p=demand)

    order_dates = date_pool[od_idx].astype("datetime64[D]", copy=False)

    # ------------------------------------------------------------
    # Order IDs: YYYYMMDD * 1e9 + random suffix
    # (no string formatting; faster)
    # ------------------------------------------------------------
    days = order_dates.astype("datetime64[D]").astype(np.int64, copy=False)
    date_int = _yyyymmdd_from_days(days)

    MOD = np.int64(1_000_000_000)

    if order_id_start is None:
        raise RuntimeError(
            "order_id_start is required to guarantee unique SalesOrderNumber "
            "(caller must assign a disjoint range per chunk)."
        )

    start = np.int64(order_id_start)
    suffix_int = start + np.arange(order_count, dtype=np.int64)

    # Safety: suffix must fit in 9 digits
    if suffix_int.size and suffix_int[-1] >= MOD:
        raise RuntimeError("SalesOrderNumber suffix overflow; increase suffix width or capacity.")

    order_ids_int = date_int * MOD + suffix_int

    # ------------------------------------------------------------
    # Assign a customer per order (preserve upstream distribution)
    # ------------------------------------------------------------
    order_customers = rng.choice(customers, size=order_count, replace=True)

    # ------------------------------------------------------------
    # Lines per order (vectorized)
    # ------------------------------------------------------------
    month_demand = build_month_demand()

    # month-of-year (0-11)
    months = (order_dates.astype("datetime64[M]").astype(np.int64) % 12).astype(np.int64, copy=False)
    month_factor = month_demand[months]
    holiday_boost = month_factor > 1.10

    # Discrete outcomes
    k = np.array([1, 2, 3, 4, 5], dtype=np.int8)

    base_p = np.array([0.55, 0.25, 0.10, 0.06, 0.04], dtype=np.float64)
    holiday_p = np.array([0.40, 0.30, 0.15, 0.10, 0.05], dtype=np.float64)

    # Vectorized categorical sampling via inverse CDF:
    # pick base/holiday cdf per order, then digitize U~[0,1)
    cdf_base = np.cumsum(base_p)
    cdf_hol = np.cumsum(holiday_p)

    u = rng.random(order_count)  # one uniform per order
    lines_per_order = np.empty(order_count, dtype=np.int8)

    # base orders
    base_mask = ~holiday_boost
    if base_mask.any():
        lines_per_order[base_mask] = k[np.searchsorted(cdf_base, u[base_mask], side="right")]
    if holiday_boost.any():
        lines_per_order[holiday_boost] = k[np.searchsorted(cdf_hol, u[holiday_boost], side="right")]

    repeats = lines_per_order.astype(np.int32, copy=False)
    expanded_len = int(repeats.sum())

    # prefix sums for line numbering
    order_starts = np.cumsum(repeats, dtype=np.int64) - repeats

    # expand to line level
    customer_keys = np.repeat(order_customers, repeats)
    order_dates_expanded = np.repeat(order_dates, repeats)
    sales_order_num_int = np.repeat(order_ids_int, repeats)

    # line number per order
    line_num = (
        np.arange(expanded_len, dtype=np.int64)
        - np.repeat(order_starts, repeats)
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
    # Output
    # ------------------------------------------------------------
    result = {
        "customer_keys": customer_keys.astype(np.int64, copy=False),
        "order_dates": order_dates_expanded.astype("datetime64[D]", copy=False),
    }

    if not skip_cols:
        result["order_ids_int"] = sales_order_num_int
        result["line_num"] = line_num
        # Keep for downstream compatibility (string conversion can be expensive but optional)
        result["order_ids_str"] = sales_order_num_int.astype(str)

    return result
