"""Order generation: month demand curve and order/line-level expansion."""

from __future__ import annotations

import numpy as np

from ..globals import State


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

    max_lines = int(getattr(State, "max_lines_per_order", 6) or 6)
    if max_lines < 1:
        raise RuntimeError(f"State.max_lines_per_order must be >= 1, got {max_lines}")

    min_orders = (int(n) + max_lines - 1) // max_lines
    order_count = max(int(order_count), int(min_orders))

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
    # Order IDs: (ExcelDayID * 1000 + RunID) * 1e9 + suffix
    #
    # ExcelDayID is an Excel-style day serial (1899-12-30 = 0):
    #   2022-09-22 -> 44826
    #
    # RunID is 0..999 and must differ across dataset runs to make
    # SalesOrderNumber globally unique across reruns.
    #
    # Suffix remains a 9-digit space allocated disjointly per chunk via order_id_start.
    # ------------------------------------------------------------
    days = order_dates.astype("datetime64[D]").astype(np.int64, copy=False)

    # Excel epoch offset: 1970-01-01 is day 25569 in Excel
    excel_day_id = days + np.int64(25569)
    if excel_day_id.size and excel_day_id.min() < 0:
        raise RuntimeError("ExcelDayID underflow (dates before 1899-12-30)")

    run_id = int(getattr(State, "order_id_run_id", 0) or 0)
    if run_id < 0 or run_id > 999:
        raise RuntimeError(f"State.order_id_run_id must be in [0,999], got {run_id}")

    prefix = excel_day_id * np.int64(1000) + np.int64(run_id)

    MOD = np.int64(1_000_000_000)  # 1e9 => 9-digit suffix

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

    order_ids_int = prefix * MOD + suffix_int

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

    # Discrete outcomes (respect max_lines_per_order)
    # max_lines already computed earlier in build_orders
    if max_lines == 1:
        k = np.array([1], dtype=np.int16)
        base_p = np.array([1.0], dtype=np.float64)
        holiday_p = np.array([1.0], dtype=np.float64)
    else:
        # Always build k up to the configured cap
        k = np.arange(1, max_lines + 1, dtype=np.int16)

        # Preserve the original Contoso-like basket depth shape for 1..5
        base_p5 = np.array([0.55, 0.25, 0.10, 0.06, 0.04], dtype=np.float64)
        hol_p5  = np.array([0.40, 0.30, 0.15, 0.10, 0.05], dtype=np.float64)

        if max_lines <= 5:
            base_p = base_p5[:max_lines].copy()
            holiday_p = hol_p5[:max_lines].copy()
            base_p /= base_p.sum()
            holiday_p /= holiday_p.sum()
        else:
            # Add a small "long tail" probability mass for 6..max_lines
            # (keeps 1..5 relative proportions nearly identical)
            base_tail_mass = 0.03   # ~3% of orders can be 6+ lines
            hol_tail_mass  = 0.06   # holidays: heavier baskets

            tail_vals = np.arange(6, max_lines + 1, dtype=np.float64)
            decay = np.exp(-0.7 * (tail_vals - 6.0))  # geometric-ish decay
            decay /= decay.sum()

            base_p = np.concatenate([base_p5 * (1.0 - base_tail_mass), decay * base_tail_mass])
            holiday_p = np.concatenate([hol_p5 * (1.0 - hol_tail_mass), decay * hol_tail_mass])

    # Vectorized categorical sampling via inverse CDF:
    # pick base/holiday cdf per order, then digitize U~[0,1)
    cdf_base = np.cumsum(base_p)
    cdf_hol = np.cumsum(holiday_p)

    u = rng.random(order_count)  # one uniform per order
    lines_per_order = np.empty(order_count, dtype=np.int16)

    # base orders
    base_mask = ~holiday_boost
    if base_mask.any():
        lines_per_order[base_mask] = k[np.searchsorted(cdf_base, u[base_mask], side="right")]
    if holiday_boost.any():
        lines_per_order[holiday_boost] = k[np.searchsorted(cdf_hol, u[holiday_boost], side="right")]

    repeats = lines_per_order.astype(np.int64, copy=False)
    expanded_len = int(repeats.sum())

    # ------------------------------------------------------------
    # Ensure we create exactly `n` line rows WITHOUT creating a single
    # giant order. Cap repeats and distribute adjustments across orders.
    # ------------------------------------------------------------

    # Ensure feasibility: need at least ceil(n / max_lines) orders
    min_orders = (int(n) + max_lines - 1) // max_lines
    if order_count < min_orders:
        order_count = int(min_orders)
        # NOTE: if you change order_count here, you must also regenerate:
        # - order_dates
        # - order_ids_int
        # - order_customers
        # In practice, keep order_count calculation earlier consistent with max_lines.
        # (Preferred: apply min_orders right after order_count is first computed.)

    delta = int(n) - int(expanded_len)

    if delta > 0:
        need = delta
        while need > 0:
            candidates = np.flatnonzero(repeats < max_lines)
            if candidates.size == 0:
                raise RuntimeError(
                    f"Need {need} more lines but all orders are at max_lines_per_order={max_lines}. "
                    "Increase max_lines_per_order or increase order_count."
                )
            take = min(need, int(candidates.size))
            chosen = rng.choice(candidates, size=take, replace=False)
            repeats[chosen] += 1
            need -= take

    elif delta < 0:
        excess = -delta
        while excess > 0:
            candidates = np.flatnonzero(repeats > 1)
            if candidates.size == 0:
                raise RuntimeError(
                    "Unable to reduce repeats to match n without violating min 1 line/order."
                )
            take = min(excess, int(candidates.size))
            chosen = rng.choice(candidates, size=take, replace=False)
            repeats[chosen] -= 1
            excess -= take

    expanded_len = int(repeats.sum())
    if expanded_len != n:
        raise RuntimeError("Internal error: repeats sum != n after adjustment")

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
        result["_order_count"] = int(order_count)


    return result
