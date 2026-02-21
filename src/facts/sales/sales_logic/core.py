"""Sales core generation logic.

Pure(ish) numpy routines shared by the sales chunk builder.
"""

from __future__ import annotations

from .globals import State

import numpy as np


try:
    import pyarrow as pa  # type: ignore
except Exception:
    pa = None

PA_AVAILABLE = pa is not None

def fmt(dt):
    """
    Format datetime64[D] as YYYYMMDD string array (fast path).
    Accepts scalar or array-like.
    """
    d = np.asarray(dt).astype("datetime64[D]", copy=False)
    y = d.astype("datetime64[Y]").astype("int64") + 1970
    m = (
        d.astype("datetime64[M]").astype("int64")
        - d.astype("datetime64[Y]").astype("datetime64[M]").astype("int64")
        + 1
    )
    day = (
        d.astype("datetime64[D]").astype("int64")
        - d.astype("datetime64[M]").astype("datetime64[D]").astype("int64")
        + 1
    )
    return (y * 10000 + m * 100 + day).astype("U8")


_C1 = np.uint64(0x9E3779B97F4A7C15)
_C2 = np.uint64(0xBF58476D1CE4E5B9)
_C3 = np.uint64(0x94D049BB133111EB)
_MASK63 = np.uint64(0x7FFFFFFFFFFFFFFF)

def _mix_u64(x: np.ndarray) -> np.ndarray:
    # SplitMix64-style mixing (vectorized)
    x ^= (x >> np.uint64(30))
    x *= _C2
    x ^= (x >> np.uint64(27))
    x *= _C3
    x ^= (x >> np.uint64(31))
    return x

def _stable_row_hash(order_dates: np.ndarray, product_keys: np.ndarray) -> np.ndarray:
    """
    Deterministic row hash used when skip_order_cols=True (order_ids_int is None).
    IMPORTANT: uses uint64 ops to avoid numpy float/object promotion and overflows.
    """
    d = np.asarray(order_dates).astype("datetime64[D]").astype("int64", copy=False).astype(np.uint64, copy=False)
    p = np.asarray(product_keys).astype("int64", copy=False).astype(np.uint64, copy=False)

    x = d * _C1
    x ^= (p + _C2)
    x = _mix_u64(x)

    # Return signed int64 in [0, 2^63-1]
    return (x & _MASK63).astype(np.int64, copy=False)


def compute_dates(rng, n, product_keys, order_ids_int, order_dates):
    """
    Compute due dates, delivery dates, delivery status, and order delay flag.

    Supports:
    - order_ids_int present  → order-level coherent behavior
    - order_ids_int is None → row-level fallback (skip_order_cols=True)

    Returns dict of numpy arrays:
      due_date: datetime64[D]
      delivery_date: datetime64[D]
      delivery_status: fixed-width unicode (U15)
      is_order_delayed: int8
    """
    n = int(n)
    if n <= 0:
        return {
            "due_date": np.empty(0, dtype="datetime64[D]"),
            "delivery_date": np.empty(0, dtype="datetime64[D]"),
            "delivery_status": np.empty(0, dtype="U15"),
            "is_order_delayed": np.empty(0, dtype=np.int8),
        }

    # Normalize inputs once
    product_keys = np.asarray(product_keys, dtype=np.int64)
    order_dates = np.asarray(order_dates).astype("datetime64[D]", copy=False)

    has_orders = order_ids_int is not None

    if has_orders:
        order_ids_int = np.asarray(order_ids_int, dtype=np.int64)

        # Map rows → order index (order-level coherence)
        unique_orders, inv_idx = np.unique(order_ids_int, return_inverse=True)

        # Order-level hash expanded to rows
        hash_vals = unique_orders.astype(np.int64, copy=False)[inv_idx]
    else:
        # Deterministic per-row hash without consuming RNG
        hash_vals = _stable_row_hash(order_dates, product_keys)

    # ------------------------------------------------------------
    # Due dates: 3..7 days after order date
    # ------------------------------------------------------------
    # (hash % 5) in [0..4] -> +3 in [3..7]
    due_offset = (hash_vals % 5) + 3
    due_date = order_dates + due_offset.astype("timedelta64[D]")

    # ------------------------------------------------------------
    # Seeds (vectorized) - reuse modular reductions
    # ------------------------------------------------------------
    # Keep semantics equivalent to original:
    # order_seed = hash % 100
    # product_seed = (hash + product_keys) % 100
    # line_seed = (product_keys + (hash % 100)) % 100
    # Note: product_seed == line_seed under mod 100; compute once.
    hs = hash_vals % 100
    pk = product_keys % 100
    order_seed = hs
    product_seed = (hs + pk) % 100
    line_seed = product_seed  # same under mod 100

    # ------------------------------------------------------------
    # Base delivery offset (relative to due date)
    # ------------------------------------------------------------
    delivery_offset = np.zeros(n, dtype=np.int64)

    # Condition C: small delay (1..4)
    mask_c = (order_seed >= 60) & (order_seed < 85) & (product_seed >= 60)
    if mask_c.any():
        delivery_offset[mask_c] = (line_seed[mask_c] % 4) + 1

    # Condition D: larger delay (2..6)
    mask_d = order_seed >= 85
    if mask_d.any():
        delivery_offset[mask_d] = (product_seed[mask_d] % 5) + 2

    # ------------------------------------------------------------
    # Early deliveries
    #   - Order-level when we have order ids
    #   - Row-level otherwise
    # NOTE: keep RNG draw shapes consistent with original to avoid
    # shifting downstream randomness consumption.
    # ------------------------------------------------------------
    if has_orders:
        n_orders = len(unique_orders)

        # One early flag per order (10%)
        early_order = rng.random(n_orders) < 0.10
        # Early days per order: 1..2
        early_days_per_order = rng.integers(1, 3, size=n_orders, dtype=np.int64)

        early_mask = early_order[inv_idx]
        if early_mask.any():
            early_days_rows = early_days_per_order[inv_idx]
            # Early overrides delay (consistent with original behavior)
            delivery_offset[early_mask] = -early_days_rows[early_mask]
    else:
        early_mask = rng.random(n) < 0.10
        if early_mask.any():
            early_days = rng.integers(1, 3, size=n, dtype=np.int64)
            delivery_offset[early_mask] = -early_days[early_mask]

    # Final delivery date
    delivery_date = due_date + delivery_offset.astype("timedelta64[D]")

    # ------------------------------------------------------------
    # Delivery status (use delivery_offset; avoids datetime compares)
    # ------------------------------------------------------------
    # 0 = On Time, 1 = Early, 2 = Delayed
    codes = np.zeros(n, dtype=np.int8)
    codes[delivery_offset < 0] = 1
    codes[delivery_offset > 0] = 2
    labels = np.array(["On Time", "Early Delivery", "Delayed"], dtype="U15")
    delivery_status = labels[codes]

    # ------------------------------------------------------------
    # Order delayed flag (order-level coherence when has order ids)
    # ------------------------------------------------------------
    delayed_line = delivery_offset > 0

    if has_orders:
        # Any delayed line → order delayed
        delayed_any = (
            np.bincount(inv_idx, weights=delayed_line.astype(np.int8), minlength=len(unique_orders)) > 0
        )
        is_order_delayed = delayed_any[inv_idx].astype(np.int8, copy=False)
    else:
        is_order_delayed = delayed_line.astype(np.int8, copy=False)

    return {
        "due_date": due_date,
        "delivery_date": delivery_date,
        "delivery_status": delivery_status,
        "is_order_delayed": is_order_delayed,
    }


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



import math
from typing import Optional

import numpy as np


def _normalize_end_month(end_month_arr, n_customers: int) -> np.ndarray:
    """
    Convert nullable end-month representations into an int64 array with -1 meaning "no end inside window".
    """
    n_customers = int(n_customers)
    if end_month_arr is None:
        return np.full(n_customers, -1, dtype="int64")

    a = np.asarray(end_month_arr)

    if np.issubdtype(a.dtype, np.integer):
        out = a.astype("int64", copy=False)
        out = np.where(out < 0, -1, out)
        return out

    if np.issubdtype(a.dtype, np.floating):
        out = np.where(np.isnan(a), -1, a).astype("int64")
        out[out < 0] = -1
        return out

    if a.dtype == object:
        try:
            import pandas as pd

            s = pd.Series(a, copy=False)
            num = pd.to_numeric(s, errors="coerce")
            out = num.fillna(-1).astype("int64").to_numpy()
            out[out < 0] = -1
            return out
        except Exception:
            out = np.full(n_customers, -1, dtype="int64")
            for i in range(min(n_customers, a.shape[0])):
                v = a[i]
                if v is None:
                    continue
                try:
                    iv = int(v)
                    out[i] = iv if iv >= 0 else -1
                except Exception:
                    pass
            return out

    try:
        out = a.astype("int64", copy=False)
        out[out < 0] = -1
        return out
    except Exception:
        return np.full(n_customers, -1, dtype="int64")


def _eligible_customer_mask_for_month(
    m_offset: int,
    is_active_in_sales: np.ndarray,
    start_month: np.ndarray,
    end_month_norm: np.ndarray,
) -> np.ndarray:
    """
    Eligibility:
      active == 1
      start_month <= m_offset
      end_month == -1 OR m_offset <= end_month
    """
    m = int(m_offset)
    is_active_in_sales = np.asarray(is_active_in_sales, dtype="int64", order="C")
    start_month = np.asarray(start_month, dtype="int64", order="C")
    end_month_norm = np.asarray(end_month_norm, dtype="int64", order="C")

    return (
        (is_active_in_sales == 1)
        & (start_month <= m)
        & ((end_month_norm < 0) | (m <= end_month_norm))
    )


def _participation_distinct_target(
    rng: np.random.Generator,
    m_offset: int,
    eligible_count: int,
    n_orders: int,
    cfg: dict,
) -> int:
    """
    Target number of distinct customers to appear in the month.
    """
    eligible_count = int(eligible_count)
    n_orders = int(n_orders)
    if eligible_count <= 0 or n_orders <= 0:
        return 0

    base_ratio = float(cfg.get("base_distinct_ratio", 0.0))
    min_k = int(cfg.get("min_distinct_customers", 0))
    max_ratio = float(cfg.get("max_distinct_ratio", 1.0))

    k = eligible_count * base_ratio

    cycles_cfg = cfg.get("cycles", {}) or {}
    if bool(cycles_cfg.get("enabled", False)):
        period = int(cycles_cfg.get("period_months", 24))
        amp = float(cycles_cfg.get("amplitude", 0.0))
        phase = float(cycles_cfg.get("phase", 0.0))
        noise_std = float(cycles_cfg.get("noise_std", 0.0))

        cyc = math.sin((2.0 * math.pi * float(m_offset) / max(period, 1)) + phase)
        mult = 1.0 + (amp * cyc)
        if noise_std > 0:
            mult += float(rng.normal(loc=0.0, scale=noise_std))
        mult = float(np.clip(mult, 0.05, 3.0))
        k *= mult

    k = max(k, float(min_k))
    k = min(k, eligible_count * max_ratio)
    k = min(k, float(eligible_count), float(n_orders))

    return int(max(1, round(k)))


# ------------------------------------------------------------
# Sampling helpers
# ------------------------------------------------------------

def _weights_for_indices(indices: np.ndarray, base_weight: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Build probability vector p aligned with a subset of dimension indices.
    This path is correct even if CustomerKey isn't dense/sequential.
    """
    if base_weight is None:
        return None
    try:
        w = base_weight[np.asarray(indices, dtype="int64")].astype("float64", copy=False)
        w = np.where(np.isfinite(w), w, 0.0)
        w = np.clip(w, 1e-12, None)
        s = float(w.sum())
        if s <= 0.0:
            return None
        return w / s
    except Exception:
        return None


def _weights_for_keys(keys: np.ndarray, base_weight: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Assumes CustomerKey is 1..N aligned to base_weight.
    If mapping fails, returns None (uniform sampling).
    """
    if base_weight is None:
        return None
    try:
        keys_i64 = np.asarray(keys).astype("int64", copy=False)
        idx = keys_i64 - 1
        if idx.size == 0:
            return None
        if idx.min() < 0 or idx.max() >= base_weight.shape[0]:
            return None
        w = base_weight[idx].astype("float64", copy=False)
        w = np.where(np.isfinite(w), w, 0.0)
        w = np.clip(w, 1e-12, None)
        s = float(w.sum())
        if s <= 0.0:
            return None
        return w / s
    except Exception:
        return None


def _choice(
    rng: np.random.Generator,
    keys: np.ndarray,
    size: int,
    *,
    replace: bool,
    p: Optional[np.ndarray],
) -> np.ndarray:
    if size <= 0:
        return np.empty(0, dtype=keys.dtype)
    if p is None:
        return rng.choice(keys, size=size, replace=replace)
    return rng.choice(keys, size=size, replace=replace, p=p)


def _sample_customers(
    rng: np.random.Generator,
    customer_keys: np.ndarray,
    eligible_mask: np.ndarray,
    seen_set: set,
    n: int,
    use_discovery: bool,
    discovery_cfg: dict,
    base_weight: np.ndarray | None = None,
    target_distinct: int | None = None,
) -> np.ndarray:
    """
    Returns array of CustomerKeys of length n, sampled from eligible customers.

    - If use_discovery: forces a slice of newly-eligible-but-unseen customers to appear.
    - If target_distinct is provided: builds a distinct pool then repeats from it.
    """
    n = int(n)
    if n <= 0:
        return np.empty(0, dtype=np.asarray(customer_keys).dtype)

    customer_keys = np.asarray(customer_keys)
    eligible_mask = np.asarray(eligible_mask, dtype=bool)

    eligible_idx = np.flatnonzero(eligible_mask)
    if eligible_idx.size == 0:
        return np.empty(0, dtype=customer_keys.dtype)

    eligible_keys = customer_keys[eligible_idx]
    if eligible_keys.size == 0:
        return np.empty(0, dtype=customer_keys.dtype)

    # Normalize target distinct
    k = None
    if target_distinct is not None:
        try:
            k0 = int(target_distinct)
            k = max(1, min(k0, int(eligible_keys.size), n))
        except Exception:
            k = None

    # Precompute eligible weights (dimension-aligned)
    p_eligible = _weights_for_indices(eligible_idx, base_weight)

    # -----------------------------
    # No discovery
    # -----------------------------
    if not use_discovery:
        if k is None:
            return _choice(rng, eligible_keys, n, replace=True, p=p_eligible)

        distinct_pool = _choice(rng, eligible_keys, k, replace=False, p=p_eligible)
        remaining = n - distinct_pool.size
        if remaining <= 0:
            out = distinct_pool
            rng.shuffle(out)
            return out

        p_distinct = _weights_for_keys(distinct_pool, base_weight)
        repeats = _choice(rng, distinct_pool, remaining, replace=True, p=p_distinct)

        out = np.concatenate([distinct_pool, repeats])
        rng.shuffle(out)
        return out

    # -----------------------------
    # Discovery mode
    # -----------------------------
    if seen_set:
        # Faster + memory-light than np.isin(seen_arr) for large seen_set
        seen_mask = np.fromiter((k in seen_set for k in eligible_keys), dtype=bool, count=eligible_keys.size)
        seen_eligible = eligible_keys[seen_mask]
        seen_eligible_idx = eligible_idx[seen_mask]

        undiscovered = eligible_keys[~seen_mask]
        undiscovered_idx = eligible_idx[~seen_mask]
    else:
        seen_eligible = np.empty(0, dtype=eligible_keys.dtype)
        seen_eligible_idx = np.empty(0, dtype="int64")
        undiscovered = eligible_keys
        undiscovered_idx = eligible_idx

    forced = np.empty(0, dtype=eligible_keys.dtype)
    if undiscovered.size > 0:
        discover_n = int(discovery_cfg.get("_target_new_customers", 1))

        # Add variance around the target (seed-deterministic but not “flat”).
        # Default True: restores the older jaggedness without requiring config edits.
        if bool(discovery_cfg.get("stochastic_discovery", True)) and discover_n > 0:
            discover_n = int(rng.poisson(lam=float(discover_n)))

        # Cap (apply to eligible pool so it behaves consistently with lifecycle eligibility)
        max_frac = discovery_cfg.get("max_fraction_per_month")
        if max_frac is not None:
            try:
                max_new = int(float(max_frac) * int(eligible_keys.size))
                discover_n = min(discover_n, max_new)
            except Exception:
                pass

        discover_n = max(0, min(discover_n, int(undiscovered.size)))
        if discover_n > 0:
            # Keep forced uniform (fast and keeps “new customer” mix broad)
            forced = rng.choice(undiscovered, size=discover_n, replace=False)

    # ------------------------------------------------------------
    # Discovery without participation target
    # ------------------------------------------------------------
    if k is None:
        remaining = n - forced.size
        if remaining <= 0:
            out = forced
            rng.shuffle(out)
            return out

        # Repeat from seen *eligible this month* (NOT global seen)
        if seen_eligible.size > 0:
            repeat_pool = seen_eligible
            p_repeat = _weights_for_indices(seen_eligible_idx, base_weight)
        else:
            repeat_pool = eligible_keys
            p_repeat = p_eligible

        repeat = _choice(rng, repeat_pool, remaining, replace=True, p=p_repeat)

        out = np.concatenate([forced, repeat])
        rng.shuffle(out)
        return out

    # ------------------------------------------------------------
    # Participation-controlled discovery
    # ------------------------------------------------------------
    if forced.size > k:
        forced = rng.choice(forced, size=k, replace=False)

    distinct_pool = forced
    need = k - distinct_pool.size

    if need > 0:
        # Fill with seen eligible first, then other undiscovered excluding already forced
        other = undiscovered
        if distinct_pool.size > 0 and other.size > 0:
            other = other[~np.isin(other, distinct_pool, assume_unique=False)]

        if seen_eligible.size > 0 and other.size > 0:
            candidates = np.concatenate([seen_eligible, other])
        elif seen_eligible.size > 0:
            candidates = seen_eligible
        else:
            candidates = other

        if candidates.size > 0:
            add_n = min(need, int(candidates.size))
            extra = rng.choice(candidates, size=add_n, replace=False)
            distinct_pool = np.concatenate([distinct_pool, extra])

    if distinct_pool.size == 0:
        return _choice(rng, eligible_keys, n, replace=True, p=p_eligible)

    remaining = n - distinct_pool.size
    if remaining <= 0:
        out = distinct_pool
        rng.shuffle(out)
        return out

    p_distinct = _weights_for_keys(distinct_pool, base_weight)
    repeats = _choice(rng, distinct_pool, remaining, replace=True, p=p_distinct)

    out = np.concatenate([distinct_pool, repeats])
    rng.shuffle(out)
    return out




import numpy as np


def _as_date32(x):
    """
    Normalize incoming date arrays to datetime64[D] for consistent comparisons.
    """
    if x is None:
        return None
    a = np.asarray(x)
    if a.size == 0:
        return a
    return a.astype("datetime64[D]", copy=False)


def _safe_clip_pct(p):
    p = np.asarray(p, dtype=np.float64)
    p = np.where(np.isfinite(p), p, 0.0)
    return np.clip(p, 0.0, 1.0)


def _sanitize_weights(promo_weight_all, promo_valid_glob):
    """
    Returns:
      weights: float64 array aligned with promo_keys_all, or None if unusable
    """
    if promo_weight_all is None:
        return None

    w = np.asarray(promo_weight_all, dtype=np.float64)
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.maximum(w, 0.0)
    w[~promo_valid_glob] = 0.0

    return w if w.sum() > 0.0 else None


def _group_by_inverse(inv: np.ndarray):
    """
    Efficient grouping for inv codes:
      inv: int array (len n) with values in [0..U-1]
    Yields tuples (code, row_idx_array)
    """
    order = np.argsort(inv, kind="stable")
    inv_sorted = inv[order]
    if inv_sorted.size == 0:
        return

    # boundaries where code changes
    cuts = np.flatnonzero(inv_sorted[1:] != inv_sorted[:-1]) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, inv_sorted.size]

    for s, e in zip(starts, ends):
        code = int(inv_sorted[int(s)])
        yield code, order[int(s):int(e)]


def apply_promotions(
    rng,
    n,
    order_dates,
    promo_keys_all,
    promo_pct_all,
    promo_start_all,
    promo_end_all,
    no_discount_key=1,
    promo_weight_all=None,
):
    """
    Assign at most one promotion per row.

    Default behavior (matches previous):
      - Uniform random choice among active promotions for that row.

    Optional:
      - promo_weight_all: 1D array aligned with promo_keys_all (non-negative weights)
        enables weighted choice among active promos.

    Returns:
      promo_keys: int64 array (len n)
      promo_pct:  float64 array (len n), clipped to [0, 1]
    """
    n = int(n)
    promo_keys = np.full(n, int(no_discount_key), dtype=np.int64)
    promo_pct = np.zeros(n, dtype=np.float64)

    if n <= 0:
        return promo_keys, promo_pct
    if promo_keys_all is None:
        return promo_keys, promo_pct

    promo_keys_all = np.asarray(promo_keys_all, dtype=np.int64)
    P = int(promo_keys_all.size)
    if P == 0:
        return promo_keys, promo_pct

    # Clip promo pct
    promo_pct_all = _safe_clip_pct(promo_pct_all)

    promo_start_all = _as_date32(promo_start_all)
    promo_end_all = _as_date32(promo_end_all)
    order_dates = _as_date32(order_dates)

    if promo_start_all is None or promo_end_all is None or order_dates is None:
        return promo_keys, promo_pct
    if order_dates.shape[0] != n:
        raise ValueError("order_dates length must match n")
    if promo_start_all.shape[0] != P or promo_end_all.shape[0] != P or promo_pct_all.shape[0] != P:
        raise ValueError("promo_*_all arrays must align with promo_keys_all length")

    # Exclude the no-discount key globally (we will fill default with no_discount_key)
    promo_valid_glob = (promo_keys_all != int(no_discount_key))
    if not promo_valid_glob.any():
        return promo_keys, promo_pct

    weights = _sanitize_weights(promo_weight_all, promo_valid_glob)
    if promo_weight_all is not None and weights is None:
        # weights provided but unusable => treat as uniform
        weights = None

    # ------------------------------------------------------------
    # Group rows by unique date (fast path for month-sliced generation)
    # ------------------------------------------------------------
    unique_dates, inv = np.unique(order_dates, return_inverse=True)

    # Precompute active promos per unique date: U x P comparisons
    # U is typically <= 31 when month-sliced; safe & fast.
    # active_u: list of index arrays of promos active that day
    active_u = []
    for d in unique_dates:
        active = promo_valid_glob & (d >= promo_start_all) & (d <= promo_end_all)
        idx = np.nonzero(active)[0]
        active_u.append(idx)

    # Assign per-group (date)
    for code, rows in _group_by_inverse(inv):
        idx = active_u[code]
        if idx.size == 0:
            continue

        if weights is None:
            # uniform among active promos
            chosen = idx[rng.integers(0, idx.size, size=rows.size)]
        else:
            # weighted among active promos, vectorized via CDF + searchsorted
            w = weights[idx]
            s = float(w.sum())
            if s <= 0.0:
                chosen = idx[rng.integers(0, idx.size, size=rows.size)]
            else:
                cdf = np.cumsum(w, dtype=np.float64)
                cdf /= cdf[-1]
                u = rng.random(rows.size)
                j = np.searchsorted(cdf, u, side="right")
                chosen = idx[np.minimum(j, idx.size - 1)]

        promo_keys[rows] = promo_keys_all[chosen]
        promo_pct[rows] = promo_pct_all[chosen]

    return promo_keys, promo_pct


import numpy as np
# ---------------------------------------------------------------------
# Caching (models_cfg is stable during a run; avoid re-parsing every call)
# ---------------------------------------------------------------------
_MD_CACHE_KEY = None
_MD_CACHE_VAL = None


def _to_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def _parse_bands(bands, default):
    """
    bands: list[dict] {max, step} -> sorted list[(max, step)]
    """
    out = []
    if isinstance(bands, list):
        for b in bands:
            if not isinstance(b, dict):
                continue
            mx = _to_float(b.get("max"), None)
            st = _to_float(b.get("step"), None)
            if mx is None or st is None or mx <= 0 or st <= 0:
                continue
            out.append((float(mx), float(st)))

    if not out:
        return list(default)

    out.sort(key=lambda t: t[0])
    return out


def _cfg_markdown():
    """
    Reads State.models_cfg.pricing.markdown.

    Returns:
      enabled: bool
      kind_codes: np.int8 array (0=none, 1=pct, 2=amt)
      values: np.float64 array (pct in [0,1], amt >=0)
      probs: np.float64 array (normalized)
      max_pct: float in [0,1]
      min_net: float >= 0
      allow_neg_margin: bool
      quantize_discount: bool
      discount_rounding: "floor"|"nearest"
      band_max: np.float64 array sorted asc
      band_step: np.float64 array aligned to band_max
    """
    global _MD_CACHE_KEY, _MD_CACHE_VAL

    models = getattr(State, "models_cfg", None) or {}
    key = id(models)
    if _MD_CACHE_KEY == key and _MD_CACHE_VAL is not None:
        return _MD_CACHE_VAL

    pricing = models.get("pricing", {}) or {}
    md = pricing.get("markdown", {}) or {}

    enabled = bool(md.get("enabled", True))

    ladder = md.get("ladder")
    if not isinstance(ladder, list) or len(ladder) == 0:
        ladder = [
            {"kind": "none", "value": 0.0,  "weight": 0.55},
            {"kind": "pct",  "value": 0.05, "weight": 0.20},
            {"kind": "pct",  "value": 0.10, "weight": 0.12},
            {"kind": "pct",  "value": 0.15, "weight": 0.08},
            {"kind": "amt",  "value": 25.0, "weight": 0.05},
        ]

    max_pct = float(md.get("max_pct_of_price", 0.50))
    max_pct = float(np.clip(max_pct, 0.0, 1.0))

    min_net = float(md.get("min_net_price", 0.01))
    min_net = max(0.0, min_net)

    allow_neg_margin = bool(md.get("allow_negative_margin", False))

    # Sanitize ladder into compact arrays
    kind_codes = []
    values = []
    weights = []

    for item in ladder:
        if not isinstance(item, dict):
            continue

        k = str(item.get("kind", "none")).strip().lower()
        w = float(item.get("weight", 0.0) or 0.0)
        if w <= 0:
            continue

        v = float(item.get("value", 0.0) or 0.0)

        if k == "pct":
            kind_codes.append(1)
            values.append(float(np.clip(v, 0.0, 1.0)))
            weights.append(w)
        elif k == "amt":
            kind_codes.append(2)
            values.append(max(0.0, v))
            weights.append(w)
        elif k == "none":
            kind_codes.append(0)
            values.append(0.0)
            weights.append(w)
        else:
            continue

    if not kind_codes:
        kind_codes = [0]
        values = [0.0]
        weights = [1.0]

    probs = np.asarray(weights, dtype=np.float64)
    s = float(probs.sum())
    probs = probs / s if s > 0 else np.array([1.0], dtype=np.float64)

    kind_codes = np.asarray(kind_codes, dtype=np.int8)
    values = np.asarray(values, dtype=np.float64)

    # Appearance
    appearance = md.get("appearance", {}) or {}
    quantize_discount = bool(appearance.get("quantize_discount", True))

    discount_rounding = str(appearance.get("discount_rounding", "floor")).strip().lower()
    if discount_rounding not in ("floor", "nearest"):
        discount_rounding = "floor"

    bands = appearance.get("discount_bands")
    if not isinstance(bands, list) or len(bands) == 0:
        bands = [
            {"max": 50, "step": 0.50},
            {"max": 200, "step": 1},
            {"max": 1000, "step": 5},
            {"max": 5000, "step": 10},
            {"max": 1e18, "step": 25},
        ]

    parsed = _parse_bands(
        bands,
        default=[(50.0, 0.50), (200.0, 1.0), (1000.0, 5.0), (5000.0, 10.0), (1e18, 25.0)],
    )

    band_max = np.asarray([mx for mx, _ in parsed], dtype=np.float64)
    band_step = np.asarray([st for _, st in parsed], dtype=np.float64)
    if band_max.size == 0:
        band_max = np.asarray([1e18], dtype=np.float64)
        band_step = np.asarray([25.0], dtype=np.float64)

    out = (
        enabled,
        kind_codes,
        values,
        probs,
        max_pct,
        min_net,
        allow_neg_margin,
        quantize_discount,
        discount_rounding,
        band_max,
        band_step,
    )

    _MD_CACHE_KEY = key
    _MD_CACHE_VAL = out
    return out


def _as_f64(x, n: int) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.shape[0] != int(n):
        raise ValueError("Array length mismatch")
    # Replace NaN/inf with 0.0 deterministically
    return np.where(np.isfinite(a), a, 0.0)


def _step_for_price(up: np.ndarray, band_max: np.ndarray, band_step: np.ndarray) -> np.ndarray:
    """
    Fast vectorized step lookup: first band where up <= max.
    """
    # band_max is sorted ascending
    idx = np.searchsorted(band_max, up, side="left")
    # If up > last max (shouldn't happen if last max is huge), clamp to last
    idx = np.minimum(idx, band_step.size - 1)
    step = band_step[idx]
    return np.where(step > 0.0, step, 0.01)


def _quantize_discount(
    disc: np.ndarray,
    up: np.ndarray,
    band_max: np.ndarray,
    band_step: np.ndarray,
    rounding: str,
) -> np.ndarray:
    """
    Quantize discount to clean increments chosen per-row based on UnitPrice,
    then apply a ".99" ending (e.g., 5.00 -> 4.99, 10.00 -> 9.99) for demo-friendly visuals.

    Note:
      - 0 stays 0
      - This makes discounts slightly smaller, so it will NOT violate max_pct/min_net constraints
        (those constraints are re-applied after quantization anyway).
    """
    step = _step_for_price(up, band_max, band_step)

    if rounding == "nearest":
        q = np.round(disc / step) * step
    else:
        q = np.floor(disc / step) * step

    # Apply ".99" ending for non-zero discounts
    q = np.where(q > 0.0, np.maximum(q - 0.01, 0.0), 0.0)

    return q


def compute_prices(
    rng,
    n,
    unit_price,
    unit_cost,
    promo_pct=None,  # accepted for backward compatibility; intentionally ignored
    *,
    price_pressure: float = 1.0,
    row_price_jitter_pct: float = 0.0,
):
    """
    Sales pricing rule:
      - UnitPrice/UnitCost come from Products (source of truth).
      - Sales.DiscountAmount is an independent markdown (NOT from Promotions).
      - Promotions affect ONLY PromotionKey; promo discounts are applied at analysis-time.

    Output columns represent "after markdown, before promo".
    """
    _ = promo_pct  # ignored by design

    n = int(n)
    if n <= 0:
        z = np.zeros(0, dtype=np.float64)
        return {"final_unit_price": z, "final_unit_cost": z, "discount_amt": z, "final_net_price": z}

    up = _as_f64(unit_price, n)
    uc = _as_f64(unit_cost, n)

    # Optional global scale
    pp = float(price_pressure) if price_pressure is not None else 1.0
    if not np.isfinite(pp) or pp <= 0.0:
        pp = 1.0
    up *= pp
    uc *= pp

    # Optional per-row jitter (defaults OFF)
    j = float(row_price_jitter_pct) if row_price_jitter_pct is not None else 0.0
    if np.isfinite(j) and j > 0.0:
        mult = rng.uniform(1.0 - j, 1.0 + j, size=n).astype(np.float64, copy=False)
        up *= mult
        uc *= mult

    up = np.maximum(up, 0.0)
    uc = np.maximum(uc, 0.0)

    (
        enabled,
        kind_codes,
        values,
        probs,
        max_pct,
        min_net,
        allow_neg_margin,
        quantize_discount,
        discount_rounding,
        band_max,
        band_step,
    ) = _cfg_markdown()

    disc = np.zeros(n, dtype=np.float64)

    if enabled:
        idx = rng.choice(kind_codes.size, size=n, replace=True, p=probs)

        kc = kind_codes[idx]      # 0/1/2
        v = values[idx]           # pct or amt

        # Vectorized ladder application
        # pct: up * v
        disc = np.where(kc == 1, up * v, disc)
        # amt: v
        disc = np.where(kc == 2, v, disc)
        # none: keep 0

    # Base constraints before quantization
    disc = np.maximum(disc, 0.0)
    disc = np.minimum(disc, up * max_pct)
    if min_net > 0.0:
        disc = np.minimum(disc, np.maximum(up - min_net, 0.0))
    disc = np.minimum(disc, up)

    # Quantize to clean increments
    if enabled and quantize_discount:
        disc = _quantize_discount(disc, up, band_max, band_step, discount_rounding)

        # Re-apply constraints after quantization
        disc = np.maximum(disc, 0.0)
        disc = np.minimum(disc, up * max_pct)
        if min_net > 0.0:
            disc = np.minimum(disc, np.maximum(up - min_net, 0.0))
        disc = np.minimum(disc, up)

    net = np.maximum(up - disc, 0.0)

    # Invariants
    uc = np.minimum(uc, up)
    if not allow_neg_margin:
        # Demo-friendly: avoid negative AND avoid exact break-even after rounding to cents.
        MIN_PROFIT = 0.01  # 1 cent
        uc = np.minimum(uc, np.maximum(net - MIN_PROFIT, 0.0))

    # Round to cents for storage
    up = np.round(up, 2)
    uc = np.round(uc, 2)
    disc = np.round(disc, 2)

    # Post-round safety (avoid rare disc>up due to rounding)
    disc = np.minimum(disc, up)
    if min_net > 0.0:
        disc = np.minimum(disc, np.maximum(up - min_net, 0.0))

    net = np.round(up - disc, 2)
    net = np.maximum(net, 0.0)

    if not allow_neg_margin:
        MIN_PROFIT = 0.01  # 1 cent
        uc = np.minimum(uc, np.maximum(net - MIN_PROFIT, 0.0))

    return {
        "final_unit_price": up,
        "final_unit_cost": uc,
        "discount_amt": disc,
        "final_net_price": net,
    }



import numpy as np


def _sched_mode_and_values(node: dict, name: str) -> tuple[str, list[float]]:
    """
    Validate schedule node:
      { mode: "repeat"|"once", values: [..numbers..] }
    """
    if not isinstance(node, dict):
        raise ValueError(f"{name} must be a mapping with keys: mode, values")

    mode = str(node.get("mode", "repeat")).strip().lower()
    if mode not in ("repeat", "once"):
        raise ValueError(f"{name}.mode must be 'repeat' or 'once'")

    values = node.get("values")
    if not isinstance(values, list) or len(values) == 0:
        raise ValueError(f"{name}.values must be a non-empty list")

    out: list[float] = []
    for v in values:
        try:
            out.append(float(v))
        except Exception as e:
            raise ValueError(f"{name}.values must contain only numbers: {e}") from e

    return mode, out


def _safe_prob(w: np.ndarray) -> np.ndarray:
    """
    Convert an array into a valid probability vector.
    """
    w = np.asarray(w, dtype="float64")
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= 0.0:
        # fallback uniform
        return np.full(w.shape[0], 1.0 / max(1, w.shape[0]), dtype="float64")
    return w / s


def _distribute_remainder_multinomial(
    rng: np.random.Generator,
    base_rows: np.ndarray,
    remainder: int,
    probs: np.ndarray,
) -> np.ndarray:
    """
    Add `remainder` rows across months according to `probs`.
    """
    if remainder <= 0:
        return base_rows
    add = rng.multinomial(int(remainder), _safe_prob(probs))
    return base_rows + add.astype("int64", copy=False)


def macro_month_weights(rng: np.random.Generator, T: int, cfg: dict) -> np.ndarray:
    """
    Create base demand weights per month, independent of eligible customer count.
    Produces a smooth trend + seasonality + optional shocks + noise.

    cfg example (models.yaml -> models.macro_demand):
      base_level: 1.0
      yearly_growth: 0.03               # 3% per year (smooth drift)
      seasonality_amplitude: 0.12       # +/-12%
      seasonality_phase: 0.0            # radians
      noise_std: 0.05                   # month-to-month multiplier noise

      # Optional: pin exact per-year levels (multipliers) OR compound YoY schedule.
      year_level_factors:
        mode: "repeat"
        values: [1.0, 1.02, 0.97, 0.94]
      yoy_growth_schedule:
        mode: "repeat"
        values: [0.06, 0.06, -0.03, -0.05]

      shock_probability: 0.06           # per month
      shock_impact: [-0.35, -0.10]      # multiplicative range (low, high)
    """
    if T <= 0:
        return np.zeros(0, dtype="float64")

    base_level = float(cfg.get("base_level", 1.0))
    yearly_growth = float(cfg.get("yearly_growth", 0.0))
    amp = float(cfg.get("seasonality_amplitude", 0.0))
    phase = float(cfg.get("seasonality_phase", 0.0))
    noise_std = float(cfg.get("noise_std", 0.0))

    shock_p = float(cfg.get("shock_probability", 0.0))
    shock_lo, shock_hi = cfg.get("shock_impact", [-0.25, -0.08])
    shock_lo = float(shock_lo)
    shock_hi = float(shock_hi)

    m = np.arange(T, dtype="float64")

    # ---- yearly drift (baseline) + optional year-pattern schedule ----
    yoy_node = cfg.get("yoy_growth_schedule")
    lvl_node = cfg.get("year_level_factors")
    if yoy_node and lvl_node:
        raise ValueError("Use only one of: yoy_growth_schedule OR year_level_factors")

    year_idx = (m // 12).astype("int64")  # year index per month, relative to dataset start

    # baseline smooth drift: per-month multiplier derived from yearly_growth
    if yearly_growth != 0.0:
        g = (1.0 + yearly_growth) ** (m / 12.0)
    else:
        g = 1.0

    # year-level factors (pin exact per-year levels)
    if lvl_node:
        mode, vals = _sched_mode_and_values(lvl_node, "year_level_factors")
        if any(v <= 0.0 for v in vals):
            raise ValueError("year_level_factors.values must be > 0")

        levels = np.asarray(vals, dtype="float64")
        if mode == "repeat":
            yfac = levels[year_idx % len(levels)]
        else:  # once
            yfac = levels[np.minimum(year_idx, len(levels) - 1)]
        g = g * yfac

    # yoy growth schedule (compounding)
    elif yoy_node:
        mode, vals = _sched_mode_and_values(yoy_node, "yoy_growth_schedule")
        if any(v <= -0.99 for v in vals):
            raise ValueError("yoy_growth_schedule.values must be > -0.99")

        yoy = np.asarray(vals, dtype="float64")
        n_years = int((T + 11) // 12)

        year_factor = np.ones(n_years, dtype="float64")
        for y in range(1, n_years):
            step = y - 1  # transition into year y
            if mode == "repeat":
                r = yoy[step % len(yoy)]
            else:  # once
                r = yoy[step] if step < len(yoy) else 0.0
            year_factor[y] = year_factor[y - 1] * (1.0 + r)

        g = g * year_factor[np.minimum(year_idx, n_years - 1)]

    # seasonality: sin wave (12-month cycle)
    if amp != 0.0:
        s = 1.0 + amp * np.sin((2.0 * np.pi * m / 12.0) + phase)
    else:
        s = 1.0

    # month-to-month noise (kept small)
    if noise_std > 0.0:
        n = rng.normal(loc=1.0, scale=noise_std, size=T)
        n = np.clip(n, 0.5, 1.5)
    else:
        n = 1.0

    # shocks: occasional multiplicative hits
    if shock_p > 0.0:
        shock = np.ones(T, dtype="float64")
        hit = rng.random(T) < shock_p
        if hit.any():
            if shock_lo > shock_hi:
                raise ValueError("shock_impact must be [low, high] with low <= high")
            shock[hit] = 1.0 + rng.uniform(shock_lo, shock_hi, size=int(hit.sum()))
            upper = max(1.0, 1.0 + shock_hi)  # allow positive shocks
            shock = np.clip(shock, 0.1, upper)
    else:
        shock = 1.0

    w = base_level * g * s * n * shock
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.clip(w, 1e-9, None)
    return w / float(w.sum())


def build_rows_per_month(
    *,
    rng: np.random.Generator,
    total_rows: int,
    eligible_counts: np.ndarray,
    macro_cfg: dict | None,
) -> np.ndarray:
    """
    Decide how many rows to generate in each month.

    - If macro_cfg is truthy: uses macro demand weights + optional early_month_cap.
    - Else: falls back to legacy eligible-count proportional allocation.

    Returns: int64 array of length T (months), sum == total_rows (unless total_rows<=0).
    """
    eligible_counts = np.asarray(eligible_counts, dtype="int64")
    T = int(eligible_counts.shape[0])

    if T <= 0:
        return np.zeros(0, dtype="int64")

    total_rows = int(total_rows)
    if total_rows <= 0:
        return np.zeros(T, dtype="int64")

    elig_sum = int(eligible_counts.sum())
    if elig_sum <= 0:
        return np.zeros(T, dtype="int64")

    eligible_nonzero = eligible_counts > 0

    macro_cfg = macro_cfg or {}
    use_macro = bool(macro_cfg)

    # ------------------------------------------------------------------
    # Macro demand allocation
    # ------------------------------------------------------------------
    if use_macro:
        macro_w = macro_month_weights(rng, T, macro_cfg)

        # months with no eligible customers cannot receive demand
        macro_w = macro_w * eligible_nonzero.astype("float64")
        if float(macro_w.sum()) <= 0.0:
            return np.zeros(T, dtype="int64")
        macro_w = macro_w / float(macro_w.sum())

        # Optional: blend macro weights with eligibility weights (0.0 = macro-only)
        blend = float(macro_cfg.get("eligible_blend", 0.0))
        if blend > 0.0:
            blend = float(np.clip(blend, 0.0, 1.0))
            elig_w = eligible_counts.astype("float64")
            elig_w = elig_w * eligible_nonzero.astype("float64")
            if float(elig_w.sum()) > 0.0:
                elig_w = elig_w / float(elig_w.sum())
                macro_w = (1.0 - blend) * macro_w + blend * elig_w
                macro_w = _safe_prob(macro_w)

        # Initial floor allocation
        rows_per_month = np.floor(macro_w * total_rows).astype("int64")

        # Fix rounding: distribute remainder stochastically (seed-deterministic)
        remainder = int(total_rows - int(rows_per_month.sum()))
        rows_per_month = _distribute_remainder_multinomial(rng, rows_per_month, remainder, macro_w)

        # --------------------------------------------------------------
        # Optional: cap by eligible base to avoid absurd early-month density
        # IMPORTANT semantic fix: cap is only applied if block exists.
        # --------------------------------------------------------------
        cap_cfg = macro_cfg.get("early_month_cap", None)
        if isinstance(cap_cfg, dict) and cap_cfg:
            cap_enabled = bool(cap_cfg.get("enabled", True))
            per_customer_cap = int(cap_cfg.get("max_rows_per_customer", 12))
            redistribute = bool(cap_cfg.get("redistribute_excess", True))

            if cap_enabled and per_customer_cap > 0:
                max_rows = eligible_counts * int(per_customer_cap)
                # months with no eligible customers stay at 0 cap
                max_rows = np.where(eligible_nonzero, max_rows, 0).astype("int64", copy=False)

                before = int(rows_per_month.sum())
                capped = np.minimum(rows_per_month, max_rows)
                after = int(capped.sum())
                excess = int(before - after)

                rows_per_month = capped

                if redistribute and excess > 0:
                    # Try to respect capacity; if impossible, relax cap as a last resort
                    capacity = (max_rows - rows_per_month).astype("int64", copy=False)
                    capacity = np.maximum(capacity, 0)

                    # iterative redistribution (small T; stays fast)
                    for _ in range(8):
                        if excess <= 0:
                            break
                        cap_months = np.flatnonzero(capacity > 0)
                        if cap_months.size == 0:
                            break

                        probs = _safe_prob(macro_w[cap_months])
                        add = rng.multinomial(excess, probs).astype("int64", copy=False)

                        # apply add, clamp to capacity
                        add = np.minimum(add, capacity[cap_months])
                        rows_per_month[cap_months] += add
                        capacity[cap_months] -= add
                        excess -= int(add.sum())

                    # Last resort: preserve total_rows even if cap is too tight.
                    # Distribute remaining excess across eligible months without capacity limit.
                    if excess > 0:
                        elig_months = np.flatnonzero(eligible_nonzero)
                        if elig_months.size > 0:
                            probs = _safe_prob(macro_w[elig_months])
                            add = rng.multinomial(excess, probs).astype("int64", copy=False)
                            rows_per_month[elig_months] += add
                            excess = 0

        # Final guard: exact total rows
        diff = int(total_rows - int(rows_per_month.sum()))
        if diff != 0:
            # Adjust stochastically to avoid deterministic bias
            eligible_months = np.flatnonzero(eligible_nonzero)
            if eligible_months.size == 0:
                return rows_per_month

            probs = _safe_prob(macro_w[eligible_months])

            if diff > 0:
                add = rng.multinomial(diff, probs).astype("int64", copy=False)
                rows_per_month[eligible_months] += add
            else:
                # remove -diff rows from months with >0 rows, weighted by probs
                need = -diff
                candidates = eligible_months[rows_per_month[eligible_months] > 0]
                if candidates.size > 0:
                    probs2 = _safe_prob(macro_w[candidates])
                    # sample months to decrement; do it in batches
                    pick = rng.choice(candidates, size=need, replace=True, p=probs2)
                    # bincount over indices in candidates space
                    # map picks to positions 0..len(candidates)-1
                    inv = np.searchsorted(candidates, pick)
                    dec = np.bincount(inv, minlength=candidates.size).astype("int64", copy=False)
                    dec = np.minimum(dec, rows_per_month[candidates])
                    rows_per_month[candidates] -= dec

        return rows_per_month

    # ------------------------------------------------------------------
    # Legacy allocation (eligible-count proportional) + stochastic remainder
    # ------------------------------------------------------------------
    month_weights = eligible_counts.astype("float64") / float(elig_sum)
    month_weights = month_weights * eligible_nonzero.astype("float64")
    month_weights = _safe_prob(month_weights)

    rows = np.floor(month_weights * total_rows).astype("int64")
    remainder = int(total_rows - int(rows.sum()))
    rows = _distribute_remainder_multinomial(rng, rows, remainder, month_weights)

    # exact guard (should already match)
    diff = int(total_rows - int(rows.sum()))
    if diff != 0:
        # apply minimal correction
        eligible_months = np.flatnonzero(eligible_nonzero)
        if eligible_months.size > 0:
            probs = _safe_prob(month_weights[eligible_months])
            if diff > 0:
                add = rng.multinomial(diff, probs).astype("int64", copy=False)
                rows[eligible_months] += add
            else:
                need = -diff
                candidates = eligible_months[rows[eligible_months] > 0]
                if candidates.size > 0:
                    probs2 = _safe_prob(month_weights[candidates])
                    pick = rng.choice(candidates, size=need, replace=True, p=probs2)
                    inv = np.searchsorted(candidates, pick)
                    dec = np.bincount(inv, minlength=candidates.size).astype("int64", copy=False)
                    dec = np.minimum(dec, rows[candidates])
                    rows[candidates] -= dec

    return rows

__all__ = [
    # date logic
    "compute_dates",

    # order logic
    "build_month_demand",
    "build_orders",

    # customer sampling
    "_normalize_end_month",
    "_eligible_customer_mask_for_month",
    "_participation_distinct_target",
    "_sample_customers",

    # promotions
    "apply_promotions",

    # pricing
    "compute_prices",

    # month planning
    "macro_month_weights",
    "build_rows_per_month",
]
