# src/facts/sales/sales_logic/customer_sampling.py

from __future__ import annotations

import math
from typing import Optional

import numpy as np


def _normalize_end_month(end_month_arr, n_customers: int) -> np.ndarray:
    """
    Convert nullable end-month representations into an int64 array with -1 meaning "no end inside window".

    Accepts:
      - None -> all -1
      - numpy array of ints -> returned as int64 (negative treated as -1)
      - pandas Int64 series/object array with pd.NA -> converted to -1

    Notes:
      - This runs on the customer dimension size (typically thousands), not fact rows.
      - Prefer correctness + stability over over-optimizing this path.
    """
    n_customers = int(n_customers)
    if end_month_arr is None:
        return np.full(n_customers, -1, dtype="int64")

    a = np.asarray(end_month_arr)

    # Fast path: numeric (int/bool)
    if np.issubdtype(a.dtype, np.integer):
        out = a.astype("int64", copy=False)
        out = np.where(out < 0, -1, out)
        return out

    # Float path (may contain NaN)
    if np.issubdtype(a.dtype, np.floating):
        out = np.where(np.isnan(a), -1, a).astype("int64")
        out[out < 0] = -1
        return out

    # Object path (may contain None/pd.NA/mixed types)
    if a.dtype == object:
        # Use pandas if available for robust NA handling and faster coercion.
        try:
            import pandas as pd  # optional dependency already in your project

            s = pd.Series(a, copy=False)
            num = pd.to_numeric(s, errors="coerce")
            out = num.fillna(-1).astype("int64").to_numpy()
            out[out < 0] = -1
            return out
        except Exception:
            # Fallback: safe per-element conversion
            out = np.full(n_customers, -1, dtype="int64")
            for i in range(min(n_customers, a.shape[0])):
                v = a[i]
                if v is None:
                    continue
                # handle numpy/pandas NaN-ish
                try:
                    if v is np.nan:  # noqa: E721
                        continue
                except Exception:
                    pass
                try:
                    iv = int(v)
                    out[i] = iv if iv >= 0 else -1
                except Exception:
                    pass
            return out

    # Last resort: try astype int64; coerce negatives to -1
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
    Returns boolean mask over customer dimension rows, true if eligible in this month.

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
    Compute target number of distinct customers to appear in the month.

    Notes:
      - Returns 0 if eligible_count == 0 or n_orders == 0.
      - Always capped by eligible_count and n_orders.
      - Returns at least 1 if both eligible_count and n_orders are positive.
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

    # floor + ratio cap (cap applies to eligible population)
    k = max(k, float(min_k))
    k = min(k, eligible_count * max_ratio)

    # final caps
    k = min(k, float(eligible_count), float(n_orders))

    return int(max(1, round(k)))


# ------------------------------------------------------------
# Sampling helpers
# ------------------------------------------------------------

def _weights_for_keys(keys: np.ndarray, base_weight: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """
    Build probability vector p aligned with `keys` for rng.choice.

    Assumption (existing behavior): CustomerKey is 1..N and base_weight is aligned to that.
    If mapping fails, returns None (uniform sampling).
    """
    if base_weight is None:
        return None

    try:
        keys_i64 = keys.astype("int64", copy=False)
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

    Features:
      - Optional discovery forcing (bring in newly-eligible-but-unseen customers).
      - Optional weighted sampling (customer_base_weight).
      - Optional participation control: target_distinct creates a distinct pool then repeats from it.
    """
    n = int(n)
    if n <= 0:
        return np.empty(0, dtype=np.asarray(customer_keys).dtype)

    customer_keys = np.asarray(customer_keys)
    eligible_mask = np.asarray(eligible_mask, dtype=bool)

    # Eligible pool
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

    # Precompute eligible weights (dimension-aligned) for the common path
    p_eligible = _weights_for_indices(eligible_idx, base_weight)

    # -----------------------------
    # No discovery
    # -----------------------------
    if not use_discovery:
        if k is None:
            return _choice(rng, eligible_keys, n, replace=True, p=p_eligible)

        # distinct pool then repeat
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
    # Build seen array once (vectorized membership)
    if seen_set:
        seen_arr = np.fromiter(seen_set, dtype=eligible_keys.dtype, count=len(seen_set))
        seen_mask = np.isin(eligible_keys, seen_arr, assume_unique=False)
        seen_eligible = eligible_keys[seen_mask]
        undiscovered = eligible_keys[~seen_mask]
    else:
        seen_arr = None
        seen_eligible = np.empty(0, dtype=eligible_keys.dtype)
        undiscovered = eligible_keys

    # Forced undiscovered
    forced = np.empty(0, dtype=eligible_keys.dtype)
    if undiscovered.size > 0:
        discover_n = int(discovery_cfg.get("_target_new_customers", 1))

        max_frac = discovery_cfg.get("max_fraction_per_month")
        if max_frac is not None:
            try:
                max_new = int(float(max_frac) * int(customer_keys.size))
                discover_n = min(discover_n, max_new)
            except Exception:
                pass

        discover_n = max(0, min(discover_n, int(undiscovered.size)))
        if discover_n > 0:
            # Keep forced sampling uniform (matches current behavior)
            forced = rng.choice(undiscovered, size=discover_n, replace=False)

    # ------------------------------------------------------------
    # Discovery without participation target (legacy discovery behavior)
    # ------------------------------------------------------------
    if k is None:
        remaining = n - forced.size
        if remaining <= 0:
            out = forced
            rng.shuffle(out)
            return out

        if seen_set:
            repeat_pool = seen_arr if seen_arr is not None else np.fromiter(seen_set, dtype=eligible_keys.dtype)
        else:
            repeat_pool = eligible_keys

        if repeat_pool.size == 0:
            out = forced
            rng.shuffle(out)
            return out

        p_repeat = _weights_for_keys(repeat_pool, base_weight) if base_weight is not None else None
        repeat = _choice(rng, repeat_pool, remaining, replace=True, p=p_repeat)

        out = np.concatenate([forced, repeat])
        rng.shuffle(out)
        return out

    # ------------------------------------------------------------
    # Participation-controlled discovery:
    # build a month distinct pool of size k, seeded with forced customers.
    # ------------------------------------------------------------
    if forced.size > k:
        forced = rng.choice(forced, size=k, replace=False)

    distinct_pool = forced
    need = k - distinct_pool.size

    if need > 0:
        # Prefer seen_eligible, then undiscovered excluding already forced
        if distinct_pool.size > 0 and undiscovered.size > 0:
            other = undiscovered[~np.isin(undiscovered, distinct_pool, assume_unique=False)]
        else:
            other = undiscovered

        # seen_eligible and undiscovered are disjoint by construction; no need for np.unique
        if seen_eligible.size > 0 and other.size > 0:
            candidates = np.concatenate([seen_eligible, other])
        elif seen_eligible.size > 0:
            candidates = seen_eligible
        else:
            candidates = other

        if candidates.size > 0:
            add_n = min(need, int(candidates.size))
            # Keep distinct fill uniform (matches current behavior)
            extra = rng.choice(candidates, size=add_n, replace=False)
            distinct_pool = np.concatenate([distinct_pool, extra])

    if distinct_pool.size == 0:
        # fallback: draw from eligible
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


__all__ = [
    "_normalize_end_month",
    "_eligible_customer_mask_for_month",
    "_participation_distinct_target",
    "_sample_customers",
]
