# src/facts/sales/sales_logic/customer_sampling.py

import math
import numpy as np


def _normalize_end_month(end_month_arr, n_customers: int) -> np.ndarray:
    """
    Convert nullable end-month representations into an int64 array with -1 meaning "no end inside window".
    Accepts:
      - None -> all -1
      - numpy array of ints -> returned as int64 (negative treated as -1)
      - pandas Int64 series/object array with pd.NA -> converted to -1
    """
    if end_month_arr is None:
        return np.full(n_customers, -1, dtype="int64")

    a = np.asarray(end_month_arr)

    # object arrays may contain pd.NA
    if a.dtype == object:
        out = np.empty(n_customers, dtype="int64")
        for i in range(n_customers):
            v = a[i]
            if v is None or v is np.nan:
                out[i] = -1
            else:
                try:
                    out[i] = int(v)
                except Exception:
                    out[i] = -1
        out[out < 0] = -1
        return out

    # pandas nullable ints often come through as float with nans depending on upstream
    if np.issubdtype(a.dtype, np.floating):
        out = np.where(np.isnan(a), -1, a).astype("int64")
        out[out < 0] = -1
        return out

    out = a.astype("int64", copy=False)
    out[out < 0] = -1
    return out


def _eligible_customer_mask_for_month(
    m_offset: int,
    is_active_in_sales: np.ndarray,
    start_month: np.ndarray,
    end_month_norm: np.ndarray,
) -> np.ndarray:
    """
    Returns boolean mask over customers dimension rows, true if eligible in this month.
    """
    # global gate
    mask = (is_active_in_sales == 1)

    # lifecycle start
    mask &= (start_month <= m_offset)

    # lifecycle end: -1 means no end
    has_end = (end_month_norm >= 0)
    mask &= (~has_end) | (m_offset <= end_month_norm)

    return mask


def _participation_distinct_target(
    rng: np.random.Generator,
    m_offset: int,
    eligible_count: int,
    n_orders: int,
    cfg: dict,
) -> int:
    """
    Compute the target number of distinct customers to appear in a given month.

    models.yaml -> models.customer_participation
      base_distinct_ratio: 0.26
      min_distinct_customers: 250
      max_distinct_ratio: 0.55
      cycles:
        enabled: true
        period_months: 24
        amplitude: 0.35
        phase: 0.0
        noise_std: 0.08

    Notes:
      - Returns 0 if eligible_count == 0 or n_orders == 0.
      - Always capped by eligible_count and n_orders.
      - Intended to shape *distinct-customer participation* independently from macro_demand row allocation.
    """
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

        # Keep sane bounds so we don't get negative/huge distinct targets
        mult = float(np.clip(mult, 0.05, 3.0))
        k *= mult

    # hard floor / cap (ratio cap applies to eligible population)
    k = max(k, float(min_k))
    k = min(k, eligible_count * max_ratio)

    # final caps
    k = min(k, float(eligible_count), float(n_orders))

    return int(max(1, round(k)))


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
    Returns an array of CustomerKeys of length n, sampling from eligible customers.

    Features:
      - Optional discovery forcing (bring in newly-eligible-but-unseen customers).
      - Optional weighted repeat sampling (customer_base_weight).
      - Optional participation control: target_distinct enforces a target number of distinct customers
        to appear in the month, then fills remaining orders with repeats from that distinct pool.

    If use_discovery is True:
      - forces a slice of newly-eligible-but-unseen customers to appear
      - (then) fills the remainder with repeat customers from seen_set (or from eligible if empty)
        unless target_distinct is provided, in which case repeats are drawn from the month distinct pool.
    """
    eligible_keys = customer_keys[eligible_mask]
    if eligible_keys.size == 0 or n <= 0:
        return np.empty(0, dtype=customer_keys.dtype)

    # Normalize target distinct
    if target_distinct is not None:
        try:
            k = int(target_distinct)
        except Exception:
            k = None
        else:
            k = max(1, min(k, int(eligible_keys.size), int(n)))
    else:
        k = None

    # Helper: weighted choice without replacement
    def _choice_unique(keys: np.ndarray, size: int) -> np.ndarray:
        if size <= 0:
            return np.empty(0, dtype=keys.dtype)
        if base_weight is None:
            return rng.choice(keys, size=size, replace=False)
        try:
            # assumes CustomerKey 1..N
            idx = (keys.astype("int64") - 1)
            ww = base_weight[idx].astype("float64", copy=False)
            ww = np.clip(ww, 1e-12, None)
            p = ww / ww.sum()
            return rng.choice(keys, size=size, replace=False, p=p)
        except Exception:
            return rng.choice(keys, size=size, replace=False)

    # Helper: sample repeats (with replacement) from a pool, optionally weighted by base_weight
    def _choice_repeat(keys: np.ndarray, size: int) -> np.ndarray:
        if size <= 0:
            return np.empty(0, dtype=keys.dtype)
        if base_weight is None:
            return rng.choice(keys, size=size, replace=True)
        try:
            idx = (keys.astype("int64") - 1)
            ww = base_weight[idx].astype("float64", copy=False)
            ww = np.clip(ww, 1e-12, None)
            p = ww / ww.sum()
            return rng.choice(keys, size=size, replace=True, p=p)
        except Exception:
            return rng.choice(keys, size=size, replace=True)

    # -----------------------------
    # No discovery: simple sampling
    # -----------------------------
    if not use_discovery:
        if k is None:
            # legacy behavior
            if base_weight is not None:
                w = base_weight[eligible_mask].astype("float64", copy=False)
                w = np.clip(w, 1e-12, None)
                p = w / w.sum()
                return rng.choice(eligible_keys, size=n, replace=True, p=p)
            return rng.choice(eligible_keys, size=n, replace=True)

        # participation-controlled: build a distinct pool then repeat from it
        distinct_pool = _choice_unique(eligible_keys, size=k)
        remaining = int(n - distinct_pool.size)
        if remaining <= 0:
            out = distinct_pool
            rng.shuffle(out)
            return out

        repeats = _choice_repeat(distinct_pool, size=remaining)
        out = np.concatenate([distinct_pool, repeats])
        rng.shuffle(out)
        return out

    # -----------------------------
    # Discovery mode
    # -----------------------------
    # Determine undiscovered among eligible
    if seen_set:
        seen_arr = np.fromiter(seen_set, dtype=eligible_keys.dtype)
        undiscovered_mask = ~np.isin(eligible_keys, seen_arr, assume_unique=False)
        undiscovered = eligible_keys[undiscovered_mask]
        seen_eligible = eligible_keys[~undiscovered_mask]
    else:
        undiscovered = eligible_keys
        seen_eligible = np.empty(0, dtype=eligible_keys.dtype)

    forced = np.empty(0, dtype=customer_keys.dtype)

    if undiscovered.size > 0:
        # NOTE: discover_n is driven by chunk_builder via discovery_cfg["_target_new_customers"]
        discover_n = int(discovery_cfg.get("_target_new_customers", 1))

        # --- HARD CAP: prevent early discovery spike ---
        max_frac = discovery_cfg.get("max_fraction_per_month")
        if max_frac is not None:
            max_new = int(max_frac * customer_keys.size)
            discover_n = min(discover_n, max_new)

        forced = rng.choice(
            undiscovered,
            size=min(discover_n, undiscovered.size),
            replace=False,
        )

    # If no participation target: keep legacy discovery behavior
    if k is None:
        remaining = max(0, n - forced.size)
        if remaining <= 0:
            out = forced
            rng.shuffle(out)
            return out

        # Repeat sampling: prefer seen customers if any, else eligible
        if seen_set:
            repeat_pool = np.fromiter(seen_set, dtype=customer_keys.dtype)
        else:
            repeat_pool = eligible_keys

        if repeat_pool.size == 0:
            out = forced
            rng.shuffle(out)
            return out

        # weighted repeats if possible
        if base_weight is not None and seen_set:
            try:
                idx = (repeat_pool.astype("int64") - 1)
                ww = base_weight[idx].astype("float64", copy=False)
                ww = np.clip(ww, 1e-12, None)
                pp = ww / ww.sum()
                repeat = rng.choice(repeat_pool, size=remaining, replace=True, p=pp)
            except Exception:
                repeat = rng.choice(repeat_pool, size=remaining, replace=True)
        else:
            repeat = rng.choice(repeat_pool, size=remaining, replace=True)

        out = np.concatenate([forced, repeat])
        rng.shuffle(out)
        return out

    # Participation-controlled discovery:
    # Build the month distinct pool of size k, seeded with forced undiscovered customers.
    if forced.size > k:
        forced = rng.choice(forced, size=k, replace=False)

    distinct_pool = forced

    need = int(k - distinct_pool.size)
    if need > 0:
        # fill remaining distinct slots: prefer seen eligible first, then undiscovered
        fill_candidates = []
        if seen_eligible.size > 0:
            fill_candidates.append(seen_eligible)
        if undiscovered.size > 0:
            # exclude those already forced
            if distinct_pool.size > 0:
                u = undiscovered[~np.isin(undiscovered, distinct_pool, assume_unique=False)]
            else:
                u = undiscovered
            if u.size > 0:
                fill_candidates.append(u)

        if fill_candidates:
            pool = np.unique(np.concatenate(fill_candidates))
            if pool.size > 0:
                add_n = min(need, int(pool.size))
                extra = rng.choice(pool, size=add_n, replace=False)
                distinct_pool = np.concatenate([distinct_pool, extra])

    # If we still don't have enough distinct customers (tiny eligible), just use what we have.
    if distinct_pool.size == 0:
        return rng.choice(eligible_keys, size=n, replace=True)

    # Guarantee at least one order per distinct customer, then repeat from distinct pool
    remaining = int(n - distinct_pool.size)
    if remaining <= 0:
        out = distinct_pool
        rng.shuffle(out)
        return out

    repeats = _choice_repeat(distinct_pool, size=remaining)
    out = np.concatenate([distinct_pool, repeats])
    rng.shuffle(out)
    return out


__all__ = [
    "_normalize_end_month",
    "_eligible_customer_mask_for_month",
    "_participation_distinct_target",
    "_sample_customers",
]
