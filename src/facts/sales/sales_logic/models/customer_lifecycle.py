"""
NEW ROLE (post-overhaul):
- Customer existence over time is encoded in customers.parquet:
    IsActiveInSales, CustomerStartMonth, CustomerEndMonth
- This module optionally applies a *monthly activity overlay* on top of that:
    temporary inactivity (dormancy), optional reactivation, mild seasonality/noise

IMPORTANT:
- Do NOT use this module to "grow the customer base" anymore.
  That conflicts with CustomerStartMonth/CustomerEndMonth generated in customers.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from src.facts.sales.sales_logic.globals import State


# -------------------------------------------------------------------
# Config model (safe defaults; all optional)
# -------------------------------------------------------------------
@dataclass(frozen=True)
class ActivityOverlayCfg:
    enabled: bool = False

    # Base probability that an eligible customer is "active" in a given month.
    # If enabled=False, activity mask == eligibility mask.
    base_monthly_activity_rate: float = 0.85

    # Noise applied to base activity rate each month (multiplicative)
    monthly_noise: float = 0.08

    # Seasonality (multiplicative): activity_rate *= 1 + seasonal_amplitude*sin(2πm/period)
    seasonal_amplitude: float = 0.15
    seasonal_period_months: int = 24

    # Reactivation: fraction of inactive customers that can return each month
    monthly_reactivation_rate: float = 0.10

    # Optional: if customer_temperature array exists, scale volatility
    # Effective noise = monthly_noise * (1 + temperature_noise_scale*(temp-1))
    temperature_noise_scale: float = 0.50


def _load_overlay_cfg() -> ActivityOverlayCfg:
    """
    Reads config from State.models_cfg if present.

    Recommended config location (doesn't break old configs):
      customer_activity_overlay:
        enabled: true
        base_monthly_activity_rate: 0.85
        monthly_noise: 0.08
        seasonal_amplitude: 0.15
        seasonal_period_months: 24
        monthly_reactivation_rate: 0.10
        temperature_noise_scale: 0.50

    If absent, returns disabled overlay (no behavior change).
    """
    cfg = (getattr(State, "models_cfg", None) or {})
    block = cfg.get("customer_activity_overlay", {}) or {}
    try:
        return ActivityOverlayCfg(
            enabled=bool(block.get("enabled", False)),
            base_monthly_activity_rate=float(block.get("base_monthly_activity_rate", 0.85)),
            monthly_noise=float(block.get("monthly_noise", 0.08)),
            seasonal_amplitude=float(block.get("seasonal_amplitude", 0.15)),
            seasonal_period_months=int(block.get("seasonal_period_months", 24)),
            monthly_reactivation_rate=float(block.get("monthly_reactivation_rate", 0.10)),
            temperature_noise_scale=float(block.get("temperature_noise_scale", 0.50)),
        )
    except Exception as e:
        raise ValueError(f"Invalid customer_activity_overlay config: {e}") from e


def _normalize_end_month(end_month_arr, n: int) -> np.ndarray:
    """
    Convert nullable end-month representations into an int64 array with -1 meaning "no end".
    """
    n = int(n)
    if end_month_arr is None:
        return np.full(n, -1, dtype=np.int64)

    a = np.asarray(end_month_arr)

    # ints
    if np.issubdtype(a.dtype, np.integer):
        out = a.astype(np.int64, copy=False)
        out = np.where(out < 0, -1, out)
        return out

    # floats (NaN -> -1)
    if np.issubdtype(a.dtype, np.floating):
        out = np.where(np.isnan(a), -1, a).astype(np.int64)
        out[out < 0] = -1
        return out

    # objects / nullable ints (pd.NA/None)
    if a.dtype == object:
        try:
            import pandas as pd

            s = pd.Series(a, copy=False)
            num = pd.to_numeric(s, errors="coerce")
            out = num.fillna(-1).astype("int64").to_numpy()
            out[out < 0] = -1
            return out
        except Exception:
            out = np.full(n, -1, dtype=np.int64)
            lim = min(n, a.shape[0])
            for i in range(lim):
                v = a[i]
                if v is None:
                    continue
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

    # last resort
    try:
        out = a.astype(np.int64, copy=False)
        out[out < 0] = -1
        return out
    except Exception:
        return np.full(n, -1, dtype=np.int64)


# -------------------------------------------------------------------
# Core: Eligibility mask from Customers dimension (authoritative)
# -------------------------------------------------------------------
def build_eligibility_by_month(
    start_month_arr: np.ndarray,
    end_month_arr: Optional[np.ndarray],
    is_active_in_sales_arr: Optional[np.ndarray],
    start_month: int,
    end_month: int,
) -> Dict[int, np.ndarray]:
    """
    Build month -> eligibility mask over customers (aligned by customer row index).

    Eligibility rule:
      - IsActiveInSales == 1 (if provided, else assume all eligible globally)
      - CustomerStartMonth <= m
      - CustomerEndMonth < 0 or missing => no end
        else m <= CustomerEndMonth
    """
    start_month_arr = np.asarray(start_month_arr, dtype=np.int64)
    n = int(start_month_arr.shape[0])

    start_m = int(start_month)
    end_m = int(end_month)
    if end_m < start_m:
        return {}

    if is_active_in_sales_arr is None:
        active_gate = np.ones(n, dtype=bool)
    else:
        active_gate = (np.asarray(is_active_in_sales_arr, dtype=np.int64) == 1)

    end_month_norm = _normalize_end_month(end_month_arr, n)

    # Initial mask for start_m (single pass comparisons)
    mask = active_gate & (start_month_arr <= start_m) & ((end_month_norm < 0) | (start_m <= end_month_norm))

    out: Dict[int, np.ndarray] = {start_m: mask.copy()}

    # Precompute start-events and end-events to update incrementally
    # Starts: indices sorted by start_month
    start_sorted = np.argsort(start_month_arr, kind="mergesort")
    start_vals = start_month_arr[start_sorted]
    ptr_start = int(np.searchsorted(start_vals, start_m + 1, side="left"))

    # Ends: consider only finite ends (>=0); remove after month m if end == m
    finite_end_idx = np.flatnonzero(end_month_norm >= 0)
    if finite_end_idx.size:
        end_vals_f = end_month_norm[finite_end_idx]
        end_sorted = finite_end_idx[np.argsort(end_vals_f, kind="mergesort")]
        end_vals_sorted = end_month_norm[end_sorted]
        ptr_end = int(np.searchsorted(end_vals_sorted, start_m, side="left"))
    else:
        end_sorted = np.empty(0, dtype=np.int64)
        end_vals_sorted = np.empty(0, dtype=np.int64)
        ptr_end = 0

    # Increment month by month
    for m in range(start_m, end_m):
        next_m = m + 1

        # Remove customers whose lifecycle ends at month m
        while ptr_end < end_sorted.size and int(end_vals_sorted[ptr_end]) == m:
            i = int(end_sorted[ptr_end])
            mask[i] = False
            ptr_end += 1

        # Add customers whose lifecycle starts at month next_m
        while ptr_start < start_sorted.size and int(start_vals[ptr_start]) == next_m:
            i = int(start_sorted[ptr_start])
            if active_gate[i] and (end_month_norm[i] < 0 or end_month_norm[i] >= next_m):
                mask[i] = True
            ptr_start += 1

        out[next_m] = mask.copy()

    return out


# -------------------------------------------------------------------
# Optional: Activity overlay (temporary inactivity)
# -------------------------------------------------------------------
def apply_activity_overlay_by_month(
    eligibility_by_month: Dict[int, np.ndarray],
    seed: int,
    customer_temperature: Optional[np.ndarray] = None,
) -> Dict[int, np.ndarray]:
    """
    Applies an optional activity overlay on top of eligibility masks.
    Returns month -> active_mask where active_mask <= eligibility_mask.

    If overlay is disabled, returns eligibility_by_month unchanged.
    """
    cfg = _load_overlay_cfg()
    if not cfg.enabled:
        return eligibility_by_month

    # Validate rates (fail fast)
    if not (0.0 <= cfg.base_monthly_activity_rate <= 1.0):
        raise ValueError("base_monthly_activity_rate must be in [0, 1]")
    if cfg.monthly_noise < 0.0:
        raise ValueError("monthly_noise must be >= 0")
    if cfg.seasonal_period_months <= 0:
        raise ValueError("seasonal_period_months must be > 0")
    if not (0.0 <= cfg.monthly_reactivation_rate <= 1.0):
        raise ValueError("monthly_reactivation_rate must be in [0, 1]")

    rng_global = np.random.default_rng(int(seed) + 9100)

    # Optional temperature (aligned with customers)
    temp = None
    if customer_temperature is not None:
        temp = np.asarray(customer_temperature, dtype=np.float64)
        temp = np.where(np.isfinite(temp), temp, 1.0)
        temp = np.clip(temp, 0.2, 3.0)

    out: Dict[int, np.ndarray] = {}
    prev_active: Optional[np.ndarray] = None

    months = sorted(eligibility_by_month.keys())
    if not months:
        return out

    for m in months:
        elig = np.asarray(eligibility_by_month[m], dtype=bool)
        n = int(elig.shape[0])

        # Month-scoped RNG for month-level jitter (keeps original determinism pattern)
        rng_m = np.random.default_rng(int(seed) + 10000 + int(m) * 7919)

        # Seasonality multiplier
        cycle = np.sin(2.0 * np.pi * (float(m) / float(cfg.seasonal_period_months)))
        seasonal_mult = 1.0 + cfg.seasonal_amplitude * cycle

        # Base activity probability for this month (scalar)
        p = cfg.base_monthly_activity_rate * seasonal_mult
        if cfg.monthly_noise > 0:
            p *= float(rng_m.uniform(1.0 - cfg.monthly_noise, 1.0 + cfg.monthly_noise))
        p = float(np.clip(p, 0.0, 1.0))

        # Per-customer probability (optional temp-driven volatility)
        if temp is not None:
            noise_scale = cfg.monthly_noise * (1.0 + cfg.temperature_noise_scale * (temp - 1.0))
            noise_scale = np.clip(noise_scale, 0.0, 0.50)
            per_cust_p = p * rng_m.uniform(1.0 - noise_scale, 1.0 + noise_scale, size=n)
            per_cust_p = np.clip(per_cust_p, 0.0, 1.0)
        else:
            per_cust_p = p  # scalar

        if prev_active is None:
            u = rng_global.random(n)
            active = elig & (u < per_cust_p)
            out[m] = active
            prev_active = active
            continue

        # Stay active
        u1 = rng_global.random(n)
        active = (prev_active & elig) & (u1 < per_cust_p)

        # Reactivate
        if cfg.monthly_reactivation_rate > 0.0:
            u2 = rng_global.random(n)
            active |= ((~prev_active) & elig) & (u2 < cfg.monthly_reactivation_rate)

        out[m] = active
        prev_active = active

    return out


# -------------------------------------------------------------------
# Backward-compatible API (kept because old code imported it)
# -------------------------------------------------------------------
def build_active_customer_pool(
    all_customers,
    start_month: int,
    end_month: int,
    seed: int,
):
    """
    BACKWARD-COMPAT SHIM.

    New behavior:
    - If State has lifecycle arrays, compute eligibility from them and
      optionally apply an activity overlay.
    - If lifecycle arrays are missing, fall back to "all active" masks
      across months (closest old behavior for compatibility).

    Returns:
        dict[int, np.ndarray]: month_idx -> boolean mask over customers (row-aligned)
    """
    cust_keys = getattr(State, "customer_keys", None)
    start_arr = getattr(State, "customer_start_month", None)
    end_arr = getattr(State, "customer_end_month", None)
    act_arr = getattr(State, "customer_is_active_in_sales", None)
    temp_arr = getattr(State, "customer_temperature", None)  # optional future

    if cust_keys is not None and start_arr is not None:
        elig = build_eligibility_by_month(
            start_month_arr=start_arr,
            end_month_arr=end_arr,
            is_active_in_sales_arr=act_arr,
            start_month=start_month,
            end_month=end_month,
        )
        return apply_activity_overlay_by_month(
            eligibility_by_month=elig,
            seed=seed,
            customer_temperature=temp_arr,
        )

    # Fallback: treat all customers as eligible+active
    all_customers = np.asarray(all_customers, dtype=np.int64)
    n = int(all_customers.shape[0])
    mask = np.ones(n, dtype=bool)
    return {m: mask.copy() for m in range(int(start_month), int(end_month) + 1)}
