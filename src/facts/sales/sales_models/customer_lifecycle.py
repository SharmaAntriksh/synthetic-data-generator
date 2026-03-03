"""
Customer lifecycle: eligibility + optional activity overlay.

Customer existence is encoded in customers.parquet:
    IsActiveInSales, CustomerStartMonth, CustomerEndMonth

This module:
  1. Builds per-month eligibility masks from those fields.
  2. Optionally applies a monthly activity overlay (dormancy / reactivation).

IMPORTANT: Do NOT use this module to "grow the customer base".
That is handled by CustomerStartMonth/CustomerEndMonth in customers.py.

Config source: models.yaml -> models.customer_activity_overlay
Runtime state:  State.models_cfg  (the inner "models" dict)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from src.facts.sales.sales_logic import State


# ---------------------------------------------------------------
# Overlay config
# ---------------------------------------------------------------

@dataclass(frozen=True)
class ActivityOverlayCfg:
    """Configuration for the optional customer activity overlay."""

    enabled: bool = False

    # Base probability that an eligible customer is "active" in a given month.
    base_monthly_activity_rate: float = 0.85

    # Smoothing: blends current month's rate with previous month's state.
    # 0 = no smoothing, 0.98 = very sticky.
    month_inertia: float = 0.75

    # Monthly reactivation: fraction of inactive-but-eligible customers
    # that re-enter the active pool each month.
    monthly_reactivation_rate: float = 0.08

    # Multiplicative seasonal cycle on the base rate.
    seasonal_amplitude: float = 0.10
    seasonal_period_months: int = 12

    # Per-customer volatility scaling (requires customer_temperature array).
    # Effective noise = per_customer_sigma * (1 + temperature_noise_scale * (temp - 1))
    per_customer_sigma: float = 0.20
    temperature_noise_scale: float = 0.50

    # Reproducibility seed offset.
    seed: int = 1234


def _load_overlay_cfg() -> ActivityOverlayCfg:
    """
    Load activity overlay config from State.models_cfg.

    Reads from ``models.customer_activity_overlay``.
    Returns a disabled config if the section is absent.
    """
    models = getattr(State, "models_cfg", None) or {}
    block = models.get("customer_activity_overlay", {}) or {}

    if not isinstance(block, dict) or not block:
        return ActivityOverlayCfg()

    try:
        return ActivityOverlayCfg(
            enabled=bool(block.get("enabled", False)),
            base_monthly_activity_rate=float(
                block.get("base_monthly_activity_rate", 0.85)),
            month_inertia=float(
                block.get("month_inertia", 0.75)),
            monthly_reactivation_rate=float(
                block.get("monthly_reactivation_rate", 0.08)),
            seasonal_amplitude=float(
                block.get("seasonal_amplitude", 0.10)),
            seasonal_period_months=int(
                block.get("seasonal_period_months", 12)),
            per_customer_sigma=float(
                block.get("per_customer_sigma", 0.20)),
            temperature_noise_scale=float(
                block.get("temperature_noise_scale", 0.50)),
            seed=int(block.get("seed", 1234)),
        )
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid customer_activity_overlay config: {e}") from e


# ---------------------------------------------------------------
# End-month normalization
# ---------------------------------------------------------------

def _normalize_end_month(end_month_arr, n: int) -> np.ndarray:
    """
    Convert nullable end-month representations to int64 with -1 = "no end".

    Handles: int arrays, float arrays (NaN → -1), object arrays (None/pd.NA → -1).
    """
    if end_month_arr is None:
        return np.full(n, -1, dtype=np.int64)

    a = np.asarray(end_month_arr)

    # Integer arrays: straightforward
    if np.issubdtype(a.dtype, np.integer):
        out = a.astype(np.int64, copy=True)
        out[out < 0] = -1
        return out

    # Float arrays: NaN → -1
    if np.issubdtype(a.dtype, np.floating):
        out = np.where(np.isnan(a), -1, a).astype(np.int64)
        out[out < 0] = -1
        return out

    # Object arrays (pd.NA, None, mixed)
    if a.dtype == object:
        try:
            import pandas as pd
            s = pd.to_numeric(pd.Series(a, copy=False), errors="coerce")
            out = s.fillna(-1).astype("int64").to_numpy()
            out[out < 0] = -1
            return out
        except Exception:
            pass

        # Manual fallback
        out = np.full(n, -1, dtype=np.int64)
        for i in range(min(n, a.shape[0])):
            v = a[i]
            if v is None:
                continue
            try:
                if np.isnan(v):
                    continue
            except (TypeError, ValueError):
                pass
            try:
                iv = int(v)
                out[i] = iv if iv >= 0 else -1
            except (TypeError, ValueError):
                pass
        return out

    # Last resort
    try:
        out = a.astype(np.int64, copy=True)
        out[out < 0] = -1
        return out
    except Exception:
        return np.full(n, -1, dtype=np.int64)


# ---------------------------------------------------------------
# Eligibility masks
# ---------------------------------------------------------------

def build_eligibility_by_month(
    start_month_arr: np.ndarray,
    end_month_arr: Optional[np.ndarray],
    is_active_in_sales_arr: Optional[np.ndarray],
    start_month: int,
    end_month: int,
) -> Dict[int, np.ndarray]:
    """
    Build month → eligibility mask over customers (aligned by row index).

    Eligibility rule per month *m*:
      - IsActiveInSales == 1  (or assumed if array is None)
      - CustomerStartMonth <= m
      - CustomerEndMonth < 0 (no end)  OR  m <= CustomerEndMonth

    Parameters
    ----------
    start_month_arr : int64 array — CustomerStartMonth per customer
    end_month_arr   : optional int64 array — CustomerEndMonth (nullable)
    is_active_in_sales_arr : optional int64 array — IsActiveInSales flag
    start_month, end_month : int — inclusive month range

    Returns
    -------
    dict[int, np.ndarray[bool]]
    """
    start_month_arr = np.asarray(start_month_arr, dtype=np.int64)
    n = start_month_arr.shape[0]
    start_m, end_m = int(start_month), int(end_month)

    if end_m < start_m:
        return {}

    # Global active gate
    if is_active_in_sales_arr is None:
        active_gate = np.ones(n, dtype=bool)
    else:
        active_gate = np.asarray(is_active_in_sales_arr, dtype=np.int64) == 1

    end_norm = _normalize_end_month(end_month_arr, n)

    # --- Initial mask for start_m ---
    mask = (
        active_gate
        & (start_month_arr <= start_m)
        & ((end_norm < 0) | (start_m <= end_norm))
    )
    out: Dict[int, np.ndarray] = {start_m: mask.copy()}

    # --- Precompute sorted start/end arrays for efficient sweep ---
    start_order = np.argsort(start_month_arr, kind="mergesort")
    start_vals = start_month_arr[start_order]
    ptr_start = int(np.searchsorted(start_vals, start_m + 1, side="left"))

    finite_end_mask = end_norm >= 0
    if finite_end_mask.any():
        finite_idx = np.flatnonzero(finite_end_mask)
        end_vals_at_finite = end_norm[finite_idx]
        end_order = finite_idx[np.argsort(end_vals_at_finite, kind="mergesort")]
        end_vals = end_norm[end_order]
        ptr_end = int(np.searchsorted(end_vals, start_m, side="left"))
    else:
        end_order = np.empty(0, dtype=np.int64)
        end_vals = np.empty(0, dtype=np.int64)
        ptr_end = 0

    # --- Sweep month by month ---
    for m in range(start_m, end_m):
        next_m = m + 1

        # Deactivate customers whose end month == m
        while ptr_end < end_order.size and int(end_vals[ptr_end]) == m:
            mask[int(end_order[ptr_end])] = False
            ptr_end += 1

        # Activate customers whose start month == next_m
        while ptr_start < start_order.size and int(start_vals[ptr_start]) == next_m:
            i = int(start_order[ptr_start])
            if active_gate[i] and (end_norm[i] < 0 or end_norm[i] >= next_m):
                mask[i] = True
            ptr_start += 1

        out[next_m] = mask.copy()

    return out


# ---------------------------------------------------------------
# Activity overlay
# ---------------------------------------------------------------

def apply_activity_overlay_by_month(
    eligibility_by_month: Dict[int, np.ndarray],
    seed: int,
    customer_temperature: Optional[np.ndarray] = None,
) -> Dict[int, np.ndarray]:
    """
    Apply an optional monthly activity overlay on top of eligibility.

    Returns month → active_mask where active_mask ⊆ eligibility_mask.
    If the overlay is disabled, returns eligibility_by_month unchanged.

    Parameters
    ----------
    eligibility_by_month : dict[int, np.ndarray[bool]]
    seed : int
    customer_temperature : optional float64 array (per customer)

    Returns
    -------
    dict[int, np.ndarray[bool]]
    """
    cfg = _load_overlay_cfg()
    if not cfg.enabled:
        return eligibility_by_month

    # Validate
    if not 0.0 <= cfg.base_monthly_activity_rate <= 1.0:
        raise ValueError("base_monthly_activity_rate must be in [0, 1]")
    if cfg.month_inertia < 0.0:
        raise ValueError("month_inertia must be >= 0")
    if cfg.seasonal_period_months <= 0:
        raise ValueError("seasonal_period_months must be > 0")
    if not 0.0 <= cfg.monthly_reactivation_rate <= 1.0:
        raise ValueError("monthly_reactivation_rate must be in [0, 1]")

    rng_global = np.random.default_rng(int(seed) + 9100)

    # Prepare temperature array (if provided)
    temp = None
    if customer_temperature is not None:
        temp = np.asarray(customer_temperature, dtype=np.float64)
        temp = np.where(np.isfinite(temp), temp, 1.0)
        temp = np.clip(temp, 0.2, 3.0)

    months = sorted(eligibility_by_month.keys())
    if not months:
        return {}

    out: Dict[int, np.ndarray] = {}
    prev_active: np.ndarray | None = None

    for m in months:
        elig = np.asarray(eligibility_by_month[m], dtype=bool)
        n = elig.shape[0]

        # Per-month deterministic RNG for reproducibility
        rng_m = np.random.default_rng(int(seed) + 10000 + int(m) * 7919)

        # Seasonal modulation
        cycle = np.sin(2.0 * np.pi * float(m) / float(cfg.seasonal_period_months))
        p = cfg.base_monthly_activity_rate * (1.0 + cfg.seasonal_amplitude * cycle)
        p = float(np.clip(p, 0.0, 1.0))

        # Per-customer probability (uses temperature if available)
        if temp is not None and cfg.per_customer_sigma > 0.0:
            noise_scale = cfg.per_customer_sigma * (
                1.0 + cfg.temperature_noise_scale * (temp - 1.0))
            noise_scale = np.clip(noise_scale, 0.0, 0.50)
            per_cust_p = np.clip(
                p * rng_m.uniform(1.0 - noise_scale, 1.0 + noise_scale, size=n),
                0.0, 1.0,
            )
        else:
            per_cust_p = p

        # First month: simple Bernoulli draw
        if prev_active is None:
            active = elig & (rng_global.random(n) < per_cust_p)
            out[m] = active
            prev_active = active
            continue

        # Subsequent months: previously-active stay active with probability p;
        # previously-inactive can reactivate
        active = (prev_active & elig) & (rng_global.random(n) < per_cust_p)

        if cfg.monthly_reactivation_rate > 0.0:
            reactivate = (~prev_active) & elig & (
                rng_global.random(n) < cfg.monthly_reactivation_rate)
            active |= reactivate

        out[m] = active
        prev_active = active

    return out


# ---------------------------------------------------------------
# Backward-compat shim
# ---------------------------------------------------------------

def build_active_customer_pool(
    all_customers,
    start_month: int,
    end_month: int,
    seed: int,
):
    """
    Build month → boolean mask over customers.

    New behavior:
      If State has lifecycle arrays → eligibility + optional overlay.
    Fallback:
      All customers active in all months.

    Parameters
    ----------
    all_customers : array-like of customer keys
    start_month, end_month : int — inclusive month range
    seed : int

    Returns
    -------
    dict[int, np.ndarray[bool]]
    """
    cust_keys = getattr(State, "customer_keys", None)
    start_arr = getattr(State, "customer_start_month", None)
    end_arr = getattr(State, "customer_end_month", None)
    act_arr = getattr(State, "customer_is_active_in_sales", None)
    temp_arr = getattr(State, "customer_temperature", None)

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

    # Fallback: all customers active every month
    n = np.asarray(all_customers, dtype=np.int64).shape[0]
    mask = np.ones(n, dtype=bool)
    return {m: mask.copy() for m in range(int(start_month), int(end_month) + 1)}
