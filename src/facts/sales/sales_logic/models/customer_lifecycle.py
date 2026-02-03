"""
customer_lifecycle.py

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
    # (only relevant when enabled=True)
    monthly_reactivation_rate: float = 0.10

    # Optional: if customer_temperature array exists, scale volatility
    # Effective noise = monthly_noise * (1 + temperature_noise_scale*(temp-1))
    temperature_noise_scale: float = 0.50


def _load_overlay_cfg() -> ActivityOverlayCfg:
    """
    Reads config from State.models_cfg if present.

    Recommended config location (doesn't break old configs):
      models:
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
      - CustomerEndMonth < 0 => no end
        else m <= CustomerEndMonth

    All arrays are expected aligned to customers dimension row order.
    """
    start_month_arr = np.asarray(start_month_arr, dtype=np.int64)
    n = start_month_arr.shape[0]

    if is_active_in_sales_arr is None:
        active_gate = np.ones(n, dtype=bool)
    else:
        active_gate = (np.asarray(is_active_in_sales_arr, dtype=np.int64) == 1)

    if end_month_arr is None:
        end_month_norm = np.full(n, -1, dtype=np.int64)
    else:
        end_month_norm = np.asarray(end_month_arr, dtype=np.int64)
        end_month_norm = np.where(end_month_norm < 0, -1, end_month_norm)

    out: Dict[int, np.ndarray] = {}

    for m in range(int(start_month), int(end_month) + 1):
        mask = active_gate & (start_month_arr <= m)
        has_end = end_month_norm >= 0
        mask &= (~has_end) | (m <= end_month_norm)
        out[m] = mask

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

    # Validate rates
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
        # clip to keep it stable
        temp = np.clip(temp, 0.2, 3.0)

    out: Dict[int, np.ndarray] = {}
    prev_active: Optional[np.ndarray] = None

    months = sorted(eligibility_by_month.keys())

    for m in months:
        elig = eligibility_by_month[m]
        n = elig.shape[0]

        rng_m = np.random.default_rng(int(seed) + 10000 + int(m) * 7919)

        # Seasonality factor
        cycle = np.sin(2 * np.pi * (m / cfg.seasonal_period_months))
        seasonal_mult = 1.0 + cfg.seasonal_amplitude * cycle

        # Base activity probability for this month
        p = cfg.base_monthly_activity_rate * seasonal_mult
        p *= rng_m.uniform(1.0 - cfg.monthly_noise, 1.0 + cfg.monthly_noise)
        p = float(np.clip(p, 0.0, 1.0))

        # Temperature-adjusted per-customer noise (optional)
        if temp is not None:
            # Expand around 1.0: higher temp -> more volatile probability
            noise_scale = cfg.monthly_noise * (1.0 + cfg.temperature_noise_scale * (temp - 1.0))
            noise_scale = np.clip(noise_scale, 0.0, 0.50)  # hard cap
            per_cust_p = p * rng_m.uniform(1.0 - noise_scale, 1.0 + noise_scale, size=n)
            per_cust_p = np.clip(per_cust_p, 0.0, 1.0)
        else:
            per_cust_p = p

        # First month: draw activity from eligible only
        if prev_active is None:
            active = np.zeros(n, dtype=bool)
            # eligible customers become active with prob per_cust_p
            u = rng_global.random(n)
            active = elig & (u < per_cust_p)
            out[m] = active
            prev_active = active
            continue

        # Subsequent months: allow reactivation for previously inactive-but-eligible
        active = np.zeros(n, dtype=bool)

        # customers who were active can remain active with prob p
        was_active = prev_active & elig
        u1 = rng_global.random(n)
        active |= was_active & (u1 < per_cust_p)

        # customers eligible but not active last month may reactivate
        can_reactivate = (~prev_active) & elig
        u2 = rng_global.random(n)
        active |= can_reactivate & (u2 < cfg.monthly_reactivation_rate)

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

    Old behavior (deprecated) used State.models_cfg["customers"] to simulate
    growth/churn and returned month->mask. :contentReference[oaicite:1]{index=1}

    New behavior:
    - If State has lifecycle arrays, we compute eligibility from them and
      optionally apply an activity overlay.
    - If lifecycle arrays are missing, we fall back to a simple "all active"
      mask across months (closest old behavior for compatibility).

    Returns:
        dict[int, np.ndarray]: month_idx -> boolean mask over customers (row-aligned)
    """
    # Attempt lifecycle-aware path
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
        return apply_activity_overlay_by_month(eligibility_by_month=elig, seed=seed, customer_temperature=temp_arr)

    # Fallback (no lifecycle arrays bound): treat all customers as eligible+active
    all_customers = np.asarray(all_customers, dtype=np.int64)
    n = len(all_customers)
    mask = np.ones(n, dtype=bool)
    return {m: mask.copy() for m in range(int(start_month), int(end_month) + 1)}
