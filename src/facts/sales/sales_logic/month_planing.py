# src/facts/sales/sales_logic/month_planing.py

from __future__ import annotations

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

    try:
        vals = [float(v) for v in values]
    except Exception as e:
        raise ValueError(f"{name}.values must be numeric") from e

    return mode, vals


def _normalize_nonneg(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype="float64")
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= 0.0:
        raise ValueError("weights must sum to > 0")
    return w / s


def macro_month_weights(rng: np.random.Generator, T: int, cfg: dict) -> np.ndarray:
    """
    Create base demand weights per month, independent of eligible customer count.
    Produces a smooth trend + seasonality + optional shocks + noise.

    cfg example (models.yaml -> models.macro_demand):
      base_level: 1.0
      yearly_growth: 0.03               # 3% per year
      seasonality_amplitude: 0.12       # +/-12%
      seasonality_phase: 0.0            # radians
      noise_std: 0.05                   # month-to-month
      shock_probability: 0.06           # per month
      shock_impact: [-0.35, -0.10]      # multiplicative range
      early_month_cap:
        enabled: true
        max_rows_per_customer: 12
        redistribute_excess: true
    """
    T = int(T)
    if T <= 0:
        return np.zeros(0, dtype="float64")

    cfg = cfg or {}

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
    year_idx = (m // 12.0).astype("int64")  # 0-based year index per month

    # ---- baseline smooth drift from yearly_growth ----
    if yearly_growth != 0.0:
        g = (1.0 + yearly_growth) ** (m / 12.0)
    else:
        g = np.ones(T, dtype="float64")

    yoy_node = cfg.get("yoy_growth_schedule")
    lvl_node = cfg.get("year_level_factors")
    if yoy_node and lvl_node:
        raise ValueError("Use only one of: yoy_growth_schedule OR year_level_factors")

    # ---- year_level_factors: pin exact year multipliers ----
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

    # ---- yoy_growth_schedule: compound year-over-year rates ----
    elif yoy_node:
        mode, vals = _sched_mode_and_values(yoy_node, "yoy_growth_schedule")
        if any(v <= -0.99 for v in vals):
            raise ValueError("yoy_growth_schedule.values must be > -0.99")

        yoy = np.asarray(vals, dtype="float64")
        n_years = int((T + 11) // 12)

        if n_years <= 1:
            year_factor = np.ones(1, dtype="float64")
        else:
            steps = np.arange(n_years - 1, dtype="int64")  # transitions into year 1..n_years-1
            if mode == "repeat":
                rates = yoy[steps % len(yoy)]
            else:  # once: rates beyond provided list are 0.0
                rates = np.zeros(n_years - 1, dtype="float64")
                k = min(len(yoy), n_years - 1)
                if k > 0:
                    rates[:k] = yoy[:k]

            year_factor = np.ones(n_years, dtype="float64")
            year_factor[1:] = np.cumprod(1.0 + rates)

        g = g * year_factor[np.minimum(year_idx, len(year_factor) - 1)]

    # ---- seasonality (12-month cycle) ----
    if amp != 0.0:
        s = 1.0 + amp * np.sin((2.0 * np.pi * m / 12.0) + phase)
    else:
        s = np.ones(T, dtype="float64")

    # ---- month-to-month noise ----
    if noise_std > 0.0:
        nn = rng.normal(loc=1.0, scale=noise_std, size=T).astype("float64", copy=False)
        nn = np.clip(nn, 0.5, 1.5)
    else:
        nn = np.ones(T, dtype="float64")

    # ---- occasional shocks ----
    if shock_p > 0.0:
        if shock_lo > shock_hi:
            raise ValueError("shock_impact must be [low, high] with low <= high")

        shock = np.ones(T, dtype="float64")
        hit = rng.random(T) < shock_p
        if hit.any():
            shock[hit] = 1.0 + rng.uniform(shock_lo, shock_hi, size=int(hit.sum()))
            upper = max(1.0, 1.0 + shock_hi)  # allow positive shocks
            shock = np.clip(shock, 0.1, upper)
    else:
        shock = np.ones(T, dtype="float64")

    w = base_level * g * s * nn * shock
    w = np.clip(w, 1e-9, None)
    return _normalize_nonneg(w)


def build_rows_per_month(
    *,
    rng: np.random.Generator,
    total_rows: int,
    eligible_counts: np.ndarray,
    macro_cfg: dict | None,
) -> np.ndarray:
    """
    Decide how many rows to generate in each month.

    - If macro_cfg is truthy: uses macro demand weights + early_month_cap logic.
    - Else: falls back to the legacy eligible-count proportional allocation.

    Returns: int64 array of length T (months).
    """
    eligible_counts = np.asarray(eligible_counts, dtype="float64")
    T = int(eligible_counts.size)

    if T <= 0:
        return np.zeros(0, dtype="int64")
    if int(total_rows) <= 0:
        return np.zeros(T, dtype="int64")
    if float(eligible_counts.sum()) <= 0.0:
        return np.zeros(T, dtype="int64")

    eligible_nonzero = eligible_counts > 0.0
    macro_cfg = macro_cfg or {}
    use_macro = bool(macro_cfg)

    if use_macro:
        macro_w = macro_month_weights(rng, T, macro_cfg)

        # months with no eligible customers cannot receive demand
        macro_w = macro_w * eligible_nonzero.astype("float64")
        if float(macro_w.sum()) <= 0.0:
            return np.zeros(T, dtype="int64")
        macro_w = macro_w / float(macro_w.sum())

        # initial allocation (floor)
        total_rows = int(total_rows)
        rows = np.floor(macro_w * total_rows).astype("int64")

        # fix rounding remainder deterministically (largest weights first)
        remainder = int(total_rows - int(rows.sum()))
        if remainder > 0:
            add_idx = np.argsort(-macro_w)[:remainder]
            rows[add_idx] += 1

        # early month cap (vectorized)
        cap_cfg = macro_cfg.get("early_month_cap", {}) or {}
        cap_enabled = bool(cap_cfg.get("enabled", True))
        per_customer_cap = int(cap_cfg.get("max_rows_per_customer", 12))
        redistribute = bool(cap_cfg.get("redistribute_excess", True))

        if cap_enabled and per_customer_cap > 0:
            elig_int = eligible_counts.astype("int64", copy=False)  # matches int(...) truncation
            max_rows = elig_int * per_customer_cap

            # Months with zero eligible customers already have 0 weight, but enforce safety:
            max_rows = np.where(eligible_nonzero, max_rows, 0).astype("int64", copy=False)

            clipped = np.minimum(rows, max_rows)
            excess = int((rows - clipped).sum())
            rows = clipped

            if redistribute and excess > 0:
                capacity = np.maximum(0, max_rows - rows)
                cap_months = np.nonzero(capacity > 0)[0]
                if cap_months.size > 0:
                    w = macro_w[cap_months]
                    w = w / float(w.sum())
                    add = rng.multinomial(excess, w)
                    rows[cap_months] += add

        return rows

    # ------------------------------------------------------------
    # Legacy behavior (eligible-count proportional) w/ exact total_rows
    # ------------------------------------------------------------
    month_weights = eligible_counts / float(eligible_counts.sum())
    month_weights = np.where(np.isfinite(month_weights), month_weights, 0.0)

    total_rows = int(total_rows)
    rows = np.floor(month_weights * total_rows).astype("int64")

    remainder = int(total_rows - int(rows.sum()))
    if remainder > 0:
        add_idx = np.argsort(-month_weights)[:remainder]
        rows[add_idx] += 1

    return rows


__all__ = ["macro_month_weights", "build_rows_per_month"]
