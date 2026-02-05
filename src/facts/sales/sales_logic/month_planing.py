# src/facts/sales/sales_logic/month_planing.py

import numpy as np


def _sched_mode_and_values(node: dict, name: str) -> tuple[str, list[float]]:
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
    if noise_std > 0:
        n = rng.normal(loc=1.0, scale=noise_std, size=T)
        n = np.clip(n, 0.5, 1.5)
    else:
        n = 1.0

    # shocks: occasional multiplicative hits
    if shock_p > 0:
        shock = np.ones(T, dtype="float64")
        hit = rng.random(T) < shock_p
        if hit.any():
            if shock_lo > shock_hi:
                raise ValueError("shock_impact must be [low, high] with low <= high")
            shock[hit] = 1.0 + rng.uniform(shock_lo, shock_hi, size=int(hit.sum()))
            upper = max(1.0, 1.0 + shock_hi)   # allow positive shocks
            shock = np.clip(shock, 0.1, upper)
    else:
        shock = 1.0

    w = base_level * g * s * n * shock
    w = np.clip(w, 1e-9, None)
    return w / w.sum()


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
    T = int(len(eligible_counts))
    if T <= 0:
        return np.zeros(0, dtype="int64")

    if total_rows <= 0:
        return np.zeros(T, dtype="int64")

    if eligible_counts.sum() <= 0:
        return np.zeros(T, dtype="int64")

    macro_cfg = macro_cfg or {}
    use_macro = bool(macro_cfg)
    eligible_nonzero = (eligible_counts > 0)

    if use_macro:
        # base demand weights independent of customer count
        macro_w = macro_month_weights(rng, T, macro_cfg)

        # months with no eligible customers cannot receive demand
        macro_w = macro_w * eligible_nonzero.astype("float64")
        if macro_w.sum() <= 0:
            return np.zeros(T, dtype="int64")
        macro_w = macro_w / macro_w.sum()

        # initial allocation
        rows_per_month = np.floor(macro_w * int(total_rows)).astype("int64")

        # ensure we allocate all rows (fix rounding)
        remainder = int(int(total_rows) - int(rows_per_month.sum()))
        if remainder > 0:
            add_idx = np.argsort(-macro_w)[:remainder]
            rows_per_month[add_idx] += 1

        # cap early months if eligible base is too small
        cap_cfg = macro_cfg.get("early_month_cap", {}) or {}
        cap_enabled = bool(cap_cfg.get("enabled", True))
        per_customer_cap = int(cap_cfg.get("max_rows_per_customer", 12))
        redistribute = bool(cap_cfg.get("redistribute_excess", True))

        if cap_enabled and per_customer_cap > 0:
            excess = 0
            for m in range(T):
                if not eligible_nonzero[m]:
                    continue
                max_rows = int(eligible_counts[m]) * per_customer_cap
                if rows_per_month[m] > max_rows:
                    excess += int(rows_per_month[m] - max_rows)
                    rows_per_month[m] = max_rows

            if redistribute and excess > 0:
                capacity = np.maximum(
                    0,
                    (eligible_counts * per_customer_cap).astype("int64") - rows_per_month,
                )
                cap_months = np.nonzero(capacity > 0)[0]
                if cap_months.size > 0:
                    w = macro_w[cap_months]
                    w = w / w.sum()
                    add = rng.multinomial(excess, w)
                    rows_per_month[cap_months] += add

        return rows_per_month

    # legacy behavior (backward compatibility) but preserve total_rows exactly
    month_weights = eligible_counts / eligible_counts.sum()
    rows = np.floor(month_weights * int(total_rows)).astype("int64")

    remainder = int(int(total_rows) - int(rows.sum()))
    if remainder > 0:
        add_idx = np.argsort(-month_weights)[:remainder]
        rows[add_idx] += 1

    return rows

__all__ = ["macro_month_weights", "build_rows_per_month"]
