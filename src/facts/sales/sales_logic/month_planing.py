# src/facts/sales/sales_logic/month_planing.py

import numpy as np


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

    # gentle growth: per-month multiplier derived from yearly_growth
    if yearly_growth != 0.0:
        g = (1.0 + yearly_growth) ** (m / 12.0)
    else:
        g = 1.0

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
            shock[hit] = 1.0 + rng.uniform(shock_lo, shock_hi, size=int(hit.sum()))
            shock = np.clip(shock, 0.1, 1.0)
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

    # legacy behavior (backward compatibility)
    month_weights = eligible_counts / eligible_counts.sum()
    return np.maximum(1, (month_weights * int(total_rows)).astype("int64"))


__all__ = ["macro_month_weights", "build_rows_per_month"]
