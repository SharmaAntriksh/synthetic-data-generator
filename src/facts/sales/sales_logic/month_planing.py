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


__all__ = ["macro_month_weights", "build_rows_per_month"]
