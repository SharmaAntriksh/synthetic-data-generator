import numpy as np
import pandas as pd


def _as_float_or_none(x):
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None


def _stretch_unit_price_to_range(
    s: pd.Series,
    target_min: float,
    target_max: float,
    q_low: float = 0.01,
    q_high: float = 0.99,
) -> pd.Series:
    """
    Linearly rescale a price distribution so that quantiles [q_low, q_high]
    map to [target_min, target_max]. Values outside that band will exceed
    the range and will be clamped later.

    This makes min/max behave like a DESIGN RANGE, not just a clamp.
    """
    if target_max <= target_min:
        raise ValueError("products.pricing.base: max_unit_price must be > min_unit_price")

    arr = s.to_numpy(dtype="float64", copy=True)
    finite = np.isfinite(arr)

    if not finite.any():
        # Nothing usable to scale
        return s

    x = arr[finite]
    q_low = float(np.clip(q_low, 0.0, 0.49))
    q_high = float(np.clip(q_high, 0.51, 1.0))

    lo = float(np.quantile(x, q_low))
    hi = float(np.quantile(x, q_high))

    # If distribution is degenerate, set finite values to mid-point.
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-12:
        arr[finite] = (target_min + target_max) / 2.0
        return pd.Series(arr, index=s.index)

    # Linear map: lo->target_min, hi->target_max
    scaled = (arr - lo) / (hi - lo)
    arr = target_min + scaled * (target_max - target_min)

    return pd.Series(arr, index=s.index)


def apply_product_pricing(
    df: pd.DataFrame,
    pricing_cfg: dict,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Apply pricing rules to Products.

    Option 2 (stretch-to-range):
      - If min_unit_price and max_unit_price are provided (and stretch_to_range is True),
        rescale UnitPrice so the chosen percentile band maps into [min, max], then clamp.

    Products become the economic source of truth.
    Sales may apply mild macro factors (e.g., inflation) but should not regenerate base UnitPrice/UnitCost.
    """
    if not pricing_cfg:
        return df

    rng = np.random.default_rng(seed)
    out = df.copy()

    # Ensure UnitCost exists (if upstream doesn't provide it)
    if "UnitCost" not in out.columns:
        out["UnitCost"] = out["UnitPrice"]

    # -------------------------------------------------
    # 1) BASE PRICE: scale + (optionally) stretch-to-range + clamp
    # -------------------------------------------------
    base_cfg = pricing_cfg.get("base", {}) or {}

    value_scale = float(base_cfg.get("value_scale", 1.0))
    if value_scale <= 0:
        raise ValueError("products.pricing.base.value_scale must be > 0")

    min_price = _as_float_or_none(base_cfg.get("min_unit_price"))
    max_price = _as_float_or_none(base_cfg.get("max_unit_price"))

    # Optional stretch controls
    stretch_to_range = bool(base_cfg.get("stretch_to_range", True))
    q_low = float(base_cfg.get("stretch_low_quantile", 0.01))
    q_high = float(base_cfg.get("stretch_high_quantile", 0.99))

    # Start from numeric prices
    out["UnitPrice"] = pd.to_numeric(out["UnitPrice"], errors="coerce").astype("float64")
    out["UnitPrice"] = out["UnitPrice"] * value_scale

    # Stretch-to-range if we have both bounds and it's enabled
    if stretch_to_range and (min_price is not None) and (max_price is not None):
        out["UnitPrice"] = _stretch_unit_price_to_range(
            out["UnitPrice"],
            target_min=min_price,
            target_max=max_price,
            q_low=q_low,
            q_high=q_high,
        )

    # Clamp AFTER stretch (or after scale if stretch disabled)
    if min_price is not None:
        out["UnitPrice"] = out["UnitPrice"].clip(lower=min_price)
    if max_price is not None:
        out["UnitPrice"] = out["UnitPrice"].clip(upper=max_price)

    # -------------------------------------------------
    # 2) COST MODEL (MARGIN-BASED)
    # -------------------------------------------------
    cost_cfg = pricing_cfg.get("cost", {}) or {}
    min_margin = _as_float_or_none(cost_cfg.get("min_margin_pct"))
    max_margin = _as_float_or_none(cost_cfg.get("max_margin_pct"))

    if (min_margin is not None) or (max_margin is not None):
        if (min_margin is None) or (max_margin is None):
            raise ValueError("Both min_margin_pct and max_margin_pct must be provided")

        if not (0.0 < min_margin < max_margin < 1.0):
            raise ValueError("Margin pct must satisfy 0 < min < max < 1")

        margin = rng.uniform(min_margin, max_margin, size=len(out))
        out["UnitCost"] = out["UnitPrice"] * (1.0 - margin)
    else:
        # Ensure numeric cost even if not recomputed
        out["UnitCost"] = pd.to_numeric(out["UnitCost"], errors="coerce").astype("float64")

    # -------------------------------------------------
    # 3) JITTER (OPTIONAL NOISE)
    # -------------------------------------------------
    jitter_cfg = pricing_cfg.get("jitter", {}) or {}
    price_jitter = float(jitter_cfg.get("price_pct", 0.0))
    cost_jitter = float(jitter_cfg.get("cost_pct", 0.0))

    if price_jitter > 0.0:
        pj = rng.uniform(1.0 - price_jitter, 1.0 + price_jitter, size=len(out))
        out["UnitPrice"] = out["UnitPrice"] * pj

    if cost_jitter > 0.0:
        cj = rng.uniform(1.0 - cost_jitter, 1.0 + cost_jitter, size=len(out))
        out["UnitCost"] = out["UnitCost"] * cj

    # Re-apply min/max AFTER jitter so bounds are hard
    if min_price is not None:
        out["UnitPrice"] = out["UnitPrice"].clip(lower=min_price)
    if max_price is not None:
        out["UnitPrice"] = out["UnitPrice"].clip(upper=max_price)

    # Cost must never exceed price
    out["UnitCost"] = np.minimum(out["UnitCost"], out["UnitPrice"])

    # -------------------------------------------------
    # 4) HARD SAFETY RULES
    # -------------------------------------------------
    out["UnitPrice"] = out["UnitPrice"].clip(lower=0.0)
    out["UnitCost"] = out["UnitCost"].clip(lower=0.0)
    out["UnitCost"] = np.minimum(out["UnitCost"], out["UnitPrice"])

    # -------------------------------------------------
    # 5) FINAL ROUNDING (STORAGE)
    # -------------------------------------------------
    out["UnitPrice"] = out["UnitPrice"].round(2)
    out["UnitCost"] = out["UnitCost"].round(2)

    return out
