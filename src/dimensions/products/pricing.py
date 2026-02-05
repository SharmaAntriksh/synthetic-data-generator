import numpy as np
import pandas as pd


def _f(x, default=None):
    return default if x is None else x


def _to_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def _to_bool(x, default=False):
    if x is None:
        return default
    return bool(x)


def _stretch_to_range(series: pd.Series, target_min: float, target_max: float, qlo: float, qhi: float) -> pd.Series:
    """
    Map quantiles [qlo, qhi] linearly onto [target_min, target_max].
    Then caller can clip to hard bounds.
    """
    arr = series.to_numpy(dtype="float64", copy=True)
    finite = np.isfinite(arr)
    if not finite.any():
        return series

    x = arr[finite]
    qlo = float(np.clip(qlo, 0.0, 0.49))
    qhi = float(np.clip(qhi, 0.51, 1.0))

    lo = float(np.quantile(x, qlo))
    hi = float(np.quantile(x, qhi))

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-12:
        arr[finite] = (target_min + target_max) / 2.0
        return pd.Series(arr, index=series.index)

    scaled = (arr - lo) / (hi - lo)
    arr = target_min + scaled * (target_max - target_min)
    return pd.Series(arr, index=series.index)


def apply_product_pricing(df: pd.DataFrame, pricing_cfg: dict, seed: int | None = None) -> pd.DataFrame:
    """
    Products are the economic source of truth.
    This function finalizes Products.UnitPrice and Products.UnitCost based on config,
    regardless of whether products are Contoso-derived or synthetic.
    """
    if not pricing_cfg:
        return df

    rng = np.random.default_rng(seed)
    out = df.copy()

    # Ensure UnitPrice exists
    if "UnitPrice" not in out.columns:
        raise ValueError("Products dataframe must contain a UnitPrice column before pricing is applied")

    out["UnitPrice"] = pd.to_numeric(out["UnitPrice"], errors="coerce").astype("float64")

    # UnitCost may or may not exist upstream (Contoso usually has it)
    if "UnitCost" in out.columns:
        out["UnitCost"] = pd.to_numeric(out["UnitCost"], errors="coerce").astype("float64")
    else:
        out["UnitCost"] = np.nan

    base_cfg = pricing_cfg.get("base", {}) or {}
    cost_cfg = pricing_cfg.get("cost", {}) or {}
    jitter_cfg = pricing_cfg.get("jitter", {}) or {}

    # ----------------------------
    # Base price: scale / (optional) stretch / clamp
    # ----------------------------
    value_scale = _to_float(base_cfg.get("value_scale"), 1.0)
    if value_scale is None or value_scale <= 0:
        raise ValueError("products.pricing.base.value_scale must be a number > 0")

    min_price = _to_float(base_cfg.get("min_unit_price"), None)
    max_price = _to_float(base_cfg.get("max_unit_price"), None)

    stretch = _to_bool(base_cfg.get("stretch_to_range"), False)  # default OFF
    qlo = _to_float(base_cfg.get("stretch_low_quantile"), 0.01)
    qhi = _to_float(base_cfg.get("stretch_high_quantile"), 0.99)

    # Apply scale
    out["UnitPrice"] = out["UnitPrice"] * value_scale

    # Optional stretch-to-range (use only when you want the bounds to be “design targets”)
    if stretch and (min_price is not None) and (max_price is not None):
        if max_price <= min_price:
            raise ValueError("products.pricing.base.max_unit_price must be > min_unit_price when stretching")
        out["UnitPrice"] = _stretch_to_range(out["UnitPrice"], min_price, max_price, qlo, qhi)

    # Hard clamp
    if min_price is not None:
        out["UnitPrice"] = out["UnitPrice"].clip(lower=min_price)
    if max_price is not None:
        out["UnitPrice"] = out["UnitPrice"].clip(upper=max_price)

    # ----------------------------
    # Cost model
    # ----------------------------
    # mode: "keep" (use existing UnitCost) or "margin" (derive from UnitPrice)
    mode = (cost_cfg.get("mode") or "").strip().lower()
    if mode not in ("", "keep", "margin"):
        raise ValueError('products.pricing.cost.mode must be one of: "keep", "margin"')

    min_margin = _to_float(cost_cfg.get("min_margin_pct"), None)
    max_margin = _to_float(cost_cfg.get("max_margin_pct"), None)

    # Choose a default mode:
    # - If margins provided -> margin mode
    # - Else if UnitCost exists -> keep
    # - Else -> margin with a conservative default
    if mode == "":
        if (min_margin is not None) or (max_margin is not None):
            mode = "margin"
        else:
            mode = "keep" if out["UnitCost"].notna().any() else "margin"

    if mode == "keep":
        # If some costs missing, fill them using a conservative margin band
        if not out["UnitCost"].notna().all():
            mm_lo = 0.20 if min_margin is None else float(min_margin)
            mm_hi = 0.45 if max_margin is None else float(max_margin)
            margin = rng.uniform(mm_lo, mm_hi, size=len(out))
            fill_cost = out["UnitPrice"].to_numpy(dtype="float64") * (1.0 - margin)
            out["UnitCost"] = out["UnitCost"].to_numpy(dtype="float64")
            mask = ~np.isfinite(out["UnitCost"])
            out.loc[mask, "UnitCost"] = fill_cost[mask]
    else:
        # margin mode
        if min_margin is None or max_margin is None:
            raise ValueError("products.pricing.cost: min_margin_pct and max_margin_pct must be provided for mode=margin")
        if not (0.0 < float(min_margin) < float(max_margin) < 1.0):
            raise ValueError("products.pricing.cost: require 0 < min_margin_pct < max_margin_pct < 1")

        margin = rng.uniform(float(min_margin), float(max_margin), size=len(out))
        out["UnitCost"] = out["UnitPrice"].to_numpy(dtype="float64") * (1.0 - margin)

    # ----------------------------
    # Jitter
    # ----------------------------
    price_pct = float(jitter_cfg.get("price_pct", 0.0) or 0.0)
    cost_pct = float(jitter_cfg.get("cost_pct", 0.0) or 0.0)

    if price_pct > 0:
        mult = rng.uniform(1.0 - price_pct, 1.0 + price_pct, size=len(out))
        out["UnitPrice"] = out["UnitPrice"].to_numpy(dtype="float64") * mult

    if cost_pct > 0:
        mult = rng.uniform(1.0 - cost_pct, 1.0 + cost_pct, size=len(out))
        out["UnitCost"] = out["UnitCost"].to_numpy(dtype="float64") * mult

    # Reapply hard bounds after jitter
    if min_price is not None:
        out["UnitPrice"] = out["UnitPrice"].clip(lower=min_price)
    if max_price is not None:
        out["UnitPrice"] = out["UnitPrice"].clip(upper=max_price)

    # Invariants
    out["UnitPrice"] = out["UnitPrice"].clip(lower=0.0)
    out["UnitCost"] = pd.to_numeric(out["UnitCost"], errors="coerce").astype("float64").clip(lower=0.0)
    out["UnitCost"] = np.minimum(out["UnitCost"].to_numpy(dtype="float64"), out["UnitPrice"].to_numpy(dtype="float64"))

    # Store format
    out["UnitPrice"] = out["UnitPrice"].round(2)
    out["UnitCost"] = out["UnitCost"].round(2)

    return out
