import numpy as np
from .globals import State


# ---------------------------------------------------------------------
# Caching (models_cfg is stable during a run; avoid re-parsing every call)
# ---------------------------------------------------------------------
_MD_CACHE_KEY = None
_MD_CACHE_VAL = None


def _to_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def _parse_bands(bands, default):
    """
    bands: list[dict] {max, step} -> sorted list[(max, step)]
    """
    out = []
    if isinstance(bands, list):
        for b in bands:
            if not isinstance(b, dict):
                continue
            mx = _to_float(b.get("max"), None)
            st = _to_float(b.get("step"), None)
            if mx is None or st is None or mx <= 0 or st <= 0:
                continue
            out.append((float(mx), float(st)))

    if not out:
        return list(default)

    out.sort(key=lambda t: t[0])
    return out


def _cfg_markdown():
    """
    Reads State.models_cfg.pricing.markdown.

    Returns:
      enabled: bool
      kind_codes: np.int8 array (0=none, 1=pct, 2=amt)
      values: np.float64 array (pct in [0,1], amt >=0)
      probs: np.float64 array (normalized)
      max_pct: float in [0,1]
      min_net: float >= 0
      allow_neg_margin: bool
      quantize_discount: bool
      discount_rounding: "floor"|"nearest"
      band_max: np.float64 array sorted asc
      band_step: np.float64 array aligned to band_max
    """
    global _MD_CACHE_KEY, _MD_CACHE_VAL

    models = getattr(State, "models_cfg", None) or {}
    key = id(models)
    if _MD_CACHE_KEY == key and _MD_CACHE_VAL is not None:
        return _MD_CACHE_VAL

    pricing = models.get("pricing", {}) or {}
    md = pricing.get("markdown", {}) or {}

    enabled = bool(md.get("enabled", True))

    ladder = md.get("ladder")
    if not isinstance(ladder, list) or len(ladder) == 0:
        ladder = [
            {"kind": "none", "value": 0.0,  "weight": 0.55},
            {"kind": "pct",  "value": 0.05, "weight": 0.20},
            {"kind": "pct",  "value": 0.10, "weight": 0.12},
            {"kind": "pct",  "value": 0.15, "weight": 0.08},
            {"kind": "amt",  "value": 25.0, "weight": 0.05},
        ]

    max_pct = float(md.get("max_pct_of_price", 0.50))
    max_pct = float(np.clip(max_pct, 0.0, 1.0))

    min_net = float(md.get("min_net_price", 0.01))
    min_net = max(0.0, min_net)

    allow_neg_margin = bool(md.get("allow_negative_margin", False))

    # Sanitize ladder into compact arrays
    kind_codes = []
    values = []
    weights = []

    for item in ladder:
        if not isinstance(item, dict):
            continue

        k = str(item.get("kind", "none")).strip().lower()
        w = float(item.get("weight", 0.0) or 0.0)
        if w <= 0:
            continue

        v = float(item.get("value", 0.0) or 0.0)

        if k == "pct":
            kind_codes.append(1)
            values.append(float(np.clip(v, 0.0, 1.0)))
            weights.append(w)
        elif k == "amt":
            kind_codes.append(2)
            values.append(max(0.0, v))
            weights.append(w)
        elif k == "none":
            kind_codes.append(0)
            values.append(0.0)
            weights.append(w)
        else:
            continue

    if not kind_codes:
        kind_codes = [0]
        values = [0.0]
        weights = [1.0]

    probs = np.asarray(weights, dtype=np.float64)
    s = float(probs.sum())
    probs = probs / s if s > 0 else np.array([1.0], dtype=np.float64)

    kind_codes = np.asarray(kind_codes, dtype=np.int8)
    values = np.asarray(values, dtype=np.float64)

    # Appearance
    appearance = md.get("appearance", {}) or {}
    quantize_discount = bool(appearance.get("quantize_discount", True))

    discount_rounding = str(appearance.get("discount_rounding", "floor")).strip().lower()
    if discount_rounding not in ("floor", "nearest"):
        discount_rounding = "floor"

    bands = appearance.get("discount_bands")
    if not isinstance(bands, list) or len(bands) == 0:
        bands = [
            {"max": 50, "step": 0.50},
            {"max": 200, "step": 1},
            {"max": 1000, "step": 5},
            {"max": 5000, "step": 10},
            {"max": 1e18, "step": 25},
        ]

    parsed = _parse_bands(
        bands,
        default=[(50.0, 0.50), (200.0, 1.0), (1000.0, 5.0), (5000.0, 10.0), (1e18, 25.0)],
    )

    band_max = np.asarray([mx for mx, _ in parsed], dtype=np.float64)
    band_step = np.asarray([st for _, st in parsed], dtype=np.float64)
    if band_max.size == 0:
        band_max = np.asarray([1e18], dtype=np.float64)
        band_step = np.asarray([25.0], dtype=np.float64)

    out = (
        enabled,
        kind_codes,
        values,
        probs,
        max_pct,
        min_net,
        allow_neg_margin,
        quantize_discount,
        discount_rounding,
        band_max,
        band_step,
    )

    _MD_CACHE_KEY = key
    _MD_CACHE_VAL = out
    return out


def _as_f64(x, n: int) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    if a.shape[0] != int(n):
        raise ValueError("Array length mismatch")
    # Replace NaN/inf with 0.0 deterministically
    return np.where(np.isfinite(a), a, 0.0)


def _step_for_price(up: np.ndarray, band_max: np.ndarray, band_step: np.ndarray) -> np.ndarray:
    """
    Fast vectorized step lookup: first band where up <= max.
    """
    # band_max is sorted ascending
    idx = np.searchsorted(band_max, up, side="left")
    # If up > last max (shouldn't happen if last max is huge), clamp to last
    idx = np.minimum(idx, band_step.size - 1)
    step = band_step[idx]
    return np.where(step > 0.0, step, 0.01)


def _quantize_discount(
    disc: np.ndarray,
    up: np.ndarray,
    band_max: np.ndarray,
    band_step: np.ndarray,
    rounding: str,
) -> np.ndarray:
    """
    Quantize discount to clean increments chosen per-row based on UnitPrice,
    then apply a ".99" ending (e.g., 5.00 -> 4.99, 10.00 -> 9.99) for demo-friendly visuals.

    Note:
      - 0 stays 0
      - This makes discounts slightly smaller, so it will NOT violate max_pct/min_net constraints
        (those constraints are re-applied after quantization anyway).
    """
    step = _step_for_price(up, band_max, band_step)

    if rounding == "nearest":
        q = np.round(disc / step) * step
    else:
        q = np.floor(disc / step) * step

    # Apply ".99" ending for non-zero discounts
    q = np.where(q > 0.0, np.maximum(q - 0.01, 0.0), 0.0)

    return q


def compute_prices(
    rng,
    n,
    unit_price,
    unit_cost,
    promo_pct=None,  # accepted for backward compatibility; intentionally ignored
    *,
    price_pressure: float = 1.0,
    row_price_jitter_pct: float = 0.0,
):
    """
    Sales pricing rule:
      - UnitPrice/UnitCost come from Products (source of truth).
      - Sales.DiscountAmount is an independent markdown (NOT from Promotions).
      - Promotions affect ONLY PromotionKey; promo discounts are applied at analysis-time.

    Output columns represent "after markdown, before promo".
    """
    _ = promo_pct  # ignored by design

    n = int(n)
    if n <= 0:
        z = np.zeros(0, dtype=np.float64)
        return {"final_unit_price": z, "final_unit_cost": z, "discount_amt": z, "final_net_price": z}

    up = _as_f64(unit_price, n)
    uc = _as_f64(unit_cost, n)

    # Optional global scale
    pp = float(price_pressure) if price_pressure is not None else 1.0
    if not np.isfinite(pp) or pp <= 0.0:
        pp = 1.0
    up *= pp
    uc *= pp

    # Optional per-row jitter (defaults OFF)
    j = float(row_price_jitter_pct) if row_price_jitter_pct is not None else 0.0
    if np.isfinite(j) and j > 0.0:
        mult = rng.uniform(1.0 - j, 1.0 + j, size=n).astype(np.float64, copy=False)
        up *= mult
        uc *= mult

    up = np.maximum(up, 0.0)
    uc = np.maximum(uc, 0.0)

    (
        enabled,
        kind_codes,
        values,
        probs,
        max_pct,
        min_net,
        allow_neg_margin,
        quantize_discount,
        discount_rounding,
        band_max,
        band_step,
    ) = _cfg_markdown()

    disc = np.zeros(n, dtype=np.float64)

    if enabled:
        idx = rng.choice(kind_codes.size, size=n, replace=True, p=probs)

        kc = kind_codes[idx]      # 0/1/2
        v = values[idx]           # pct or amt

        # Vectorized ladder application
        # pct: up * v
        disc = np.where(kc == 1, up * v, disc)
        # amt: v
        disc = np.where(kc == 2, v, disc)
        # none: keep 0

    # Base constraints before quantization
    disc = np.maximum(disc, 0.0)
    disc = np.minimum(disc, up * max_pct)
    if min_net > 0.0:
        disc = np.minimum(disc, np.maximum(up - min_net, 0.0))
    disc = np.minimum(disc, up)

    # Quantize to clean increments
    if enabled and quantize_discount:
        disc = _quantize_discount(disc, up, band_max, band_step, discount_rounding)

        # Re-apply constraints after quantization
        disc = np.maximum(disc, 0.0)
        disc = np.minimum(disc, up * max_pct)
        if min_net > 0.0:
            disc = np.minimum(disc, np.maximum(up - min_net, 0.0))
        disc = np.minimum(disc, up)

    net = np.maximum(up - disc, 0.0)

    # Invariants
    uc = np.minimum(uc, up)
    if not allow_neg_margin:
        # Demo-friendly: avoid negative AND avoid exact break-even after rounding to cents.
        MIN_PROFIT = 0.01  # 1 cent
        uc = np.minimum(uc, np.maximum(net - MIN_PROFIT, 0.0))

    # Round to cents for storage
    up = np.round(up, 2)
    uc = np.round(uc, 2)
    disc = np.round(disc, 2)

    # Post-round safety (avoid rare disc>up due to rounding)
    disc = np.minimum(disc, up)
    if min_net > 0.0:
        disc = np.minimum(disc, np.maximum(up - min_net, 0.0))

    net = np.round(up - disc, 2)
    net = np.maximum(net, 0.0)

    if not allow_neg_margin:
        MIN_PROFIT = 0.01  # 1 cent
        uc = np.minimum(uc, np.maximum(net - MIN_PROFIT, 0.0))

    return {
        "final_unit_price": up,
        "final_unit_cost": uc,
        "discount_amt": disc,
        "final_net_price": net,
    }
