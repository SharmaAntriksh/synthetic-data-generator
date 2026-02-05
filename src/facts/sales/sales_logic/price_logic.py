import numpy as np
from .globals import State


def _to_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def _cfg_markdown():
    """
    models:
      pricing:
        markdown:
          enabled: true
          ladder:
            - {kind: none, value: 0.0,  weight: 0.55}
            - {kind: pct,  value: 0.05, weight: 0.20}
            - {kind: pct,  value: 0.10, weight: 0.12}
            - {kind: pct,  value: 0.15, weight: 0.08}
            - {kind: amt,  value: 25.0, weight: 0.05}
          max_pct_of_price: 0.50
          min_net_price: 0.01
          allow_negative_margin: false
          appearance:
            quantize_discount: true
            discount_rounding: floor   # floor | nearest
            discount_bands:
              - {max: 50, step: 0.50}
              - {max: 200, step: 1}
              - {max: 1000, step: 5}
              - {max: 5000, step: 10}
              - {max: 1e18, step: 25}
    """
    models = getattr(State, "models_cfg", None) or {}
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

    # sanitize ladder
    kinds, values, weights = [], [], []
    for item in ladder:
        if not isinstance(item, dict):
            continue
        k = str(item.get("kind", "none")).strip().lower()
        v = float(item.get("value", 0.0) or 0.0)
        w = float(item.get("weight", 0.0) or 0.0)
        if w <= 0:
            continue
        if k not in ("pct", "amt", "none"):
            continue
        if k == "pct":
            v = float(np.clip(v, 0.0, 1.0))
        else:
            v = max(0.0, v)
        kinds.append(k)
        values.append(v)
        weights.append(w)

    if not kinds:
        kinds = ["none"]
        values = [0.0]
        weights = [1.0]

    w = np.array(weights, dtype=np.float64)
    w = w / w.sum()

    # appearance (optional)
    appearance = md.get("appearance", {}) or {}
    quantize_discount = bool(appearance.get("quantize_discount", True))
    discount_rounding = str(appearance.get("discount_rounding", "floor")).strip().lower()
    if discount_rounding not in ("floor", "nearest"):
        discount_rounding = "floor"

    bands = appearance.get("discount_bands")
    if not isinstance(bands, list) or len(bands) == 0:
        # Default: scales well from low to high ticket items
        bands = [
            {"max": 50, "step": 0.50},
            {"max": 200, "step": 1},
            {"max": 1000, "step": 5},
            {"max": 5000, "step": 10},
            {"max": 1e18, "step": 25},
        ]

    discount_bands = _parse_bands(bands, default=[(50.0, 0.50), (200.0, 1.0), (1000.0, 5.0), (5000.0, 10.0), (1e18, 25.0)])

    return (
        enabled,
        kinds,
        np.array(values, dtype=np.float64),
        w,
        max_pct,
        min_net,
        allow_neg_margin,
        quantize_discount,
        discount_rounding,
        discount_bands,
    )


def _parse_bands(bands, default):
    """
    bands: list of dicts {max, step} -> sorted list[(max, step)]
    """
    out = []
    for b in bands:
        if not isinstance(b, dict):
            continue
        mx = _to_float(b.get("max"), None)
        st = _to_float(b.get("step"), None)
        if mx is None or st is None or mx <= 0 or st <= 0:
            continue
        out.append((float(mx), float(st)))

    if not out:
        return default

    out.sort(key=lambda t: t[0])
    return out


def _as_f64(x, n):
    a = np.asarray(x, dtype=np.float64)
    if a.shape[0] != int(n):
        raise ValueError("Array length mismatch")
    a = np.where(np.isfinite(a), a, 0.0)
    return a


def _step_for_price(up: np.ndarray, bands):
    """
    Vectorized step lookup based on UnitPrice magnitude.
    """
    step = np.empty_like(up, dtype=np.float64)
    step.fill(bands[-1][1])
    for mx, st in bands:
        step = np.where(up <= mx, st, step)
    return step


def _quantize_discount(disc: np.ndarray, up: np.ndarray, bands, rounding: str) -> np.ndarray:
    """
    Quantize discount to clean increments (0.5, 1, 5, 10, 25, ...)
    chosen per-row based on UnitPrice.
    """
    step = _step_for_price(up, bands)
    step = np.where(step > 0, step, 0.01)

    if rounding == "nearest":
        q = np.round(disc / step) * step
    else:
        # floor: avoids rounding up discounts (more realistic)
        q = np.floor(disc / step) * step

    q = np.maximum(q, 0.0)
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

    # Optional tiny per-row perturbations (keep defaults off)
    pp = float(price_pressure) if price_pressure is not None else 1.0
    if not np.isfinite(pp) or pp <= 0:
        pp = 1.0
    up = up * pp
    uc = uc * pp

    if row_price_jitter_pct:
        j = float(row_price_jitter_pct)
        if np.isfinite(j) and j > 0:
            mult = rng.uniform(1.0 - j, 1.0 + j, size=n).astype(np.float64, copy=False)
            up = up * mult
            uc = uc * mult

    up = np.maximum(up, 0.0)
    uc = np.maximum(uc, 0.0)

    (
        enabled,
        kinds,
        values,
        probs,
        max_pct,
        min_net,
        allow_neg_margin,
        quantize_discount,
        discount_rounding,
        discount_bands,
    ) = _cfg_markdown()

    disc = np.zeros(n, dtype=np.float64)

    if enabled:
        idx = rng.choice(len(kinds), size=n, replace=True, p=probs)

        # Apply ladder
        for i, k in enumerate(kinds):
            m = (idx == i)
            if not np.any(m):
                continue
            v = float(values[i])
            if k == "pct":
                disc[m] = up[m] * v
            elif k == "amt":
                disc[m] = v
            else:
                disc[m] = 0.0

    # Base constraints before quantization
    disc = np.maximum(disc, 0.0)
    disc = np.minimum(disc, up * max_pct)
    if min_net > 0.0:
        disc = np.minimum(disc, np.maximum(up - min_net, 0.0))
    disc = np.minimum(disc, up)

    # Quantize to clean increments (the main “looks realistic” fix)
    if quantize_discount and enabled:
        disc = _quantize_discount(disc, up, discount_bands, discount_rounding)

        # Re-apply constraints after quantization (quantization can bump slightly)
        disc = np.maximum(disc, 0.0)
        disc = np.minimum(disc, up * max_pct)
        if min_net > 0.0:
            disc = np.minimum(disc, np.maximum(up - min_net, 0.0))
        disc = np.minimum(disc, up)

    net = up - disc
    net = np.maximum(net, 0.0)

    # Invariants
    uc = np.minimum(uc, up)
    if not allow_neg_margin:
        uc = np.minimum(uc, net)

    # Round to cents for storage
    up = np.round(up, 2)
    uc = np.round(uc, 2)
    disc = np.round(disc, 2)
    net = np.round(up - disc, 2)

    return {
        "final_unit_price": up,
        "final_unit_cost": uc,
        "discount_amt": disc,
        "final_net_price": net,
    }
