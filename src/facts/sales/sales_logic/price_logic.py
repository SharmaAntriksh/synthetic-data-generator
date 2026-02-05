import numpy as np
from .globals import State


def _cfg_markdown():
    """
    models:
      pricing:
        markdown:
          enabled: true
          # ladder items: each is {kind: pct|amt|none, value: float, weight: float}
          ladder:
            - {kind: none, value: 0.0,  weight: 0.55}
            - {kind: pct,  value: 0.05, weight: 0.20}
            - {kind: pct,  value: 0.10, weight: 0.12}
            - {kind: pct,  value: 0.15, weight: 0.08}
            - {kind: amt,  value: 25.0, weight: 0.05}
          max_pct_of_price: 0.50
          min_net_price: 0.01
          allow_negative_margin: false
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

    return enabled, kinds, np.array(values, dtype=np.float64), w, max_pct, min_net, allow_neg_margin


def _as_f64(x, n):
    a = np.asarray(x, dtype=np.float64)
    if a.shape[0] != int(n):
        raise ValueError("Array length mismatch")
    a = np.where(np.isfinite(a), a, 0.0)
    return a


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

    enabled, kinds, values, probs, max_pct, min_net, allow_neg_margin = _cfg_markdown()

    disc = np.zeros(n, dtype=np.float64)

    if enabled:
        idx = rng.choice(len(kinds), size=n, replace=True, p=probs)

        # Vectorized application by kind
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

    # Hard constraints: discount cannot exceed price, and preserve min_net if requested
    disc = np.maximum(disc, 0.0)
    disc = np.minimum(disc, up * max_pct)
    if min_net > 0.0:
        disc = np.minimum(disc, np.maximum(up - min_net, 0.0))
    disc = np.minimum(disc, up)

    net = up - disc
    net = np.maximum(net, 0.0)
    if min_net > 0.0:
        net = np.maximum(net, np.minimum(up, min_net) if min_net > 0 else 0.0) if False else net  # no-op; net already ok

    # Invariants
    uc = np.minimum(uc, up)
    if not allow_neg_margin:
        uc = np.minimum(uc, net)

    # Round to cents
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
