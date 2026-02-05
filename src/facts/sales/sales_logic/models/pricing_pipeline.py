import numpy as np
from src.facts.sales.sales_logic.globals import State


def _parse_bands(bands, default):
    out = []
    if isinstance(bands, list):
        for b in bands:
            if not isinstance(b, dict):
                continue
            mx = b.get("max", None)
            st = b.get("step", None)
            try:
                mx = float(mx)
                st = float(st)
            except Exception:
                continue
            if mx <= 0 or st <= 0:
                continue
            out.append((mx, st))
    if not out:
        return default
    out.sort(key=lambda t: t[0])
    return out


def _choose_steps(x, bands):
    # vectorized step selection based on magnitude of x
    step = np.empty_like(x, dtype=np.float64)
    step.fill(float(bands[-1][1]))
    for mx, st in bands:
        step = np.where(x <= mx, float(st), step)
    step = np.where(step > 0, step, 0.01)
    return step


def _quantize(x, step, rounding: str):
    if rounding == "floor":
        return np.floor(x / step) * step
    return np.round(x / step) * step


def _snap_unit_price(rng, up, cfg):
    """
    Snap to nice retail price-points:
      anchor to band-step (e.g. 5s or 10s),
      then apply ending (.99/.50/.00) with weights.
    """
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return up

    unit_cfg = cfg.get("unit_price", {}) or {}

    rounding = str(unit_cfg.get("rounding", "nearest")).strip().lower()
    if rounding not in ("nearest", "floor"):
        rounding = "nearest"

    bands = _parse_bands(
        unit_cfg.get("bands", None),
        default=[(200.0, 1.0), (1000.0, 5.0), (10000.0, 5.0), (1e18, 10.0)],
    )

    endings = unit_cfg.get("endings", None)
    if not isinstance(endings, list) or len(endings) == 0:
        endings = [{"value": 0.99, "weight": 0.70}, {"value": 0.50, "weight": 0.25}, {"value": 0.00, "weight": 0.05}]

    end_vals = []
    end_w = []
    for e in endings:
        if not isinstance(e, dict):
            continue
        try:
            v = float(e.get("value", 0.0))
            w = float(e.get("weight", 0.0))
        except Exception:
            continue
        if w <= 0:
            continue
        # endings should be [0, 0.99]-ish
        v = float(np.clip(v, 0.0, 0.99))
        end_vals.append(v)
        end_w.append(w)

    if not end_vals:
        end_vals = [0.99]
        end_w = [1.0]

    end_w = np.asarray(end_w, dtype=np.float64)
    end_w = end_w / end_w.sum()

    up = np.asarray(up, dtype=np.float64)
    up = np.where(np.isfinite(up), up, 0.0)
    up = np.maximum(up, 0.0)

    step = _choose_steps(up, bands)
    anchor = _quantize(up, step, rounding=rounding)

    # choose ending per-row
    idx = rng.choice(len(end_vals), size=up.shape[0], p=end_w)
    ending = np.asarray(end_vals, dtype=np.float64)[idx]

    snapped = anchor + ending

    # Ensure snapped stays within [anchor, anchor + step)
    # If rounding=floor, anchor is already <= up; snapped might exceed up slightly but still realistic.
    # Clamp non-negative.
    snapped = np.maximum(snapped, 0.01)
    return snapped


def _snap_cost(uc, cfg):
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return uc

    unit_cfg = cfg.get("unit_cost", {}) or {}
    rounding = str(unit_cfg.get("rounding", "nearest")).strip().lower()
    if rounding not in ("nearest", "floor"):
        rounding = "nearest"

    bands = _parse_bands(
        unit_cfg.get("bands", None),
        default=[(200.0, 0.05), (1000.0, 0.10), (10000.0, 1.0), (1e18, 5.0)],
    )

    uc = np.asarray(uc, dtype=np.float64)
    uc = np.where(np.isfinite(uc), uc, 0.0)
    uc = np.maximum(uc, 0.0)

    step = _choose_steps(uc, bands)
    snapped = _quantize(uc, step, rounding=rounding)
    snapped = np.maximum(snapped, 0.0)
    return snapped


def _snap_discount(rng, disc, up, cfg):
    enabled = bool(cfg.get("enabled", False))
    if not enabled:
        return disc

    dcfg = cfg.get("discount", {}) or {}
    rounding = str(dcfg.get("rounding", "floor")).strip().lower()
    if rounding not in ("nearest", "floor"):
        rounding = "floor"

    bands = _parse_bands(
        dcfg.get("bands", None),
        default=[(50.0, 0.50), (200.0, 1.0), (1000.0, 5.0), (5000.0, 10.0), (1e18, 25.0)],
    )

    disc = np.asarray(disc, dtype=np.float64)
    disc = np.where(np.isfinite(disc), disc, 0.0)
    disc = np.maximum(disc, 0.0)

    step = _choose_steps(np.maximum(up, 0.0), bands)
    snapped = _quantize(disc, step, rounding=rounding)
    snapped = np.maximum(snapped, 0.0)
    return snapped


def _cfg():
    """
    Minimal sales-time drift. Keep this small.

    models:
      pricing:
        inflation:
          annual_rate: 0.04
          month_volatility_sigma: 0.01
          factor_clip: [0.90, 1.15]
          scale_discount: true
          volatility_seed: 123
        appearance:
          enabled: true
          ...
    """
    models = getattr(State, "models_cfg", None) or {}
    p = models.get("pricing", {}) or {}
    infl = p.get("inflation", {}) or {}

    annual_rate = float(infl.get("annual_rate", 0.0))
    month_sigma = float(infl.get("month_volatility_sigma", 0.0))
    scale_discount = bool(infl.get("scale_discount", True))

    clip = infl.get("factor_clip", [0.0, 10.0])
    if not (isinstance(clip, (list, tuple)) and len(clip) == 2):
        clip = [0.0, 10.0]
    lo, hi = float(clip[0]), float(clip[1])
    if hi < lo:
        lo, hi = hi, lo

    vol_seed = int(infl.get("volatility_seed", 0))

    annual_rate = float(np.clip(annual_rate, -0.50, 2.0))
    month_sigma = float(np.clip(month_sigma, 0.0, 0.25))
    lo = float(max(lo, 0.0))
    hi = float(max(hi, lo))

    appearance = p.get("appearance", {}) or {}
    return annual_rate, month_sigma, lo, hi, scale_discount, vol_seed, appearance


def _global_start_month_int(order_dates: np.ndarray) -> int:
    dp = getattr(State, "date_pool", None)
    if dp is not None:
        try:
            if len(dp) > 0:
                d0 = np.min(dp.astype("datetime64[D]"))
                return d0.astype("datetime64[M]").astype("int64")
        except Exception:
            pass

    d0 = np.min(order_dates.astype("datetime64[D]"))
    return d0.astype("datetime64[M]").astype("int64")


def _month_noise(month_int: int, seed: int, sigma: float) -> float:
    if sigma <= 0.0:
        return 1.0
    s = (int(seed) ^ (int(month_int) * 1000003)) & 0xFFFFFFFF
    rng = np.random.default_rng(s)

    mu = -0.5 * (float(sigma) ** 2)  # unbiased: expected multiplier = 1
    return float(rng.lognormal(mean=mu, sigma=float(sigma), size=1)[0])



def build_prices(rng, order_dates, qty, price):
    """
    Apply only mild time drift to the prices computed from Products,
    then snap to nice retail-looking price points.

    Expects price dict from compute_prices():
      final_unit_price, final_unit_cost, discount_amt, final_net_price
    """
    annual_rate, month_sigma, clip_lo, clip_hi, scale_discount, vol_seed, appearance = _cfg()

    order_dates = np.asarray(order_dates)
    n = int(order_dates.shape[0])
    if n <= 0:
        return price

    order_month_i = order_dates.astype("datetime64[M]").astype("int64")
    uniq_months, inv = np.unique(order_month_i, return_inverse=True)

    start_m = _global_start_month_int(order_dates)
    months_since = (uniq_months - start_m).astype(np.float64)

    infl = (1.0 + annual_rate) ** (months_since / 12.0)
    infl = np.where(np.isfinite(infl), infl, 1.0)

    if month_sigma > 0.0:
        noises = np.array([_month_noise(int(m), vol_seed, month_sigma) for m in uniq_months], dtype=np.float64)
    else:
        noises = np.ones_like(infl, dtype=np.float64)

    factor_u = infl * noises
    factor_u = np.clip(factor_u, clip_lo, clip_hi)
    factor = factor_u[inv]

    up = np.asarray(price["final_unit_price"], dtype=np.float64) * factor
    uc = np.asarray(price["final_unit_cost"], dtype=np.float64) * factor

    disc = np.asarray(price["discount_amt"], dtype=np.float64)
    if scale_discount:
        disc = disc * factor

    # Invariants
    up = np.where(np.isfinite(up), up, 0.0)
    uc = np.where(np.isfinite(uc), uc, 0.0)
    disc = np.where(np.isfinite(disc), disc, 0.0)

    up = np.maximum(up, 0.0)
    uc = np.maximum(uc, 0.0)
    disc = np.maximum(disc, 0.0)

    disc = np.minimum(disc, up)
    net = np.clip(up - disc, 0.0, up)

    # keep cost <= net (avoid negative margin after scaling)
    bad = uc > net
    if np.any(bad):
        uc[bad] = net[bad]

    # ---------------------------------------------------------
    # SNAP / APPEARANCE (THIS IS THE IMPORTANT NEW BIT)
    # ---------------------------------------------------------
    up = _snap_unit_price(rng, up, appearance)

    # Re-quantize discount AFTER snapping price, so it looks clean too
    disc = np.minimum(disc, up)
    disc = _snap_discount(rng, disc, up, appearance)

    # recompute net after snapping
    disc = np.minimum(disc, up)
    net = np.clip(up - disc, 0.0, up)

    # snap cost and re-enforce invariants
    uc = _snap_cost(uc, appearance)
    uc = np.minimum(uc, net)

    # cents
    up = np.round(up, 2)
    uc = np.round(uc, 2)
    disc = np.round(np.maximum(up - net, 0.0), 2)
    net = np.round(up - disc, 2)

    price["final_unit_price"] = up
    price["final_unit_cost"] = uc
    price["discount_amt"] = disc
    price["final_net_price"] = net
    return price
