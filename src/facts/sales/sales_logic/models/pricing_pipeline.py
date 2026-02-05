import numpy as np
from src.facts.sales.sales_logic.globals import State


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

    return annual_rate, month_sigma, lo, hi, scale_discount, vol_seed


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

    # unbiased lognormal: E[exp(N(mu, sigma^2))] = 1 when mu = -0.5*sigma^2
    mu = -0.5 * (float(sigma) ** 2)
    return float(rng.lognormal(mean=mu, sigma=float(sigma), size=1)[0])


def build_prices(rng, order_dates, qty, price):
    """
    Apply only mild time drift to the prices computed from Products.

    Expects price dict from compute_prices():
      final_unit_price, final_unit_cost, discount_amt, final_net_price
    """
    annual_rate, month_sigma, clip_lo, clip_hi, scale_discount, vol_seed = _cfg()

    order_dates = np.asarray(order_dates)
    n = int(order_dates.shape[0])
    if n <= 0:
        return price

    order_month_i = order_dates.astype("datetime64[M]").astype("int64")
    uniq_months, inv = np.unique(order_month_i, return_inverse=True)

    start_m = _global_start_month_int(order_dates)
    months_since = (uniq_months - start_m).astype(np.float64)

    # Inflation/deflation
    infl = (1.0 + annual_rate) ** (months_since / 12.0)
    infl = np.where(np.isfinite(infl), infl, 1.0)

    # Month volatility (deterministic per month)
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

    # discount cannot exceed price
    disc = np.minimum(disc, up)

    net = up - disc
    net = np.maximum(net, 0.0)
    net = np.minimum(net, up)

    # keep cost <= net (avoid negative margin after scaling)
    bad = uc > net
    if np.any(bad):
        uc[bad] = net[bad]

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
