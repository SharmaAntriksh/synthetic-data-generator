import numpy as np
from src.facts.sales.sales_logic.globals import State


def _get_cfg():
    models = getattr(State, "models_cfg", None) or {}
    pricing = models.get("pricing", {}) or {}
    infl = pricing.get("inflation", {}) or {}

    # Defaults: mild, predictable
    annual_rate = float(infl.get("annual_rate", 0.04))          # 4%/year
    scale_discount = bool(infl.get("scale_discount", True))     # keep discount proportional

    # clamp annual_rate so deflation doesn't go crazy
    annual_rate = float(np.clip(annual_rate, -0.50, 2.0))
    return annual_rate, scale_discount


def _global_start_month_int(order_dates: np.ndarray) -> int:
    """
    Anchor inflation to global timeline.
    Prefer State.date_pool (authoritative), else fallback to batch min.
    """
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


def build_prices(rng, order_dates, qty, price):
    """
    Simple pricing pipeline:
      - Products.UnitPrice/UnitCost are the baseline (already used by compute_prices)
      - Sales applies only time-based inflation/deflation
      - Optional: scale discount by inflation to keep discount rate stable

    Expects 'price' dict from compute_prices with:
      final_unit_price, final_unit_cost, discount_amt, final_net_price
    """
    annual_rate, scale_discount = _get_cfg()

    order_dates = np.asarray(order_dates)
    n = int(order_dates.shape[0])
    if n <= 0:
        return price

    # months since global start
    global_start_m = _global_start_month_int(order_dates)
    order_month_i = order_dates.astype("datetime64[M]").astype("int64")
    months_since = (order_month_i - global_start_m).astype(np.float64)

    # inflation factor (compounded smoothly)
    factor = (1.0 + annual_rate) ** (months_since / 12.0)
    factor = np.where(np.isfinite(factor), factor, 1.0)

    # Apply to price and cost
    price["final_unit_price"] = np.asarray(price["final_unit_price"], dtype=np.float64) * factor
    price["final_unit_cost"] = np.asarray(price["final_unit_cost"], dtype=np.float64) * factor

    # Discount handling
    if scale_discount:
        price["discount_amt"] = np.asarray(price["discount_amt"], dtype=np.float64) * factor
    else:
        price["discount_amt"] = np.asarray(price["discount_amt"], dtype=np.float64)

    # Recompute net and enforce invariants
    up = np.maximum(price["final_unit_price"], 0.0)
    uc = np.maximum(price["final_unit_cost"], 0.0)
    uc = np.minimum(uc, up)

    disc = np.maximum(price["discount_amt"], 0.0)
    disc = np.minimum(disc, up)   # never discount above price

    net = up - disc
    net = np.minimum(net, up)
    net = np.maximum(net, 0.0)

    # Ensure cost never exceeds net (otherwise margin goes negative)
    bad = uc > net
    if np.any(bad):
        uc[bad] = net[bad]

    # round to cents
    price["final_unit_price"] = np.round(up, 2)
    price["final_unit_cost"] = np.round(uc, 2)
    price["discount_amt"] = np.round(np.maximum(price["final_unit_price"] - net, 0.0), 2)
    price["final_net_price"] = np.round(price["final_unit_price"] - price["discount_amt"], 2)

    return price
