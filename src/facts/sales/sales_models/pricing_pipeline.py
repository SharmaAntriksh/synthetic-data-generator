"""
Sales pricing pipeline.

Applies time drift to UnitPrice, snaps prices/costs, and applies markdown discounts
using a ladder + caps (configured in models.yaml).

"""
from __future__ import annotations

import numpy as np

from src.facts.sales.sales_logic import State


# ---------------------------------------------------------------------
# Markdown/discount ladder helpers (pricing_pipeline)
# ---------------------------------------------------------------------
_MD_PIPE_CACHE_KEY = None
_MD_PIPE_CACHE_VAL = None


def _md_pipeline_cfg():
    """
    Pull the markdown ladder + constraints from State.models_cfg (models.yaml).
    Returns:
      enabled, ladder_vals(np.ndarray sorted), rounding(str),
      max_pct(float), min_net(float), allow_neg_margin(bool)
    """
    global _MD_PIPE_CACHE_KEY, _MD_PIPE_CACHE_VAL

    models = getattr(State, "models_cfg", None) or {}
    p = models.get("pricing", {}) or {}
    md = p.get("markdown", {}) or {}

    key = (id(models), id(p), id(md))
    if key == _MD_PIPE_CACHE_KEY and _MD_PIPE_CACHE_VAL is not None:
        return _MD_PIPE_CACHE_VAL

    enabled = bool(md.get("enabled", False))

    max_pct = md.get("max_pct_of_price", None)
    if max_pct is None:
        max_pct = (p.get("discount", {}) or {}).get("max_pct_of_price", 1.0)
    max_pct = float(max_pct) if max_pct is not None else 1.0
    if not np.isfinite(max_pct) or max_pct <= 0.0:
        max_pct = 1.0

    min_net = md.get("min_net_price", None)
    if min_net is None:
        min_net = (p.get("floors", {}) or {}).get("min_net_price", 0.01)
    min_net = float(min_net) if min_net is not None else 0.01
    if not np.isfinite(min_net) or min_net < 0.0:
        min_net = 0.0

    allow_neg_margin = bool(md.get("allow_negative_margin", False))

    rounding = (((p.get("appearance", {}) or {}).get("discount", {}) or {}).get("rounding", "nearest")) or "nearest"
    rounding = str(rounding).strip().lower()
    if rounding not in ("nearest", "floor"):
        rounding = "nearest"

    ladder = md.get("ladder", []) or []
    vals = [0.0]
    for it in ladder:
        if not isinstance(it, dict):
            continue
        kind = str(it.get("kind", "")).strip().lower()
        if kind == "none":
            vals.append(0.0)
        elif kind == "amt":
            try:
                v = float(it.get("value", 0.0))
            except Exception:
                v = 0.0
            if np.isfinite(v) and v >= 0.0:
                vals.append(v)

    ladder_vals = np.unique(np.asarray(vals, dtype=np.float64))
    ladder_vals = ladder_vals[np.isfinite(ladder_vals)]
    ladder_vals = np.sort(ladder_vals)
    if ladder_vals.size == 0:
        ladder_vals = np.asarray([0.0], dtype=np.float64)

    _MD_PIPE_CACHE_KEY = key
    _MD_PIPE_CACHE_VAL = (enabled, ladder_vals, rounding, max_pct, min_net, allow_neg_margin)
    return _MD_PIPE_CACHE_VAL


def _snap_to_ladder(disc: np.ndarray, cap: np.ndarray, ladder_vals: np.ndarray, rounding: str) -> np.ndarray:
    """
    Snap discount to ladder values, while respecting per-row cap.
    - If rounding == "nearest": pick nearest ladder value, but if it exceeds cap, snap DOWN to <=cap.
    - If rounding == "floor": always snap DOWN to <=min(disc, cap).
    """
    disc = np.asarray(disc, dtype=np.float64)
    cap = np.asarray(cap, dtype=np.float64)

    disc = np.where(np.isfinite(disc), disc, 0.0)
    cap = np.where(np.isfinite(cap), cap, 0.0)

    disc = np.maximum(disc, 0.0)
    cap = np.maximum(cap, 0.0)

    target = np.minimum(disc, cap)

    L = np.asarray(ladder_vals, dtype=np.float64)
    if L.size == 0:
        return np.zeros_like(target, dtype=np.float64)

    def floor_ladder(x: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(L, x, side="right") - 1
        idx = np.clip(idx, 0, L.size - 1)
        return L[idx]

    if rounding == "floor":
        return floor_ladder(target)

    idx = np.searchsorted(L, target, side="left")
    idx_hi = np.clip(idx, 0, L.size - 1)
    idx_lo = np.clip(idx - 1, 0, L.size - 1)

    lo = L[idx_lo]
    hi = L[idx_hi]

    choose_hi = (np.abs(hi - target) < np.abs(target - lo))
    snapped = np.where(choose_hi, hi, lo)

    snapped = np.minimum(snapped, cap)
    snapped = floor_ladder(snapped)
    return snapped


# ---------------------------------------------------------------------
# Module caches (process-local; safe with multiprocessing)
# ---------------------------------------------------------------------
_APPEAR_CACHE_KEY = None
_APPEAR_CACHE_VAL = None

_MONTH_NOISE_CACHE = {}  # (vol_seed, sigma, month_int) -> multiplier


def _as_f64(x):
    a = np.asarray(x, dtype=np.float64)
    return np.where(np.isfinite(a), a, 0.0)


def _parse_bands_to_arrays(bands, default):
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
        out = list(default)

    out.sort(key=lambda t: t[0])
    maxs = np.asarray([m for m, _ in out], dtype=np.float64)
    steps = np.asarray([s for _, s in out], dtype=np.float64)
    if maxs.size == 0:
        maxs = np.asarray([1e18], dtype=np.float64)
        steps = np.asarray([0.01], dtype=np.float64)
    return maxs, steps


def _choose_step_by_magnitude(x: np.ndarray, band_max: np.ndarray, band_step: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    idx = np.searchsorted(band_max, x, side="left")
    idx = np.minimum(idx, band_step.size - 1)
    step = band_step[idx]
    return np.where(step > 0.0, step, 0.01)


def _quantize(x: np.ndarray, step: np.ndarray, rounding: str) -> np.ndarray:
    if rounding == "floor":
        return np.floor(x / step) * step
    return np.round(x / step) * step


def _safe_prob(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.clip(w, 0.0, None)
    s = float(w.sum())
    if s <= 0.0:
        return np.full(w.shape[0], 1.0 / max(1, w.shape[0]), dtype=np.float64)
    return w / s


def _parse_endings(endings, *, default_if_missing: bool):
    if not isinstance(endings, list) or len(endings) == 0:
        if not default_if_missing:
            return None, None
        endings = [
            {"value": 0.99, "weight": 0.70},
            {"value": 0.50, "weight": 0.25},
            {"value": 0.00, "weight": 0.05},
        ]

    vals = []
    wts = []
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
        v = float(np.clip(v, 0.0, 0.99))
        vals.append(v)
        wts.append(w)

    if not vals:
        if not default_if_missing:
            return None, None
        vals = [0.99]
        wts = [1.0]

    w = _safe_prob(np.asarray(wts, dtype=np.float64))
    return np.asarray(vals, dtype=np.float64), w


def _appearance_cfg():
    global _APPEAR_CACHE_KEY, _APPEAR_CACHE_VAL

    models = getattr(State, "models_cfg", None) or {}
    key = id(models)
    if _APPEAR_CACHE_KEY == key and _APPEAR_CACHE_VAL is not None:
        return _APPEAR_CACHE_VAL

    p = models.get("pricing", {}) or {}
    appearance = p.get("appearance", {}) or {}

    enabled = bool(appearance.get("enabled", False))

    unit_cfg = appearance.get("unit_price", {}) or {}
    up_round = str(unit_cfg.get("rounding", "nearest")).strip().lower()
    if up_round not in ("nearest", "floor"):
        up_round = "nearest"

    up_max, up_step = _parse_bands_to_arrays(
        unit_cfg.get("bands", None),
        default=[(200.0, 1.0), (1000.0, 5.0), (10000.0, 5.0), (1e18, 10.0)],
    )
    up_end_vals, up_end_w = _parse_endings(unit_cfg.get("endings", None), default_if_missing=True)

    cost_cfg = appearance.get("unit_cost", {}) or {}
    uc_round = str(cost_cfg.get("rounding", "nearest")).strip().lower()
    if uc_round not in ("nearest", "floor"):
        uc_round = "nearest"

    uc_max, uc_step = _parse_bands_to_arrays(
        cost_cfg.get("bands", None),
        default=[(200.0, 0.05), (1000.0, 0.10), (10000.0, 1.0), (1e18, 5.0)],
    )
    uc_end_vals, uc_end_w = _parse_endings(cost_cfg.get("endings", None), default_if_missing=False)

    disc_cfg = appearance.get("discount", {}) or {}
    d_round = str(disc_cfg.get("rounding", "floor")).strip().lower()
    if d_round not in ("nearest", "floor"):
        d_round = "floor"

    d_max, d_step = _parse_bands_to_arrays(
        disc_cfg.get("bands", None),
        default=[(50.0, 0.50), (200.0, 1.0), (1000.0, 5.0), (5000.0, 10.0), (1e18, 25.0)],
    )

    out = {
        "enabled": enabled,
        "up_round": up_round,
        "up_band_max": up_max,
        "up_band_step": up_step,
        "up_end_vals": up_end_vals,
        "up_end_w": up_end_w,
        "uc_round": uc_round,
        "uc_band_max": uc_max,
        "uc_band_step": uc_step,
        "uc_end_vals": uc_end_vals,
        "uc_end_w": uc_end_w,
        "d_round": d_round,
        "d_band_max": d_max,
        "d_band_step": d_step,
    }

    _APPEAR_CACHE_KEY = key
    _APPEAR_CACHE_VAL = out
    return out


def _snap_unit_price(rng, up: np.ndarray, appearance_cfg: dict) -> np.ndarray:
    if not appearance_cfg.get("enabled", False):
        return up

    up = _as_f64(up)
    up = np.maximum(up, 0.0)

    step = _choose_step_by_magnitude(up, appearance_cfg["up_band_max"], appearance_cfg["up_band_step"])
    step = np.maximum(step, 1.0)

    anchor = _quantize(up, step, rounding=appearance_cfg["up_round"])

    end_vals = appearance_cfg["up_end_vals"]
    end_w = appearance_cfg["up_end_w"]
    idx = rng.choice(end_vals.size, size=up.shape[0], p=end_w)
    ending = end_vals[idx]

    snapped = anchor + ending
    return np.maximum(snapped, 0.01)


def _snap_cost(rng, uc: np.ndarray, appearance_cfg: dict) -> np.ndarray:
    if not appearance_cfg.get("enabled", False):
        return uc

    uc = _as_f64(uc)
    uc = np.maximum(uc, 0.0)

    step = _choose_step_by_magnitude(uc, appearance_cfg["uc_band_max"], appearance_cfg["uc_band_step"])
    anchor = _quantize(uc, step, rounding=appearance_cfg["uc_round"])

    end_vals = appearance_cfg.get("uc_end_vals", None)
    end_w = appearance_cfg.get("uc_end_w", None)

    if end_vals is not None and end_w is not None and end_vals.size > 0:
        idx = rng.choice(end_vals.size, size=uc.shape[0], p=end_w)
        ending = end_vals[idx]

        snapped = anchor.copy()
        mask = step >= 1.0
        if mask.any():
            snapped[mask] = np.floor(anchor[mask]) + ending[mask]
    else:
        snapped = anchor

    return np.maximum(snapped, 0.0)


def _snap_discount(disc: np.ndarray, up: np.ndarray, appearance_cfg: dict) -> np.ndarray:
    if not appearance_cfg.get("enabled", False):
        return disc

    disc = _as_f64(disc)
    disc = np.maximum(disc, 0.0)

    up = _as_f64(up)
    up = np.maximum(up, 0.0)

    step = _choose_step_by_magnitude(up, appearance_cfg["d_band_max"], appearance_cfg["d_band_step"])
    step = np.maximum(step, 1.0)

    snapped = _quantize(disc, step, rounding=appearance_cfg["d_round"])
    snapped = np.minimum(snapped, up)
    return np.maximum(snapped, 0.0)


def _cfg():
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
                d0 = np.min(np.asarray(dp).astype("datetime64[D]"))
                return int(d0.astype("datetime64[M]").astype("int64"))
        except Exception:
            pass

    d0 = np.min(np.asarray(order_dates).astype("datetime64[D]"))
    return int(d0.astype("datetime64[M]").astype("int64"))


def _month_noise(month_int: int, seed: int, sigma: float) -> float:
    if sigma <= 0.0:
        return 1.0

    key = (int(seed), float(sigma), int(month_int))
    v = _MONTH_NOISE_CACHE.get(key)
    if v is not None:
        return float(v)

    s = (int(seed) ^ (int(month_int) * 1000003)) & 0xFFFFFFFF
    rng = np.random.default_rng(s)

    mu = -0.5 * (float(sigma) ** 2)
    v = float(rng.lognormal(mean=mu, sigma=float(sigma), size=1)[0])

    _MONTH_NOISE_CACHE[key] = v
    return v


def build_prices(rng, order_dates, qty, price):
    _ = qty  # intentionally unused

    annual_rate, month_sigma, clip_lo, clip_hi, scale_discount, vol_seed = _cfg()

    order_dates = np.asarray(order_dates)
    n = int(order_dates.shape[0])
    if n <= 0:
        return price

    order_month_i = order_dates.astype("datetime64[M]").astype("int64")
    uniq_months, inv = np.unique(order_month_i, return_inverse=True)

    start_m = _global_start_month_int(order_dates)
    months_since = (uniq_months.astype(np.int64) - int(start_m)).astype(np.float64)

    infl = (1.0 + annual_rate) ** (months_since / 12.0)
    infl = np.where(np.isfinite(infl), infl, 1.0)

    if month_sigma > 0.0:
        noises = np.fromiter(
            (_month_noise(int(m), vol_seed, month_sigma) for m in uniq_months),
            dtype=np.float64,
            count=uniq_months.size,
        )
    else:
        noises = np.ones_like(infl, dtype=np.float64)

    factor_u = np.clip(infl * noises, clip_lo, clip_hi)
    factor = factor_u[inv]

    base_up = _as_f64(price["final_unit_price"])
    base_uc = _as_f64(price["final_unit_cost"])
    base_disc = _as_f64(price["discount_amt"])

    up = np.maximum(base_up * factor, 0.0)

    disc = np.maximum(base_disc, 0.0)
    if scale_discount:
        disc = np.maximum(disc * factor, 0.0)

    appearance = _appearance_cfg()
    up = _snap_unit_price(rng, up, appearance)

    uc = np.maximum(base_uc, 0.0)
    uc = np.minimum(uc, up)
    uc = _snap_cost(rng, uc, appearance)
    uc = np.minimum(uc, up)

    md_enabled, ladder_vals, md_rounding, max_pct, min_net, allow_neg_margin = _md_pipeline_cfg()

    cap = np.minimum(up * max_pct, up)
    if min_net > 0.0:
        cap = np.minimum(cap, np.maximum(up - min_net, 0.0))

    if not allow_neg_margin:
        MIN_PROFIT = 0.01
        cap = np.minimum(cap, np.maximum(up - (uc + MIN_PROFIT), 0.0))

    cap = np.clip(cap, 0.0, up)

    if md_enabled and ladder_vals is not None and np.asarray(ladder_vals).size > 0:
        disc = _snap_to_ladder(disc, cap, ladder_vals, md_rounding)
    else:
        disc = np.minimum(disc, up)
        disc = _snap_discount(disc, up, appearance)
        disc = np.minimum(disc, cap)

    disc = np.minimum(disc, up)
    net = np.maximum(up - disc, 0.0)

    up = np.where(np.isfinite(up), up, 0.0)
    uc = np.where(np.isfinite(uc), uc, 0.0)
    disc = np.where(np.isfinite(disc), disc, 0.0)
    net = np.where(np.isfinite(net), net, 0.0)

    up = np.round(up, 2)
    uc = np.round(uc, 2)
    disc = np.round(np.clip(disc, 0.0, up), 2)
    net = np.round(np.maximum(up - disc, 0.0), 2)

    uc = np.minimum(uc, up)
    if not allow_neg_margin:
        uc = np.minimum(uc, np.maximum(net - 0.01, 0.0))

    uc = np.round(uc, 2)

    price["final_unit_price"] = up
    price["final_unit_cost"] = uc
    price["discount_amt"] = disc
    price["final_net_price"] = net
    return price