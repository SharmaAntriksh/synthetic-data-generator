import numpy as np
import pandas as pd


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


def _parse_bands(bands, default_bands):
    """
    bands: list of dicts with keys: max, step
    """
    if not isinstance(bands, list) or len(bands) == 0:
        return default_bands

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
        return default_bands

    out.sort(key=lambda t: t[0])
    return out


def _step_for_value(x: np.ndarray, bands):
    """
    Correct vectorized step assignment by bands.

    For each value x:
      pick the step for the first band where x <= band.max
      else use the last band's step.
    """
    a = np.asarray(x, dtype=np.float64)
    a2 = np.where(np.isfinite(a), a, np.inf)

    mx = np.asarray([b[0] for b in bands], dtype=np.float64)
    st = np.asarray([b[1] for b in bands], dtype=np.float64)
    if mx.size == 0:
        return np.ones_like(a2, dtype=np.float64)

    idx = np.searchsorted(mx, a2, side="left")
    idx = np.clip(idx, 0, mx.size - 1)
    return st[idx]


def _snap_unit_price_to_points(unit_price: np.ndarray, bands, ending: float) -> np.ndarray:
    """
    Snap prices to magnitude-based steps with a psychological ending (e.g., .99).
    """
    p = np.asarray(unit_price, dtype=np.float64)
    p = np.where(np.isfinite(p), p, np.nan)
    p = np.maximum(p, 0.0)

    ending = float(np.clip(float(ending), 0.0, 0.99))
    step = _step_for_value(p, bands)

    base = np.round(p / step) * step
    if ending <= 0.0:
        snapped = base
    else:
        snapped = base - (1.0 - ending)

    snapped = np.maximum(snapped, 0.0)
    return snapped


def _round_to_step(x: np.ndarray, bands) -> np.ndarray:
    """
    Round values to clean increments (no psychological endings).
    Keeps NaNs as NaNs (caller must sanitize upstream).
    """
    a = np.asarray(x, dtype=np.float64)
    a = np.where(np.isfinite(a), a, np.nan)
    a = np.maximum(a, 0.0)

    step = _step_for_value(a, bands)
    rounded = np.round(a / step) * step
    rounded = np.maximum(rounded, 0.0)
    return rounded


def _sanitize_unit_price(out: pd.DataFrame, min_price: float | None, max_price: float | None) -> None:
    """
    Ensure UnitPrice is finite and > 0 (or >= min_price when provided).
    Replace bad values with min_price (preferred) or median of valid prices.
    """
    up = out["UnitPrice"].to_numpy(dtype=np.float64, copy=False)
    bad = ~np.isfinite(up) | (up <= 0.0)
    if bad.any():
        if min_price is not None and np.isfinite(min_price) and min_price > 0:
            fallback = float(min_price)
        else:
            good = up[np.isfinite(up) & (up > 0.0)]
            fallback = float(np.nanmedian(good)) if good.size else 1.0

        up2 = up.copy()
        up2[bad] = fallback
        out["UnitPrice"] = up2

    if min_price is not None:
        out["UnitPrice"] = out["UnitPrice"].clip(lower=float(min_price))
    if max_price is not None:
        out["UnitPrice"] = out["UnitPrice"].clip(upper=float(max_price))


def _sample_margin(rng: np.random.Generator, n: int, min_margin: float, max_margin: float) -> np.ndarray:
    lo = float(np.clip(min_margin, 0.01, 0.95))
    hi = float(np.clip(max_margin, lo + 0.01, 0.99))
    return rng.uniform(lo, hi, size=n).astype(np.float64)


def _sanitize_unit_cost_from_price(
    out: pd.DataFrame,
    rng: np.random.Generator,
    min_margin: float | None,
    max_margin: float | None,
) -> None:
    """
    If UnitCost has NaN/inf/negative values, fill using UnitPrice and a margin band.
    """
    up = out["UnitPrice"].to_numpy(dtype=np.float64, copy=False)
    uc = out["UnitCost"].to_numpy(dtype=np.float64, copy=False)

    bad = ~np.isfinite(uc) | (uc < 0.0)
    if not bad.any():
        return

    # fallbacks if margin bounds missing
    mm_lo = 0.20 if min_margin is None else float(min_margin)
    mm_hi = 0.35 if max_margin is None else float(max_margin)
    if not (np.isfinite(mm_lo) and np.isfinite(mm_hi) and 0.0 < mm_lo < mm_hi < 1.0):
        mm_lo, mm_hi = 0.20, 0.35

    m = _sample_margin(rng, int(bad.sum()), mm_lo, mm_hi)
    fill = up[bad] * (1.0 - m)

    uc2 = uc.copy()
    uc2[bad] = fill
    out["UnitCost"] = uc2


def _apply_brand_price_normalization(
    out: pd.DataFrame,
    brand_cfg: dict,
    rng: np.random.Generator,
    min_price: float | None,
    max_price: float | None,
) -> None:
    """
    Brand-level price normalization in log space; low-count brands are left unchanged.
    """
    if not isinstance(brand_cfg, dict) or not _to_bool(brand_cfg.get("enabled"), False):
        return

    brand_col = brand_cfg.get("brand_col")
    if not brand_col:
        if "Brand" in out.columns:
            brand_col = "Brand"
        elif "BrandName" in out.columns:
            brand_col = "BrandName"
        else:
            return
    if brand_col not in out.columns:
        return

    alpha = _to_float(brand_cfg.get("alpha"), 0.7)
    if alpha is None:
        alpha = 0.7
    alpha = float(np.clip(alpha, 0.0, 1.0))

    min_factor = _to_float(brand_cfg.get("min_factor"), 0.6)
    max_factor = _to_float(brand_cfg.get("max_factor"), 1.6)
    if min_factor is None:
        min_factor = 0.6
    if max_factor is None:
        max_factor = 1.6
    if max_factor <= 0 or min_factor <= 0 or max_factor < min_factor:
        return

    try:
        min_count = int(brand_cfg.get("min_count", 10))
    except Exception:
        min_count = 10
    min_count = max(1, min_count)

    noise_sd = _to_float(brand_cfg.get("noise_sd"), 0.0) or 0.0
    noise_sd = float(max(0.0, noise_sd))

    up = out["UnitPrice"].to_numpy(dtype=np.float64, copy=False)
    finite = np.isfinite(up) & (up > 0.0)
    if not finite.any():
        return

    logp = np.full(up.shape[0], np.nan, dtype=np.float64)
    logp[finite] = np.log(up[finite])

    global_med = float(np.nanmedian(logp))
    if not np.isfinite(global_med):
        return

    tmp = pd.DataFrame({"_brand": out[brand_col].astype("string"), "_logp": logp})
    grp = tmp.groupby("_brand")["_logp"]
    brand_med = grp.median()
    brand_cnt = grp.count()

    delta = (global_med - brand_med).astype("float64")
    factor = np.exp(alpha * delta.to_numpy(dtype=np.float64))
    factor = pd.Series(factor, index=brand_med.index, dtype="float64")

    if noise_sd > 0.0 and len(factor) > 0:
        eps = rng.normal(loc=0.0, scale=noise_sd, size=len(factor))
        factor = factor * np.exp(eps)

    factor = factor.clip(lower=float(min_factor), upper=float(max_factor))
    factor[brand_cnt < min_count] = 1.0

    f_row = tmp["_brand"].map(factor).to_numpy(dtype=np.float64, copy=False)
    f_row = np.where(np.isfinite(f_row), f_row, 1.0)

    out["UnitPrice"] = up * f_row

    if min_price is not None:
        out["UnitPrice"] = out["UnitPrice"].clip(lower=float(min_price))
    if max_price is not None:
        out["UnitPrice"] = out["UnitPrice"].clip(upper=float(max_price))


def _scale_unit_cost_keep_mode(
    rng: np.random.Generator,
    orig_up: np.ndarray,
    orig_uc: np.ndarray,
    new_up: np.ndarray,
    min_margin: float | None,
    max_margin: float | None,
) -> np.ndarray:
    """
    KEEP mode, but sane:
      - scale UnitCost by (new_up / orig_up) so costs track UnitPrice transformations
      - fill missing/invalid with margin-derived cost
      - optionally clip into configured margin band if provided
    """
    oup = np.asarray(orig_up, dtype=np.float64)
    ouc = np.asarray(orig_uc, dtype=np.float64)
    nup = np.asarray(new_up, dtype=np.float64)

    # scale factor (safe)
    good_price = np.isfinite(oup) & (oup > 0.0)
    factor = np.full_like(nup, np.nan, dtype=np.float64)
    factor[good_price] = nup[good_price] / oup[good_price]

    # scaled cost
    uc = ouc * factor

    # mark invalid
    bad = ~np.isfinite(uc) | (uc < 0.0)

    # fill bad via margin
    mm_lo = 0.20 if min_margin is None else float(min_margin)
    mm_hi = 0.35 if max_margin is None else float(max_margin)
    if not (np.isfinite(mm_lo) and np.isfinite(mm_hi) and 0.0 < mm_lo < mm_hi < 1.0):
        mm_lo, mm_hi = 0.20, 0.35

    if bad.any():
        m = _sample_margin(rng, int(bad.sum()), mm_lo, mm_hi)
        uc = uc.copy()
        uc[bad] = nup[bad] * (1.0 - m)

    # clip to plausible margin band if margins configured
    if min_margin is not None and max_margin is not None:
        lo = float(min_margin)
        hi = float(max_margin)
        if np.isfinite(lo) and np.isfinite(hi) and 0.0 < lo < hi < 1.0:
            # margin in [lo, hi] => cost in [price*(1-hi), price*(1-lo)]
            c_min = nup * (1.0 - hi)
            c_max = nup * (1.0 - lo)
            uc = np.clip(uc, np.maximum(0.0, c_min), np.maximum(0.0, c_max))

    # always enforce <= price
    uc = np.minimum(np.maximum(uc, 0.0), nup)
    return uc


def apply_product_pricing(df: pd.DataFrame, pricing_cfg: dict, seed: int | None = None) -> pd.DataFrame:
    """
    Finalize Products.UnitPrice and Products.UnitCost based on config.

    Changes vs old behavior:
      - In keep mode, UnitCost is scaled by the UnitPrice transformation ratio, so it stays consistent.
      - Prevent NaN/inf from turning into 0.00 via rounding.
      - Correctly assign rounding steps from bands.
    """
    if not pricing_cfg:
        return df

    rng = np.random.default_rng(seed)
    out = df.copy()

    if "UnitPrice" not in out.columns:
        raise ValueError("Products dataframe must contain a UnitPrice column before pricing is applied")

    out["UnitPrice"] = pd.to_numeric(out["UnitPrice"], errors="coerce").astype("float64")

    if "UnitCost" in out.columns:
        out["UnitCost"] = pd.to_numeric(out["UnitCost"], errors="coerce").astype("float64")
    else:
        out["UnitCost"] = np.nan

    # snapshot originals for keep-mode scaling
    orig_up = out["UnitPrice"].to_numpy(dtype=np.float64, copy=True)
    orig_uc = out["UnitCost"].to_numpy(dtype=np.float64, copy=True)

    base_cfg = pricing_cfg.get("base", {}) or {}
    cost_cfg = pricing_cfg.get("cost", {}) or {}
    jitter_cfg = pricing_cfg.get("jitter", {}) or {}

    appearance_cfg = pricing_cfg.get("appearance", None)
    if not isinstance(appearance_cfg, dict):
        appearance_cfg = base_cfg.get("appearance", {}) or {}

    # ----------------------------
    # Base price: scale / (optional) stretch / clamp
    # ----------------------------
    value_scale = _to_float(base_cfg.get("value_scale"), 1.0)
    if value_scale is None or value_scale <= 0:
        raise ValueError("products.pricing.base.value_scale must be a number > 0")

    min_price = _to_float(base_cfg.get("min_unit_price"), None)
    max_price = _to_float(base_cfg.get("max_unit_price"), None)

    stretch = _to_bool(base_cfg.get("stretch_to_range"), False)
    qlo = _to_float(base_cfg.get("stretch_low_quantile"), 0.01)
    qhi = _to_float(base_cfg.get("stretch_high_quantile"), 0.99)

    out["UnitPrice"] = out["UnitPrice"] * float(value_scale)

    if stretch and (min_price is not None) and (max_price is not None):
        if float(max_price) <= float(min_price):
            raise ValueError("products.pricing.base.max_unit_price must be > min_unit_price when stretching")
        out["UnitPrice"] = _stretch_to_range(out["UnitPrice"], float(min_price), float(max_price), float(qlo), float(qhi))

    if min_price is not None:
        out["UnitPrice"] = out["UnitPrice"].clip(lower=float(min_price))
    if max_price is not None:
        out["UnitPrice"] = out["UnitPrice"].clip(upper=float(max_price))

    _sanitize_unit_price(out, min_price, max_price)

    # ----------------------------
    # Brand normalization
    # ----------------------------
    _apply_brand_price_normalization(
        out=out,
        brand_cfg=pricing_cfg.get("brand_normalization", {}) or {},
        rng=rng,
        min_price=min_price,
        max_price=max_price,
    )
    _sanitize_unit_price(out, min_price, max_price)

    # ----------------------------
    # Price jitter
    # ----------------------------
    price_pct = float(jitter_cfg.get("price_pct", 0.0) or 0.0)
    if price_pct > 0:
        mult = rng.uniform(1.0 - price_pct, 1.0 + price_pct, size=len(out))
        out["UnitPrice"] = out["UnitPrice"].to_numpy(dtype="float64") * mult
        _sanitize_unit_price(out, min_price, max_price)

    # ----------------------------
    # Appearance: snap UnitPrice
    # ----------------------------
    snap_unit_price = _to_bool(appearance_cfg.get("snap_unit_price"), False)
    round_unit_cost = _to_bool(appearance_cfg.get("round_unit_cost"), False)

    default_price_bands = [(100.0, 1.0), (1000.0, 10.0), (10000.0, 50.0), (1e18, 100.0)]
    default_cost_bands = [(100.0, 1.0), (500.0, 5.0), (2000.0, 25.0), (1e18, 50.0)]

    price_bands = _parse_bands(appearance_cfg.get("price_bands"), default_price_bands)
    cost_bands = _parse_bands(appearance_cfg.get("cost_bands"), default_cost_bands)

    price_ending = _to_float(appearance_cfg.get("price_ending"), 0.99)
    if price_ending is None:
        price_ending = 0.99

    if snap_unit_price:
        up = out["UnitPrice"].to_numpy(dtype=np.float64, copy=False)
        up = _snap_unit_price_to_points(up, price_bands, price_ending)
        out["UnitPrice"] = up
        _sanitize_unit_price(out, min_price, max_price)

    # ----------------------------
    # Cost model
    # ----------------------------
    mode = (cost_cfg.get("mode") or "").strip().lower()
    if mode not in ("", "keep", "margin"):
        raise ValueError('products.pricing.cost.mode must be one of: "keep", "margin"')

    min_margin = _to_float(cost_cfg.get("min_margin_pct"), None)
    max_margin = _to_float(cost_cfg.get("max_margin_pct"), None)

    # Default behavior: if margin bounds exist, prefer margin mode even if UnitCost exists.
    if mode == "":
        if (min_margin is not None) and (max_margin is not None):
            mode = "margin"
        else:
            mode = "keep" if out["UnitCost"].notna().any() else "margin"

    if mode == "margin":
        if min_margin is None or max_margin is None:
            raise ValueError("products.pricing.cost: min_margin_pct and max_margin_pct must be provided for mode=margin")
        if not (0.0 < float(min_margin) < float(max_margin) < 1.0):
            raise ValueError("products.pricing.cost: require 0 < min_margin_pct < max_margin_pct < 1")
        m = _sample_margin(rng, len(out), float(min_margin), float(max_margin))
        out["UnitCost"] = out["UnitPrice"].to_numpy(dtype=np.float64, copy=False) * (1.0 - m)
    else:
        # Keep-but-scale: cost tracks UnitPrice changes (scale/stretch/brand/snap/jitter)
        new_up = out["UnitPrice"].to_numpy(dtype=np.float64, copy=False)
        out["UnitCost"] = _scale_unit_cost_keep_mode(
            rng=rng,
            orig_up=orig_up,
            orig_uc=orig_uc,
            new_up=new_up,
            min_margin=min_margin,
            max_margin=max_margin,
        )

    # ----------------------------
    # Cost jitter (optional)
    # ----------------------------
    cost_pct = float(jitter_cfg.get("cost_pct", 0.0) or 0.0)
    if cost_pct > 0:
        mult = rng.uniform(1.0 - cost_pct, 1.0 + cost_pct, size=len(out))
        out["UnitCost"] = out["UnitCost"].to_numpy(dtype="float64") * mult

    # Never round NaNs -> 0; fill first
    _sanitize_unit_cost_from_price(out, rng=rng, min_margin=min_margin, max_margin=max_margin)

    # ----------------------------
    # Round UnitCost to bands (no pennies)
    # ----------------------------
    if round_unit_cost:
        uc = out["UnitCost"].to_numpy(dtype=np.float64, copy=False)
        uc = _round_to_step(uc, cost_bands)
        out["UnitCost"] = uc

    # ----------------------------
    # Final invariants
    # ----------------------------
    out["UnitPrice"] = pd.to_numeric(out["UnitPrice"], errors="coerce").astype("float64")
    out["UnitCost"] = pd.to_numeric(out["UnitCost"], errors="coerce").astype("float64")

    _sanitize_unit_price(out, min_price, max_price)
    _sanitize_unit_cost_from_price(out, rng=rng, min_margin=min_margin, max_margin=max_margin)

    out["UnitPrice"] = out["UnitPrice"].clip(lower=0.0)
    out["UnitCost"] = out["UnitCost"].clip(lower=0.0)

    up_arr = out["UnitPrice"].to_numpy(dtype="float64", copy=False)
    uc_arr = out["UnitCost"].to_numpy(dtype="float64", copy=False)
    out["UnitCost"] = np.minimum(uc_arr, up_arr)

    out["UnitPrice"] = out["UnitPrice"].round(2)
    out["UnitCost"] = out["UnitCost"].round(2)

    return out