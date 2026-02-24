from __future__ import annotations

from pathlib import Path
import datetime as _dt

import numpy as np
import pandas as pd

from src.utils import info, skip
from src.versioning import should_regenerate, save_version

from .contoso_loader import load_contoso_products
from .contoso_expander import expand_contoso_products
from .pricing import apply_product_pricing


# Prefer the config shim (stable API surface)
try:
    # typical in this repo layout (engine/config)
    from src.engine.config.config_loader import get_global_dates  # type: ignore
except Exception:  # pragma: no cover
    try:
        # fallback if you later move it
        from src.config_loader import get_global_dates  # type: ignore
    except Exception:  # pragma: no cover
        get_global_dates = None  # last-resort fallback to raw parsing


# ---------------------------------------------------------------------
# Deterministic hashing utilities (no row-order dependence)
# ---------------------------------------------------------------------
_U64 = np.uint64
_MASK64 = _U64(0xFFFFFFFFFFFFFFFF)


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """Vectorized splitmix64 hash; returns uint64 array."""
    z = (x + _U64(0x9E3779B97F4A7C15)) & _MASK64
    z = (z ^ (z >> _U64(30))) * _U64(0xBF58476D1CE4E5B9) & _MASK64
    z = (z ^ (z >> _U64(27))) * _U64(0x94D049BB133111EB) & _MASK64
    return (z ^ (z >> _U64(31))) & _MASK64


def _u01(u: np.ndarray) -> np.ndarray:
    """Map uint64 -> float64 in [0,1)."""
    return ((u >> _U64(11)).astype(np.float64)) * (1.0 / (1 << 53))


def _base_uniform(uniq_base_u64: np.ndarray, seed: int, salt: int) -> np.ndarray:
    s = _U64(int(seed) ^ int(salt))
    return _u01(_splitmix64(uniq_base_u64 ^ s))


def _resolve_global_date_range(cfg: dict) -> tuple[pd.Timestamp, pd.Timestamp]:
    """
    Returns (start_ts, end_ts) normalized.
    Prefers get_global_dates() if available; else falls back to cfg['defaults'].
    """
    if get_global_dates is not None:
        d = get_global_dates(cfg)  # {"start": "...", "end": "..."}
        start = pd.to_datetime(d["start"]).normalize()
        end = pd.to_datetime(d["end"]).normalize()
        return start, end

    defaults = cfg.get("defaults") or cfg.get("_defaults") or {}
    dates = defaults.get("dates") or {}
    start = pd.to_datetime(dates.get("start")).normalize()
    end = pd.to_datetime(dates.get("end")).normalize()
    return start, end


# ---------------------------------------------------------------------
# Lifecycle enrichment (Launch/Discontinued)  -> stored as DATE (not text)
# ---------------------------------------------------------------------
def _apply_product_lifecycle(df: pd.DataFrame, cfg: dict, *, seed: int) -> pd.DataFrame:
    """
    Adds/overwrites:
      - LaunchDate           (python datetime.date; parquet date32)
      - LaunchDateKey        (int64 YYYYMMDD)
      - DiscontinuedDate     (python datetime.date or None; parquet date32)
      - DiscontinuedDateKey  (nullable Int64 YYYYMMDD)
      - IsDiscontinued       (int64 0/1)

    Deterministic per BaseProductKey; variants inherit the same lifecycle.
    """
    p = cfg.get("products") or {}
    lc = p.get("lifecycle") or {}
    if not isinstance(lc, dict) or not lc:
        return df

    lookback_years = int(lc.get("lookback_years", 0))
    lookahead_years = int(lc.get("lookahead_years", 0))
    preexisting_share = float(lc.get("preexisting_share", 0.70))
    discontinue_ratio = float(lc.get("discontinue_ratio", 0.20))

    if not (0.0 <= preexisting_share <= 1.0):
        raise ValueError("products.lifecycle.preexisting_share must be in [0,1]")
    if not (0.0 <= discontinue_ratio <= 1.0):
        raise ValueError("products.lifecycle.discontinue_ratio must be in [0,1]")
    if lookback_years < 0 or lookahead_years < 0:
        raise ValueError("products.lifecycle.lookback_years/lookahead_years must be >= 0")

    global_start, global_end = _resolve_global_date_range(cfg)

    early_start = (global_start - pd.DateOffset(years=lookback_years)).normalize()
    early_end = (global_start - pd.Timedelta(days=1)).normalize()
    disc_end = (global_end + pd.DateOffset(years=lookahead_years)).normalize()

    early_start_np = np.datetime64(early_start.date())
    global_start_np = np.datetime64(global_start.date())
    global_end_np = np.datetime64(global_end.date())
    disc_end_np = np.datetime64(disc_end.date())

    early_days = int((early_end - early_start).days) + 1 if early_end >= early_start else 0
    in_days = int((global_end - global_start).days) + 1
    if in_days <= 0:
        raise ValueError("Invalid global dates: defaults.dates.start must be < end")

    base = pd.to_numeric(df.get("BaseProductKey", df["ProductKey"]), errors="coerce").fillna(0).astype("int64").to_numpy()
    uniq_base, inv = np.unique(base, return_inverse=True)
    b_u64 = uniq_base.astype("uint64", copy=False)

    u_pre = _base_uniform(b_u64, seed, 0xA5A5A5A5)
    u_day = _base_uniform(b_u64, seed, 0x5A5A5A5A)

    use_early = (u_pre < preexisting_share) & (early_days > 0)

    launch_off = np.empty(uniq_base.size, dtype=np.int64)
    if early_days > 0:
        launch_off[use_early] = np.floor(u_day[use_early] * early_days).astype(np.int64)
    launch_off[~use_early] = np.floor(u_day[~use_early] * in_days).astype(np.int64)

    launch_np = np.empty(uniq_base.size, dtype="datetime64[D]")
    if early_days > 0:
        launch_np[use_early] = early_start_np + launch_off[use_early].astype("timedelta64[D]")
    launch_np[~use_early] = global_start_np + launch_off[~use_early].astype("timedelta64[D]")

    # Discontinued
    u_disc = _base_uniform(b_u64, seed, 0xC3C3C3C3)
    do_disc = u_disc < discontinue_ratio

    disc_np = np.full(uniq_base.size, np.datetime64("NaT"), dtype="datetime64[D]")
    if np.any(do_disc):
        disc_start_np = launch_np.copy()
        max_days = (disc_end_np - disc_start_np).astype("timedelta64[D]").astype(np.int64)
        valid = do_disc & (max_days >= 0)
        if np.any(valid):
            u_disc_day = _base_uniform(b_u64, seed, 0x3C3C3C3C)
            disc_off = np.floor(u_disc_day[valid] * (max_days[valid] + 1)).astype(np.int64)
            disc_np[valid] = disc_start_np[valid] + disc_off.astype("timedelta64[D]")

    # Broadcast to rows
    launch_ts = pd.Series(pd.to_datetime(launch_np[inv]), index=df.index).dt.normalize()
    disc_ts = pd.Series(pd.to_datetime(disc_np[inv]), index=df.index).dt.normalize()

    # Store as *date* for Parquet (Power Query-friendly)
    launch_date = launch_ts.dt.date
    disc_date = disc_ts.dt.date.where(disc_ts.notna(), None)

    out = df.copy()
    out["LaunchDate"] = launch_date
    out["LaunchDateKey"] = launch_ts.dt.strftime("%Y%m%d").astype("int64")

    out["DiscontinuedDate"] = disc_date
    out["DiscontinuedDateKey"] = pd.to_numeric(disc_ts.dt.strftime("%Y%m%d"), errors="coerce").astype("Int64")
    out["IsDiscontinued"] = disc_ts.notna().astype("int64")

    return out


# ---------------------------------------------------------------------
# Attribute enrichment (merch, channel, logistics, quality)
# ---------------------------------------------------------------------
_COLOR_FAMILY_RULES: list[tuple[tuple[str, ...], str]] = [
    (("black",), "Black"),
    (("white", "ivory", "cream"), "White"),
    (("gray", "grey", "silver"), "Gray"),
    (("blue", "navy", "cyan", "teal", "turquoise"), "Blue"),
    (("red", "maroon", "crimson"), "Red"),
    (("green", "olive", "lime"), "Green"),
    (("yellow", "gold"), "Yellow"),
    (("orange",), "Orange"),
    (("purple", "violet", "lavender"), "Purple"),
    (("pink", "magenta"), "Pink"),
    (("brown", "tan", "beige"), "Brown"),
    (("metal", "chrome", "steel"), "Metallic"),
]


_MATERIAL_KEYWORDS: list[tuple[str, str]] = [
    ("leather", "Leather"),
    ("cotton", "Cotton"),
    ("wool", "Wool"),
    ("silk", "Silk"),
    ("polyester", "Polyester"),
    ("nylon", "Nylon"),
    ("denim", "Denim"),
    ("rubber", "Rubber"),
    ("plastic", "Plastic"),
    ("aluminum", "Aluminum"),
    ("aluminium", "Aluminum"),
    ("steel", "Steel"),
    ("stainless", "Steel"),
    ("wood", "Wood"),
    ("glass", "Glass"),
    ("ceramic", "Ceramic"),
]


_STYLE_KEYWORDS: list[tuple[str, str]] = [
    ("modern", "Modern"),
    ("vintage", "Vintage"),
    ("classic", "Classic"),
    ("minimal", "Minimalist"),
    ("sport", "Sport"),
    ("outdoor", "Outdoor"),
    ("formal", "Formal"),
    ("casual", "Casual"),
    ("premium", "Premium"),
]


_AGE_KEYWORDS: list[tuple[tuple[str, ...], str]] = [
    (("baby", "infant", "toddler"), "Baby"),
    (("kid", "kids", "child", "children", "youth"), "Kids"),
    (("teen", "teenage"), "Teen"),
]


_SEASON_KEYWORDS: list[tuple[tuple[str, ...], str]] = [
    (("holiday", "christmas", "xmas"), "Holiday"),
    (("winter", "snow"), "Winter"),
    (("summer", "beach"), "Summer"),
    (("spring",), "Spring"),
    (("back to school", "school"), "BackToSchool"),
]


def _lower_series(s: pd.Series) -> pd.Series:
    return s.astype("string").fillna("").str.lower()


def _pick_from_list(u: np.ndarray, choices: list[str]) -> np.ndarray:
    if len(choices) == 1:
        return np.full(u.shape[0], choices[0], dtype=object)
    idx = np.floor(u * len(choices)).astype(np.int64)
    idx = np.clip(idx, 0, len(choices) - 1)
    return np.array([choices[i] for i in idx], dtype=object)


def _enrich_products_attributes(df: pd.DataFrame, cfg: dict, *, seed: int) -> pd.DataFrame:
    """
    Adds:
      BrandTier, ProductLine, ColorFamily, Material, Style, AgeGroup,
      SeasonalityProfile,
      EligibleStore/Online/Marketplace/B2B,
      logistics & fulfillment,
      quality & returns.
    """
    out = df.copy()

    # Base keys
    base = pd.to_numeric(out.get("BaseProductKey", out["ProductKey"]), errors="coerce").fillna(0).astype("int64").to_numpy()
    uniq_base, inv = np.unique(base, return_inverse=True)
    b_u64 = uniq_base.astype("uint64", copy=False)

    # --- Merch / taxonomy ---
    # ColorFamily from existing Color (rules-based; no RNG)
    if "Color" in out.columns:
        col_l = _lower_series(out["Color"])
        fam = pd.Series("Other", index=out.index, dtype="string")
        for keys, name in _COLOR_FAMILY_RULES:
            mask = False
            for k in keys:
                mask = mask | col_l.str.contains(k, regex=False)
            fam = fam.mask(mask, name)
        out["ColorFamily"] = fam.astype("string")
    else:
        out["ColorFamily"] = pd.Series("Other", index=out.index, dtype="string")

    # Material / Style / AgeGroup from keywords, else deterministic fallback by BaseProductKey
    name_l = _lower_series(out.get("ProductName", pd.Series("", index=out.index)))
    desc_l = _lower_series(out.get("ProductDescription", pd.Series("", index=out.index)))
    text_l = (name_l + " " + desc_l).astype("string")

    # Material
    mat = pd.Series("", index=out.index, dtype="string")
    for kw, val in _MATERIAL_KEYWORDS:
        mat = mat.mask((mat == "") & text_l.str.contains(kw, regex=False), val)
    if (mat == "").any():
        # fallback deterministic on BaseProductKey
        u_mat = _base_uniform(b_u64, seed, 0x11111111)[inv]
        fallback = _pick_from_list(u_mat, ["Plastic", "Steel", "Wood", "Cotton", "Polyester", "Aluminum", "Rubber"])
        mat = mat.mask(mat == "", pd.Series(fallback, index=out.index, dtype="string"))
    out["Material"] = mat.astype("string")

    # Style
    sty = pd.Series("", index=out.index, dtype="string")
    for kw, val in _STYLE_KEYWORDS:
        sty = sty.mask((sty == "") & text_l.str.contains(kw, regex=False), val)
    if (sty == "").any():
        u_sty = _base_uniform(b_u64, seed, 0x22222222)[inv]
        fallback = _pick_from_list(u_sty, ["Classic", "Modern", "Casual", "Outdoor", "Minimalist", "Sport"])
        sty = sty.mask(sty == "", pd.Series(fallback, index=out.index, dtype="string"))
    out["Style"] = sty.astype("string")

    # AgeGroup
    age = pd.Series("", index=out.index, dtype="string")
    for keys, val in _AGE_KEYWORDS:
        mask = False
        for k in keys:
            mask = mask | text_l.str.contains(k, regex=False)
        age = age.mask((age == "") & mask, val)
    # default
    age = age.mask(age == "", "Adult")
    out["AgeGroup"] = age.astype("string")

    # ProductLine deterministic by (SubcategoryKey, BaseProductKey)
    sub = pd.to_numeric(out.get("SubcategoryKey", 0), errors="coerce").fillna(0).astype("int64").to_numpy()
    u_line = _base_uniform(b_u64, seed, 0x33333333)[inv]
    lines = ["Core", "Essentials", "Classic", "Pro", "Elite", "Studio", "Urban", "Outdoor", "Home", "Active", "Travel"]
    # bias line selection slightly by subcategory key (stable)
    u_line2 = (u_line + ((sub % 7) / 7.0) * 0.23) % 1.0
    out["ProductLine"] = pd.Series(_pick_from_list(u_line2, lines), index=out.index, dtype="string")

    # SeasonalityProfile: keyword first, else weighted deterministic fallback
    seas = pd.Series("", index=out.index, dtype="string")
    for keys, val in _SEASON_KEYWORDS:
        mask = False
        for k in keys:
            mask = mask | text_l.str.contains(k, regex=False)
        seas = seas.mask((seas == "") & mask, val)

    if (seas == "").any():
        u_seas = _base_uniform(b_u64, seed, 0x44444444)[inv]
        # 70% None, else split across profiles
        seas_f = np.where(
            u_seas < 0.70,
            "None",
            np.where(
                u_seas < 0.78, "Holiday",
                np.where(u_seas < 0.86, "Winter",
                         np.where(u_seas < 0.93, "Summer", "BackToSchool")),
            ),
        )
        seas = seas.mask(seas == "", pd.Series(seas_f, index=out.index, dtype="string"))
    out["SeasonalityProfile"] = seas.astype("string")

    # --- Logistics & fulfillment (base-level, variants inherit) ---
    u_size = _base_uniform(b_u64, seed, 0x51515151)
    u_w = _base_uniform(b_u64, seed, 0x61616161)
    u_d1 = _base_uniform(b_u64, seed, 0x71717171)
    u_d2 = _base_uniform(b_u64, seed, 0x81818181)
    u_d3 = _base_uniform(b_u64, seed, 0x91919191)

    size_bucket = np.where(u_size < 0.65, 0, np.where(u_size < 0.93, 1, 2))  # 0 small, 1 med, 2 large

    w_kg = np.empty_like(u_w, dtype=np.float64)
    # small: 0.05..2.0, medium: 1..12, large: 8..60
    w_kg[size_bucket == 0] = 0.05 + u_w[size_bucket == 0] * (2.0 - 0.05)
    w_kg[size_bucket == 1] = 1.0 + u_w[size_bucket == 1] * (12.0 - 1.0)
    w_kg[size_bucket == 2] = 8.0 + u_w[size_bucket == 2] * (60.0 - 8.0)

    # dims in cm
    L = np.empty_like(u_d1, dtype=np.float64)
    W = np.empty_like(u_d2, dtype=np.float64)
    H = np.empty_like(u_d3, dtype=np.float64)

    # small: 5..35, medium: 20..90, large: 50..250
    def _rng(lo: float, hi: float, u: np.ndarray) -> np.ndarray:
        return lo + u * (hi - lo)

    L[size_bucket == 0] = _rng(5, 35, u_d1[size_bucket == 0])
    W[size_bucket == 0] = _rng(5, 35, u_d2[size_bucket == 0])
    H[size_bucket == 0] = _rng(2, 25, u_d3[size_bucket == 0])

    L[size_bucket == 1] = _rng(20, 90, u_d1[size_bucket == 1])
    W[size_bucket == 1] = _rng(10, 70, u_d2[size_bucket == 1])
    H[size_bucket == 1] = _rng(5, 60, u_d3[size_bucket == 1])

    L[size_bucket == 2] = _rng(50, 250, u_d1[size_bucket == 2])
    W[size_bucket == 2] = _rng(30, 200, u_d2[size_bucket == 2])
    H[size_bucket == 2] = _rng(10, 180, u_d3[size_bucket == 2])

    vol = (L * W * H)

    # thresholds (can be made configurable later)
    freight = (w_kg >= 35.0) | (vol >= 150_000.0)
    oversize = (~freight) & ((w_kg >= 15.0) | (vol >= 90_000.0))
    ship_class = np.where(freight, "Freight", np.where(oversize, "Oversize", "Standard"))

    # Fragile/Hazmat (base)
    u_frag = _base_uniform(b_u64, seed, 0xA0A0A0A0)
    u_haz = _base_uniform(b_u64, seed, 0xB0B0B0B0)

    # material-driven bump
    mat_base = pd.Series(_pick_from_list(_base_uniform(b_u64, seed, 0x12121212), ["Plastic"]), index=pd.RangeIndex(len(uniq_base)))
    # but we already computed Material per row; use base material by taking first per base
    mat_row = out["Material"].astype("string")
    base_mat = mat_row.groupby(pd.Series(base, index=out.index)).first()
    base_mat = base_mat.reindex(uniq_base, fill_value="Plastic").astype("string").to_numpy()

    fragile_base = (u_frag < 0.10) | np.isin(base_mat, ["Glass", "Ceramic"])
    haz_base = (u_haz < 0.01) | pd.Series(base_mat).astype("string").str.contains("chemical", regex=False).to_numpy(dtype=bool, copy=False)

    # Lead time depends on shipping class
    u_lt = _base_uniform(b_u64, seed, 0xC0C0C0C0)
    lead = np.empty_like(u_lt, dtype=np.int64)
    lead[ship_class == "Standard"] = (2 + np.floor(u_lt[ship_class == "Standard"] * 9)).astype(np.int64)          # 2..10
    lead[ship_class == "Oversize"] = (5 + np.floor(u_lt[ship_class == "Oversize"] * 17)).astype(np.int64)          # 5..21
    lead[ship_class == "Freight"] = (10 + np.floor(u_lt[ship_class == "Freight"] * 36)).astype(np.int64)           # 10..45

    # Case pack qty
    u_cp = _base_uniform(b_u64, seed, 0xD0D0D0D0)
    case_opts = [1, 2, 4, 6, 8, 12, 24]
    case_pack = np.array(case_opts, dtype=np.int64)[np.clip((u_cp * len(case_opts)).astype(np.int64), 0, len(case_opts) - 1)]

    # Fulfillment type
    u_f = _base_uniform(b_u64, seed, 0xE0E0E0E0)
    fulfil = np.where(u_f < 0.70, "Stocked", np.where(u_f < 0.88, "3PL", "DropShip"))

    # Broadcast logistics to rows
    out["WeightKg"] = pd.Series(w_kg[inv], index=out.index).astype("float32")
    out["LengthCm"] = pd.Series(L[inv], index=out.index).astype("float32")
    out["WidthCm"] = pd.Series(W[inv], index=out.index).astype("float32")
    out["HeightCm"] = pd.Series(H[inv], index=out.index).astype("float32")
    out["VolumeCm3"] = pd.Series(vol[inv], index=out.index).astype("float32")
    out["ShippingClass"] = pd.Series(ship_class[inv], index=out.index, dtype="string")
    out["IsFragile"] = pd.Series(fragile_base[inv].astype("int64"), index=out.index)
    out["IsHazmat"] = pd.Series(haz_base[inv].astype("int64"), index=out.index)
    out["LeadTimeDays"] = pd.Series(lead[inv], index=out.index).astype("int32")
    out["CasePackQty"] = pd.Series(case_pack[inv], index=out.index).astype("int32")
    out["FulfillmentType"] = pd.Series(fulfil[inv], index=out.index, dtype="string")

    # --- BrandTier from UnitPrice (robust: rank-based so Mainstream isn't empty) ---
    up = pd.to_numeric(out.get("UnitPrice", 0.0), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    N = len(out)

    tier = np.full(N, "Mainstream", dtype=object)
    finite = np.isfinite(up)

    if finite.any() and N >= 3:
        # stable tie-breaker: ProductKey (or BaseProductKey) to break equal prices deterministically
        tie = pd.to_numeric(out.get("ProductKey", np.arange(N)), errors="coerce").fillna(0).to_numpy(dtype=np.int64, copy=False)

        idx = np.where(finite)[0]
        # sort by (price asc, productkey asc)
        order = idx[np.lexsort((tie[idx], up[idx]))]

        n = order.size
        c1 = n // 3
        c2 = (2 * n) // 3

        tier[order[:c1]] = "Value"
        tier[order[c1:c2]] = "Mainstream"
        tier[order[c2:]] = "Premium"
    else:
        # small N or all non-finite: keep default "Mainstream"
        pass

    out["BrandTier"] = pd.Series(tier, index=out.index, dtype="string")

    # --- Channel eligibility ---
    u_ch = _base_uniform(b_u64, seed, 0xF0F0F0F0)[inv]
    is_haz = out["IsHazmat"].to_numpy(dtype=np.int64, copy=False) == 1
    ship = out["ShippingClass"].astype("string")

    out["EligibleStore"] = 1
    out["EligibleOnline"] = 1

    # Marketplace: avoid hazmat/freight, then probabilistic
    eligible_mkt = (~is_haz) & (ship != "Freight") & (u_ch < 0.70)
    out["EligibleMarketplace"] = eligible_mkt.astype("int64")

    # B2B: small share + bias if Class contains business/enterprise
    cls = _lower_series(out.get("Class", pd.Series("", index=out.index)))
    b2b_bias = cls.str.contains("business", regex=False) | cls.str.contains("enterprise", regex=False)
    eligible_b2b = (u_ch < 0.10) | b2b_bias
    out["EligibleB2B"] = eligible_b2b.astype("int64")

    # --- Quality / returns ---
    bt = out["BrandTier"].astype("string")
    u_q = _base_uniform(b_u64, seed, 0xABABABAB)[inv]

    # Warranty
    warr = np.where(
        bt == "Premium",
        np.where(u_q < 0.50, 24, np.where(u_q < 0.85, 36, 12)),
        np.where(
            bt == "Value",
            np.where(u_q < 0.65, 6, 12),
            np.where(u_q < 0.70, 12, 24),
        ),
    )
    out["WarrantyMonths"] = pd.Series(warr, index=out.index).astype("int32")

    # Return / defect baseline rates
    fragile = out["IsFragile"].to_numpy(dtype=np.int64, copy=False) == 1
    ship_freight = (ship == "Freight").to_numpy(dtype=bool, copy=False)

    base_return = 0.02 + (fragile * 0.05) + (ship_freight * 0.03)
    base_return = base_return - (bt == "Premium").to_numpy(dtype=np.float64, copy=False) * 0.01
    jitter_r = (u_q * 0.03)
    rr = np.clip(base_return + jitter_r, 0.005, 0.25)
    out["ReturnRateBase"] = pd.Series(rr, index=out.index).astype("float32")

    base_def = 0.002 + (bt == "Value").to_numpy(dtype=np.float64, copy=False) * 0.006 + is_haz.astype(np.float64) * 0.003
    jitter_d = (_base_uniform(b_u64, seed, 0xCDCDCDCD)[inv] * 0.005)
    dr = np.clip(base_def + jitter_d, 0.0005, 0.05)
    out["DefectRateBase"] = pd.Series(dr, index=out.index).astype("float32")

    # Return window
    rw = np.where(bt == "Premium", 90, np.where(bt == "Mainstream", 60, 30))
    out["ReturnWindowDays"] = pd.Series(rw, index=out.index).astype("int32")

    return out


# ---------------------------------------------------------------------
# Supplier assignment
# ---------------------------------------------------------------------
def _load_supplier_keys(output_folder: Path) -> np.ndarray:
    """
    Loads SupplierKey values from suppliers.parquet in the same dims folder.
    Returns sorted unique int64 keys.
    """
    sup_path = output_folder / "suppliers.parquet"
    if not sup_path.exists():
        raise FileNotFoundError(
            f"Missing suppliers dimension parquet: {sup_path}. "
            "Generate dimensions first (Suppliers)."
        )

    sup = pd.read_parquet(sup_path)
    key_col = None
    for c in ["SupplierKey", "Key"]:
        if c in sup.columns:
            key_col = c
            break
    if key_col is None:
        raise KeyError(f"suppliers.parquet missing SupplierKey/Key. Available: {list(sup.columns)}")

    keys = pd.to_numeric(sup[key_col], errors="coerce").dropna().astype("int64").to_numpy()
    keys = np.unique(keys)
    if keys.size == 0:
        raise ValueError("suppliers.parquet has zero valid SupplierKey values")
    return np.sort(keys)


# ---------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------
def load_product_dimension(config, output_folder: Path):
    """
    Product dimension loader (single method):

      - Load Contoso catalog as base (2517 rows typically)
      - Scale to target row count via expand_contoso_products():
          * stratified trim by SubcategoryKey when target < base_count
          * repeat/variants when target > base_count
      - Apply lifecycle (Launch/Discontinued) from products.lifecycle (date-typed)
      - Apply pricing from products.pricing (UnitPrice/UnitCost finalized here)
      - Enrich attributes (merch/channel/logistics/quality)
      - Assign SupplierKey (optional)
      - Mark IsActiveInSales based on active_ratio
      - Write products.parquet

    Returns:
        (DataFrame, regenerated: bool)
    """
    p = config["products"]
    seed = int(p.get("seed", 42))

    active_ratio = p.get("active_ratio", 1.0)
    if not isinstance(active_ratio, (int, float)) or not (0 < float(active_ratio) <= 1.0):
        raise ValueError("products.active_ratio must be a number in the range (0, 1]")

    # Supplier assignment
    sup_cfg = p.get("supplier_assignment") or {}
    sup_enabled = bool(sup_cfg.get("enabled", True))
    sup_seed = int(sup_cfg.get("seed", seed))
    sup_strategy = str(sup_cfg.get("strategy", "by_base_product")).lower()

    supplier_keys = None
    supplier_sig = None
    if sup_enabled:
        supplier_keys = _load_supplier_keys(output_folder)
        supplier_sig = {"n": int(supplier_keys.size), "min": int(supplier_keys.min()), "max": int(supplier_keys.max())}

    # Versioning / skip
    parquet_path = output_folder / "products.parquet"
    version_key = _version_key(p)

    if sup_enabled:
        version_key = dict(version_key)
        version_key["supplier_assignment"] = {"enabled": True, "strategy": sup_strategy, "seed": sup_seed}
        version_key["supplier_sig"] = supplier_sig

    force = bool(p.get("_force_regenerate", False))
    if not force and not should_regenerate("products", version_key, parquet_path):
        skip("Products up-to-date; skipping regeneration")
        return pd.read_parquet(parquet_path), False

    # Base catalog
    base_df = load_contoso_products(output_folder)
    base_count = int(len(base_df))

    # Target row count
    target_n = p.get("num_products", None)
    if target_n is None:
        target_n = base_count
    target_n = int(target_n)

    if target_n <= 0:
        raise ValueError("products.num_products must be a positive integer")

    if "num_products" in p and "use_contoso_products" in p:
        info("products.use_contoso_products is deprecated; ignoring it because products.num_products is set.")

    if target_n < base_count:
        info(f"TRIMMING CONTOSO PRODUCTS: {base_count} -> {target_n} (stratified by SubcategoryKey)")
    elif target_n == base_count:
        info(f"USING CONTOSO PRODUCTS (standardized identity): {target_n}")
    else:
        info(f"EXPANDING CONTOSO PRODUCTS: {base_count} -> {target_n} (variants)")

    df = expand_contoso_products(
        base_products=base_df,
        num_products=target_n,
        seed=seed,
    )

    # Defensive: ensure ProductCode exists
    if "ProductCode" not in df.columns:
        df["ProductCode"] = df["ProductKey"].astype(str).str.zfill(7)

    # Lifecycle (date-typed)
    df = _apply_product_lifecycle(df, config, seed=seed)

    # Pricing (authoritative)
    df = apply_product_pricing(
        df=df,
        pricing_cfg=p.get("pricing"),
        seed=seed,
    )

    # Enrichment columns
    df = _enrich_products_attributes(df, config, seed=seed)

    # SupplierKey (deterministic)
    if sup_enabled:
        n_sup = int(supplier_keys.size)
        base = pd.to_numeric(df.get("BaseProductKey", df["ProductKey"]), errors="coerce").fillna(0).astype("int64").to_numpy()

        if sup_strategy == "by_subcategory" and "SubcategoryKey" in df.columns:
            sub = pd.to_numeric(df["SubcategoryKey"], errors="coerce").fillna(0).astype("int64").to_numpy()
            idx = np.mod(sub, n_sup)
        elif sup_strategy == "uniform":
            rng_sup = np.random.default_rng(sup_seed)
            idx = rng_sup.integers(0, n_sup, size=len(df), dtype=np.int64)
        else:
            idx = np.mod(base, n_sup)

        df["SupplierKey"] = supplier_keys[idx].astype("int64")

    # Active products for Sales
    N = len(df)
    active_count = int(N * float(active_ratio))
    if active_count <= 0:
        raise ValueError("products.active_ratio results in zero active products; increase active_ratio or product count")

    product_keys = df["ProductKey"].to_numpy(dtype="int64", copy=False)
    if active_count < N:
        rng = np.random.default_rng(seed)
        active_product_keys = rng.choice(product_keys, size=active_count, replace=False)
        active_product_set = set(active_product_keys.tolist())
    else:
        active_product_set = set(product_keys.tolist())

    df["IsActiveInSales"] = df["ProductKey"].isin(active_product_set).astype("int64")

    # Minimal required fields for Sales
    required = [
        "ProductKey",
        "BaseProductKey",
        "VariantIndex",
        "SubcategoryKey",
        "UnitPrice",
        "UnitCost",
        "IsActiveInSales",
        "LaunchDate",
        "LaunchDateKey",
        "DiscontinuedDate",
        "DiscontinuedDateKey",
        "IsDiscontinued",
    ]
    if sup_enabled:
        required.append("SupplierKey")

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required field(s) in Products: {missing}")

    df.to_parquet(parquet_path, index=False)
    save_version("products", version_key, parquet_path)
    return df, True


def _version_key(p: dict) -> dict:
    """
    Version key for Products. Pricing is the economic source of truth.
    """
    return {
        "num_products": p.get("num_products"),
        "seed": p.get("seed"),
        "pricing": p.get("pricing"),
        "active_ratio": p.get("active_ratio", 1.0),
        "lifecycle": p.get("lifecycle"),
        # bump whenever you add/remove enrichment columns (forces one regen)
        "enrichment_v": 1,
    }