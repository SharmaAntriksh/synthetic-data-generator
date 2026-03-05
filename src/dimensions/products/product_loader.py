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



# ---------------------------------------------------------------------


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

    # --- Sourcing & Compliance ---
    _COUNTRY_OF_ORIGIN = ["China", "USA", "Germany", "India", "Vietnam", "Mexico", "Japan", "Taiwan"]
    _COO_PROBS_BY_MATERIAL = {
        "Steel":      [0.30, 0.15, 0.20, 0.10, 0.05, 0.05, 0.10, 0.05],
        "Aluminum":   [0.35, 0.10, 0.15, 0.10, 0.05, 0.10, 0.10, 0.05],
        "Plastic":    [0.40, 0.10, 0.10, 0.10, 0.15, 0.10, 0.03, 0.02],
        "Cotton":     [0.20, 0.05, 0.05, 0.30, 0.25, 0.05, 0.05, 0.05],
        "Wood":       [0.25, 0.15, 0.10, 0.10, 0.20, 0.10, 0.05, 0.05],
    }
    _COO_DEFAULT_PROBS = [0.35, 0.12, 0.10, 0.12, 0.12, 0.08, 0.06, 0.05]

    u_coo = _base_uniform(b_u64, seed, 0x1A1A1A1A)[inv]
    mat_vals = out["Material"].astype(str).to_numpy()
    coo = np.empty(N, dtype=object)
    assigned = np.zeros(N, dtype=bool)
    for mat_key, probs in _COO_PROBS_BY_MATERIAL.items():
        mask = (~assigned) & (mat_vals == mat_key)
        if mask.any():
            cum = np.cumsum(probs)
            idx = np.searchsorted(cum, u_coo[mask])
            idx = np.clip(idx, 0, len(_COUNTRY_OF_ORIGIN) - 1)
            coo[mask] = np.array(_COUNTRY_OF_ORIGIN)[idx]
            assigned[mask] = True
    if (~assigned).any():
        cum = np.cumsum(_COO_DEFAULT_PROBS)
        idx = np.searchsorted(cum, u_coo[~assigned])
        idx = np.clip(idx, 0, len(_COUNTRY_OF_ORIGIN) - 1)
        coo[~assigned] = np.array(_COUNTRY_OF_ORIGIN)[idx]
    out["CountryOfOrigin"] = pd.Series(coo, index=out.index, dtype="string")

    u_sus = _base_uniform(b_u64, seed, 0x2A2A2A2A)[inv]
    tier_num = np.where(bt == "Premium", 0.7, np.where(bt == "Mainstream", 0.5, 0.3))
    mat_eco_bonus = np.where(
        np.isin(mat_vals, ["Cotton", "Wood", "Ceramic"]), 0.15, 0.0
    )
    sus_raw = (tier_num + mat_eco_bonus + u_sus * 0.3) / 1.15 * 100
    out["SustainabilityScore"] = pd.Series(np.clip(sus_raw, 1, 100).astype("int32"), index=out.index)

    _CERT_TYPES = ["CE", "UL", "ISO", "FDA", "None"]
    u_cert = _base_uniform(b_u64, seed, 0x3A3A3A3A)[inv]
    cert = np.where(
        u_cert < 0.25, "CE",
        np.where(u_cert < 0.45, "UL",
        np.where(u_cert < 0.60, "ISO",
        np.where(u_cert < 0.68, "FDA", "None"))),
    )
    out["CertificationType"] = pd.Series(cert, index=out.index, dtype="string")

    u_asm = _base_uniform(b_u64, seed, 0x4A4A4A4A)[inv]
    asm_prob = np.where(size_bucket[inv] == 2, 0.55, np.where(size_bucket[inv] == 1, 0.25, 0.08))
    out["AssemblyRequired"] = pd.Series(
        np.where(u_asm < asm_prob, "Yes", "No"), index=out.index, dtype="string"
    )

    # --- Inventory & Supply Chain ---
    u_pop = _base_uniform(b_u64, seed, 0x5A5A5A5A)[inv]
    tier_pop_base = np.where(bt == "Premium", 0.6, np.where(bt == "Value", 0.3, 0.45))
    pop_raw = (tier_pop_base + u_pop * 0.5) / 1.1 * 100
    PopularityScore = np.clip(pop_raw, 1, 100).astype("int32")
    out["PopularityScore"] = pd.Series(PopularityScore, index=out.index)

    up_arr = pd.to_numeric(out.get("UnitPrice", 0.0), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    velocity_est = np.clip(PopularityScore / 100.0, 0.01, 1.0)
    abc_value = up_arr * velocity_est
    abc_rank = np.argsort(np.argsort(-abc_value))
    abc_n = len(abc_rank)
    abc_class = np.where(
        abc_rank < abc_n * 0.20, "A",
        np.where(abc_rank < abc_n * 0.50, "B", "C"),
    )
    out["ABCClassification"] = pd.Series(abc_class, index=out.index, dtype="string")

    lead_arr = out["LeadTimeDays"].to_numpy(dtype=np.int64, copy=False)
    u_rop = _base_uniform(b_u64, seed, 0x6A6A6A6A)[inv]
    rop_base = velocity_est * 30 * (lead_arr / 10.0)
    ReorderPointUnits = np.clip((rop_base * (0.5 + u_rop)).astype("int64"), 10, 5000)
    out["ReorderPointUnits"] = pd.Series(ReorderPointUnits, index=out.index).astype("int32")

    u_ss = _base_uniform(b_u64, seed, 0x7A7A7A7A)[inv]
    SafetyStockUnits = np.clip((ReorderPointUnits * (0.2 + u_ss * 0.3)).astype("int64"), 5, 1000)
    out["SafetyStockUnits"] = pd.Series(SafetyStockUnits, index=out.index).astype("int32")

    _PKG_TYPES = ["Box", "Blister", "Bag", "Bulk", "Pallet"]
    u_pkg = _base_uniform(b_u64, seed, 0x8A8A8A8A)[inv]
    pkg = np.where(
        size_bucket[inv] == 2,
        np.where(u_pkg < 0.40, "Pallet", np.where(u_pkg < 0.75, "Bulk", "Box")),
        np.where(
            size_bucket[inv] == 1,
            np.where(u_pkg < 0.60, "Box", np.where(u_pkg < 0.85, "Bag", "Blister")),
            np.where(u_pkg < 0.35, "Blister", np.where(u_pkg < 0.70, "Box", "Bag")),
        ),
    )
    out["PackagingType"] = pd.Series(pkg, index=out.index, dtype="string")

    # --- Market & Customer Perception ---
    u_rat = _base_uniform(b_u64, seed, 0x9A9A9A9A)[inv]
    defect_arr = out["DefectRateBase"].to_numpy(dtype=np.float64, copy=False)
    rat_base = 4.2 - defect_arr * 30
    tier_rat_bonus = np.where(bt == "Premium", 0.3, np.where(bt == "Value", -0.2, 0.0))
    AvgCustomerRating = np.clip(
        np.round((rat_base + tier_rat_bonus + u_rat * 0.6), 1), 1.0, 5.0
    )
    out["AvgCustomerRating"] = pd.Series(AvgCustomerRating, index=out.index).astype("float32")

    u_rev = _base_uniform(b_u64, seed, 0xAA1A1A1A)[inv]
    review_base = PopularityScore * 50 + u_rev * 2000
    ReviewCount = np.clip(review_base.astype("int64"), 0, 10000)
    out["ReviewCount"] = pd.Series(ReviewCount, index=out.index).astype("int32")

    u_cpi = _base_uniform(b_u64, seed, 0xBB1B1B1B)[inv]
    cpi_base = np.where(bt == "Premium", 1.05, np.where(bt == "Value", 0.85, 0.95))
    CompetitorPriceIndex = np.clip(
        np.round(cpi_base + (u_cpi - 0.5) * 0.3, 2), 0.70, 1.40
    )
    out["CompetitorPriceIndex"] = pd.Series(CompetitorPriceIndex, index=out.index).astype("float32")

    cost_arr = pd.to_numeric(out.get("UnitCost", 0.0), errors="coerce").to_numpy(dtype=np.float64, copy=False)
    margin_pct = np.where(up_arr > 0, (up_arr - cost_arr) / up_arr, 0.0)
    margin_cat = np.where(
        margin_pct < 0.15, "Low",
        np.where(margin_pct < 0.30, "Standard",
        np.where(margin_pct < 0.50, "High", "Premium")),
    )
    out["MarginCategory"] = pd.Series(margin_cat, index=out.index, dtype="string")

    # --- Fun / Beginner-Friendly ---
    u_gift = _base_uniform(b_u64, seed, 0xCC1C1C1C)[inv]
    gift_prob = np.where(size_bucket[inv] == 0, 0.70, np.where(size_bucket[inv] == 1, 0.50, 0.20))
    out["IsGiftEligible"] = pd.Series(
        np.where(u_gift < gift_prob, "Yes", "No"), index=out.index, dtype="string"
    )

    pop_rank = np.argsort(np.argsort(-PopularityScore))
    out["IsBestseller"] = pd.Series(
        np.where(pop_rank < N * 0.15, "Yes", "No"), index=out.index, dtype="string"
    )

    sus_arr = out["SustainabilityScore"].to_numpy(dtype=np.int32, copy=False)
    out["IsEcoFriendly"] = pd.Series(
        np.where(sus_arr > 70, "Yes", "No"), index=out.index, dtype="string"
    )

    _GENDER_MALE_KW = ("men", " male", "boy", "gentleman")
    _GENDER_FEMALE_KW = ("women", "woman", " female", "girl", "lady", "ladies")
    tg = pd.Series("Unisex", index=out.index, dtype="string")
    for kw in _GENDER_FEMALE_KW:
        tg = tg.mask((tg == "Unisex") & text_l.str.contains(kw, regex=False), "Female")
    for kw in _GENDER_MALE_KW:
        tg = tg.mask((tg == "Unisex") & text_l.str.contains(kw, regex=False), "Male")
    out["TargetGender"] = tg

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
def load_product_dimension(config, output_folder: Path, *, log_skip: bool = True):
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
        if log_skip:
            skip("Products up-to-date; skipping regeneration")
        profile_path = output_folder / "product_profile.parquet"
        profile_df = pd.read_parquet(profile_path) if profile_path.exists() else pd.DataFrame()
        return pd.read_parquet(parquet_path), profile_df, False

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
        info("products.use_contoso_products is deprecated; ignoring (num_products is set)")

    if target_n < base_count:
        info(f"Trimming Contoso: {base_count:,} -> {target_n:,} (stratified by SubcategoryKey)")
    elif target_n == base_count:
        info(f"Using Contoso catalog (standardized): {target_n:,}")
    else:
        info(f"Expanding Contoso: {base_count:,} -> {target_n:,} (variants)")

    df = expand_contoso_products(
        base_products=base_df,
        num_products=target_n,
        seed=seed,
    )

    # Defensive: ensure ProductCode exists
    if "ProductCode" not in df.columns:
        df["ProductCode"] = df["ProductKey"].astype(str).str.zfill(7)


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
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required field(s) in Products: {missing}")

    # -----------------------------------------------------------------
    # Split into Products (core) and ProductProfile (analytical)
    # -----------------------------------------------------------------
    _PRODUCTS_CORE_COLS = [
        "ProductKey", "ProductCode", "ProductName", "ProductDescription",
        "SubcategoryKey", "Brand", "Class", "Color",
        "StockTypeCode", "StockType",
        "UnitCost", "UnitPrice",
        "BaseProductKey", "VariantIndex",
        "IsActiveInSales",
    ]

    core_cols = [c for c in _PRODUCTS_CORE_COLS if c in df.columns]
    profile_cols = ["ProductKey"] + [c for c in df.columns if c not in core_cols]

    products_df = df[core_cols].copy()
    profile_df = df[profile_cols].copy()

    profile_path = output_folder / "product_profile.parquet"
    products_df.to_parquet(parquet_path, index=False)
    profile_df.to_parquet(profile_path, index=False)

    save_version("products", version_key, parquet_path)
    return products_df, profile_df, True


def _version_key(p: dict) -> dict:
    """
    Version key for Products. Pricing is the economic source of truth.
    """
    return {
        "num_products": p.get("num_products"),
        "seed": p.get("seed"),
        "pricing": p.get("pricing"),
        "active_ratio": p.get("active_ratio", 1.0),
        # bump whenever you add/remove enrichment columns (forces one regen)
        "enrichment_v": 3,
    }