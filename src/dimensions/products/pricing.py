import numpy as np
import pandas as pd


def apply_product_pricing(
    df: pd.DataFrame,
    pricing_cfg: dict,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Apply pricing rules to Products.

    Products become the economic source of truth.
    Sales must never rescale or clamp prices again.
    """

    if not pricing_cfg:
        return df

    rng = np.random.default_rng(seed)
    out = df.copy()

    # -------------------------------------------------
    # 1. BASE PRICE SCALING
    # -------------------------------------------------
    base_cfg = pricing_cfg.get("base", {})

    value_scale = float(base_cfg.get("value_scale", 1.0))
    min_price = base_cfg.get("min_unit_price")
    max_price = base_cfg.get("max_unit_price")

    if value_scale <= 0:
        raise ValueError("products.pricing.base.value_scale must be > 0")

    out["UnitPrice"] = out["UnitPrice"] * value_scale

    # Apply min / max AFTER scaling
    if min_price is not None:
        out["UnitPrice"] = out["UnitPrice"].clip(lower=min_price)

    if max_price is not None:
        out["UnitPrice"] = out["UnitPrice"].clip(upper=max_price)

    # -------------------------------------------------
    # 2. COST MODEL (MARGIN-BASED)
    # -------------------------------------------------
    cost_cfg = pricing_cfg.get("cost", {})
    min_margin = cost_cfg.get("min_margin_pct")
    max_margin = cost_cfg.get("max_margin_pct")

    if min_margin is not None or max_margin is not None:
        if min_margin is None or max_margin is None:
            raise ValueError(
                "Both min_margin_pct and max_margin_pct must be provided"
            )

        if not (0 < min_margin < max_margin < 1):
            raise ValueError(
                "Margin pct must satisfy 0 < min < max < 1"
            )

        margin = rng.uniform(
            min_margin,
            max_margin,
            size=len(out),
        )

        out["UnitCost"] = out["UnitPrice"] * (1 - margin)

    # -------------------------------------------------
    # 3. JITTER (OPTIONAL NOISE)
    # -------------------------------------------------
    jitter_cfg = pricing_cfg.get("jitter", {})
    price_jitter = float(jitter_cfg.get("price_pct", 0.0))
    cost_jitter = float(jitter_cfg.get("cost_pct", 0.0))

    if price_jitter > 0:
        pj = rng.uniform(
            1 - price_jitter,
            1 + price_jitter,
            size=len(out),
        )
        out["UnitPrice"] = out["UnitPrice"] * pj

    if cost_jitter > 0:
        cj = rng.uniform(
            1 - cost_jitter,
            1 + cost_jitter,
            size=len(out),
        )
        out["UnitCost"] = out["UnitCost"] * cj

    # -------------------------------------------------
    # 4. HARD SAFETY RULES
    # -------------------------------------------------
    out["UnitPrice"] = out["UnitPrice"].clip(lower=0)
    out["UnitCost"] = out["UnitCost"].clip(lower=0)

    # Cost must never exceed price
    out["UnitCost"] = np.minimum(
        out["UnitCost"],
        out["UnitPrice"],
    )

    # -------------------------------------------------
    # 5. FINAL ROUNDING (STORAGE)
    # -------------------------------------------------
    out["UnitPrice"] = out["UnitPrice"].round(2)
    out["UnitCost"] = out["UnitCost"].round(2)

    return out
