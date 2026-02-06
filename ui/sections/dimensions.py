# ui/sections/dimensions.py
from __future__ import annotations

import streamlit as st


# =========================================================
# Dimension size controls (config-driven)
# =========================================================

DIMENSION_SIZE_FIELDS = {
    "customers": "total_customers",
    "products": "num_products",
    "stores": "num_stores",
}

# Dimensions that support force regeneration from UI (used by render_regeneration)
FORCE_REGENERATABLE_DIMENSIONS = [
    "customers",
    "products",
    "stores",
    "promotions",
    "dates",
    "currency",
    "exchange_rates",
]


def _ensure(cfg: dict, section: str, field: str, default: int) -> None:
    cfg.setdefault(section, {})
    cfg[section].setdefault(field, default)


def _as_int(v, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _get_promotions_total(promotions_cfg: dict) -> int:
    """
    Promotions are commonly modeled as multiple buckets.
    Prefer summing buckets if they exist; else fall back to total_promotions; else 0.
    """
    keys = ["num_seasonal", "num_clearance", "num_limited"]
    if all(k in promotions_cfg for k in keys):
        return sum(_as_int(promotions_cfg.get(k), 0) for k in keys)
    return _as_int(promotions_cfg.get("total_promotions", 0), 0)


def _set_promotions_total(promotions_cfg: dict, total: int) -> None:
    """
    Write total promotions back into buckets (proportional if existing; else 1/3 split).
    Also writes total_promotions as a back-compat/summary key.
    """
    total = max(0, int(total))
    keys = ["num_seasonal", "num_clearance", "num_limited"]

    if all(k in promotions_cfg for k in keys):
        cur = [_as_int(promotions_cfg.get(k), 0) for k in keys]
        s = sum(cur)
        if s <= 0:
            # default equal split
            base = [1, 1, 1]
            s = 3
        else:
            base = cur

        scaled = [b * total / s for b in base]
        floors = [int(x) for x in scaled]
        remainder = total - sum(floors)

        # distribute remainder to largest fractional parts
        fracs = sorted(
            [(i, scaled[i] - floors[i]) for i in range(3)],
            key=lambda t: t[1],
            reverse=True,
        )
        for i in range(remainder):
            floors[fracs[i % 3][0]] += 1

        promotions_cfg["num_seasonal"] = floors[0]
        promotions_cfg["num_clearance"] = floors[1]
        promotions_cfg["num_limited"] = floors[2]
    else:
        # minimal schema: just keep a total
        promotions_cfg["total_promotions"] = total

    promotions_cfg["total_promotions"] = total


def render_dimensions(cfg: dict) -> None:
    st.subheader("4️⃣ Dimensions")

    # Ensure baseline keys (safe defaults)
    _ensure(cfg, "customers", "total_customers", 10_000)
    _ensure(cfg, "stores", "num_stores", 100)
    _ensure(cfg, "products", "num_products", 5_000)
    cfg.setdefault("promotions", {})

    col1, col2 = st.columns(2)

    with col1:
        cfg["customers"]["total_customers"] = st.number_input(
            "Customers (entities)",
            min_value=1,
            step=1_000,
            value=_as_int(cfg["customers"]["total_customers"], 10_000),
            help="Controls the size of the Customers dimension. Higher values increase dimension generation time.",
        )
        cfg["stores"]["num_stores"] = st.number_input(
            "Physical stores",
            min_value=1,
            step=10,
            value=_as_int(cfg["stores"]["num_stores"], 100),
            help="Controls the size of the Stores dimension.",
        )

    with col2:
        cfg["products"]["num_products"] = st.number_input(
            "Products (SKUs)",
            min_value=1,
            step=500,
            value=_as_int(cfg["products"]["num_products"], 5_000),
            help="Controls the size of the Products dimension. Higher values can increase sales generation variance.",
        )

        # Promotions as total (distributed into buckets if bucketed schema exists)
        promo_total = _get_promotions_total(cfg["promotions"])
        promo_total_new = st.number_input(
            "Active promotions",
            min_value=0,
            step=5,
            value=int(promo_total),
            help="Total active promotions. If promotions are bucketed (seasonal/clearance/limited), this will distribute the total across buckets.",
        )
        _set_promotions_total(cfg["promotions"], int(promo_total_new))

    st.caption("Tip: Dimension sizes affect generation time and file sizes; tune them before increasing sales rows.")

    # -----------------------------------------------------
    # NOTE:
    # Pricing (product behavior) is rendered AFTER this
    # function by the caller. Regeneration controls must
    # therefore stay LAST in this section.
    # -----------------------------------------------------
