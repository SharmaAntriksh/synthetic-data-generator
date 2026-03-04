# ui/sections/dimensions.py
from __future__ import annotations

import streamlit as st
from ui.helpers import as_int
from ui.constants import DIMENSION_SIZE_FIELDS


def _ensure(cfg, section, field, default):
    cfg.setdefault(section, {})
    cfg[section].setdefault(field, default)


def _get_promotions_total(promotions_cfg):
    keys = ["num_seasonal", "num_clearance", "num_limited"]
    if all(k in promotions_cfg for k in keys):
        return sum(as_int(promotions_cfg.get(k), 0) for k in keys)
    return as_int(promotions_cfg.get("total_promotions", 0), 0)


def _set_promotions_total(promotions_cfg, total):
    total = max(0, int(total))
    keys = ["num_seasonal", "num_clearance", "num_limited"]

    if all(k in promotions_cfg for k in keys):
        cur = [as_int(promotions_cfg.get(k), 0) for k in keys]
        s = sum(cur)
        base = [1, 1, 1] if s <= 0 else cur
        s = s if s > 0 else 3

        scaled = [b * total / s for b in base]
        floors = [int(x) for x in scaled]
        remainder = total - sum(floors)

        fracs = sorted(
            [(i, scaled[i] - floors[i]) for i in range(3)],
            key=lambda t: t[1], reverse=True,
        )
        for i in range(remainder):
            floors[fracs[i % 3][0]] += 1

        promotions_cfg["num_seasonal"] = floors[0]
        promotions_cfg["num_clearance"] = floors[1]
        promotions_cfg["num_limited"] = floors[2]
    else:
        promotions_cfg["total_promotions"] = total

    promotions_cfg["total_promotions"] = total


def render_dimensions(cfg):
    st.subheader("4\ufe0f\u20e3 Dimensions")

    _ensure(cfg, "customers", "total_customers", 10_000)
    _ensure(cfg, "stores", "num_stores", 100)
    _ensure(cfg, "products", "num_products", 5_000)
    cfg.setdefault("promotions", {})

    col1, col2 = st.columns(2)

    with col1:
        cfg["customers"]["total_customers"] = st.number_input(
            "Customers (entities)", min_value=1, step=1_000,
            value=as_int(cfg["customers"]["total_customers"], 10_000),
            help="Controls the size of the Customers dimension.",
        )
        cfg["stores"]["num_stores"] = st.number_input(
            "Physical stores", min_value=1, step=10,
            value=as_int(cfg["stores"]["num_stores"], 100),
            help="Controls the size of the Stores dimension.",
        )

    with col2:
        cfg["products"]["num_products"] = st.number_input(
            "Products (SKUs)", min_value=1, step=500,
            value=as_int(cfg["products"]["num_products"], 5_000),
            help="Controls the size of the Products dimension.",
        )

        promo_total = _get_promotions_total(cfg["promotions"])
        promo_total_new = st.number_input(
            "Active promotions", min_value=0, step=5,
            value=int(promo_total),
            help="Total active promotions. Distributed across buckets if bucketed schema exists.",
        )
        _set_promotions_total(cfg["promotions"], int(promo_total_new))

    st.caption("Tip: Dimension sizes affect generation time and file sizes; tune them before increasing sales rows.")
