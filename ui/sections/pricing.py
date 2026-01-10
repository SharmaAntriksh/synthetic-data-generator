import streamlit as st

def render_pricing(cfg):
    with st.expander("💰 Pricing (product behavior)"):
        pricing_base = (
            cfg["products"]
            .setdefault("pricing", {})
            .setdefault("base", {})
        )

        pricing_base["value_scale"] = st.number_input(
            "Base product value scale",
            min_value=0.01,
            max_value=10.0,
            step=0.05,
            format="%.2f",
            value=pricing_base.get("value_scale", 1.0),
            help="Scales base product prices (e.g. 0.2 = cheaper products)",
        )

        st.caption(
            f"Effective prices will be scaled by ~{pricing_base['value_scale']}×"
        )
