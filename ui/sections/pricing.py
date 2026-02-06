import streamlit as st


def _as_float(v, default: float) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _as_int(v, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def render_pricing(cfg: dict) -> None:
    with st.expander("💰 Pricing (product behavior)"):
        products = cfg.setdefault("products", {})
        pricing = products.setdefault("pricing", {})
        pricing_base = pricing.setdefault("base", {})

        # Defaults (reasonable)
        pricing_base.setdefault("value_scale", 1.0)
        pricing_base.setdefault("min_unit_price", 10)
        pricing_base.setdefault("max_unit_price", 5000)

        col1, col2 = st.columns(2)

        with col1:
            pricing_base["value_scale"] = st.number_input(
                "Base product value scale",
                min_value=0.01,
                max_value=10.0,
                step=0.05,
                format="%.2f",
                value=_as_float(pricing_base.get("value_scale"), 1.0),
                help="Scales base product prices (e.g. 0.20 = cheaper products, 2.00 = premium).",
            )

        with col2:
            st.caption(
                f"Effective prices will be scaled by ~{pricing_base['value_scale']:.2f}×"
            )

        st.divider()

        st.caption("Optional bounds (applied after scaling)")

        b1, b2 = st.columns(2)
        with b1:
            pricing_base["min_unit_price"] = st.number_input(
                "Min unit price",
                min_value=0,
                max_value=10_000_000,
                step=10,
                value=_as_int(pricing_base.get("min_unit_price"), 10),
                help="Lower bound on generated unit price after scaling.",
            )
        with b2:
            pricing_base["max_unit_price"] = st.number_input(
                "Max unit price",
                min_value=1,
                max_value=10_000_000,
                step=50,
                value=_as_int(pricing_base.get("max_unit_price"), 5000),
                help="Upper bound on generated unit price after scaling.",
            )

        min_p = int(pricing_base["min_unit_price"])
        max_p = int(pricing_base["max_unit_price"])

        if max_p <= min_p:
            st.warning("Max unit price should be greater than min unit price.")
        else:
            scaled_min = min_p * float(pricing_base["value_scale"])
            scaled_max = max_p * float(pricing_base["value_scale"])
            st.caption(f"Scaled bounds preview: ~{scaled_min:,.0f} → {scaled_max:,.0f}")
