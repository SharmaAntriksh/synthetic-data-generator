import streamlit as st

DIMENSIONS = [
    "customers",
    "products",
    "stores",
    "promotions",
    "dates",
    "currency",
    "exchange_rates",
]


def render_regeneration():
    st.subheader("🔄 Regenerate dimensions")

    # --------------------------------------------------
    # One-shot UI reset after Generate
    # --------------------------------------------------
    if st.session_state.pop("_clear_regen_ui", False):
        st.session_state.pop("regen_all_dims", None)
        for dim in DIMENSIONS:
            st.session_state.pop(f"regen_dim_{dim}", None)

    # --------------------------------------------------
    # Detect regen_all toggle OFF → clear stale flags
    # --------------------------------------------------
    prev_all = st.session_state.get("_prev_regen_all_dims", False)
    current_all = st.session_state.get("regen_all_dims", False)

    if prev_all and not current_all:
        for dim in DIMENSIONS:
            st.session_state.pop(f"regen_dim_{dim}", None)

    st.session_state["_prev_regen_all_dims"] = current_all

    # --------------------------------------------------
    # Widgets
    # --------------------------------------------------
    regen_all = st.checkbox(
        "Regenerate all dimensions",
        key="regen_all_dims",
        help="Applies to the next Generate run only.",
    )

    cols = st.columns(3)
    for i, dim in enumerate(DIMENSIONS):
        with cols[i % 3]:
            st.checkbox(
                dim.replace("_", " ").title(),
                key=f"regen_dim_{dim}",
                disabled=regen_all,
            )
