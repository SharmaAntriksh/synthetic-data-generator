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


def _clear_dim_flags():
    st.session_state.pop("regen_all_dims", None)
    for dim in DIMENSIONS:
        st.session_state.pop(f"regen_dim_{dim}", None)


def get_regen_dimensions() -> set[str]:
    """
    Read current UI selection from session_state and return the set of dimensions
    to force regenerate. If "Regenerate all" is checked, returns all DIMENSIONS.
    """
    if st.session_state.get("regen_all_dims", False):
        return set(DIMENSIONS)

    selected = set()
    for dim in DIMENSIONS:
        if st.session_state.get(f"regen_dim_{dim}", False):
            selected.add(dim)
    return selected


def render_regeneration() -> set[str]:
    st.subheader("🔄 Regenerate dimensions")
    st.caption("Applies to the next Generate run only.")

    # One-shot UI reset after Generate
    # (Generate section should set: st.session_state["_clear_regen_ui"] = True)
    if st.session_state.pop("_clear_regen_ui", False):
        _clear_dim_flags()

    # NOTE: Keep the checkbox simple; no Clear button.

    regen_all = st.checkbox(
        "Regenerate all dimensions",
        key="regen_all_dims",
        help="When enabled, individual selections are disabled.",
    )

    cols = st.columns(3)
    for i, dim in enumerate(DIMENSIONS):
        with cols[i % 3]:
            st.checkbox(
                dim.replace("_", " ").title(),
                key=f"regen_dim_{dim}",
                disabled=regen_all,
            )

    return get_regen_dimensions()
