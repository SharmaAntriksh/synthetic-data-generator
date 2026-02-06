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

    # --------------------------------------------------
    # One-shot UI reset after Generate
    # (Generate section should set: st.session_state["_clear_regen_ui"] = True)
    # --------------------------------------------------
    if st.session_state.pop("_clear_regen_ui", False):
        _clear_dim_flags()

    # --------------------------------------------------
    # Detect regen_all toggle OFF → clear stale flags
    # --------------------------------------------------
    prev_all = st.session_state.get("_prev_regen_all_dims", False)
    current_all = st.session_state.get("regen_all_dims", False)

    if prev_all and not current_all:
        # Clear individual flags when turning "all" off, to avoid stale selections
        for dim in DIMENSIONS:
            st.session_state.pop(f"regen_dim_{dim}", None)

    st.session_state["_prev_regen_all_dims"] = current_all

    # --------------------------------------------------
    # Top-row actions
    # --------------------------------------------------
    left, right = st.columns([3, 1])

    with left:
        regen_all = st.checkbox(
            "Regenerate all dimensions",
            key="regen_all_dims",
            help="When enabled, individual selections are disabled.",
        )

    with right:
        if st.button("Clear", use_container_width=True, help="Clear all regeneration selections."):
            _clear_dim_flags()
            st.rerun()

    # --------------------------------------------------
    # Individual dimension checkboxes
    # --------------------------------------------------
    cols = st.columns(3)
    for i, dim in enumerate(DIMENSIONS):
        with cols[i % 3]:
            st.checkbox(
                dim.replace("_", " ").title(),
                key=f"regen_dim_{dim}",
                disabled=regen_all,
            )

    # Return selection for the caller (render_generate / app.py)
    return get_regen_dimensions()
