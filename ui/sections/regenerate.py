# ui/sections/regenerate.py
import streamlit as st
from ui.constants import REGENERATABLE_DIMENSIONS


def _clear_dim_flags():
    st.session_state.pop("regen_all_dims", None)
    for dim in REGENERATABLE_DIMENSIONS:
        st.session_state.pop(f"regen_dim_{dim}", None)


def get_regen_dimensions():
    if st.session_state.get("regen_all_dims", False):
        return set(REGENERATABLE_DIMENSIONS)
    selected = set()
    for dim in REGENERATABLE_DIMENSIONS:
        if st.session_state.get(f"regen_dim_{dim}", False):
            selected.add(dim)
    return selected


def render_regeneration():
    st.subheader("Regenerate dimensions")
    st.caption("Applies to the next Generate run only.")

    if st.session_state.pop("_clear_regen_ui", False):
        _clear_dim_flags()

    regen_all = st.checkbox(
        "Regenerate all dimensions",
        key="regen_all_dims",
        help="When enabled, individual selections are disabled.",
    )

    cols = st.columns(3)
    for i, dim in enumerate(REGENERATABLE_DIMENSIONS):
        with cols[i % 3]:
            st.checkbox(
                dim.replace("_", " ").title(),
                key=f"regen_dim_{dim}",
                disabled=regen_all,
            )

    return get_regen_dimensions()
