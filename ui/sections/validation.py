# ui/sections/validation.py
import streamlit as st
from ui.validators import validate


def render_validation(cfg):
    st.subheader("5️⃣ Validation")

    errors, warnings = validate(cfg)
    n_err, n_warn = len(errors), len(warnings)

    # Summary
    if n_err:
        st.error(f"Configuration has {n_err} error(s) and {n_warn} warning(s). Fix errors before generating.")
    elif n_warn:
        st.warning(f"Configuration is valid, but has {n_warn} warning(s).")
    else:
        st.success("Configuration is valid.")

    # Details (collapsed when long)
    if errors:
        with st.expander("Show errors", expanded=True):
            for e in errors:
                st.error(e)

    if warnings:
        with st.expander("Show warnings", expanded=(n_err == 0)):
            for w in warnings:
                st.warning(w)

    return errors, warnings
