# ui/sections/validation.py
import streamlit as st
from ui.validators import validate, validate_models_config


def render_validation(cfg):
    st.subheader("5\ufe0f\u20e3 Validation")

    errors, warnings = validate(cfg)

    # Models config validation (if path is available in session state)
    models_path = st.session_state.get("models_config_path")
    if models_path:
        m_errors, m_warnings = validate_models_config(models_path)
        errors.extend(m_errors)
        warnings.extend(m_warnings)

    n_err, n_warn = len(errors), len(warnings)

    if n_err:
        st.error(f"Configuration has {n_err} error(s) and {n_warn} warning(s). Fix errors before generating.")
    elif n_warn:
        st.warning(f"Configuration is valid, but has {n_warn} warning(s).")
    else:
        st.success("Configuration is valid.")

    if errors:
        with st.expander("Show errors", expanded=True):
            for e in errors:
                st.error(e)

    if warnings:
        with st.expander("Show warnings", expanded=(n_err == 0)):
            for w in warnings:
                st.warning(w)

    return errors, warnings
