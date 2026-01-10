# validate(cfg) + UI rendering
import streamlit as st
from ui.validators import validate

def render_validation(cfg):
    st.subheader("5️⃣ Validation")

    errors, warnings = validate(cfg)

    if errors:
        for e in errors:
            st.error(e)
    elif warnings:
        for w in warnings:
            st.warning(w)
        st.success("Configuration is valid (with warnings).")
    else:
        st.success("Configuration is valid.")

    return errors, warnings
