# defaults.dates + dates.*
import streamlit as st

def render_dates(cfg, require_key):
    st.subheader("2️⃣ Date range")

    defaults_dates = require_key(cfg, ["defaults", "dates"])
    defaults_dates["start"] = st.date_input(
        "Start date", value=defaults_dates["start"]
    )
    defaults_dates["end"] = st.date_input(
        "End date", value=defaults_dates["end"]
    )

    with st.expander("📅 Calendar options"):
        dates_cfg = require_key(cfg, ["dates"])

        dates_cfg["fiscal_month_offset"] = st.number_input(
            "Fiscal month offset",
            min_value=0,
            max_value=11,
            value=dates_cfg.get("fiscal_month_offset", 0),
        )

        include = dates_cfg.setdefault("include", {})
        include["calendar"] = st.checkbox(
            "Include calendar columns",
            value=include.get("calendar", True),
            disabled=True,
        )
        include["iso"] = st.checkbox(
            "Include ISO week columns",
            value=include.get("iso", False),
        )
        include["fiscal"] = st.checkbox(
            "Include fiscal calendar columns",
            value=include.get("fiscal", False),
        )
