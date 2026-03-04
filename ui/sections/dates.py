# ui/sections/dates.py
from __future__ import annotations

from datetime import date, datetime
import streamlit as st


_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
           "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _to_date(v) -> date:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        try:
            return date.fromisoformat(v)
        except ValueError:
            pass
    return date.today()


def _set_iso(dct: dict, key: str, d: date) -> None:
    dct[key] = d.isoformat()


def render_dates(cfg, require_key):
    st.subheader("2\ufe0f\u20e3 Date range")

    defaults_dates = require_key(cfg, ["defaults", "dates"])
    defaults_dates.setdefault("start", date.today().isoformat())
    defaults_dates.setdefault("end", date.today().isoformat())

    start_d = _to_date(defaults_dates.get("start"))
    end_d = _to_date(defaults_dates.get("end"))

    col1, col2 = st.columns(2)

    with col1:
        picked = st.date_input("Start date", value=start_d)
        _set_iso(defaults_dates, "start", picked)

    with col2:
        picked = st.date_input("End date", value=end_d)
        _set_iso(defaults_dates, "end", picked)

    if _to_date(defaults_dates["end"]) < _to_date(defaults_dates["start"]):
        st.warning("End date is before start date.")

    with st.expander("Calendar options"):
        dates_cfg = cfg.setdefault("dates", {})
        include = dates_cfg.setdefault("include", {})

        current = int(dates_cfg.get("fiscal_month_offset", 0) or 0)
        current = max(0, min(11, current))

        month = st.selectbox(
            "First month of fiscal year",
            _MONTHS,
            index=current,
            help="Stored as fiscal_month_offset (0=Jan ... 11=Dec).",
        )
        dates_cfg["fiscal_month_offset"] = _MONTHS.index(month)

        include["calendar"] = True
        st.caption("Calendar columns are always included.")

        include["iso"] = st.checkbox(
            "Include ISO week columns",
            value=bool(include.get("iso", False)),
        )
        include["fiscal"] = st.checkbox(
            "Include fiscal calendar columns",
            value=bool(include.get("fiscal", False)),
        )
