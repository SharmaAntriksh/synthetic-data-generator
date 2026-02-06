# ui/sections/output.py
import streamlit as st


_FORMATS = ["csv", "parquet", "deltaparquet"]


def _normalize_format(v) -> str:
    v = str(v or "").strip().lower()
    if v == "delta":
        return "deltaparquet"
    if v in _FORMATS:
        return v
    return "csv"


def render_output(cfg: dict) -> None:
    st.subheader("1️⃣ Output")

    sales = cfg.setdefault("sales", {})
    sales["file_format"] = _normalize_format(sales.get("file_format", "csv"))
    sales.setdefault("skip_order_cols", False)

    col1, col2 = st.columns([1, 1])

    with col1:
        # Safe index selection
        current_fmt = sales["file_format"]
        idx = _FORMATS.index(current_fmt) if current_fmt in _FORMATS else 0

        sales["file_format"] = st.selectbox(
            "Output format",
            _FORMATS,
            index=idx,
            help="CSV is easiest to inspect but biggest. Parquet/DeltaParquet are smaller and faster for analytics.",
        )

    with col2:
        sales["skip_order_cols"] = st.checkbox(
            "Skip order columns (smaller file size)",
            value=bool(sales.get("skip_order_cols", False)),
            help="Removes SalesOrderHeader/Detail-style columns from the fact output. Use this if you only need aggregated analysis.",
        )

    # Optional contextual hint
    if sales["file_format"] == "csv":
        st.caption("Tip: For large runs, prefer parquet/deltaparquet to reduce disk usage and improve load speed.")
