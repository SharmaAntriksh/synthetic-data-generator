# sales.file_format, sales.skip_order_cols
import streamlit as st

def render_output(cfg):
    sales = cfg["sales"]

    st.subheader("1️⃣ Output")

    sales["file_format"] = st.selectbox(
        "Output format",
        ["csv", "parquet", "deltaparquet"],
        index=["csv", "parquet", "deltaparquet"].index(sales["file_format"]),
    )

    sales["skip_order_cols"] = st.checkbox(
        "Skip order columns (smaller file size)",
        value=sales["skip_order_cols"],
    )
