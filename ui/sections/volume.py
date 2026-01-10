# sales.total_rows, workers, chunk_size
import streamlit as st
from ui.validators import cpu_count_safe

def render_volume(cfg):
    sales = cfg["sales"]

    st.subheader("3️⃣ Volume")

    sales["total_rows"] = st.number_input(
        "Sales rows",
        min_value=1,
        step=100_000,
        value=sales["total_rows"],
    )

    with st.expander("Performance tuning (advanced)"):
        sales["workers"] = st.number_input(
            "Worker processes",
            min_value=1,
            max_value=cpu_count_safe(),
            value=sales["workers"],
        )

        sales["chunk_size"] = st.number_input(
            "Chunk size",
            min_value=10_000,
            step=100_000,
            value=sales["chunk_size"],
        )
