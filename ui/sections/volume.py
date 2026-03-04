# ui/sections/volume.py
import streamlit as st
from ui.helpers import as_int
from ui.validators import cpu_count_safe


def render_volume(cfg: dict) -> None:
    sales = cfg.setdefault("sales", {})
    sales.setdefault("total_rows", 100_000)
    sales.setdefault("chunk_size", 200_000)
    if "workers" not in sales:
        sales["workers"] = None

    st.subheader("3\ufe0f\u20e3 Volume")

    total_rows = as_int(sales.get("total_rows"), 100_000)
    step_rows = 10_000 if total_rows < 500_000 else 100_000

    sales["total_rows"] = st.number_input(
        "Sales rows",
        min_value=1,
        step=step_rows,
        value=total_rows,
        help="Total number of rows to generate in the Sales fact table.",
    )

    with st.expander("Performance tuning (advanced)"):
        st.caption("These settings affect generation speed and memory usage.")

        auto_workers = st.checkbox(
            "Auto-detect workers",
            value=(sales.get("workers") is None),
            help="When enabled, the generator decides worker count automatically.",
        )

        if auto_workers:
            sales["workers"] = None
            st.caption(f"Workers: auto (CPU cores detected: {cpu_count_safe()})")
        else:
            current_workers = sales.get("workers")
            if current_workers is None:
                current_workers = min(6, cpu_count_safe())
            current_workers = as_int(current_workers, 1)

            sales["workers"] = st.number_input(
                "Worker processes",
                min_value=1,
                max_value=cpu_count_safe(),
                value=current_workers,
                help="More workers can speed up generation, but increases CPU/RAM usage.",
            )

        chunk_size = sales.get("chunk_size")
        if chunk_size is None:
            chunk_size = 200_000
        chunk_size = as_int(chunk_size, 200_000)

        sales["chunk_size"] = st.number_input(
            "Chunk size",
            min_value=10_000,
            step=100_000,
            value=chunk_size,
            help="Rows generated per chunk. Larger chunks reduce overhead but use more memory.",
        )

        if as_int(sales["chunk_size"], 0) > as_int(sales["total_rows"], 0):
            st.warning("Chunk size exceeds total rows; consider lowering chunk size.")
