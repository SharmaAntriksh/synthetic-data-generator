# ui/sections/volume.py
import streamlit as st
from ui.validators import cpu_count_safe


def _as_int(v, default: int) -> int:
    try:
        return int(v)
    except Exception:
        return default


def render_volume(cfg: dict) -> None:
    sales = cfg.setdefault("sales", {})
    sales.setdefault("total_rows", 100_000)
    sales.setdefault("chunk_size", 200_000)
    # workers may be None for auto
    if "workers" not in sales:
        sales["workers"] = None

    st.subheader("3️⃣ Volume")

    total_rows = _as_int(sales.get("total_rows"), 100_000)
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

        # Workers: allow Auto (None)
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
            current_workers = _as_int(current_workers, 1)

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
        chunk_size = _as_int(chunk_size, 200_000)

        sales["chunk_size"] = st.number_input(
            "Chunk size",
            min_value=10_000,
            step=100_000,
            value=chunk_size,
            help="Rows generated per chunk. Larger chunks reduce overhead but use more memory.",
        )

        if _as_int(sales["chunk_size"], 0) > _as_int(sales["total_rows"], 0):
            st.warning("Chunk size exceeds total rows; consider lowering chunk size.")
