import streamlit as st
import yaml
from pathlib import Path
import sys

from ui.presets import apply_preset, build_presets_by_sales
from ui.sections import (
    render_output,
    render_dates,
    render_volume,
    render_dimensions,
    render_pricing,
    render_validation,
    render_generate,
)

# ------------------------------------------------------------------
# Bootstrap
# ------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_CONFIG_PATH = Path("config.yaml")


def load_base_config():
    with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def require_key(cfg, path):
    cur = cfg
    for p in path:
        if p not in cur:
            st.error(f"config.yaml is missing: {'.'.join(path)}")
            st.stop()
        cur = cur[p]
    return cur


# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------

st.set_page_config(
    page_title="Contoso Fake Data Generator",
    layout="wide",
)

st.title("Contoso Fake Data Generator")
st.caption("Generate large, realistic datasets using a schema-safe web UI")

if "config" not in st.session_state:
    st.session_state.config = load_base_config()

cfg = st.session_state.config

# ------------------------------------------------------------------
# Sidebar – Presets (calm, compact)
# ------------------------------------------------------------------

with st.sidebar:
    st.header("Presets")

    presets_by_sales = build_presets_by_sales()

    sales_bucket = st.selectbox(
        "Sales volume",
        list(presets_by_sales.keys()),
        label_visibility="collapsed",
    )

    variants = presets_by_sales[sales_bucket]

    preset_name = st.radio(
        "Dataset variant",
        list(variants.keys()),
        label_visibility="collapsed",
    )

    st.caption(preset_name)

    if st.button("Apply preset", type="primary", use_container_width=True):
        apply_preset(cfg, load_base_config, preset_name)

# ------------------------------------------------------------------
# Main UI – unchanged structure, slight breathing room
# ------------------------------------------------------------------

render_output(cfg)
st.write("")

render_dates(cfg, require_key)
st.write("")

render_volume(cfg)
st.write("")

render_dimensions(cfg)
render_pricing(cfg)
st.write("")

errors, warnings = render_validation(cfg)

st.caption("Review the summary below, then generate the dataset.")
render_generate(cfg, errors)

# ------------------------------------------------------------------
# Advanced
# ------------------------------------------------------------------

with st.expander("Advanced ▸ Resolved config"):
    st.code(
        yaml.safe_dump(cfg, sort_keys=False),
        language="yaml",
    )
