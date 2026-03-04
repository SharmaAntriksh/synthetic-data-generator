# ui/app.py
import argparse
import sys
from pathlib import Path

import streamlit as st
import yaml

from ui.helpers import find_repo_root, ensure_root_on_path


# ------------------------------------------------------------------
# Bootstrap (no Streamlit calls before set_page_config)
# ------------------------------------------------------------------

ROOT = find_repo_root()
ensure_root_on_path(ROOT)


def parse_app_args():
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--models-config", default="models.yaml")
    return p.parse_known_args()[0]


APP_ARGS = parse_app_args()
BASE_CONFIG_PATH = Path(APP_ARGS.config)
MODELS_CONFIG_PATH = Path(APP_ARGS.models_config)


def _resolve_under_root(p: Path) -> Path:
    if p.is_absolute():
        return p
    return (ROOT / p).resolve()


def load_base_config():
    p = _resolve_under_root(BASE_CONFIG_PATH)
    if not p.exists():
        st.error(f"Config file not found: {p}")
        st.stop()
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_models_config():
    p = _resolve_under_root(MODELS_CONFIG_PATH)
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def require_key(cfg, path):
    cur = cfg
    for key in path:
        if key not in cur:
            st.error(f"config.yaml is missing: {'.'.join(path)}")
            st.stop()
        cur = cur[key]
    return cur


# ------------------------------------------------------------------
# Page setup
# ------------------------------------------------------------------

st.set_page_config(page_title="Synthetic Retail Data Generator", layout="wide")

st.markdown(
    "<style>.block-container{padding-top:1.5rem;padding-bottom:1.5rem;"
    "padding-left:2rem;padding-right:2rem;}</style>",
    unsafe_allow_html=True,
)

st.title("Retail Data Generator")
st.caption("Generate large, realistic datasets using a schema-safe web UI")

# ------------------------------------------------------------------
# Imports that depend on ROOT being in sys.path
# ------------------------------------------------------------------

from ui.presets import apply_preset, build_presets_by_sales
from ui.sections import (
    render_output,
    render_dates,
    render_volume,
    render_dimensions,
    render_pricing,
    render_validation,
    render_generate,
    render_regeneration,
)

# ------------------------------------------------------------------
# Session state init
# ------------------------------------------------------------------

if "config" not in st.session_state:
    st.session_state.config = load_base_config()

if "config_path" not in st.session_state:
    st.session_state.config_path = str(_resolve_under_root(BASE_CONFIG_PATH))

if "models_config_path" not in st.session_state:
    st.session_state.models_config_path = str(_resolve_under_root(MODELS_CONFIG_PATH))

cfg = st.session_state.config

# ------------------------------------------------------------------
# Sidebar
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
        st.toast(f"Applied: {preset_name}")

    st.divider()

    # --- Models tuning knobs ---
    st.header("Models Tuning")
    st.caption("Quick sliders for customer acquisition behaviour.")

    models_raw = _load_models_config()
    tuning = {}
    if models_raw and isinstance(models_raw, dict):
        tuning = models_raw.get("tuning", {}) or {}

    tuning_intensity = st.slider(
        "Acquisition intensity",
        min_value=0.0, max_value=1.0, step=0.05,
        value=float(tuning.get("acquisition_intensity", 0.55)),
        help="0 = few new customers, 1 = many new customers.",
    )
    tuning_smoothness = st.slider(
        "Acquisition smoothness",
        min_value=0.0, max_value=1.0, step=0.05,
        value=float(tuning.get("acquisition_smoothness", 0.90)),
        help="0 = spiky month-to-month, 1 = very flat spread.",
    )
    tuning_cycles = st.slider(
        "Acquisition cycles",
        min_value=0.0, max_value=1.0, step=0.05,
        value=float(tuning.get("acquisition_cycles", 0.35)),
        help="0 = no multi-year waves, 1 = dramatic waves.",
    )

    st.session_state["_tuning_overrides"] = {
        "acquisition_intensity": tuning_intensity,
        "acquisition_smoothness": tuning_smoothness,
        "acquisition_cycles": tuning_cycles,
    }

    st.divider()

    # --- Config download ---
    st.header("Config")
    st.caption(f"Config: {st.session_state.config_path}")
    st.caption(f"Models: {st.session_state.models_config_path}")

    config_yaml = yaml.safe_dump(cfg, sort_keys=False)
    st.download_button(
        "Download current config",
        data=config_yaml,
        file_name="config.yaml",
        mime="text/yaml",
        use_container_width=True,
    )

# ------------------------------------------------------------------
# Main UI
# ------------------------------------------------------------------

render_output(cfg)
st.write("")

render_dates(cfg, require_key)
st.write("")

render_volume(cfg)
st.write("")

render_dimensions(cfg)
render_pricing(cfg)
render_regeneration()
st.write("")

errors, warnings = render_validation(cfg)

st.caption("Review the summary below, then generate the dataset.")
render_generate(cfg, errors)

# ------------------------------------------------------------------
# Advanced
# ------------------------------------------------------------------

with st.expander("Advanced: Resolved config"):
    st.code(yaml.safe_dump(cfg, sort_keys=False), language="yaml")
