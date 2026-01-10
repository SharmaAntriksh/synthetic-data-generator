import streamlit as st
import yaml
import subprocess
import tempfile
import sys
from pathlib import Path
import re

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

# --- Streamlit import bootstrap ---
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui.presets import PRESETS, apply_preset
from ui.validators import validate, cpu_count_safe


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------
BASE_CONFIG_PATH = Path("config.yaml")

DIMENSION_SIZE_FIELDS = {
    "customers": "total_customers",
    "products": "num_products",
    "stores": "num_stores",
    "promotions": "num_seasonal",
}

# ------------------------------------------------------------
# App setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Contoso Fake Data Generator",
    layout="wide",
)

st.title("Contoso Fake Data Generator")
st.caption("Generate large, realistic datasets using a schema-safe web UI")

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def load_base_config():
    if not BASE_CONFIG_PATH.exists():
        st.error("config.yaml not found in project root.")
        st.stop()
    with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def require_key(cfg, path: list[str]):
    cur = cfg
    for p in path:
        if p not in cur:
            st.error(f"config.yaml is missing: {'.'.join(path)}")
            st.stop()
        cur = cur[p]
    return cur


def apply_global_dates(cfg):
    start = cfg["defaults"]["dates"]["start"]
    end = cfg["defaults"]["dates"]["end"]

    for section in ["sales", "stores", "promotions", "dates", "exchange_rates"]:
        if section not in cfg:
            continue
        cfg[section].setdefault("dates", {})
        cfg[section]["dates"]["start"] = start
        cfg[section]["dates"]["end"] = end


def run_generator(config: dict):
    tmp_dir = tempfile.TemporaryDirectory()
    cfg_path = Path(tmp_dir.name) / "config.yaml"

    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    process = subprocess.Popen(
        [sys.executable, "main.py", "--config", str(cfg_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=Path.cwd(),
    )

    return process, tmp_dir


# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------
if "config" not in st.session_state:
    st.session_state.config = load_base_config()

cfg = st.session_state.config

# ------------------------------------------------------------
# Sidebar — Presets
# ------------------------------------------------------------
with st.sidebar:
    st.header("Presets")

    for name in PRESETS:
        if st.button(name):
            apply_preset(cfg, load_base_config, name)


# ------------------------------------------------------------
# Main UI
# ------------------------------------------------------------
sales = require_key(cfg, ["sales"])

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

st.subheader("2️⃣ Date range")

st.subheader("📅 Calendar options")

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
    disabled=True,  # safe default, always on
)

include["iso"] = st.checkbox(
    "Include ISO week columns",
    value=include.get("iso", False),
)

include["fiscal"] = st.checkbox(
    "Include fiscal calendar columns",
    value=include.get("fiscal", False),
)

defaults_dates = require_key(cfg, ["defaults", "dates"])

defaults_dates["start"] = st.date_input("Start date", value=defaults_dates["start"])
defaults_dates["end"] = st.date_input("End date", value=defaults_dates["end"])

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

st.subheader("4️⃣ Dimensions")

def dim(section, label, step, min_val=1):
    field = DIMENSION_SIZE_FIELDS[section]
    cfg[section][field] = st.number_input(
        label,
        min_value=min_val,
        step=step,
        value=cfg[section][field],
    )

dim("customers", "Customers (entities)", step=1_000)
dim("products", "Products (SKUs)", step=500)
dim("stores", "Physical stores", step=10)
dim("promotions", "Active promotions", step=5, min_val=0)

st.subheader("💰 Pricing")

pricing_base = (
    cfg["products"]
    .setdefault("pricing", {})
    .setdefault("base", {})
)

pricing_base["value_scale"] = st.number_input(
    "Base product value scale",
    min_value=0.01,
    max_value=10.0,
    step=0.05,
    format="%.2f",
    value=pricing_base.get("value_scale", 1.0),
    help="Scales base product prices (e.g. 0.2 = cheaper products)",
)

# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# Actions
# ------------------------------------------------------------
st.subheader("6️⃣ Generate")

st.markdown(
    f"""
**This will generate:**
- **{sales['total_rows']:,}** sales rows
- **{cfg['customers'][DIMENSION_SIZE_FIELDS['customers']]:,}** customers
- **{cfg['products'][DIMENSION_SIZE_FIELDS['products']]:,}** products
- Output format: **{sales['file_format'].upper()}**
"""
)

if st.button("▶ Generate Data", type="primary"):
    if errors:
        st.error("Fix validation errors before running.")
    else:
        apply_global_dates(cfg)
        st.info("Running pipeline...")
        log_area = st.empty()

        process, tmp_dir = run_generator(cfg)
        logs = []

        for line in process.stdout:
            clean = ANSI_ESCAPE_RE.sub("", line)
            logs.append(clean)
            log_area.code("".join(logs), language="text")

        process.wait()
        tmp_dir.cleanup()

        if process.returncode == 0:
            st.success("Data generation completed successfully.")
        else:
            st.error("Generation failed. See logs above.")

# ------------------------------------------------------------
# Resolved config (advanced)
# ------------------------------------------------------------
with st.expander("Resolved config (advanced)"):
    st.code(yaml.safe_dump(cfg, sort_keys=False), language="yaml")

    st.download_button(
        "Download config.yaml",
        data=yaml.safe_dump(cfg, sort_keys=False),
        file_name="config.yaml",
        mime="text/yaml",
    )
