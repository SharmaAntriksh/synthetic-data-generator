# run button + execution
import streamlit as st
import yaml
from pathlib import Path
import tempfile
import subprocess
import sys
import re

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

def apply_global_dates(cfg):
    start = cfg["defaults"]["dates"]["start"]
    end = cfg["defaults"]["dates"]["end"]

    for section in ["sales", "stores", "promotions", "dates", "exchange_rates"]:
        if section not in cfg:
            continue
        cfg[section].setdefault("dates", {})
        cfg[section]["dates"]["start"] = start
        cfg[section]["dates"]["end"] = end


def render_generate(cfg, errors):
    st.subheader("6️⃣ Generate")

    sales = cfg["sales"]

    st.markdown(
        f"""
**This will generate:**
- **{sales['total_rows']:,}** sales rows
- **{cfg['customers']['total_customers']:,}** customers
- **{cfg['products']['num_products']:,}** products
- Output format: **{sales['file_format'].upper()}**
"""
    )

    if st.button("▶ Generate Data", type="primary"):
        if errors:
            st.error("Fix validation errors before running.")
            return

        apply_global_dates(cfg)
        st.info("Running pipeline...")
        log_area = st.empty()

        tmp_dir = tempfile.TemporaryDirectory()
        cfg_path = Path(tmp_dir.name) / "config.yaml"

        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        process = subprocess.Popen(
            [sys.executable, "main.py", "--config", str(cfg_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=Path.cwd(),
        )

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
