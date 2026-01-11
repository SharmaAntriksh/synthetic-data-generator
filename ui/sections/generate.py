# run button + execution
import streamlit as st
import yaml
from pathlib import Path
import tempfile
import subprocess
import sys
import re

ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

DIMENSIONS = [
    "customers",
    "products",
    "stores",
    "promotions",
    "dates",
    "currency",
    "exchange_rates",
]


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

        # --------------------------------------------------
        # Derive regeneration intent (CORRECT)
        # --------------------------------------------------
        force_regen = set()

        if st.session_state.get("regen_all_dims", False):
            force_regen = set(DIMENSIONS)
        else:
            for dim in DIMENSIONS:
                if st.session_state.get(f"regen_dim_{dim}", False):
                    force_regen.add(dim)

        apply_global_dates(cfg)
        st.info("Running pipeline...")
        log_area = st.empty()

        # --------------------------------------------------
        # Write temp config
        # --------------------------------------------------
        tmp_dir = tempfile.TemporaryDirectory()
        cfg_path = Path(tmp_dir.name) / "config.yaml"

        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        # --------------------------------------------------
        # Resolve main.py robustly
        # --------------------------------------------------
        project_root = Path(__file__).resolve().parents[2]
        main_py = project_root / "main.py"

        cmd = [
            sys.executable,
            str(main_py),
            "--config",
            str(cfg_path),
        ]

        if force_regen:
            cmd.extend(["--regen-dimensions", *sorted(force_regen)])

        # --------------------------------------------------
        # Run pipeline
        # --------------------------------------------------
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # --------------------------------------------------
        # Stream logs (plain text)
        # --------------------------------------------------
        logs = []

        for line in process.stdout:
            clean = ANSI_ESCAPE_RE.sub("", line)
            logs.append(clean)
            log_area.code("".join(logs), language="text")

        process.wait()
        tmp_dir.cleanup()

        # --------------------------------------------------
        # Trigger SAFE UI reset (DO NOT touch widget keys)
        # --------------------------------------------------
        if process.returncode == 0:
            st.success("Data generation completed successfully.")
            st.session_state["_clear_regen_ui"] = True
        else:
            st.error("Generation failed. See logs above.")
