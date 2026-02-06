# ui/sections/generate.py

from __future__ import annotations

import os
import re
import sys
import time
import tempfile
import subprocess
from collections import deque
from pathlib import Path

import streamlit as st
import yaml


ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

# Single source of truth
DIMENSIONS = [
    "customers",
    "products",
    "stores",
    "geography",
    "promotions",
    "dates",
    "currency",
    "exchange_rates",
]


def _find_project_root() -> Path:
    """
    Robustly locate the repo root by searching upwards for main.py.
    Assumes ui/sections/generate.py lives somewhere inside the repo.
    """
    here = Path(__file__).resolve()
    for p in [here.parent, *here.parents]:
        if (p / "main.py").exists():
            return p
    # Fallback to previous assumption (ui/sections -> repo root is parents[2])
    return here.parents[2]


def _resolve_path(project_root: Path, p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (project_root / p).resolve()


def _derive_regen_dims() -> set[str]:
    regen_all = st.session_state.get("regen_all_dims", False)
    if regen_all:
        return set(DIMENSIONS)
    return {d for d in DIMENSIONS if st.session_state.get(f"regen_dim_{d}", False)}


def _render_summary(cfg: dict, regen_dims: set[str]) -> None:
    sales = cfg.get("sales", {})
    customers = cfg.get("customers", {})
    products = cfg.get("products", {})
    defaults_dates = cfg.get("defaults", {}).get("dates", {})

    start = defaults_dates.get("start", "—")
    end = defaults_dates.get("end", "—")

    file_format = str(sales.get("file_format", "—")).upper()
    sales_rows = sales.get("total_rows", "—")
    cust_n = customers.get("total_customers", "—")
    prod_n = products.get("num_products", "—")

    st.markdown(
        f"""
**This will generate:**
- **{sales_rows:,}** sales rows
- **{cust_n:,}** customers
- **{prod_n:,}** products
- Date range: **{start} → {end}**
- Output format: **{file_format}**
"""
    )

    final_out = cfg.get("final_output_folder")
    if final_out:
        st.caption(f"Final output folder: {final_out}")

    if regen_dims:
        st.markdown(
            "**Regenerating dimensions:** "
            + ", ".join(d.replace("_", " ").title() for d in sorted(regen_dims))
        )


def _stream_logs(process: subprocess.Popen, log_area, max_lines: int = 2000) -> None:
    """
    Stream stdout into a bounded log buffer and update UI at a throttled rate.
    """
    buf = deque(maxlen=max_lines)
    last_render = 0.0

    if process.stdout is None:
        return

    for line in process.stdout:
        clean = ANSI_ESCAPE_RE.sub("", line).rstrip("\n")
        buf.append(clean)

        now = time.time()
        if now - last_render > 0.15:
            log_area.code("\n".join(buf), language="text")
            last_render = now

    # final render
    log_area.code("\n".join(buf), language="text")


def render_generate(cfg: dict, errors: list[str]):
    st.subheader("6️⃣ Generate")

    project_root = _find_project_root()

    regen_dims = _derive_regen_dims()
    _render_summary(cfg, regen_dims)

    # --------------------------------------------------
    # Advanced run controls (optional but useful)
    # --------------------------------------------------
    with st.expander("Run options"):
        col1, col2 = st.columns(2)

        with col1:
            clean = st.checkbox(
                "Clean final outputs before run",
                value=False,
                help="Passes --clean (deletes FINAL output folders before running).",
            )

        with col2:
            only = st.selectbox(
                "Run scope",
                ["all", "dimensions", "sales"],
                index=0,
                help="Passes --only dimensions/sales (or runs full pipeline).",
            )

        st.caption(
            "Logs shown below are truncated to the most recent lines to keep the UI responsive."
        )

    # Disable button if validation errors exist
    disabled = bool(errors)

    if st.button("▶ Generate Data", type="primary", disabled=disabled):
        if errors:
            st.error("Fix validation errors before running.")
            return

        # Resolve models-config path (UI should set this in session_state)
        models_cfg_path = st.session_state.get("models_config_path", "models.yaml")
        models_cfg_path = _resolve_path(project_root, models_cfg_path)

        # --------------------------------------------------
        # Write temp config (resolved UI config)
        # --------------------------------------------------
        st.info("Running pipeline…")
        log_area = st.empty()

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.yaml"
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)

            main_py = project_root / "main.py"
            if not main_py.exists():
                st.error(f"Could not find main.py under: {project_root}")
                return

            # Build CLI command
            cmd = [
                sys.executable,
                "-u",  # unbuffered
                str(main_py),
                "--config",
                str(cfg_path),
                "--models-config",
                str(models_cfg_path),
            ]

            if clean:
                cmd.append("--clean")

            if only in ("dimensions", "sales"):
                cmd.extend(["--only", only])

            if regen_dims:
                cmd.extend(["--regen-dimensions", *sorted(regen_dims)])

            with st.expander("Command"):
                st.code(" ".join(cmd), language="bash")

            # --------------------------------------------------
            # Run pipeline
            # --------------------------------------------------
            env = dict(os.environ)
            env["PYTHONUNBUFFERED"] = "1"

            try:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                    cwd=str(project_root),
                )
            except Exception as e:
                st.error(f"Failed to start generator: {e}")
                return

            _stream_logs(process, log_area)
            rc = process.wait()

        if rc == 0:
            st.success("Data generation completed successfully.")
            # Optional one-shot reset for regen UI (safe if regen section uses it)
            st.session_state["_clear_regen_ui"] = True
        else:
            st.error("Generation failed. See logs above.")
