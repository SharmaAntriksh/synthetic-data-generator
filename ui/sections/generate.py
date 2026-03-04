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

from ui.constants import REGENERATABLE_DIMENSIONS
from ui.helpers import find_repo_root


ANSI_ESCAPE_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _resolve_path(project_root: Path, p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (project_root / p).resolve()


def _derive_regen_dims() -> set[str]:
    regen_all = st.session_state.get("regen_all_dims", False)
    if regen_all:
        return set(REGENERATABLE_DIMENSIONS)
    return {d for d in REGENERATABLE_DIMENSIONS if st.session_state.get(f"regen_dim_{d}", False)}


def _normalize_format(v: str) -> str:
    v = str(v or "").strip().lower()
    if v == "delta":
        return "deltaparquet"
    return v


def _format_bytes(n: int) -> str:
    n = int(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


# ------------------------------------------------------------------
# Pre-run summary
# ------------------------------------------------------------------

def _render_summary(cfg: dict, regen_dims: set[str]) -> None:
    sales = cfg.get("sales", {})
    customers = cfg.get("customers", {})
    products = cfg.get("products", {})
    defaults_dates = cfg.get("defaults", {}).get("dates", {})

    start = defaults_dates.get("start", "\u2014")
    end = defaults_dates.get("end", "\u2014")

    file_format = str(sales.get("file_format", "\u2014")).upper()
    sales_rows = sales.get("total_rows", "\u2014")
    cust_n = customers.get("total_customers", "\u2014")
    prod_n = products.get("num_products", "\u2014")

    st.markdown(
        f"**This will generate:** "
        f"**{sales_rows:,}** sales rows, "
        f"**{cust_n:,}** customers, "
        f"**{prod_n:,}** products | "
        f"**{start}** to **{end}** | "
        f"Format: **{file_format}**"
    )

    final_out = cfg.get("final_output_folder")
    if final_out:
        st.caption(f"Final output folder: {final_out}")

    if regen_dims:
        st.markdown(
            "**Regenerating dimensions:** "
            + ", ".join(d.replace("_", " ").title() for d in sorted(regen_dims))
        )


# ------------------------------------------------------------------
# Subprocess execution (extracted from render_generate)
# ------------------------------------------------------------------

def _stream_logs(process, log_area, timer_area=None, max_lines: int = 2000) -> None:
    buf = deque(maxlen=max_lines)
    last_render = 0.0
    start_time = time.time()

    if process.stdout is None:
        return

    for line in process.stdout:
        clean = ANSI_ESCAPE_RE.sub("", line).rstrip("\n")
        buf.append(clean)

        now = time.time()
        if now - last_render > 0.15:
            log_area.code("\n".join(buf), language="text")
            if timer_area is not None:
                elapsed = now - start_time
                timer_area.caption(f"Elapsed: {_fmt_elapsed(elapsed)}")
            last_render = now

    # final render
    log_area.code("\n".join(buf), language="text")
    if timer_area is not None:
        elapsed = time.time() - start_time
        timer_area.caption(f"Total elapsed: {_fmt_elapsed(elapsed)}")


def _fmt_elapsed(sec: float) -> str:
    if sec < 60:
        return f"{sec:.1f}s"
    m, s = divmod(int(sec), 60)
    if m < 60:
        return f"{m}m {s}s"
    h, m = divmod(m, 60)
    return f"{h}h {m}m {s}s"


def _build_cli_command(
    project_root: Path,
    cfg_path: Path,
    models_cfg_path: Path,
    clean: bool,
    only: str,
    regen_dims: set[str],
) -> list[str]:
    main_py = project_root / "main.py"

    cmd = [
        sys.executable,
        "-u",
        str(main_py),
        "--config", str(cfg_path),
        "--models-config", str(models_cfg_path),
    ]

    if clean:
        cmd.append("--clean")

    if only in ("dimensions", "sales"):
        cmd.extend(["--only", only])

    if regen_dims:
        cmd.extend(["--regen-dimensions", *sorted(regen_dims)])

    return cmd


def _run_pipeline_subprocess(
    project_root: Path,
    cmd: list[str],
    log_area,
    timer_area=None,
) -> int:
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
        return 1

    _stream_logs(process, log_area, timer_area)
    return process.wait()


# ------------------------------------------------------------------
# Post-run artifacts
# ------------------------------------------------------------------

def _find_pbip_output_dir(run_folder: Path, fmt: str):
    if not run_folder.exists() or not run_folder.is_dir():
        return None

    want = None
    if fmt == "csv":
        want = "pbip csv"
    elif fmt in ("parquet", "deltaparquet"):
        want = "pbip parquet"

    if want:
        for p in run_folder.iterdir():
            if p.is_dir() and p.name.lower() == want:
                return p

    for p in run_folder.iterdir():
        if p.is_dir() and p.name.lower() in ("pbip csv", "pbip parquet"):
            return p

    return None


def _looks_like_pbip(folder: Path) -> bool:
    if folder.name.lower().endswith(".pbip"):
        return True
    try:
        top = list(folder.iterdir())
    except Exception:
        return False

    if any(p.is_file() and p.suffix.lower() == ".pbip" for p in top):
        return True

    typical_dirs = {"dataset", "report"}
    top_dirnames = {p.name.lower() for p in top if p.is_dir()}
    return bool(typical_dirs & top_dirnames)


def _collect_artifacts(folder: Path, *, recursive: bool, limit: int = 250) -> list[dict]:
    rows = []
    if not folder.exists():
        return rows

    skip_dirs = {
        "generated_datasets", "generate_datasets",
        "__pycache__", ".git", ".venv", "node_modules",
    }

    def _scan(base: Path, depth: int = 0):
        if len(rows) >= limit:
            return
        try:
            entries = sorted(base.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        except PermissionError:
            return

        for entry in entries:
            if len(rows) >= limit:
                return

            if entry.is_dir():
                if entry.name in skip_dirs:
                    continue
                if recursive and depth < 4:
                    _scan(entry, depth + 1)
                else:
                    rows.append({
                        "Name": entry.name + "/",
                        "Size": "",
                        "Type": "folder",
                    })
            else:
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = 0
                rows.append({
                    "Name": str(entry.relative_to(folder)),
                    "Size": _format_bytes(size),
                    "Type": entry.suffix.lstrip(".") or "file",
                })

    _scan(folder)
    return rows


def _render_artifacts(cfg: dict, project_root: Path) -> None:
    final_out_cfg = cfg.get("final_output_folder")
    if not final_out_cfg:
        st.info("final_output_folder not set in config; skipping artifacts list.")
        return

    final_out_path = _resolve_path(project_root, final_out_cfg)
    fmt = _normalize_format((cfg.get("sales", {}) or {}).get("file_format", "csv"))

    pbip_dir = _find_pbip_output_dir(final_out_path, fmt)

    if pbip_dir is not None:
        scan_root = pbip_dir
        recursive = False
        st.info(f"PBIP output detected. Showing top-level only: {pbip_dir}")
    else:
        scan_root = final_out_path
        is_pbip = _looks_like_pbip(final_out_path)
        recursive = not (is_pbip and fmt in ("csv", "parquet", "deltaparquet"))

    st.subheader("Artifacts")
    st.caption(f"Artifacts root: {scan_root}")

    rows = _collect_artifacts(scan_root, recursive=recursive, limit=250 if recursive else 200)

    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("No files found (or output folder does not exist yet).")


# ------------------------------------------------------------------
# Main render function
# ------------------------------------------------------------------

def render_generate(cfg: dict, errors: list[str]):
    st.subheader("6\ufe0f\u20e3 Generate")

    project_root = find_repo_root()

    regen_dims = _derive_regen_dims()
    _render_summary(cfg, regen_dims)

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

    disabled = bool(errors)

    if st.button("Generate Data", type="primary", disabled=disabled):
        if errors:
            st.error("Fix validation errors before running.")
            return

        models_cfg_path = st.session_state.get("models_config_path", "models.yaml")
        models_cfg_path = _resolve_path(project_root, models_cfg_path)

        main_py = project_root / "main.py"
        if not main_py.exists():
            st.error(f"Could not find main.py under: {project_root}")
            return

        st.info("Running pipeline...")
        log_area = st.empty()
        timer_area = st.empty()

        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.yaml"
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)

            cmd = _build_cli_command(
                project_root, cfg_path, models_cfg_path,
                clean=clean,
                only=only if only != "all" else "",
                regen_dims=regen_dims,
            )

            with st.expander("Command"):
                st.code(" ".join(cmd), language="bash")

            rc = _run_pipeline_subprocess(project_root, cmd, log_area, timer_area)

        if rc == 0:
            st.success("Data generation completed successfully.")
            st.session_state["_clear_regen_ui"] = True
            _render_artifacts(cfg, project_root)
        else:
            st.error("Generation failed. See logs above.")
