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

def _find_pbip_output_dir(run_folder: Path, fmt: str) -> Path | None:
    """
    PBIP outputs are written under a nested subfolder like:
      <run_folder>/PBIP CSV
      <run_folder>/PBIP Parquet

    If found, we use that as the artifacts root and DO NOT recurse past its first level.
    """
    if not run_folder.exists() or not run_folder.is_dir():
        return None

    want = None
    if fmt == "csv":
        want = "pbip csv"
    elif fmt in ("parquet", "deltaparquet"):
        want = "pbip parquet"

    # Prefer the format-matching PBIP folder if present
    if want:
        for p in run_folder.iterdir():
            if p.is_dir() and p.name.lower() == want:
                return p

    # Fallback: if either PBIP folder exists, return it
    for p in run_folder.iterdir():
        if p.is_dir() and p.name.lower() in ("pbip csv", "pbip parquet"):
            return p

    return None


def _format_bytes(n: int) -> str:
    n = int(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def _looks_like_pbip(folder: Path) -> bool:
    """
    Heuristic: treat as PBIP if:
      - folder itself endswith .pbip (rare), OR
      - contains any *.pbip file at top-level (common), OR
      - contains typical PBIP folders at top-level.
    """
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


def _normalize_format(v: str) -> str:
    v = str(v or "").strip().lower()
    if v == "delta":
        return "deltaparquet"
    return v

def _collect_artifacts(folder: Path, *, recursive: bool, limit: int = 250) -> list[dict]:
    """
    Returns list of dict rows suitable for st.dataframe.

    - If recursive=False: only top-level entries (files + dirs), no recursion.
    - If recursive=True: list files recursively, BUT do not descend into huge dataset folders
      like generated_datasets / generate_datasets (show the folder entry only).
    """
    if not folder or not folder.exists() or not folder.is_dir():
        return []

    STOP_DIRS = {"generated_datasets", "generate_datasets"}

    # If the artifacts root itself is a stop-folder, force top-level only.
    if recursive and folder.name.lower() in STOP_DIRS:
        recursive = False

    rows: list[dict] = []

    if not recursive:
        for p in sorted(folder.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            try:
                stt = p.stat()
                rows.append({
                    "path": p.name,
                    "type": "dir" if p.is_dir() else "file",
                    "size": "" if p.is_dir() else _format_bytes(stt.st_size),
                    "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stt.st_mtime)),
                })
            except OSError:
                continue
            if len(rows) >= limit:
                break
        return rows

    # Recursive mode: list files, but prune STOP_DIRS so we don't explode the UI
    seen_stop_dirs: set[str] = set()

    for root, dirnames, filenames in os.walk(folder, topdown=True):
        root_path = Path(root)

        # If a stop-dir is present here, add it as a single dir row (so user sees it),
        # then prune it from traversal so we don't list its contents.
        for d in list(dirnames):
            if d.lower() in STOP_DIRS:
                stop_path = root_path / d
                rel = stop_path.relative_to(folder).as_posix()
                if rel not in seen_stop_dirs:
                    try:
                        stt = stop_path.stat()
                        rows.append({
                            "path": f"{rel}/",
                            "type": "dir",
                            "size": "",
                            "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stt.st_mtime)),
                        })
                        seen_stop_dirs.add(rel)
                    except OSError:
                        pass

        # Prune traversal into stop dirs
        dirnames[:] = [d for d in dirnames if d.lower() not in STOP_DIRS]

        # Collect files
        for fn in filenames:
            p = root_path / fn
            try:
                stt = p.stat()
                rows.append({
                    "path": p.relative_to(folder).as_posix(),
                    "type": "file",
                    "size": _format_bytes(stt.st_size),
                    "modified": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stt.st_mtime)),
                })
            except OSError:
                continue

            if len(rows) >= limit:
                return rows

    return rows


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
            st.session_state["_clear_regen_ui"] = True

            # --- Artifacts panel (post-run only) ---
            final_out_cfg = cfg.get("final_output_folder")
            if final_out_cfg:
                final_out_path = _resolve_path(project_root, final_out_cfg)

                fmt = _normalize_format((cfg.get("sales", {}) or {}).get("file_format", "csv"))

                # If PBIP output is nested, stop at the PBIP folder itself (first-level only)
                pbip_dir = _find_pbip_output_dir(final_out_path, fmt)

                if pbip_dir is not None:
                    scan_root = pbip_dir
                    recursive = False
                    st.info(f"PBIP output detected. Showing top-level only: {pbip_dir}")
                else:
                    scan_root = final_out_path
                    # keep your existing behavior for non-PBIP folders
                    is_pbip = _looks_like_pbip(final_out_path)
                    recursive = not (is_pbip and fmt in ("csv", "parquet", "deltaparquet"))

                st.subheader("Artifacts")
                st.caption(f"Artifacts root: {scan_root}")

                rows = _collect_artifacts(scan_root, recursive=recursive, limit=250 if recursive else 200)

                if rows:
                    st.dataframe(rows, use_container_width=True, hide_index=True)
                else:
                    st.info("No files found (or output folder does not exist yet).")
            else:
                st.info("final_output_folder not set in config; skipping artifacts list.")

        else:
            st.error("Generation failed. See logs above.")