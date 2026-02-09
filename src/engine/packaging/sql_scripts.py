import os
import shutil
from pathlib import Path

from src.tools.sql.generate_bulk_insert_sql import generate_bulk_insert_script
from src.tools.sql.generate_create_table_scripts import generate_all_create_tables
from src.utils.logging_utils import stage, skip, done

from .paths import tables_from_sales_cfg


# ------------------------------------------------------------
# Repo utilities
# ------------------------------------------------------------

def _find_repo_root(start: Path) -> Path:
    """
    Find repo root by walking parents and looking for both 'src' and 'scripts'.
    """
    for p in [start, *start.parents]:
        if (p / "src").is_dir() and (p / "scripts").is_dir():
            return p
    # fallback: common layout <repo>/src/engine/packaging/sql_scripts.py
    return start.parents[4]


def _read_text(path: Path) -> str:
    """
    Robust SQL text reader:
      - Handles UTF-8, UTF-8-BOM, UTF-16 (LE/BE)
      - Heuristic-detects UTF-16 via NUL bytes
      - Strips embedded NULs that break SQL Server parsing
    """
    raw = path.read_bytes()

    # BOM detection
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16")
    elif raw.startswith(b"\xef\xbb\xbf"):
        text = raw.decode("utf-8-sig")
    else:
        # Heuristic: lots of NULs early usually means UTF-16
        if b"\x00" in raw[:4096]:
            try:
                text = raw.decode("utf-16")
            except UnicodeDecodeError:
                text = raw.decode("utf-8", errors="replace")
        else:
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("utf-16")

    # SQL Server will choke on NULs in identifiers (dbo -> d\0b\0o\0)
    if "\x00" in text:
        text = text.replace("\x00", "")

    return text


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _sales_mode(sales_cfg: dict) -> str:
    """
    Normalize sales output mode from config.
    Expected values: "sales", "sales_order", "both"
    """
    mode = str(sales_cfg.get("sales_output") or "").strip().lower()
    return mode or "sales"


# ------------------------------------------------------------
# Static SQL assets (non-conditional)
# ------------------------------------------------------------

def copy_static_sql_assets(*, sql_root: Path) -> None:
    """
    Copy only truly static SQL assets into the packaged output.

    Current scope:
      - Indexes (CCI helper): scripts/sql/columnstore/create_drop_cci.sql
        -> <final>/sql/indexes/create_drop_cci.sql

    Constraints are mode-dependent and must be written by compose_constraints_sql().
    Views are copied by copy_views_sql().
    """
    repo_root = _find_repo_root(Path(__file__).resolve())

    cci_src = repo_root / "scripts" / "sql" / "columnstore" / "create_drop_cci.sql"
    if not cci_src.exists():
        skip(f"CCI script not found; skipping: {cci_src}")
        return

    dst_dir = sql_root / "indexes"
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst = dst_dir / "create_drop_cci.sql"
    shutil.copy2(cci_src, dst)
    done(f"Copied CCI script: {dst.relative_to(sql_root)}")


def copy_views_sql(*, sql_root: Path) -> None:
    """
    Copy the views script into the packaged output.

    Source:
      scripts/sql/views/create_views.sql
    Destination:
      <final>/sql/schema/04_create_views.sql
    """
    repo_root = _find_repo_root(Path(__file__).resolve())
    src = repo_root / "scripts" / "sql" / "views" / "create_views.sql"

    if not src.exists():
        skip(f"Views script not found; skipping: {src}")
        return

    dst = sql_root / "schema" / "04_create_views.sql"
    _write_text(dst, _read_text(src))
    done(f"Copied views script: {dst.relative_to(sql_root)}")


# ------------------------------------------------------------
# Constraints composition (conditional by sales_output)
# ------------------------------------------------------------

def compose_constraints_sql(*, sql_root: Path, sales_cfg: dict) -> None:
    """
    Compose a single constraints script based on sales_output mode.

    Preferred source (modular):
      scripts/sql/bootstrap/constraints/*.sql

    Fallback (legacy, monolithic):
      scripts/sql/bootstrap/create_constraints.sql

    Destination:
      <final>/sql/schema/03_create_constraints.sql

    Important: This function ONLY writes the composed single file.
    It does NOT copy the constraints parts folder into the output.
    """
    repo_root = _find_repo_root(Path(__file__).resolve())
    modular_dir = repo_root / "scripts" / "sql" / "bootstrap" / "constraints"
    legacy_file = repo_root / "scripts" / "sql" / "bootstrap" / "create_constraints.sql"

    out_path = sql_root / "schema" / "03_create_constraints.sql"
    mode = _sales_mode(sales_cfg)

    if modular_dir.exists() and modular_dir.is_dir():
        # Always include dimensions constraints.
        parts: list[Path] = [modular_dir / "00_dimensions.sql"]

        if mode in {"sales", "both"}:
            parts.append(modular_dir / "10_sales.sql")

        if mode in {"sales_order", "both"}:
            parts.extend(
                [
                    modular_dir / "20_sales_order_header.sql",
                    modular_dir / "21_sales_order_detail.sql",
                    modular_dir / "22_sales_order_relations.sql",
                ]
            )

        existing = [p for p in parts if p.exists()]
        missing = [p.name for p in parts if not p.exists()]

        if not existing:
            skip(f"No modular constraint parts found in: {modular_dir}; skipping constraints.")
            return

        if missing:
            skip(f"Missing constraint parts: {', '.join(missing)} (mode={mode}); composing partial constraints.")

        chunks: list[str] = []
        chunks.append("-- Auto-generated by packaging: composed constraints\n")
        chunks.append(f"-- mode: {mode}\n")

        for p in existing:
            chunks.append(
                "\n\n-- ============================================================\n"
                f"-- {p.name}\n"
                "-- ============================================================\n"
            )
            chunks.append(_read_text(p).rstrip())

        _write_text(out_path, "\n".join(chunks).rstrip() + "\n")
        done(f"Composed constraints: {out_path.relative_to(sql_root)}")
        return

    # Fallback to legacy constraints file
    if legacy_file.exists():
        _write_text(out_path, _read_text(legacy_file))
        done(f"Copied legacy constraints: {out_path.relative_to(sql_root)}")
        return

    skip(f"No constraints source found (modular or legacy). Skipping: {out_path}")


# ------------------------------------------------------------
# SQL generation from packaged CSVs
# ------------------------------------------------------------

def write_bulk_insert_scripts(*, dims_out: Path, facts_out: Path, sql_root: Path, sales_cfg: dict) -> None:
    load_root = sql_root / "load"
    sql_root.mkdir(parents=True, exist_ok=True)
    load_root.mkdir(parents=True, exist_ok=True)

    with stage("Generating BULK INSERT Scripts"):
        dims_csv = sorted(dims_out.glob("*.csv"))
        facts_csv = sorted(facts_out.rglob("*.csv"))

        if not dims_csv and not facts_csv:
            skip("No CSV files found — skipping BULK INSERT scripts.")
            return

        # dims
        generate_bulk_insert_script(
            csv_folder=str(dims_out),
            table_name=None,
            output_sql_file=str(load_root / "01_bulk_insert_dims.sql"),
            mode="csv",
        )

        # facts (single script, recursive scan; filtered by sales_output)
        out_sql = load_root / "02_bulk_insert_facts.sql"
        generate_bulk_insert_script(
            csv_folder=str(facts_out),
            table_name=None,
            output_sql_file=str(out_sql),
            mode="legacy",
            row_terminator="0x0a",
            recursive=True,
            allowed_tables=set(tables_from_sales_cfg(sales_cfg)),
        )


def write_create_table_scripts(*, dims_out: Path, facts_out: Path, sql_root: Path, cfg: dict) -> None:
    with stage("Generating CREATE TABLE Scripts"):
        # Many SQL generators assume facts are in ONE flat folder.
        # Create a temp flat folder via hardlinks (fallback to copy).
        tmp_flat = sql_root / "_tmp_facts_flat_for_sql"

        if tmp_flat.exists():
            shutil.rmtree(tmp_flat, ignore_errors=True)
        tmp_flat.mkdir(parents=True, exist_ok=True)

        try:
            for f in facts_out.rglob("*.csv"):
                dst = tmp_flat / f.name
                try:
                    os.link(f, dst)
                except Exception:
                    shutil.copy2(f, dst)

            generate_all_create_tables(
                dim_folder=dims_out,
                fact_folder=tmp_flat,
                output_folder=sql_root,
                cfg=cfg,
            )
        finally:
            shutil.rmtree(tmp_flat, ignore_errors=True)
