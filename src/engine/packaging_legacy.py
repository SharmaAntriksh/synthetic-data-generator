import os
import re
import shutil
from pathlib import Path
from urllib.parse import unquote
from typing import Optional

from src.utils.output_utils import create_final_output_folder
from src.tools.sql.generate_bulk_insert_sql import generate_bulk_insert_script
from src.tools.sql.generate_create_table_scripts import generate_all_create_tables
from src.utils.logging_utils import stage, info, skip, done

from src.facts.sales.output_paths import (
    TABLE_SALES,
    TABLE_SALES_ORDER_DETAIL,
    TABLE_SALES_ORDER_HEADER,
)

# Canonical folder names for scratch + packaged outputs
_TABLE_DIR_MAP = {
    TABLE_SALES: "sales",
    TABLE_SALES_ORDER_DETAIL: "sales_order_detail",
    TABLE_SALES_ORDER_HEADER: "sales_order_header",
}


def _get_first_existing_path(cfg: dict, keys: list[str]) -> Optional[Path]:
    """
    Resolve config/model yaml paths from cfg keys.
    Checks:
      1) absolute path
      2) cwd-relative
      3) repo-root-relative
    Returns first that exists, else None.
    """
    project_root = Path(__file__).resolve().parents[2]  # repo root

    def _resolve_existing(v: str) -> Optional[Path]:
        p = Path(str(v)).expanduser()
        if p.is_absolute() and p.exists():
            return p

        cwd_candidate = (Path.cwd() / p).resolve()
        if cwd_candidate.exists():
            return cwd_candidate

        repo_candidate = (project_root / p).resolve()
        if repo_candidate.exists():
            return repo_candidate

        return None

    for k in keys:
        v = cfg.get(k)
        if not v:
            continue
        resolved = _resolve_existing(v)
        if resolved:
            return resolved
    return None


def _to_snake(s: str) -> str:
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", s)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


def _tables_from_sales_cfg(sales_cfg: dict) -> list[str]:
    sales_output = str(sales_cfg.get("sales_output", "sales")).lower().strip()
    if sales_output not in {"sales", "sales_order", "both"}:
        raise ValueError(f"Invalid sales_output: {sales_output}")

    tables: list[str] = []
    if sales_output in {"sales", "both"}:
        tables.append(TABLE_SALES)
    if sales_output in {"sales_order", "both"}:
        tables += [TABLE_SALES_ORDER_DETAIL, TABLE_SALES_ORDER_HEADER]
    return tables


def _table_dir_name(table: str) -> str:
    return _TABLE_DIR_MAP.get(table, _to_snake(table))


def _resolve_merged_parquet(fact_out: Path, sales_cfg: dict, table: str) -> Optional[Path]:
    """
    Locate merged parquet output in scratch fact_out.

    We intentionally search multiple layouts because scratch folders have evolved:
      - <fact_out>/parquet/<snake_table>/<TableName>.parquet   (current order tables)
      - <fact_out>/<snake_table>/<TableName>.parquet          (older)
      - Sales merged file may sit at <fact_out>/parquet/sales.parquet or <fact_out>/sales.parquet
    """
    parquet_root = fact_out / "parquet"
    tdir = _table_dir_name(table)

    if table == TABLE_SALES:
        merged_name = str(sales_cfg.get("merged_file") or "sales.parquet")
        candidates = [
            parquet_root / merged_name,
            parquet_root / tdir / merged_name,
            fact_out / merged_name,
            fact_out / tdir / merged_name,
        ]
    else:
        merged_name = f"{_table_dir_name(table)}.parquet"  # snake_case file
        candidates = [
            parquet_root / tdir / merged_name,
            parquet_root / table / merged_name,     # fallback if CamelCase dir
            parquet_root / merged_name,             # fallback if written to root
            fact_out / tdir / merged_name,
            fact_out / table / merged_name,
            fact_out / merged_name,
            # legacy fallback: older runs may still have CamelCase filename
            parquet_root / tdir / f"{table}.parquet",
            fact_out / tdir / f"{table}.parquet",
        ]

    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def _looks_like_delta_table_dir(p: Path) -> bool:
    """
    Correct signature for a Delta table root:
      - it has a _delta_log directory

    Do NOT require parquet files at the root because partitioned delta tables
    store data under subfolders (e.g., Year=.../Month=...).
    """
    try:
        return (p / "_delta_log").is_dir()
    except Exception:
        return False


def _delta_name_variants(table: str) -> list[str]:
    """
    Plausible directory names for delta tables.
    Includes both snake_case and CamelCase variants.
    """
    snake = _to_snake(table)
    canonical = _table_dir_name(table)

    variants = [
        table,
        table.lower(),
        snake,
        snake.replace("_", ""),
        canonical,
        canonical.replace("_", ""),
        table.replace("_", ""),
        table.replace(" ", ""),
    ]

    # de-dup while preserving order
    out: list[str] = []
    seen = set()
    for v in variants:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _find_delta_table_dir(fact_out: Path, sales_cfg: dict, table: str) -> Optional[Path]:
    """
    Find the Delta table directory for `table` across known layouts.

    Your current run writes delta tables under:
      data/fact_out/sales/
        _delta_log                      (Sales)
        sales_order_detail/_delta_log   (SalesOrderDetail)
        sales_order_header/_delta_log   (SalesOrderHeader)

    But it may also be under:
      data/fact_out/delta/<table_subdir>/_delta_log
    depending on config / legacy behavior.
    """

    def _to_snake(name: str) -> str:
        s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
        return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()

    snake = _to_snake(table)

    # Roots to probe (keep order; include legacy scratch roots)
    roots: list[Path] = []

    cfg_root = sales_cfg.get("delta_output_folder")
    if cfg_root:
        roots.append(Path(cfg_root))

    roots += [
        fact_out / "delta",
        fact_out / "deltaparquet",
        fact_out / "sales",  # <-- IMPORTANT: matches your observed layout
    ]

    # De-dup roots (preserve order)
    seen = set()
    uniq_roots: list[Path] = []
    for r in roots:
        key = str(r)
        if key not in seen:
            seen.add(key)
            uniq_roots.append(r)

    # Candidate dirs per root.
    # IMPORTANT: do NOT treat the root itself as the match for non-Sales tables,
    # because fact_out/sales is the Sales delta table root.
    candidates: list[Path] = []
    for root in uniq_roots:
        # root itself only if it *looks like* it was intended for this table
        if root.name in {snake, table, table.lower()}:
            candidates.append(root)

        candidates += [
            root / snake,          # sales_order_detail
            root / table,          # SalesOrderDetail
            root / table.lower(),  # salesorderdetail
        ]

    # De-dup candidates
    seen_c = set()
    uniq_candidates: list[Path] = []
    for c in candidates:
        key = str(c)
        if key not in seen_c:
            seen_c.add(key)
            uniq_candidates.append(c)

    for cand in uniq_candidates:
        if _looks_like_delta_table_dir(cand):
            return cand

    return None


def _copy_delta_table_dir(src: Path, dst: Path, skip_dirnames: set[str]) -> None:
    """
    Copy a delta table directory snapshot (keep _delta_log and data files),
    but skip internal scratch dirs like _tmp_parts.
    """
    if dst.exists():
        shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        if item.name in skip_dirnames:
            continue
        target = dst / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def _resolve_csv_table_dir(fact_out: Path, table: str) -> Optional[Path]:
    """
    Locate CSV chunk folder for a table.
    We accept both:
      - <fact_out>/csv/<snake_table>/
      - <fact_out>/csv/<TableName>/
      - <fact_out>/csv/ (Sales may be rooted)
    """
    csv_root = fact_out / "csv"
    if not csv_root.exists():
        return None

    tdir = _table_dir_name(table)
    candidates: list[Path] = []

    if table == TABLE_SALES:
        candidates += [csv_root / tdir, csv_root / table, csv_root]
    else:
        candidates += [csv_root / tdir, csv_root / table, csv_root / table.lower()]

    for p in candidates:
        if p.exists() and p.is_dir():
            return p
    return None


def package_output(cfg, sales_cfg, parquet_dims: Path, fact_out: Path):
    """
    Handles:
    - Creating final packaged folder (dims + config copied by output_utils)
    - Copying fact outputs (Sales / SalesOrderHeader / SalesOrderDetail)
    - Generating SQL scripts (CSV only)
    """
    file_format = str(sales_cfg["file_format"]).lower()
    is_csv = file_format == "csv"

    # Normalize final output root ONCE
    final_root = Path(unquote(str(cfg["final_output_folder"]))).resolve()

    # Resolve config/model yaml paths (optional)
    config_yaml_path = _get_first_existing_path(
        cfg,
        keys=["config_yaml_path", "config_path", "config_file", "config_yaml", "config"],
    )
    model_yaml_path = _get_first_existing_path(
        cfg,
        keys=["model_yaml_path", "model_path", "model_file", "model_yaml", "model"],
    )

    with stage("Creating Final Output Folder"):
        final_folder = create_final_output_folder(
            final_folder_root=final_root,
            parquet_dims=parquet_dims,
            fact_folder=fact_out,
            sales_cfg=sales_cfg,
            file_format=file_format,
            sales_rows_expected=sales_cfg["total_rows"],
            cfg=cfg,
            config_yaml_path=config_yaml_path,
            model_yaml_path=model_yaml_path,
            package_facts=False,
        )

        # Remove URL-encoded duplicate run folder (%20)
        parent = final_folder.parent
        real_name = final_folder.name
        for sibling in parent.iterdir():
            if sibling.is_dir() and "%20" in sibling.name and unquote(sibling.name) == real_name:
                shutil.rmtree(sibling, ignore_errors=True)

        dims_out = final_folder / "dimensions"
        facts_out = final_folder / "facts"
        facts_out.mkdir(parents=True, exist_ok=True)

        tables = _tables_from_sales_cfg(sales_cfg)

        # ============================================================
        # PARQUET MODE — copy merged parquet(s), then exit
        # ============================================================
        if file_format == "parquet":
            copied = 0
            missing: list[str] = []

            for t in tables:
                src = _resolve_merged_parquet(fact_out, sales_cfg, t)
                if src is None:
                    missing.append(t)
                    info(
                        f"No merged parquet found for {t}. "
                        f"Checked: {fact_out / 'parquet'} and {fact_out}."
                    )
                    continue

                # Keep Sales filename untouched; normalize others to <snake_table>.parquet
                dst_name = src.name if t == TABLE_SALES else f"{_table_dir_name(t)}.parquet"
                dst = facts_out / dst_name

                if dst.exists():
                    dst.unlink()

                shutil.copy2(src, dst)
                copied += 1

            if missing:
                raise RuntimeError(
                    "No merged parquet found for table(s): "
                    + ", ".join(missing)
                    + f". Scratch roots checked: {fact_out} and {fact_out / 'parquet'}."
                )

            done(f"Copied {copied} parquet fact file(s).")
            return final_folder

        # ============================================================
        # DELTA MODE — copy delta tables (folder per table)
        # ============================================================
        if file_format == "deltaparquet":
            copied = 0
            missing: list[str] = []

            for t in tables:
                src_dir = _find_delta_table_dir(fact_out, sales_cfg, t)
                if src_dir is None:
                    missing.append(t)
                    info(
                        f"No delta output found for {t}. "
                        f"Checked delta_output_folder plus: {fact_out / 'delta'}, {fact_out / 'deltaparquet'}, {fact_out / 'sales'}."
                    )
                    continue

                dst_dir = facts_out / _table_dir_name(t)
                skip_dirs = {"_tmp_parts"}

                # If we're packaging multiple tables (sales_output="both"),
                # don't allow the Sales delta root to drag nested tables with it.
                if t == TABLE_SALES and len(tables) > 1:
                    skip_dirs |= {_table_dir_name(x) for x in tables if x != TABLE_SALES}

                _copy_delta_table_dir(src_dir, dst_dir, skip_dirnames=skip_dirs)

                copied += 1

            if missing:
                raise RuntimeError(
                    "No delta fact outputs found for table(s): "
                    + ", ".join(missing)
                    + ". Checked delta_output_folder plus these scratch roots: "
                    + ", ".join(
                        [
                            str(p)
                            for p in [
                                fact_out / "delta",
                                fact_out / "deltaparquet",
                                fact_out / "sales",
                            ]
                        ]
                    )
                )

            done(f"Copied {copied} delta table snapshot(s).")
            return final_folder

        # ============================================================
        # CSV MODE — copy chunks into per-table folders
        # ============================================================
        if file_format != "csv":
            raise ValueError(f"Unsupported file_format in packaging: {file_format!r}")

        copied_files = 0
        missing_dirs: list[str] = []

        for t in tables:
            src_dir = _resolve_csv_table_dir(fact_out, t)
            if src_dir is None:
                missing_dirs.append(t)
                info(f"No CSV folder found for {t}; expected under {fact_out / 'csv'}.")
                continue

            multi_table = len(tables) > 1
            if t == TABLE_SALES:
                dst_dir = (facts_out / _table_dir_name(t)) if multi_table else facts_out
            else:
                dst_dir = facts_out / _table_dir_name(t)

            dst_dir.mkdir(parents=True, exist_ok=True)
            csv_files = sorted(src_dir.glob("*.csv"))
            info(f"Copying {len(csv_files)} CSV file(s) for {t} from: {src_dir}")

            for f in csv_files:
                target = dst_dir / f.name
                if target.exists():
                    raise RuntimeError(f"Duplicate CSV filename during packaging: {target}")
                shutil.copy2(f, target)
                copied_files += 1

        if missing_dirs:
            raise RuntimeError(
                "No CSV fact outputs found for table(s): "
                + ", ".join(missing_dirs)
                + f". Expected under: {fact_out / 'csv'}"
            )

        done(f"Copied {copied_files} CSV fact file(s).")

    # ============================================================
    # SQL SCRIPT GENERATION — CSV ONLY
    # ============================================================
    if is_csv:
        sql_root = final_folder / "sql"
        load_root = sql_root / "load"
        sql_root.mkdir(parents=True, exist_ok=True)
        load_root.mkdir(parents=True, exist_ok=True)

        with stage("Generating BULK INSERT Scripts"):
            dims_csv = sorted(dims_out.glob("*.csv"))
            facts_csv = sorted(facts_out.rglob("*.csv"))

            if not dims_csv and not facts_csv:
                skip("No CSV files found — skipping BULK INSERT scripts.")
            else:
                # dims
                generate_bulk_insert_script(
                    csv_folder=str(dims_out),
                    table_name=None,
                    output_sql_file=str(load_root / "01_bulk_insert_dims.sql"),
                    mode="csv",
                )

                # facts: single script for all enabled fact tables (conditional via sales_output)
                out_sql = load_root / "02_bulk_insert_facts.sql"
                generate_bulk_insert_script(
                    csv_folder=str(facts_out),
                    table_name=None,
                    output_sql_file=str(out_sql),
                    mode="legacy",
                    row_terminator="0x0a",
                    recursive=True,
                    allowed_tables=set(_tables_from_sales_cfg(sales_cfg)),
                )

        with stage("Generating CREATE TABLE Scripts"):
            # Many SQL generators assume facts are in ONE flat folder.
            # Create a temp flat folder via hardlinks (fallback to copy).
            tmp_flat = final_folder / "_facts_flat_for_sql"
            if tmp_flat.exists():
                shutil.rmtree(tmp_flat, ignore_errors=True)
            tmp_flat.mkdir(parents=True, exist_ok=True)

            for f in facts_out.rglob("*.csv"):
                # guard: dims live elsewhere
                if dims_out in f.parents:
                    continue
                dst = tmp_flat / f.name
                try:
                    os.link(f, dst)  # cheap if supported
                except Exception:
                    shutil.copy2(f, dst)

            generate_all_create_tables(
                dim_folder=dims_out,
                fact_folder=tmp_flat,
                output_folder=sql_root,
                cfg=cfg,
            )

            shutil.rmtree(tmp_flat, ignore_errors=True)

    else:
        info("Skipping SQL script generation for non-CSV format.")

    return final_folder
