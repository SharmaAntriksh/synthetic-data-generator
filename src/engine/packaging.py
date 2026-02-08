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


def _get_first_existing_path(cfg: dict, keys: list[str]) -> Optional[Path]:
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


_TABLE_DIR_MAP = {
    TABLE_SALES: "sales",
    TABLE_SALES_ORDER_DETAIL: "sales_order_detail",
    TABLE_SALES_ORDER_HEADER: "sales_order_header",
}


def _to_snake(s: str) -> str:
    # CamelCase -> snake_case fallback (only used if table not in map)
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", s)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


def _tables_from_sales_cfg(sales_cfg: dict) -> list[str]:
    sales_output = str(sales_cfg.get("sales_output", "sales")).lower()
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


def _table_out_dir(base: Path, table: str) -> Path:
    # OutputPaths keeps Sales in root, other tables under subfolder == table name
    # (e.g., .../csv/SalesOrderHeader/*.csv)
    return base if table == TABLE_SALES else (base / table)


def _resolve_merged_parquet(fact_out: Path, sales_cfg: dict, table: str) -> Optional[Path]:
    # fact_out is the root (e.g., data/fact_out)
    parquet_root = fact_out / "parquet"

    if table == TABLE_SALES:
        merged_name = str(sales_cfg.get("merged_file", "sales.parquet"))
        candidates = [
            parquet_root / merged_name,
            parquet_root / table / merged_name,   # fallback
            fact_out / merged_name,               # legacy fallback
            fact_out / table / merged_name,       # fallback
        ]
    else:
        merged_name = f"{table}.parquet"
        candidates = [
            parquet_root / table / merged_name,   # expected (per OutputPaths)
            parquet_root / merged_name,           # fallback
            fact_out / table / merged_name,       # fallback
            fact_out / merged_name,               # fallback
        ]

    for p in candidates:
        if p.exists():
            return p

    return None


def _copy_delta_table_dir(src: Path, dst: Path, skip_dirnames: set[str]) -> None:
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


def package_output(cfg, sales_cfg, parquet_dims: Path, fact_out: Path):
    """
    Handles:
    - Creating final packaged folder (dims + facts)
    - Copying fact outputs (Sales / SalesOrderHeader / SalesOrderDetail)
    - Generating SQL scripts (CSV only)
    - Cleaning stale output
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

    # Create final output folder (dims packaged here)
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

        # HARD FIX: remove URL-encoded duplicate run folder (%20)
        parent = final_folder.parent
        real_name = final_folder.name
        for sibling in parent.iterdir():
            if sibling.is_dir() and "%20" in sibling.name and unquote(sibling.name) == real_name:
                shutil.rmtree(sibling)

        dims_out = final_folder / "dimensions"
        facts_out = final_folder / "facts"

        tables = _tables_from_sales_cfg(sales_cfg)

        # Clean old packaged fact folders (idempotent)
        for t in tables:
            variants = {
                _table_dir_name(t),  # canonical snake name
                t,                   # CamelCase
                t.lower(),           # lower CamelCase
                _to_snake(t),         # snake fallback
            }
            for v in variants:
                out_dir = facts_out / v
                if out_dir.exists():
                    shutil.rmtree(out_dir, ignore_errors=True)

        # ============================================================
        # PARQUET MODE — copy merged parquet(s), then exit
        # ============================================================
        if file_format == "parquet":
            copied = 0
            for t in tables:
                merged = _resolve_merged_parquet(fact_out, sales_cfg, t)
                if merged is None:
                    info(f"No merged parquet found for {t}; skipping.")
                    continue

                facts_out.mkdir(parents=True, exist_ok=True)

                # Normalize merged filename if needed (optional but nice)
                # Prefer "<TableName>.parquet" for non-sales tables
                if t != TABLE_SALES:
                    dst_name = f"{t}.parquet"
                else:
                    dst_name = merged.name  # keep existing sales filename (e.g., sales.parquet)

                dst_file = facts_out / dst_name
                if dst_file.exists():
                    dst_file.unlink()

                shutil.copy2(merged, dst_file)
                copied += 1

            done(f"Copied {copied} parquet fact file(s).")

            # Parquet never generates SQL scripts
            return final_folder

        # ============================================================
        # DELTA MODE — copy delta tables (folder per table)
        # ============================================================
        if file_format == "deltaparquet":
            import re
            from src.facts.sales.output_paths import build_output_paths_from_sales_cfg

            out_paths = build_output_paths_from_sales_cfg(sales_cfg)

            def _looks_like_delta_table_dir(p: Path) -> bool:
                return p.is_dir() and (p / "_delta_log").exists()

            def _table_name_variants(table: str) -> list[str]:
                # SalesOrderDetail -> sales_order_detail, salesorderdetail, etc.
                snake = re.sub(r"(?<!^)(?=[A-Z])", "_", table).lower()
                return [
                    table,
                    table.lower(),
                    snake,
                    snake.replace("_", ""),
                    table.replace("_", ""),
                    table.replace(" ", ""),
                ]

            delta_roots: list[Path] = []

            def _add_root(p: Optional[Path]) -> None:
                if not p:
                    return
                try:
                    rp = p.expanduser().resolve()
                except Exception:
                    return
                if rp.exists() and rp.is_dir() and rp not in delta_roots:
                    delta_roots.append(rp)

            # 1) explicit config root (if any)
            cfg_delta = sales_cfg.get("delta_output_folder")
            if cfg_delta:
                _add_root(Path(str(cfg_delta)))

            # 2) common new layout: <fact_out>/delta/<TableName or snake>
            _add_root(fact_out / "delta")

            # 3) legacy-ish / alternate layouts (resilience)
            _add_root(fact_out / "deltaparquet")
            _add_root(fact_out / "sales")  # some older flows

            # 4) add parents of the “expected” per-table path from OutputPaths
            expected_by_table: dict[str, Path] = {}
            for t in tables:
                exp = Path(out_paths.delta_table_dir(t)).resolve()
                expected_by_table[t] = exp
                _add_root(exp.parent)

            def _find_delta_table_dir(table: str) -> Optional[Path]:
                # First: accept the exact expected path if it’s already a delta table
                exp = expected_by_table.get(table)
                if exp and _looks_like_delta_table_dir(exp):
                    return exp

                # Then: try root/<variant> for each configured root
                for root in delta_roots:
                    # root itself could be a delta table dir (edge case)
                    if _looks_like_delta_table_dir(root) and root.name.lower() in {table.lower(), _table_dir_name(table)}:
                        return root

                    for name in _table_name_variants(table) + [_table_dir_name(table)]:
                        cand = root / name
                        if _looks_like_delta_table_dir(cand):
                            return cand

                return None

            copied = 0
            searched = [str(p) for p in delta_roots]

            for t in tables:
                src_dir = _find_delta_table_dir(t)
                if src_dir is None:
                    info(f"No delta output found for {t} at {expected_by_table[t]}; skipping.")
                    continue

                dst_dir = facts_out / _table_dir_name(t)
                _copy_delta_table_dir(src_dir, dst_dir, skip_dirnames={"_tmp_parts"})
                copied += 1

            if copied == 0:
                raise RuntimeError(
                    f"No delta fact outputs found for tables={tables}. "
                    f"Checked delta roots: {searched}"
                )

            done(f"Copied {copied} delta table snapshot(s).")
            return final_folder

        # ============================================================
        # CSV MODE — copy chunks into per-table folders
        # ============================================================
        if file_format != "csv":
            raise ValueError(f"Unsupported file_format in packaging: {file_format!r}")

        csv_root = fact_out / "csv"
        if not csv_root.exists():
            raise RuntimeError(f"Expected CSV output folder not found: {csv_root}")

        copied_files = 0
        for t in tables:
            src_dir = _table_out_dir(csv_root, t)
            if not src_dir.exists():
                info(f"No CSV folder found for {t} at {src_dir}; skipping.")
                continue

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

                # facts: one script per fact table folder
                any_fact = False
                for t in _tables_from_sales_cfg(sales_cfg):
                    folder = facts_out / _table_dir_name(t)
                    if not folder.exists():
                        continue
                    if not list(folder.glob("*.csv")):
                        continue

                    any_fact = True
                    out_sql = load_root / f"02_bulk_insert_{_table_dir_name(t)}.sql"
                    generate_bulk_insert_script(
                        csv_folder=str(folder),
                        table_name=str(t),
                        output_sql_file=str(out_sql),
                        mode="legacy",
                        row_terminator="0x0a",
                    )

                if not any_fact:
                    skip("No fact CSV files found — skipping fact BULK INSERT scripts.")

        with stage("Generating CREATE TABLE Scripts"):
            # Many SQL generators assume facts are in ONE flat folder.
            # Create a temp flat folder via hardlinks (fallback to copy) so we don't change the generator yet.
            tmp_flat = final_folder / "_facts_flat_for_sql"
            if tmp_flat.exists():
                shutil.rmtree(tmp_flat, ignore_errors=True)
            tmp_flat.mkdir(parents=True, exist_ok=True)

            for f in facts_out.rglob("*.csv"):
                # only facts; dims are elsewhere, but guard anyway
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
                skip_order_cols=sales_cfg.get("skip_order_cols", False),
            )

            shutil.rmtree(tmp_flat, ignore_errors=True)

    else:
        info("Skipping SQL script generation for non-CSV format.")

    return final_folder
