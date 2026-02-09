import os
import shutil
from pathlib import Path

from src.tools.sql.generate_bulk_insert_sql import generate_bulk_insert_script
from src.tools.sql.generate_create_table_scripts import generate_all_create_tables
from src.utils.logging_utils import stage, info, skip

from .paths import tables_from_sales_cfg


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
        tmp_flat = sql_root.parent / "_facts_flat_for_sql"
        if tmp_flat.exists():
            shutil.rmtree(tmp_flat, ignore_errors=True)
        tmp_flat.mkdir(parents=True, exist_ok=True)

        for f in facts_out.rglob("*.csv"):
            if dims_out in f.parents:
                continue
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

        shutil.rmtree(tmp_flat, ignore_errors=True)
