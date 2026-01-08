import time
from pathlib import Path

from src.utils.output_utils import create_final_output_folder
from src.tools.sql.generate_bulk_insert_sql import generate_bulk_insert_script
from src.tools.sql.generate_create_table_scripts import generate_all_create_tables
from src.utils.logging_utils import stage, info, skip, done
import shutil
from urllib.parse import unquote


def package_output(cfg, sales_cfg, parquet_dims: Path, fact_out: Path):
    """
    Handles:
    - Creating final packaged folder (dims + facts)
    - Copying Sales fact (Delta/Parquet/CSV)
    - Generating SQL Scripts (CSV)
    - Generating CREATE TABLE scripts
    - Cleaning stale output
    """

    # ---------------------------------------------------------
    # Create final folder with dims/ and facts/
    # ---------------------------------------------------------
    with stage("Creating Final Output Folder"):
        final_folder = create_final_output_folder(
            final_folder_root=Path(
                str(cfg["final_output_folder"]).replace("%20", " ")
            ).resolve(),
            parquet_dims=parquet_dims,
            fact_folder=fact_out,
            sales_cfg=sales_cfg,
            file_format=sales_cfg["file_format"],
            sales_rows_expected=sales_cfg["total_rows"],
            cfg=cfg
        )
        # ---------------------------------------------------------
        # HARD FIX: remove URL-encoded duplicate run folder (%20)
        # ---------------------------------------------------------
        parent = final_folder.parent
        real_name = final_folder.name

        for sibling in parent.iterdir():
            if not sibling.is_dir():
                continue

            if "%20" in sibling.name:
                decoded = unquote(sibling.name)

                # Same logical dataset, encoded name → delete it
                if decoded == real_name:
                    # info(f"Removing URL-encoded duplicate run folder: {sibling}")
                    shutil.rmtree(sibling)

        dims_out = final_folder / "dimensions"
        facts_out = final_folder / "facts"

        # ---------------------------------------------------------
        # Clean OLD packaged sales folder
        # ---------------------------------------------------------
        packaged_sales_folder = facts_out / "sales"
        if packaged_sales_folder.exists():
            # info("Cleaning packaged facts/sales folder to remove stale parquet files...")
            shutil.rmtree(packaged_sales_folder)

        # ---------------------------------------------------------
        # Determine source sales folder
        # ---------------------------------------------------------
        file_format = sales_cfg["file_format"].lower()

        if file_format == "deltaparquet":
            dst_sales = facts_out / "sales"
            dst_sales.mkdir(parents=True, exist_ok=True)
        else:
            dst_sales = facts_out   # parquet and csv output directly under facts/

        # ---------------------------------------------------------
        # SPECIAL CASE: PARQUET OUTPUT = SINGLE FILE
        # ---------------------------------------------------------
        if file_format == "parquet":
            src_file = fact_out / "parquet" / "sales.parquet"
            dst_file = facts_out / "sales.parquet"

            if not src_file.exists():
                raise RuntimeError(f"Expected parquet file not found: {src_file}")

            # Remove old file if exists
            if dst_file.exists():
                dst_file.unlink()

            shutil.copy2(src_file, dst_file)
            done("Sales fact copied (single parquet file).")
            return final_folder

        # ---------------------------------------------------------
        # Other formats: deltaparquet or csv (folder copy)
        # ---------------------------------------------------------
        if file_format == "deltaparquet":
            src_sales = fact_out / "sales"
        else:
            src_sales = fact_out / "csv"

        # ---------------------------------------------------------
        # CSV MODE — flat copy into /facts
        # ---------------------------------------------------------
        if file_format == "csv":
            info(f"Copying CSV sales fact from: {src_sales}")

            for child in src_sales.rglob("*.csv"):
                out_path = dst_sales / child.name  # flat structure
                shutil.copy2(child, out_path)

            done("Sales fact copied (CSV flat).")
            # return final_folder


        if not src_sales.exists():
            raise RuntimeError(f"Expected sales output folder not found: {src_sales}")

        info(f"Copying sales fact from: {src_sales}")

        # Full Delta table snapshot copy
        for item in src_sales.iterdir():

            # Skip tmp / transient folders
            if item.name == "_tmp_parts":
                continue

            target = dst_sales / item.name

            if item.is_dir():
                shutil.copytree(item, target, dirs_exist_ok=True)
            else:
                shutil.copy2(item, target)


                done("Sales fact copied.")

    # ============================================================
    # SQL SCRIPT GENERATION — ONLY for CSV
    # ============================================================
    if file_format == "csv":

        with stage("Generating BULK INSERT Scripts"):
            dims_folder = dims_out
            facts_folder = facts_out

            dims_csv = sorted(p for p in dims_folder.glob("*.csv"))
            facts_csv = sorted(p for p in facts_folder.glob("*.csv"))

            if not dims_csv and not facts_csv:
                skip("No CSV files found — skipping BULK INSERT scripts.")
            else:
                generate_bulk_insert_script(
                    csv_folder=str(dims_folder),
                    table_name=None,
                    output_sql_file=str(final_folder / "bulk_insert_dims.sql"),
                    mode="csv",
                )
                generate_bulk_insert_script(
                    csv_folder=str(facts_folder),
                    table_name="Sales",
                    output_sql_file=str(final_folder / "bulk_insert_facts.sql"),
                    mode="legacy",
                    row_terminator="0x0a",
                )

        with stage("Generating CREATE TABLE Scripts"):
            generate_all_create_tables(
                dim_folder=dims_out,
                fact_folder=facts_out,
                output_folder=final_folder,
                cfg=cfg,
                skip_order_cols=sales_cfg.get("skip_order_cols", False),
            )

    else:
        info("Skipping SQL script generation for non-CSV format.")

    return final_folder
