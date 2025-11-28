import time
from contextlib import contextmanager
from pathlib import Path

from src.utils.output_utils import create_final_output_folder, clear_folder
from src.sql.generate_bulk_insert_sql import generate_bulk_insert_script
from src.sql.generate_create_table_scripts import generate_all_create_tables


@contextmanager
def stage(label: str):
    print(f"\n=== {label}... ===")
    t = time.time()
    yield
    print(f"\n✔ {label} completed in {time.time() - t:.2f} seconds")


def package_output(cfg, sales_cfg, parquet_dims: Path, fact_out: Path):
    """
    Handles:
    - Creating final packaged folder (dims + facts)
    - Generating SQL scripts (CSV mode only)
    - Creating CREATE TABLE scripts
    - Cleaning intermediate fact folder
    """

    # ---------------------------------------------------------
    # Create final folder with dims/ and facts/
    # ---------------------------------------------------------
    with stage("Creating Final Output Folder"):
        final_folder = create_final_output_folder(
            parquet_dims=parquet_dims,
            fact_folder=fact_out,
            file_format=sales_cfg["file_format"],
        )

        dims_out = final_folder / "dims"
        facts_out = final_folder / "facts"

    # ---------------------------------------------------------
    # SQL SCRIPTS (CSV only)
    # ---------------------------------------------------------
    if sales_cfg.get("file_format") == "csv":
        with stage("Generating BULK INSERT Scripts"):
            dims_folder = dims_out
            facts_folder = facts_out

            dims_csv = sorted(p for p in dims_folder.glob("*.csv"))
            facts_csv = sorted(p for p in facts_folder.glob("*.csv"))

            if not dims_csv and not facts_csv:
                print("⚠ No CSV files found — skipping BULK INSERT scripts.")
            else:
                # Bulk insert for dimensions (infer table names automatically)
                generate_bulk_insert_script(
                    csv_folder=str(dims_folder),
                    table_name=None,
                    output_sql_file=str(final_folder / "bulk_insert_dims.sql"),
                )

                # Bulk insert for facts (explicit Sales table name)
                generate_bulk_insert_script(
                    csv_folder=str(facts_folder),
                    table_name="Sales",
                    output_sql_file=str(final_folder / "bulk_insert_facts.sql"),
                )

                print("\n✔ Bulk Insert scripts generated successfully.")

        # ---------------------------------------------------------
        # CREATE TABLE SCRIPTS — ALWAYS
        # ---------------------------------------------------------
        with stage("Generating CREATE TABLE Scripts"):
            generate_all_create_tables(
                dim_folder=dims_out,
                fact_folder=facts_out,
                output_folder=final_folder,
                skip_order_cols=sales_cfg.get("skip_order_cols", False),
            )

    # ---------------------------------------------------------
    # Cleanup: remove intermediate fact_out folder
    # ---------------------------------------------------------
    with stage("Cleaning intermediate fact_out folder!"):
        clear_folder(fact_out)

    return final_folder
