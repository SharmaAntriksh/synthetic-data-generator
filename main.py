import json
import time
from pathlib import Path
from contextlib import contextmanager

from src.customers import generate_synthetic_customers
from src.promotions import generate_promotions_catalog
from src.stores import generate_store_table
from src.dates import generate_date_table
from src.sales import generate_sales_fact
from src.output_utils import clear_folder, create_final_output_folder
from src.generate_bulk_insert_sql import generate_bulk_insert_script
from src.generate_create_table_scripts import generate_all_create_tables


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

@contextmanager
def stage(label: str):
    print(f"\n=== {label}... ===")
    t = time.time()
    yield
    print(f"✔ {label} completed in {time.time() - t:.2f} seconds")


def validate_config(cfg, section, required_keys):
    for k in required_keys:
        if k not in cfg:
            raise KeyError(f"Missing '{k}' in config section '{section}'")


def normalize_sales_config(cfg):
    """
    Removes parquet-only config keys when generating CSV.
    Ensures sales generator receives a clean dictionary.
    """
    if cfg.get("file_format") == "csv":
        parquet_only = ("row_group_size", "compression", "merge_parquet", "merged_file")
        cfg = {k: v for k, v in cfg.items() if k not in parquet_only}
    return cfg


# ---------------------------------------------------------
# Dimension Generation
# ---------------------------------------------------------

def generate_dimensions(cfg, parquet_dims: Path):
    cust_cfg = cfg["customers"]
    promo_cfg = cfg["promotions"]
    store_cfg = cfg["stores"]
    date_cfg = cfg["dates"]

    with stage("Generating Customers and Geography"):
        customers, geo = generate_synthetic_customers(
            total_customers=cust_cfg["total_customers"],
            total_geos=cust_cfg["total_geos"],
            pct_india=cust_cfg["pct_india"],
            pct_us=cust_cfg["pct_us"],
            pct_eu=cust_cfg["pct_eu"],
            pct_org=cust_cfg["pct_org"],
            seed=cust_cfg["seed"],
            names_folder=cust_cfg["names_folder"],
            save_customer_csv=False,
            save_geography_csv=False
        )
        customers.to_parquet(parquet_dims / "customers.parquet", index=False)
        geo.to_parquet(parquet_dims / "geography.parquet", index=False)

    with stage("Generating Promotions"):
        promotions = generate_promotions_catalog(
            years=range(promo_cfg["years"][0], promo_cfg["years"][1] + 1),
            num_seasonal=promo_cfg["num_seasonal"],
            num_clearance=promo_cfg["num_clearance"],
            num_limited=promo_cfg["num_limited"],
            seed=promo_cfg["seed"]
        )
        promotions.to_parquet(parquet_dims / "promotions.parquet", index=False)

    with stage("Generating Stores"):
        stores = generate_store_table(
            geography_parquet_path=store_cfg["geography_path"],
            num_stores=store_cfg["num_stores"],
            opening_start=store_cfg["opening_start"],
            opening_end=store_cfg["opening_end"],
            closing_end=store_cfg["closing_end"],
            seed=store_cfg["seed"]
        )
        stores.to_parquet(parquet_dims / "stores.parquet", index=False)

    with stage("Generating Dates"):
        dates = generate_date_table(
            date_cfg["start_date"],
            date_cfg["end_date"],
            date_cfg["fiscal_month_offset"]
        )
        dates.to_parquet(parquet_dims / "dates.parquet", index=False)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    total_start = time.time()

    # ----------------------------------
    # Load config
    # ----------------------------------
    with open("config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    sales_cfg = normalize_sales_config(cfg["sales"])

    parquet_dims = Path(sales_cfg["parquet_folder"])
    fact_out = Path(sales_cfg["out_folder"])

    validate_config(sales_cfg, "sales", ["total_rows", "chunk_size", "file_format"])

    # ----------------------------------
    # Dimension tables
    # ----------------------------------
    generate_dimensions(cfg, parquet_dims)

    # ----------------------------------
    # Sales Fact Generation
    # ----------------------------------
    with stage("Generating Sales"):
        clear_folder(fact_out)

        # Safety: CSV mode should not merge or delete chunks
        if sales_cfg.get("file_format") == "csv":
            sales_cfg["merge_parquet"] = False
            sales_cfg["delete_chunks"] = False

        generate_sales_fact(
            parquet_folder=sales_cfg["parquet_folder"],
            out_folder=sales_cfg["out_folder"],
            total_rows=sales_cfg["total_rows"],
            chunk_size=sales_cfg["chunk_size"],
            start_date=sales_cfg["start_date"],
            end_date=sales_cfg["end_date"],
            delete_chunks=sales_cfg["delete_chunks"],
            heavy_pct=sales_cfg["heavy_pct"],
            heavy_mult=sales_cfg["heavy_mult"],
            seed=sales_cfg["seed"],
            **{k: sales_cfg.get(k) for k in (
                "file_format", "row_group_size", "compression",
                "merge_parquet", "merged_file"
            )}
        )

    # ----------------------------------
    # Final Output Packaging
    # ----------------------------------
    with stage("Creating Final Output Folder"):
        final_folder = create_final_output_folder(
            parquet_dims=parquet_dims,
            fact_folder=fact_out,
            file_format=sales_cfg["file_format"]
        )

    # ----------------------------------
    # SQL Scripts (CSV only)
    # ----------------------------------
    if sales_cfg.get("file_format") == "csv":
        with stage("Generating BULK INSERT Scripts"):
            dims_folder = final_folder / "dims"
            facts_folder = final_folder / "facts"

            generate_bulk_insert_script(
                csv_folder=facts_folder,
                table_name="Sales",
                output_sql_file=final_folder / "bulk_insert_sales.sql"
            )

            generate_bulk_insert_script(
                csv_folder=dims_folder,
                table_name=None,
                output_sql_file=final_folder / "bulk_insert_dimensions.sql"
            )

        with stage("Generating CREATE TABLE Scripts"):
            generate_all_create_tables(
                dim_folder=dims_folder,
                fact_folder=facts_folder,
                output_folder=final_folder
            )

    # ----------------------------------
    # Summary
    # ----------------------------------
    print("\n✔ All tables generated and packaged successfully!\n")
    print(f"📂 Output Folder: {final_folder}")
    print(f"\n=== ALL STEPS COMPLETED IN {time.time() - total_start:.2f} SECONDS ===\n")


if __name__ == "__main__":
    main()
