import json
import os

from src.customers import generate_synthetic_customers
from src.promotions import generate_promotions_catalog
from src.stores import generate_store_table
from src.dates import generate_date_table
from src.sales import generate_sales_fact


from src.output_utils import (
    clear_folder,
    create_final_output_folder
)

from src.generate_bulk_insert_sql import generate_bulk_insert_script
from src.generate_create_table_scripts import generate_all_create_tables


def main():

    # ------------------------------
    # Load configuration
    # ------------------------------
    with open("config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    cust_cfg = cfg["customers"]
    promo_cfg = cfg["promotions"]
    store_cfg = cfg["stores"]
    date_cfg = cfg["dates"]
    sales_cfg = cfg["sales"]

    parquet_dims = sales_cfg["parquet_folder"]
    fact_out = sales_cfg["out_folder"]

    # ------------------------------
    # Generate dimension tables
    # ------------------------------

    # Customers + Geography
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

    customers.to_parquet(os.path.join(parquet_dims, "customers.parquet"), index=False)
    geo.to_parquet(os.path.join(parquet_dims, "geography.parquet"), index=False)

    # Promotions
    promotions = generate_promotions_catalog(
        years=range(promo_cfg["years"][0], promo_cfg["years"][1] + 1),
        num_seasonal=promo_cfg["num_seasonal"],
        num_clearance=promo_cfg["num_clearance"],
        num_limited=promo_cfg["num_limited"],
        seed=promo_cfg["seed"]
    )
    promotions.to_parquet(os.path.join(parquet_dims, "promotions.parquet"), index=False)

    # Stores
    stores = generate_store_table(
        geography_parquet_path=store_cfg["geography_path"],
        num_stores=store_cfg["num_stores"],
        opening_start=store_cfg["opening_start"],
        opening_end=store_cfg["opening_end"],
        closing_end=store_cfg["closing_end"],
        seed=store_cfg["seed"]
    )

    stores.to_parquet(os.path.join(parquet_dims, "stores.parquet"), index=False)

    # Dates
    dates = generate_date_table(
        date_cfg["start_date"],
        date_cfg["end_date"],
        date_cfg["fiscal_month_offset"]
    )
    dates.to_parquet(os.path.join(parquet_dims, "dates.parquet"), index=False)

    # ------------------------------
    # Prepare FACT folder
    # ------------------------------
    clear_folder(fact_out)

    # Remove parquet-only settings if CSV mode
    if sales_cfg.get("file_format") == "csv":
        for key in ["row_group_size", "compression", "merge_parquet", "merged_file"]:
            sales_cfg.pop(key, None)

    # ------------------------------
    # Generate Sales Fact (CSV or Parquet)
    # ------------------------------
    generate_sales_fact(
        parquet_folder=sales_cfg["parquet_folder"],
        out_folder=sales_cfg["out_folder"],
        total_rows=sales_cfg["total_rows"],
        chunk_size=sales_cfg["chunk_size"],
        start_date=sales_cfg["start_date"],
        end_date=sales_cfg["end_date"],
        file_format=sales_cfg["file_format"],

        # Parquet-only fields (safe even if missing)
        row_group_size=sales_cfg.get("row_group_size"),
        compression=sales_cfg.get("compression"),
        merge_parquet=sales_cfg.get("merge_parquet", False),
        merged_file=sales_cfg.get("merged_file", "sales.parquet"),

        heavy_pct=sales_cfg["heavy_pct"],
        heavy_mult=sales_cfg["heavy_mult"],
        seed=sales_cfg["seed"]
    )

    # ------------------------------
    # Package final dataset folder
    # ------------------------------
    final_folder = create_final_output_folder(
        parquet_dims=parquet_dims,
        fact_folder=fact_out,
        file_format=sales_cfg["file_format"]
    )

    # ------------------------------
    # Generate BULK INSERT SQL (only in CSV mode)
    # ------------------------------
    if sales_cfg.get("file_format") == "csv":

        dims_folder = os.path.join(final_folder, "dims")
        facts_folder = os.path.join(final_folder, "facts")

        # --------------------------
        # Generate BULK INSERT scripts that use XML format files
        # --------------------------
        generate_bulk_insert_script(
            csv_folder=facts_folder,
            output_sql_file=os.path.join(final_folder, "bulk_insert_sales.sql")
        )

        generate_bulk_insert_script(
            csv_folder=dims_folder,
            table_name=None,
            output_sql_file=os.path.join(final_folder, "bulk_insert_dimensions.sql")
        )

        # Generate CREATE TABLE scripts FROM CSVs
        generate_all_create_tables(
            dim_folder=dims_folder,
            fact_folder=facts_folder,
            output_folder=final_folder
        )


    print("\n✔ All tables generated and packaged successfully!\n")
    print(f"📂 Output Folder: {final_folder}")


if __name__ == "__main__":
    main()
