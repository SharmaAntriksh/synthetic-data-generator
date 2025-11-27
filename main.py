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

from src.currency import generate_currency_dimension
from src.exchange_rates import generate_exchange_rate_table
from src.geography_builder import build_dim_geography

from src.versioning import should_regenerate, save_version


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
    if cfg.get("file_format") == "csv":
        parquet_only = ("row_group_size", "compression", "merge_parquet", "merged_file")
        cfg = {k: v for k, v in cfg.items() if k not in parquet_only}
    return cfg


# ---------------------------------------------------------
# Dimension Generation (with SMART VERSIONING)
# ---------------------------------------------------------

def generate_dimensions(cfg, parquet_dims: Path):
    cust_cfg = cfg["customers"]
    promo_cfg = cfg["promotions"]
    store_cfg = cfg["stores"]
    date_cfg = cfg["dates"]
    exch_cfg = cfg["exchange_rates"]

    # Geography
    geo_out = parquet_dims / "geography.parquet"
    if should_regenerate("geography", cust_cfg["geography_source"], geo_out):
        with stage("Generating Geography"):
            geo_cfg = cust_cfg["geography_source"]
            build_dim_geography(
                source_path=geo_cfg["path"],
                output_path=geo_out,
                max_rows=geo_cfg["max_geos"]
            )
            save_version("geography", geo_cfg)
    else:
        print("✔ Geography up-to-date; skipping regeneration")

    # Customers
    cust_out = parquet_dims / "customers.parquet"
    if should_regenerate("customers", cust_cfg, cust_out):
        with stage("Generating Customers"):
            customers = generate_synthetic_customers(cfg)
            customers.to_parquet(cust_out, index=False)
            save_version("customers", cust_cfg)
    else:
        print("✔ Customers up-to-date; skipping regeneration")

    # Promotions
    promo_out = parquet_dims / "promotions.parquet"
    if should_regenerate("promotions", promo_cfg, promo_out):
        with stage("Generating Promotions"):
            promotions = generate_promotions_catalog(
                years=promo_cfg["years"],
                num_seasonal=promo_cfg["num_seasonal"],
                num_clearance=promo_cfg["num_clearance"],
                num_limited=promo_cfg["num_limited"],
                seed=promo_cfg["seed"]
            )
            promotions.to_parquet(promo_out, index=False)
            save_version("promotions", promo_cfg)
    else:
        print("✔ Promotions up-to-date; skipping regeneration")

    # Stores
    store_out = parquet_dims / "stores.parquet"
    if should_regenerate("stores", store_cfg, store_out):
        with stage("Generating Stores"):
            stores = generate_store_table(
                geography_parquet_path=store_cfg["geography_path"],
                num_stores=store_cfg["num_stores"],
                opening_start=store_cfg["opening_start"],
                opening_end=store_cfg["opening_end"],
                closing_end=store_cfg["closing_end"],
                seed=store_cfg["seed"]
            )
            stores.to_parquet(store_out, index=False)
            save_version("stores", store_cfg)
    else:
        print("✔ Stores up-to-date; skipping regeneration")

    # Dates
    dates_out = parquet_dims / "dates.parquet"
    if should_regenerate("dates", date_cfg, dates_out):
        with stage("Generating Dates"):
            dates = generate_date_table(
                date_cfg["start_date"],
                date_cfg["end_date"],
                date_cfg["fiscal_month_offset"]
            )
            dates.to_parquet(dates_out, index=False)
            save_version("dates", date_cfg)
    else:
        print("✔ Dates up-to-date; skipping regeneration")

    # Currency
    curr_out = parquet_dims / "currency.parquet"
    if should_regenerate("currency", exch_cfg, curr_out):
        with stage("Generating Currency Dimension"):
            currency_df = generate_currency_dimension(exch_cfg["currencies"])
            currency_df.to_parquet(curr_out, index=False)
            save_version("currency", exch_cfg)
    else:
        print("✔ Currency dimension up-to-date; skipping regeneration")

    # Exchange Rates
    fx_out = parquet_dims / "exchange_rates.parquet"
    if should_regenerate("exchange_rates", exch_cfg, fx_out):
        with stage("Generating Exchange Rates"):
            fx_df = generate_exchange_rate_table(
                exch_cfg["start_date"],
                exch_cfg["end_date"],
                exch_cfg["currencies"],
                exch_cfg["base_currency"],
                exch_cfg["volatility"],
                exch_cfg["seed"]
            )
            fx_df.to_parquet(fx_out, index=False)
            save_version("exchange_rates", exch_cfg)
    else:
        print("✔ Exchange Rates up-to-date; skipping regeneration")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    total_start = time.time()

    with open("config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    sales_cfg = normalize_sales_config(cfg["sales"])

    parquet_dims = Path(sales_cfg["parquet_folder"])
    fact_out = Path(sales_cfg["out_folder"])

    parquet_dims.mkdir(parents=True, exist_ok=True)
    fact_out.mkdir(parents=True, exist_ok=True)

    validate_config(sales_cfg, "sales", ["total_rows", "chunk_size", "file_format"])

    # Dimensions
    generate_dimensions(cfg, parquet_dims)

    # Sales Fact
    with stage("Generating Sales"):
        clear_folder(fact_out)

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

    # Final output
    with stage("Creating Final Output Folder"):
        final_folder = create_final_output_folder(
            parquet_dims=parquet_dims,
            fact_folder=fact_out,
            file_format=sales_cfg["file_format"]
        )

    # SQL Scripts (CSV only)
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

    print("\n✔ All tables generated and packaged successfully!\n")
    print(f"📂 Output Folder: {final_folder}")
    print(f"\n=== ALL STEPS COMPLETED IN {time.time() - total_start:.2f} SECONDS ===\n")


if __name__ == "__main__":
    main()
