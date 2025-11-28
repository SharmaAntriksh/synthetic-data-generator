import time
from pathlib import Path
from contextlib import contextmanager

from src.dimensions.customers import generate_synthetic_customers
from src.dimensions.promotions import generate_promotions_catalog
from src.dimensions.stores import generate_store_table
from src.dimensions.dates import generate_date_table
from src.dimensions.currency import generate_currency_dimension
from src.dimensions.exchange_rates import generate_exchange_rate_table
from src.dimensions.geography_builder import build_dim_geography

from src.utils.versioning import should_regenerate, save_version


@contextmanager
def stage(label: str):
    print(f"\n=== {label}... ===")
    t = time.time()
    yield
    print(f"\n✔ {label} completed in {time.time() - t:.2f} seconds")


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
                max_rows=geo_cfg["max_geos"],
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
                seed=promo_cfg["seed"],
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
                seed=store_cfg["seed"],
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
                date_cfg["fiscal_month_offset"],
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
                exch_cfg["seed"],
            )
            fx_df.to_parquet(fx_out, index=False)
            save_version("exchange_rates", exch_cfg)
    else:
        print("✔ Exchange Rates up-to-date; skipping regeneration")
