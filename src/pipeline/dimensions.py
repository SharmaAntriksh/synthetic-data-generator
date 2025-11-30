import time
from pathlib import Path
from datetime import datetime

from src.dimensions.customers import generate_synthetic_customers
from src.dimensions.promotions import generate_promotions_catalog
from src.dimensions.stores import generate_store_table
from src.dimensions.dates import generate_date_table
from src.dimensions.currency import generate_currency_dimension
from src.dimensions.exchange_rates import generate_exchange_rate_table
from src.dimensions.geography_builder import build_dim_geography

from src.utils.versioning import should_regenerate, save_version
from src.utils.logging_utils import stage, info, skip, done


def expand_date_ranges(ranges):
    """
    Convert date_ranges into precise per-year windows:
    [
        {2023: (2023-04-01, 2023-12-31)},
        {2024: (2024-01-01, 2024-02-01)},
    ]
    """
    year_windows = {}

    for r in ranges:
        s = datetime.fromisoformat(r["start"])
        e = datetime.fromisoformat(r["end"])

        for y in range(s.year, e.year + 1):

            # full default year range
            y_start = datetime(y, 1, 1)
            y_end   = datetime(y, 12, 31)

            # clamp to actual date_window
            if y == s.year:
                y_start = s
            if y == e.year:
                y_end = e

            year_windows[y] = (y_start, y_end)

    return year_windows


def generate_dimensions(cfg, parquet_dims: Path):

    def out(name):
        return parquet_dims / f"{name}.parquet"

    def changed(name, section):
        return should_regenerate(name, section, out(name))

    # --------------------------------------------------
    # Geography (root)
    # --------------------------------------------------
    if changed("geography", cfg["geography"]):
        with stage("Generating Geography"):
            build_dim_geography(cfg)
            save_version("geography", cfg["geography"])
    else:
        skip("Geography up-to-date; skipping regeneration")

    # --------------------------------------------------
    # Customers (depends on geography)
    # --------------------------------------------------
    customer_need = changed("customers", cfg["customers"]) or changed("geography", cfg["geography"])

    if customer_need:
        info("Dependency triggered: Customers will regenerate.")
        with stage("Generating Customers"):
            df = generate_synthetic_customers(cfg)
            df.to_parquet(out("customers"), index=False)
            save_version("customers", cfg["customers"])
    else:
        skip("Customers up-to-date; skipping regeneration")

    # --------------------------------------------------
    # Promotions (independent)
    # --------------------------------------------------
    if changed("promotions", cfg["promotions"]):
        with stage("Generating Promotions"):

            # precise expansion of date_ranges → year windows
            ranges = cfg["promotions"]["date_ranges"]
            year_windows = expand_date_ranges(ranges)
            years = sorted(year_windows.keys())

            df = generate_promotions_catalog(
                years=years,
                year_windows=year_windows,   # <-- new precise boundaries
                num_seasonal=cfg["promotions"]["num_seasonal"],
                num_clearance=cfg["promotions"]["num_clearance"],
                num_limited=cfg["promotions"]["num_limited"],
                seed=cfg["promotions"]["seed"],
            )
            df.to_parquet(out("promotions"), index=False)
            save_version("promotions", cfg["promotions"])
    else:
        skip("Promotions up-to-date; skipping regeneration")

    # --------------------------------------------------
    # Stores (depends on geography)
    # --------------------------------------------------
    store_need = changed("stores", cfg["stores"]) or changed("geography", cfg["geography"])

    if store_need:
        info("Dependency triggered: Stores will regenerate.")
        with stage("Generating Stores"):
            df = generate_store_table(
                geography_parquet_path=cfg["stores"]["paths"]["geography"],
                num_stores=cfg["stores"]["num_stores"],
                opening_start=cfg["stores"]["opening"]["start"],
                opening_end=cfg["stores"]["opening"]["end"],
                closing_end=cfg["stores"]["closing_end"],
                seed=cfg["stores"]["seed"],
            )
            df.to_parquet(out("stores"), index=False)
            save_version("stores", cfg["stores"])
    else:
        skip("Stores up-to-date; skipping regeneration")

    # --------------------------------------------------
    # Dates
    # --------------------------------------------------
    if changed("dates", cfg["dates"]):
        with stage("Generating Dates"):
            df = generate_date_table(
                cfg["dates"]["dates"]["start"],
                cfg["dates"]["dates"]["end"],
                cfg["dates"]["fiscal_month_offset"]
            )
            df.to_parquet(out("dates"), index=False)
            save_version("dates", cfg["dates"])
    else:
        skip("Dates up-to-date; skipping regeneration")

    # --------------------------------------------------
    # Currency
    # --------------------------------------------------
    if changed("currency", cfg["exchange_rates"]):
        with stage("Generating Currency Dimension"):
            df = generate_currency_dimension(cfg["exchange_rates"]["currencies"])
            df.to_parquet(out("currency"), index=False)
            save_version("currency", cfg["exchange_rates"])
    else:
        skip("Currency dimension up-to-date; skipping regeneration")

    # --------------------------------------------------
    # Exchange Rates
    # --------------------------------------------------
    fx_need = changed("exchange_rates", cfg["exchange_rates"]) or changed("currency", cfg["exchange_rates"])

    if fx_need:
        info("Dependency triggered: Exchange Rates will regenerate.")
        with stage("Generating Exchange Rates"):
            df = generate_exchange_rate_table(
                cfg["exchange_rates"]["dates"]["start"],
                cfg["exchange_rates"]["dates"]["end"],
                cfg["exchange_rates"]["currencies"],
                cfg["exchange_rates"]["base_currency"],
                cfg["exchange_rates"]["volatility"],
                cfg["exchange_rates"]["seed"],
            )
            df.to_parquet(out("exchange_rates"), index=False)
            save_version("exchange_rates", cfg["exchange_rates"])
    else:
        skip("Exchange Rates up-to-date; skipping regeneration")

    done("All dimensions generated.")
