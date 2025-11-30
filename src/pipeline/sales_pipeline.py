# sales_pipeline.py — cleaned + dependency-aware

from pathlib import Path
from src.utils.logging_utils import info, skip, stage, done
from src.utils.versioning import should_regenerate, save_version
from src.facts.sales.sales import generate_sales_fact


def run_sales_pipeline(sales_cfg, fact_out: Path, parquet_dims: Path, cfg):
    """
    Fully dependency-aware Sales pipeline.
    Sales regenerates automatically if ANY upstream dimension changes.
    """

    # Path helper
    def out(name):
        return parquet_dims / f"{name}.parquet"

    # ------------------------------------------------------------
    # Dependency checks
    # ------------------------------------------------------------

    # Common helper
    def changed(name, section):
        return should_regenerate(name, section, out(name))

    geography_changed      = changed("geography", cfg["geography"])
    customers_changed      = changed("customers", cfg["customers"])
    stores_changed         = changed("stores", cfg["stores"])
    promotions_changed     = changed("promotions", cfg["promotions"])
    dates_changed          = changed("dates", cfg["dates"])
    currency_changed       = changed("currency", cfg["exchange_rates"])
    exchange_rates_changed = changed("exchange_rates", cfg["exchange_rates"])

    # Sales should regenerate if ANY dimension changed
    sales_dependencies_changed = any([
        geography_changed,
        customers_changed,
        promotions_changed,
        stores_changed,
        dates_changed,
        currency_changed,
        exchange_rates_changed,
    ])

    # ------------------------------------------------------------
    # Run or skip Sales regeneration
    # ------------------------------------------------------------
    sales_out = fact_out / "sales.parquet"

    if sales_dependencies_changed or should_regenerate("sales", sales_cfg, sales_out):
        info("Dependency triggered: Sales will regenerate.")

        with stage("Generating Sales"):
            generate_sales_fact(
                parquet_folder=sales_cfg["parquet_folder"],
                out_folder=sales_cfg["out_folder"],
                total_rows=sales_cfg["total_rows"],
                chunk_size=sales_cfg["chunk_size"],
                start_date=sales_cfg["dates"]["start"],
                end_date=sales_cfg["dates"]["end"],
                row_group_size=sales_cfg["row_group_size"],
                compression=sales_cfg["compression"],
                merge_parquet=sales_cfg["merge_parquet"],
                merged_file=sales_cfg["merged_file"],
                delete_chunks=sales_cfg["delete_chunks"],
                heavy_pct=sales_cfg["heavy_pct"],
                heavy_mult=sales_cfg["heavy_mult"],
                seed=sales_cfg["seed"],
                file_format=sales_cfg["file_format"],
                workers=sales_cfg["workers"],
                tune_chunk=sales_cfg["tune_chunk"],
                write_delta=sales_cfg["write_delta"],
                delta_output_folder=sales_cfg["delta_output_folder"],
                skip_order_cols=sales_cfg.get("skip_order_cols", False),
                write_pyarrow=sales_cfg.get("write_pyarrow", True),
            )

        save_version("sales", sales_cfg)

    else:
        skip("Sales up-to-date; skipping regeneration")

    done("Sales pipeline complete.")
