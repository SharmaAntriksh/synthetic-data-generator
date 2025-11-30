from pathlib import Path
from src.utils.logging_utils import info, skip, stage, done
from src.utils.versioning import should_regenerate, save_version
from src.facts.sales.sales import generate_sales_fact
import os


def run_sales_pipeline(sales_cfg, fact_out: Path, parquet_dims: Path, cfg):
    """
    Fully dependency-aware Sales pipeline.
    Sales regenerates automatically if ANY upstream dimension changes.
    """

    # Path helper
    def out(name):
        return parquet_dims / f"{name}.parquet"

    # ------------------------------------------------------------
    # Dependency detection
    # ------------------------------------------------------------
    def changed(name, section):
        return should_regenerate(name, section, out(name))

    sales_dependencies_changed = any([
        changed("geography", cfg["geography"]),
        changed("customers", cfg["customers"]),
        changed("promotions", cfg["promotions"]),
        changed("stores", cfg["stores"]),
        changed("dates", cfg["dates"]),
        changed("currency", cfg["exchange_rates"]),
        changed("exchange_rates", cfg["exchange_rates"]),
    ])

    # Output path (used only for Parquet single-file output)
    sales_out = fact_out / "sales.parquet"

    # ------------------------------------------------------------
    # Run or skip Sales regeneration
    # ------------------------------------------------------------
    if sales_dependencies_changed or should_regenerate("sales", sales_cfg, sales_out):
        info("Dependency triggered: Sales will regenerate.")

        # Extract global dates (your new config layout)
        start_date = cfg["dates"]["dates"]["start"]
        end_date   = cfg["dates"]["dates"]["end"]

        # Partitioning config for Delta
        partition_cfg = sales_cfg.get("partitioning", {})
        partition_enabled = partition_cfg.get("enabled", False)
        partition_cols    = partition_cfg.get("columns", [])

    # ------------------------------------------------------------
    # Run Sales regardless of format when regeneration is needed
    # ------------------------------------------------------------
    if sales_dependencies_changed or should_regenerate("sales", sales_cfg, sales_out):
        info("Dependency triggered: Sales will regenerate.")

        # Enforce DeltaParquet behavior
        if sales_cfg["file_format"] == "deltaparquet":
            sales_cfg["write_delta"] = True
            sales_cfg["merge_parquet"] = False

        with stage("Generating Sales"):
            generate_sales_fact(
                parquet_folder=sales_cfg["parquet_folder"],
                out_folder=fact_out,

                total_rows=sales_cfg["total_rows"],
                chunk_size=sales_cfg["chunk_size"],

                start_date=start_date,
                end_date=end_date,

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
                write_pyarrow=sales_cfg.get("write_pyarrow", True),

                write_delta=sales_cfg["write_delta"],
                delta_output_folder=os.path.join(fact_out, sales_cfg["delta_output_folder"]),

                partition_enabled=partition_enabled,
                partition_cols=partition_cols,

                skip_order_cols=sales_cfg.get("skip_order_cols", False),
            )

        save_version("sales", sales_cfg)

    else:
        skip("Sales up-to-date; skipping regeneration")


    done("Sales pipeline complete.")
