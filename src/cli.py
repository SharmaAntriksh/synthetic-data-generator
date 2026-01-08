import argparse
import time
import shutil
from pathlib import Path
from pprint import pprint

from src.engine.config.config_loader import load_config_file, load_config
from src.engine.runners.dimensions_runner import generate_dimensions
from src.engine.runners.sales_runner import run_sales_pipeline
from src.utils.logging_utils import info, fail, PIPELINE_START_TIME, fmt_sec


def main():
    parser = argparse.ArgumentParser(
        prog="contoso",
        description="Contoso Fake Data Generator"
    )

    # ----------------- OVERRIDES -------------------

    parser.add_argument(
        "--format",
        choices=["csv", "parquet", "delta", "deltaparquet"],
        help="Override sales.file_format"
    )

    parser.add_argument(
        "--skip-order-cols",
        action="store_true",
        help="Exclude OrderNumber and LineNumber columns from sales output"
    )
    
    parser.add_argument(
        "--sales-rows",
        type=int,
        help="Override sales.total_rows"
    )

    parser.add_argument(
        "--workers",
        type=int,
        help="Override sales.workers"
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Override sales.chunk_size"
    )

    parser.add_argument(
        "--start-date",
        help="Override _defaults.dates.start (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end-date",
        help="Override _defaults.dates.end (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--xrates-start",
        help="Override exchange_rates.dates.start (if use_global_dates=false)"
    )

    parser.add_argument(
        "--xrates-end",
        help="Override exchange_rates.dates.end (if use_global_dates=false)"
    )

    parser.add_argument(
        "--only",
        choices=["dimensions", "sales"],
        help="Run only a specific pipeline"
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete fact_out and generated_datasets before running"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved configuration and exit without running pipelines"
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--customers",
        type=int,
        help="Override customers.total_customers"
    )

    parser.add_argument(
        "--stores",
        type=int,
        help="Override stores.total_stores"
    )

    parser.add_argument(
        "--products",
        type=int,
        help="Override products.total_products"
    )

    parser.add_argument(
        "--promotions",
        type=int,
        help="Override promotions.total_promotions"
    )

    args = parser.parse_args()

    try:
        # ---------------- LOAD CONFIG -----------------
        raw_cfg = load_config_file(args.config)
        cfg = load_config(raw_cfg)

        # ==================================================
        # APPLY OVERRIDES (EXPLICIT AND SAFE)
        # ==================================================

        sales_cfg = cfg["sales"]

        # ----- SALES overrides -----
        if args.format:
            sales_cfg["file_format"] = args.format.lower()

        if args.sales_rows is not None:
            sales_cfg["total_rows"] = args.sales_rows

        if args.workers is not None:
            sales_cfg["workers"] = args.workers

        if args.chunk_size is not None:
            sales_cfg["chunk_size"] = args.chunk_size

        if args.skip_order_cols:
            sales_cfg["skip_order_cols"] = True

        # ----- DEFAULT DATE overrides -----
        if args.start_date:
            cfg["_defaults"]["dates"]["start"] = args.start_date

        if args.end_date:
            cfg["_defaults"]["dates"]["end"] = args.end_date

        # ----- FX overrides -----
        fx_cfg = cfg["exchange_rates"]

        if not fx_cfg.get("use_global_dates", True):
            fx_cfg.setdefault("dates", {})

            if args.xrates_start:
                fx_cfg["dates"]["start"] = args.xrates_start

            if args.xrates_end:
                fx_cfg["dates"]["end"] = args.xrates_end

        # ---------------- DIMENSION ROW OVERRIDES ------------------

        if args.customers is not None:
            cfg["customers"]["total_customers"] = args.customers

        if args.stores is not None:
            cfg["stores"]["total_stores"] = args.stores

        if args.products is not None:
            cfg["products"]["total_products"] = args.products

        if args.promotions is not None:
            cfg["promotions"]["total_promotions"] = args.promotions

        # ---------------- CLEAN (OPTIONAL) ----------------
        if args.clean:
            info("Cleaning output folders before run...")

            sales_cfg = cfg["sales"]

            # Clean fact output
            if "out_folder" in sales_cfg:
                shutil.rmtree(sales_cfg["out_folder"], ignore_errors=True)

            # Clean generated datasets (if configured)
            gen_root = cfg.get("generated_datasets_root")
            if gen_root:
                shutil.rmtree(gen_root, ignore_errors=True)

        # ---------------- DRY RUN ----------------
        if args.dry_run:
            info("Dry run enabled. Resolved configuration:")
            pprint(cfg)
            return

        # ---------------- RUN PIPELINE ------------------
        info("Starting full Contoso pipeline...")

        parquet_dims = Path(sales_cfg["parquet_folder"])
        fact_out = Path(sales_cfg["out_folder"])

        if args.only != "sales":
            generate_dimensions(cfg, parquet_dims)

        if args.only != "dimensions":
            run_sales_pipeline(sales_cfg, fact_out, parquet_dims, cfg)

        elapsed = time.time() - PIPELINE_START_TIME
        info(f"All pipelines completed in {fmt_sec(elapsed)}.")

    except Exception as ex:
        fail(str(ex))
        raise
