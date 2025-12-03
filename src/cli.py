import argparse
import time
from pathlib import Path

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

    parser.add_argument("--format",
                        choices=["csv", "parquet", "delta", "deltaparquet"],
                        help="Override sales.file_format")

    parser.add_argument("--rows",
                        type=int,
                        help="Override sales.total_rows")

    parser.add_argument("--start-date",
                        help="Override _defaults.dates.start (YYYY-MM-DD)")

    parser.add_argument("--end-date",
                        help="Override _defaults.dates.end (YYYY-MM-DD)")

    parser.add_argument("--fx-start",
                        help="Override exchange_rates.dates.start (if use_global_dates=false)")

    parser.add_argument("--fx-end",
                        help="Override exchange_rates.dates.end (if use_global_dates=false)")

    parser.add_argument("--config",
                        default="config.yaml",
                        help="Path to configuration file")

    args = parser.parse_args()

    try:
        # ---------------- LOAD CONFIG -----------------
        raw_cfg = load_config_file(args.config)
        cfg = load_config(raw_cfg)

        # ==================================================
        # APPLY OVERRIDES EXACTLY MATCHING YOUR CONFIG LOADER
        # ==================================================

        # ----- SALES overrides -----
        if args.format:
            cfg["sales"]["file_format"] = args.format

        if args.rows:
            cfg["sales"]["total_rows"] = args.rows

        # ----- DEFAULT DATES overrides (always available) -----
        if args.start_date:
            cfg["_defaults"]["dates"]["start"] = args.start_date

        if args.end_date:
            cfg["_defaults"]["dates"]["end"] = args.end_date

        # ----- FX overrides -----
        fx_cfg = cfg["exchange_rates"]

        # Only apply if override dates are actually used
        if not fx_cfg.get("use_global_dates", True):

            fx_cfg.setdefault("dates", {})

            if args.fx_start:
                fx_cfg["dates"]["start"] = args.fx_start

            if args.fx_end:
                fx_cfg["dates"]["end"] = args.fx_end

        # ---------------- RUN PIPELINE ------------------

        info("Starting full Contoso pipeline...")

        sales_cfg = cfg["sales"]
        parquet_dims = Path(sales_cfg["parquet_folder"])
        fact_out = Path(sales_cfg["out_folder"])

        # Full run = dimensions + sales
        generate_dimensions(cfg, parquet_dims)
        run_sales_pipeline(sales_cfg, fact_out, parquet_dims, cfg)

        elapsed = time.time() - PIPELINE_START_TIME
        info(f"All pipelines completed in {fmt_sec(elapsed)}.")

    except Exception as ex:
        fail(str(ex))
        raise
