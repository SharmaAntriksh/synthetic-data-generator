import argparse
import time
import shutil
from pathlib import Path
from pprint import pprint

from src.engine.config.config_loader import load_config_file, load_config
from src.engine.runners.dimensions_runner import generate_dimensions
from src.engine.runners.sales_runner import run_sales_pipeline
from src.utils.logging_utils import info, fail, PIPELINE_START_TIME, fmt_sec


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "1"):
        return True
    if v.lower() in ("no", "false", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    parser = argparse.ArgumentParser(
        prog="contoso",
        description="Contoso Fake Data Generator"
    )

    # ----------------- SALES / OUTPUT OVERRIDES -----------------

    parser.add_argument(
        "--format",
        choices=["csv", "parquet", "delta", "deltaparquet"],
        help="Override sales.file_format"
    )

    parser.add_argument(
        "--skip-order-cols",
        type=str2bool,
        nargs="?",
        const=True,
        default=None
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
        "--row-group-size",
        type=int,
        help="Parquet / Delta row group size"
    )

    # ----------------- DATE OVERRIDES -----------------

    parser.add_argument(
        "--start-date",
        help="Override global start date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end-date",
        help="Override global end date (YYYY-MM-DD)"
    )

    # ----------------- PIPELINE CONTROL -----------------

    parser.add_argument(
        "--only",
        choices=["dimensions", "sales"],
        help="Run only a specific pipeline"
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete FINAL output folders before running"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved configuration and exit"
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file"
    )

    parser.add_argument(
        "--models-config",
        default="models.yaml",
        help="Path to models configuration file"
    )

    parser.add_argument(
        "--regen-dimensions",
        nargs="+",
        help=(
            "Force regeneration of selected dimensions "
            "(e.g. customers products stores or 'all')"
        ),
    )

    # ----------------- DIMENSION SIZE OVERRIDES -----------------

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
        help="Override products.num_products"
    )

    parser.add_argument(
        "--promotions",
        type=int,
        help="Override promotions.total_promotions"
    )

    args = parser.parse_args()

    # Normalize force regeneration intent
    force_regenerate = (
        set(args.regen_dimensions)
        if args.regen_dimensions
        else set()
    )

    # --------------------------------------------------
    # NORMALIZE FORMAT (delta → deltaparquet)
    # --------------------------------------------------
    if args.format == "delta":
        args.format = "deltaparquet"

    try:
        # ==================================================
        # LOAD CONFIG
        # ==================================================
        raw_cfg = load_config_file(args.config)
        cfg = load_config(raw_cfg)
        sales_cfg = cfg["sales"]

        models_raw = load_config_file(args.models_config)

        if "models" not in models_raw or not isinstance(models_raw["models"], dict):
            fail("models.yaml must contain a top-level 'models' section")

        models_cfg = models_raw["models"]

        # ==================================================
        # APPLY OVERRIDES
        # ==================================================

        if args.format:
            sales_cfg["file_format"] = args.format.lower()

        if args.sales_rows is not None:
            sales_cfg["total_rows"] = args.sales_rows

        if args.workers is not None:
            sales_cfg["workers"] = args.workers

        if args.chunk_size is not None:
            sales_cfg["chunk_size"] = args.chunk_size

        if args.skip_order_cols is not None:
            sales_cfg["skip_order_cols"] = args.skip_order_cols

        if args.row_group_size is not None:
            fmt = sales_cfg.get("file_format")
            if fmt not in ("parquet", "deltaparquet"):
                fail("--row-group-size is only valid for parquet or deltaparquet output")

            sales_cfg.setdefault(fmt, {})
            sales_cfg[fmt]["row_group_size"] = args.row_group_size

        # ----- Global date overrides -----
        cfg.setdefault("_defaults", {}).setdefault("dates", {})

        if args.start_date:
            cfg["_defaults"]["dates"]["start"] = args.start_date

        if args.end_date:
            cfg["_defaults"]["dates"]["end"] = args.end_date

        # ----- FX always follows global dates -----
        fx_cfg = cfg["exchange_rates"]
        fx_cfg["use_global_dates"] = True
        fx_cfg.pop("dates", None)

        # ----- Dimension size overrides -----
        if args.customers is not None:
            cfg["customers"]["total_customers"] = args.customers

        if args.stores is not None:
            cfg["stores"]["total_stores"] = args.stores

        if args.products is not None:
            cfg["products"]["num_products"] = args.products

        if args.promotions is not None:
            cfg["promotions"]["total_promotions"] = args.promotions

        # ==================================================
        # DRY RUN
        # ==================================================
        if args.dry_run:
            info("Dry run enabled. Resolved configuration:")
            pprint(cfg)
            return

        # ==================================================
        # ATTACH MODELS CONFIG TO RUNTIME STATE
        # ==================================================
        from src.facts.sales.sales_logic.globals import State
        State.models_cfg = models_cfg

        # ==================================================
        # HARD RESET FACT OUTPUT
        # ==================================================
        fact_out = Path(sales_cfg["out_folder"]).resolve()

        info(f"Resetting fact output folder: {fact_out}")
        if fact_out.exists():
            shutil.rmtree(fact_out)
        fact_out.mkdir(parents=True, exist_ok=True)

        # ==================================================
        # OPTIONAL CLEAN
        # ==================================================
        if args.clean:
            info("Cleaning final output folders before run...")
            gen_root = cfg.get("generated_datasets_root")
            if gen_root:
                shutil.rmtree(gen_root, ignore_errors=True)

        # ==================================================
        # RUN PIPELINES
        # ==================================================
        info("Starting full Contoso pipeline...")

        parquet_dims = Path(sales_cfg["parquet_folder"]).resolve()

        if args.only != "sales":
            generate_dimensions(
                cfg,
                parquet_dims,
                force_regenerate=force_regenerate,
            )

        if args.only != "dimensions":
            run_sales_pipeline(sales_cfg, fact_out, parquet_dims, cfg)

        # ==================================================
        # FINAL CLEANUP
        # ==================================================
        info(f"Cleaning scratch fact_out folder: {fact_out}")
        shutil.rmtree(fact_out, ignore_errors=True)

        elapsed = time.time() - PIPELINE_START_TIME
        info(f"All pipelines completed in {fmt_sec(elapsed)}.")

    except Exception as ex:
        fail(str(ex))
        raise
