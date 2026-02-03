import argparse
import time
import shutil
from pathlib import Path
from pprint import pprint

from src.engine.config.config_loader import load_config, load_config_file
from src.engine.runners.dimensions_runner import generate_dimensions
from src.engine.runners.sales_runner import run_sales_pipeline
from src.utils.logging_utils import info, fail, PIPELINE_START_TIME, fmt_sec


def _resolve_input_path(p: str) -> Path:
    """
    Resolve an input file path robustly:
    - expanduser
    - if absolute or exists relative to CWD, use that
    - else try relative to repo root (one level above /src)
    """
    raw = Path(p).expanduser()

    if raw.is_absolute() and raw.exists():
        return raw.resolve()

    cwd_candidate = (Path.cwd() / raw).resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    repo_root = Path(__file__).resolve().parents[1]  # .../src/cli.py -> repo root
    repo_candidate = (repo_root / raw).resolve()
    return repo_candidate  # may or may not exist; caller can decide


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in ("yes", "true", "1", "y"):
        return True
    if v in ("no", "false", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def _ensure_defaults_dates(cfg: dict):
    """
    Ensure cfg has canonical defaults.dates dict for CLI overrides.

    Note: config_loader.load_config already canonicalizes defaults/_defaults.
    This is a final safety guard so CLI overrides never write into _defaults.
    """
    cfg.setdefault("defaults", {})
    cfg["defaults"].setdefault("dates", {})
    cfg["defaults"]["dates"].setdefault("start", None)
    cfg["defaults"]["dates"].setdefault("end", None)


def main():
    parser = argparse.ArgumentParser(
        prog="contoso",
        description="Contoso Fake Data Generator",
    )

    # ----------------- SALES / OUTPUT OVERRIDES -----------------

    parser.add_argument(
        "--format",
        choices=["csv", "parquet", "delta", "deltaparquet"],
        help="Override sales.file_format",
    )

    parser.add_argument(
        "--skip-order-cols",
        type=str2bool,
        nargs="?",
        const=True,
        default=None,
        help="Override sales.skip_order_cols (true/false)",
    )

    parser.add_argument(
        "--sales-rows",
        type=int,
        help="Override sales.total_rows",
    )

    parser.add_argument(
        "--workers",
        type=int,
        help="Override sales.workers",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Override sales.chunk_size",
    )

    parser.add_argument(
        "--row-group-size",
        type=int,
        help="Override sales.row_group_size (parquet/deltaparquet only)",
    )

    # ----------------- DATE OVERRIDES -----------------

    parser.add_argument(
        "--start-date",
        help="Override global start date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--end-date",
        help="Override global end date (YYYY-MM-DD)",
    )

    # ----------------- PIPELINE CONTROL -----------------

    parser.add_argument(
        "--only",
        choices=["dimensions", "sales"],
        help="Run only a specific pipeline",
    )

    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete FINAL output folders before running",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved configuration and exit",
    )

    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--models-config",
        default="models.yaml",
        help="Path to models configuration file",
    )

    parser.add_argument(
        "--regen-dimensions",
        nargs="+",
        help="Force regeneration of selected dimensions (e.g. customers products stores or 'all')",
    )

    # ----------------- DIMENSION SIZE OVERRIDES -----------------

    parser.add_argument(
        "--customers",
        type=int,
        help="Override customers.total_customers",
    )

    parser.add_argument(
        "--stores",
        type=int,
        help="Override stores.total_stores",
    )

    parser.add_argument(
        "--products",
        type=int,
        help="Override products.num_products",
    )

    parser.add_argument(
        "--promotions",
        type=int,
        help="Override promotions.total_promotions",
    )

    args = parser.parse_args()

    # Normalize force regeneration intent
    force_regenerate = set(args.regen_dimensions) if args.regen_dimensions else set()

    # Normalize format (delta → deltaparquet)
    if args.format == "delta":
        args.format = "deltaparquet"

    try:
        # ==================================================
        # LOAD CONFIGS
        # ==================================================
        cfg = load_config(args.config)  # returns resolved modules + _defaults summary

        if "sales" not in cfg or not isinstance(cfg["sales"], dict):
            fail("Config must contain a 'sales' section")

        sales_cfg = cfg["sales"]

        models_raw = load_config_file(args.models_config)
        if "models" not in models_raw or not isinstance(models_raw["models"], dict):
            fail("models.yaml must contain a top-level 'models' section")
        models_cfg = models_raw["models"]

        cfg["config_yaml_path"] = str(_resolve_input_path(args.config))
        cfg["model_yaml_path"]  = str(_resolve_input_path(args.models_config))
        info(f"CLI attached run spec paths: config={cfg['config_yaml_path']} model={cfg['model_yaml_path']}")
        # ==================================================
        # APPLY OVERRIDES (SALES)
        # ==================================================
        if args.format:
            sales_cfg["file_format"] = args.format.lower()

        if args.sales_rows is not None:
            sales_cfg["total_rows"] = int(args.sales_rows)

        if args.workers is not None:
            sales_cfg["workers"] = int(args.workers)

        if args.chunk_size is not None:
            sales_cfg["chunk_size"] = int(args.chunk_size)

        if args.skip_order_cols is not None:
            sales_cfg["skip_order_cols"] = bool(args.skip_order_cols)

        if args.row_group_size is not None:
            fmt = str(sales_cfg.get("file_format", "")).lower()
            if fmt not in ("parquet", "deltaparquet"):
                fail("--row-group-size is only valid for parquet or deltaparquet output")
            sales_cfg["row_group_size"] = int(args.row_group_size)

        # ==================================================
        # APPLY OVERRIDES (GLOBAL DATES)
        # ==================================================
        _ensure_defaults_dates(cfg)

        if args.start_date:
            cfg["defaults"]["dates"]["start"] = args.start_date

        if args.end_date:
            cfg["defaults"]["dates"]["end"] = args.end_date

        # ==================================================
        # FX ALWAYS FOLLOWS GLOBAL DATES
        # ==================================================
        if "exchange_rates" in cfg and isinstance(cfg["exchange_rates"], dict):
            fx_cfg = cfg["exchange_rates"]
            fx_cfg["use_global_dates"] = True
            # Remove any locally resolved dates so dimension uses injected global_dates
            fx_cfg.pop("dates", None)

        # ==================================================
        # DIMENSION SIZE OVERRIDES
        # ==================================================
        if args.customers is not None:
            cfg.setdefault("customers", {})
            cfg["customers"]["total_customers"] = int(args.customers)

        if args.stores is not None:
            cfg.setdefault("stores", {})
            cfg["stores"]["total_stores"] = int(args.stores)

        if args.products is not None:
            cfg.setdefault("products", {})
            cfg["products"]["num_products"] = int(args.products)

        if args.promotions is not None:
            cfg.setdefault("promotions", {})
            cfg["promotions"]["total_promotions"] = int(args.promotions)

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
        # OPTIONAL CLEAN (FINAL OUTPUT ROOT)
        # ==================================================
        if args.clean:
            info("Cleaning final output folders before run...")
            gen_root = cfg.get("generated_datasets_root")
            if gen_root:
                shutil.rmtree(gen_root, ignore_errors=True)

        # ==================================================
        # RESOLVE PATHS
        # ==================================================
        # CLI remains compatible with existing config contracts:
        # sales.parquet_folder and sales.out_folder are used by runners.
        if "parquet_folder" not in sales_cfg or "out_folder" not in sales_cfg:
            fail("sales.parquet_folder and sales.out_folder must be set in config (or added to CLI later)")

        parquet_dims = Path(sales_cfg["parquet_folder"]).resolve()
        fact_out = Path(sales_cfg["out_folder"]).resolve()

        parquet_dims.mkdir(parents=True, exist_ok=True)
        fact_out.mkdir(parents=True, exist_ok=True)

        # ==================================================
        # HARD RESET SCRATCH FACT OUTPUT (as before)
        # ==================================================
        info(f"Resetting fact output folder: {fact_out}")
        if fact_out.exists():
            shutil.rmtree(fact_out, ignore_errors=True)
        fact_out.mkdir(parents=True, exist_ok=True)

        # ==================================================
        # RUN PIPELINES
        # ==================================================
        info("Starting full Contoso pipeline...")

        if args.only != "sales":
            generate_dimensions(
                cfg,
                parquet_dims,
                force_regenerate=force_regenerate,
            )

        if args.only != "dimensions":
            run_sales_pipeline(sales_cfg, fact_out, parquet_dims, cfg)

        # ==================================================
        # FINAL CLEANUP (scratch)
        # ==================================================
        info(f"Cleaning scratch fact_out folder: {fact_out}")
        shutil.rmtree(fact_out, ignore_errors=True)

        elapsed = time.time() - PIPELINE_START_TIME
        info(f"All pipelines completed in {fmt_sec(elapsed)}.")

    except Exception as ex:
        fail(str(ex))
        raise
