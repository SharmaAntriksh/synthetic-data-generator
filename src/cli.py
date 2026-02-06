# src/cli.py
from __future__ import annotations

import argparse
import sys
from typing import Optional, Sequence

from src.engine.runners.pipeline_runner import PipelineOverrides, run_pipeline
from src.utils.logging_utils import fail


def str2bool(v):
    if isinstance(v, bool):
        return v
    v = str(v).strip().lower()
    if v in ("yes", "true", "1", "y"):
        return True
    if v in ("no", "false", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def build_parser() -> argparse.ArgumentParser:
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
        help="Override stores count (preferred: stores.num_stores; legacy: stores.total_stores)",
    )

    parser.add_argument(
        "--products",
        type=int,
        help="Override products.num_products",
    )

    parser.add_argument(
        "--promotions",
        type=int,
        help="Override total promotions (distributed across promotion buckets when possible)",
    )

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    overrides = PipelineOverrides(
        file_format=args.format,
        sales_rows=args.sales_rows,
        workers=args.workers,
        chunk_size=args.chunk_size,
        skip_order_cols=args.skip_order_cols,
        row_group_size=args.row_group_size,
        start_date=args.start_date,
        end_date=args.end_date,
        customers=args.customers,
        stores=args.stores,
        products=args.products,
        promotions=args.promotions,
    )

    try:
        run_pipeline(
            config_path=args.config,
            models_config_path=args.models_config,
            only=args.only,
            clean=bool(args.clean),
            dry_run=bool(args.dry_run),
            regen_dimensions=args.regen_dimensions,
            overrides=overrides,
        )
        return 0
    except Exception as ex:
        # pipeline_runner already logs fail(); keep a last-resort message + exit non-zero.
        fail(str(ex))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
