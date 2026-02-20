from __future__ import annotations

import time
import shutil
from pathlib import Path

from src.utils.logging_utils import stage, info, done
from src.engine.packaging import package_output
from src.engine.powerbi_packaging import maybe_attach_pbip_project
from src.facts.sales.sales import generate_sales_fact
from src.facts.sales.sales_logic import bind_globals, State


def run_sales_pipeline(sales_cfg, fact_out, parquet_dims, cfg):
    """
    Run the sales fact pipeline.

    UPDATED CONTRACT:
    - Customer lifecycle is fully driven by customers.parquet
      (IsActiveInSales, CustomerStartMonth, CustomerEndMonth, etc.)
    - This runner MUST NOT filter customers anymore.
    - Product filtering (IsActiveInSales) is still valid.
    - All lifecycle logic happens inside sales.py + chunk_builder.py

    Invariants:
    - Sales schema is determined entirely by sales_cfg
    - CSV / Delta outputs are regenerated per run
    - Parquet output is preserved for packaging
    """

    # ------------------------------------------------------------
    # Resolve and normalize paths
    # ------------------------------------------------------------
    fact_out = Path(fact_out).resolve()
    parquet_dims = Path(parquet_dims).resolve()

    fact_out.mkdir(parents=True, exist_ok=True)

    info("Sales will regenerate (forced).")

    # ------------------------------------------------------------
    # Resolve file format and output folder
    # ------------------------------------------------------------
    fmt = sales_cfg["file_format"].lower()

    if fmt == "csv":
        sales_out_folder = fact_out / "csv"
    elif fmt == "parquet":
        sales_out_folder = fact_out / "parquet"
    else:  # deltaparquet
        sales_out_folder = fact_out / "sales"

    # ------------------------------------------------------------
    # IMPORTANT: clean output folders ONLY where safe
    # ------------------------------------------------------------
    # CSV and Delta must be regenerated every run
    # Parquet must NOT be deleted before packaging
    if fmt != "parquet" and sales_out_folder.exists():
        shutil.rmtree(sales_out_folder, ignore_errors=True)

    sales_out_folder.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Validate critical config early
    # ------------------------------------------------------------
    if "skip_order_cols" not in sales_cfg:
        raise RuntimeError("sales.skip_order_cols must be explicitly defined in config")

    # Needed for PBIP selection (sales | sales_order | both)
    if "sales_output" not in sales_cfg:
        raise RuntimeError("sales.sales_output must be explicitly defined in config")

    skip_order_cols = bool(sales_cfg["skip_order_cols"])
    # ------------------------------------------------------------
    # Returns: compute "effective" enabled for THIS run
    # ------------------------------------------------------------
    returns_cfg = cfg.get("returns", {}) if isinstance(cfg, dict) else {}
    returns_enabled_requested = bool(returns_cfg.get("enabled", False))

    sales_output = str(sales_cfg["sales_output"]).strip().lower()
    returns_enabled_effective = returns_enabled_requested

    # Your rule:
    # - Generate returns for sales / sales_order / both
    # - BUT if sales_output == "sales" and skip_order_cols == True: warn + skip returns (do not fail)
    if returns_enabled_requested and sales_output == "sales" and skip_order_cols:
        info(
            "WARNING: returns.enabled=true but sales_output='sales' with skip_order_cols=true "
            "removes order identifiers. SalesReturn will be skipped. "
            "Set skip_order_cols=false or use sales_output='sales_order'/'both' to generate returns."
        )
        returns_enabled_effective = False

    # Make cfg consistent for downstream (sales.py + packaging)
    cfg_for_run = cfg
    if isinstance(cfg, dict) and (returns_enabled_effective != returns_enabled_requested):
        cfg_for_run = dict(cfg)
        rr = dict(cfg.get("returns", {}) or {})
        rr["enabled"] = returns_enabled_effective
        cfg_for_run["returns"] = rr

    # ------------------------------------------------------------
    # Load ACTIVE PRODUCTS ONLY (customers are no longer filtered here)
    # ------------------------------------------------------------
    products_path = parquet_dims / "products.parquet"

    import pandas as pd

    products_df = pd.read_parquet(
        products_path,
        columns=["ProductKey", "IsActiveInSales", "UnitPrice", "UnitCost"],
    )

    active_products_df = products_df.loc[products_df["IsActiveInSales"] == 1]

    if active_products_df.empty:
        raise RuntimeError("No active products found for sales generation")

    active_product_np = active_products_df[["ProductKey", "UnitPrice", "UnitCost"]].to_numpy()

    # ------------------------------------------------------------
    # Bind ONLY runner-level globals
    # ------------------------------------------------------------
    # NOTE:
    # - We do NOT bind active_customer_keys anymore
    # - Customer lifecycle is resolved inside sales.py
    bind_globals(
        {
            "skip_order_cols": skip_order_cols,
            "active_product_np": active_product_np,
            "models_cfg": State.models_cfg,
        }
    )

    # ------------------------------------------------------------
    # Run sales fact generation
    # ------------------------------------------------------------
    stage("Generating Sales")
    t0 = time.time()
    
    # Partitioning / delta
    partition_enabled = bool(sales_cfg.get("partition_enabled", False))

    # Only provide partition cols when enabled; otherwise force None
    partition_cols = sales_cfg.get("partition_cols")
    if not partition_enabled:
        partition_cols = None
    elif partition_cols is None:
        partition_cols = ["Year", "Month"]

    generate_sales_fact(
        cfg_for_run,
        parquet_folder=str(parquet_dims),
        out_folder=str(sales_out_folder),
        total_rows=sales_cfg["total_rows"],
        file_format=sales_cfg["file_format"],
        # Parquet merge options
        merge_parquet=sales_cfg.get("merge_parquet", False),
        merged_file=sales_cfg.get("merged_file", "sales.parquet"),
        # Performance / execution
        row_group_size=sales_cfg.get("row_group_size", 2_000_000),
        compression=sales_cfg.get("compression", "snappy"),
        chunk_size=sales_cfg.get("chunk_size", 1_000_000),
        workers=sales_cfg.get("workers"),
        # Partitioning / delta
        partition_enabled=partition_enabled,
        partition_cols=partition_cols,
        delta_output_folder=str(sales_out_folder),
        skip_order_cols=skip_order_cols,
    )

    done(f"Generating Sales completed in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------
    # Packaging (consumes Parquet output)
    # ------------------------------------------------------------
    t1 = time.time()

    final_folder = package_output(cfg_for_run, sales_cfg, parquet_dims, fact_out)

    # PBIP templates now live under:
    #   samples/powerbi/templates/{csv|parquet}/{Sales|Orders|Sales and Orders}
    # deltaparquet intentionally skips PBIP.
    maybe_attach_pbip_project(
        final_folder=final_folder,
        file_format=sales_cfg["file_format"],
        sales_output=sales_cfg["sales_output"],
    )

    done(f"Creating Final Output Folder completed in {time.time() - t1:.1f}s")
