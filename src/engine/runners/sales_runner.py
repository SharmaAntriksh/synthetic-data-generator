from __future__ import annotations

import time
import shutil
from pathlib import Path
import pandas as pd

from src.utils.logging_utils import stage, info, done
from src.engine.packaging import package_output
from src.facts.sales.sales_logic.globals import bind_globals
from src.engine.powerbi_packaging import attach_pbip_project


def run_sales_pipeline(sales_cfg, fact_out, parquet_dims, cfg):
    """
    Run the sales fact pipeline.

    Invariants:
    - Sales schema is determined entirely by sales_cfg
    - No implicit defaults override config
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
        raise RuntimeError(
            "sales.skip_order_cols must be explicitly defined in config"
        )

    skip_order_cols = bool(sales_cfg["skip_order_cols"])

    # ------------------------------------------------------------
    # Load active Customers for Sales generation
    # ------------------------------------------------------------

    customers_path = parquet_dims / "customers.parquet"

    customers_df = pd.read_parquet(
        customers_path,
        columns=["CustomerKey", "IsActiveInSales"],
    )

    active_customer_keys = customers_df.loc[
        customers_df["IsActiveInSales"],
        "CustomerKey",
    ].to_numpy()

    if len(active_customer_keys) == 0:
        raise RuntimeError("No active customers found for sales generation")

    # ------------------------------------------------------------
    # Run sales fact generation
    # ------------------------------------------------------------
    from src.facts.sales.sales import generate_sales_fact

    stage("Generating Sales")
    t0 = time.time()

    # Bind only runner-level globals
    bind_globals({
        "skip_order_cols": skip_order_cols,
        "active_customer_keys": active_customer_keys,
    })

    generate_sales_fact(
        cfg,
        parquet_folder=str(parquet_dims),
        out_folder=str(sales_out_folder),
        total_rows=sales_cfg["total_rows"],
        file_format=sales_cfg["file_format"],

        # REQUIRED FOR PARQUET MODE
        merge_parquet=sales_cfg.get("merge_parquet", False),
        merged_file=sales_cfg.get("merged_file", "sales.parquet"),

        # existing args
        row_group_size=sales_cfg.get("row_group_size", 2_000_000),
        compression=sales_cfg.get("compression", "snappy"),
        chunk_size=sales_cfg.get("chunk_size", 1_000_000),
        workers=sales_cfg.get("workers"),
        partition_enabled=sales_cfg.get("partition_enabled", False),
        partition_cols=sales_cfg.get("partition_cols", ["Year", "Month"]),
        delta_output_folder=str(sales_out_folder),
        skip_order_cols=skip_order_cols,
    )

    done(f"Generating Sales completed in {time.time() - t0:.1f}s")

    # ------------------------------------------------------------
    # Packaging (consumes Parquet output)
    # ------------------------------------------------------------
    t1 = time.time()
    
    final_folder = package_output(cfg, sales_cfg, parquet_dims, fact_out)

    fmt = sales_cfg["file_format"].lower()

    pbip_template = None

    if fmt == "csv":
        pbip_template = Path("samples/powerbi/templates/PBIP CSV")

    elif fmt == "parquet":
        pbip_template = Path("samples/powerbi/templates/PBIP Parquet")

    # deltaparquet → intentionally skip PBIP

    if pbip_template is not None:
        attach_pbip_project(
            final_folder=final_folder,
            pbip_template_root=pbip_template,
        )

    done(f"Creating Final Output Folder completed in {time.time() - t1:.1f}s")
