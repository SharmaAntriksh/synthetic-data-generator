"""Sales fact pipeline coordinator.

Resolves config, loads dimension data, binds worker state, and
orchestrates the multi-step sales → budget → inventory → packaging
pipeline.
"""
from __future__ import annotations

import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.exceptions import ConfigError, SalesError
from src.utils.logging_utils import stage, info
from src.engine.packaging import package_output
from src.engine.powerbi_packaging import maybe_attach_pbip_project
from src.facts.sales.sales import generate_sales_fact
from src.facts.sales.sales_logic import bind_globals, State


# =========================================================
# Helpers
# =========================================================

def _require_key(d: Dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise ConfigError(f"Missing required config key: {ctx}.{key}")
    return d[key]


def _normalize_file_format(sales_cfg) -> str:
    fmt = str(sales_cfg.file_format).strip().lower()
    if fmt not in {"csv", "parquet", "deltaparquet"}:
        raise ConfigError("sales.file_format must be one of: csv | parquet | deltaparquet")
    return fmt


def _resolve_sales_out_folder(fact_out: Path, fmt: str, *, merge_enabled: bool = False) -> Path:
    # When merge is enabled for parquet, write directly under fact_out
    # (no extra subfolder).  Chunks-only runs keep the parquet/ subfolder.
    if fmt == "csv":
        return fact_out / "csv"
    if fmt == "parquet":
        return fact_out if merge_enabled else fact_out / "parquet"
    return fact_out / "sales"  # deltaparquet


def _safe_clean_folder(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _compute_returns_effective(cfg, sales_cfg) -> Tuple[Any, bool]:
    """
    Apply your rule:
      - returns.enabled can be requested at top-level cfg
      - but if sales_output == 'sales' AND skip_order_cols == True -> warn and skip returns (do not fail)
    Returns (cfg_for_run, returns_enabled_effective)
    """
    returns_cfg = cfg.returns if hasattr(cfg, "returns") else None
    requested = bool(getattr(returns_cfg, "enabled", False)) if returns_cfg else False

    sales_output = str(sales_cfg.sales_output).strip().lower()
    skip_order_cols = bool(sales_cfg.skip_order_cols)

    effective = requested
    if requested and sales_output == "sales" and skip_order_cols:
        info(
            "WARNING: returns.enabled=true but sales_output='sales' with skip_order_cols=true "
            "removes order identifiers. SalesReturn will be skipped. "
            "Set skip_order_cols=false or use sales_output='sales_order'/'both' to generate returns."
        )
        effective = False

    # If we changed effective value, deep-copy cfg and override returns.enabled
    if effective != requested:
        cfg_for_run = cfg.model_copy(deep=True) if hasattr(cfg, "model_copy") else dict(cfg)
        if hasattr(cfg_for_run, "returns") and cfg_for_run.returns is not None:
            cfg_for_run.returns.enabled = effective
        return cfg_for_run, effective

    return cfg, effective


def _load_active_products(parquet_dims: Path) -> np.ndarray:
    """
    Loads active products pool: (ProductKey, UnitPrice, UnitCost)
    Raises clear errors if file/columns missing or no active products.
    """
    products_path = parquet_dims / "products.parquet"
    if not products_path.exists():
        raise SalesError(f"Missing products parquet: {products_path}")

    wanted_cols = ["ProductKey", "IsActiveInSales", "UnitPrice", "UnitCost"]
    try:
        products_df = pd.read_parquet(products_path, columns=wanted_cols)
    except (KeyError, ValueError, OSError) as e:
        raise RuntimeError(
            f"Failed reading {products_path} with columns {wanted_cols}. "
            f"Underlying error: {type(e).__name__}: {e}"
        ) from e

    missing = [c for c in wanted_cols if c not in products_df.columns]
    if missing:
        raise RuntimeError(f"products.parquet missing required columns: {missing}. Found: {list(products_df.columns)}")

    active = products_df.loc[products_df["IsActiveInSales"] == 1, ["ProductKey", "UnitPrice", "UnitCost"]]
    if active.empty:
        raise RuntimeError("No active products found for sales generation (IsActiveInSales == 1)")

    return active.to_numpy()


def _bind_runner_globals(skip_order_cols: bool, active_product_np: Any) -> None:
    """
    Bind ONLY runner-level globals (keep your contract).
    Guard State.models_cfg access.
    """
    models_cfg = getattr(State, "models_cfg", None)
    bind_globals(
        {
            "skip_order_cols": skip_order_cols,
            "active_product_np": active_product_np,
            "models_cfg": models_cfg,
        }
    )


def _normalize_partitioning(sales_cfg) -> Tuple[bool, Optional[list]]:
    partition_enabled = bool(sales_cfg.partition_enabled)
    if not partition_enabled:
        return False, None

    partition_cols = sales_cfg.partition_cols
    if partition_cols is None:
        partition_cols = ["Year", "Month"]
    return True, partition_cols


def _run_step(label: str, fn) -> None:
    with stage(label):
        fn()


# =========================================================
# Pipeline Context
# =========================================================

@dataclass
class SalesRunContext:
    sales_cfg: Dict[str, Any]
    cfg: Dict[str, Any]
    fact_out: Path
    parquet_dims: Path

    fmt: str
    sales_out_folder: Path
    cfg_for_run: Dict[str, Any]
    returns_enabled_effective: bool
    skip_order_cols: bool
    active_product_np: Any
    partition_enabled: bool
    partition_cols: Optional[list]


# =========================================================
# Public Runner
# =========================================================

def run_sales_pipeline(sales_cfg, fact_out, parquet_dims, cfg, *, force_regenerate: bool = True, report: bool = False):
    """
    Run the sales fact pipeline.

    Contract (preserved from current runner):
    - Customer lifecycle is fully driven by customers.parquet; this runner does NOT filter customers.
    - Product filtering (IsActiveInSales) remains valid.
    - CSV / Delta outputs are regenerated per run; Parquet output folder is preserved (not pre-deleted).
    """

    # --- Resolve paths
    fact_out_p = Path(fact_out).resolve()
    parquet_dims_p = Path(parquet_dims).resolve()
    fact_out_p.mkdir(parents=True, exist_ok=True)

    fmt = _normalize_file_format(sales_cfg)
    merge_enabled = bool(sales_cfg.merge_parquet)
    sales_out_folder = _resolve_sales_out_folder(fact_out_p, fmt, merge_enabled=merge_enabled)

    skip_order_cols = bool(sales_cfg.skip_order_cols)

    if force_regenerate:
        info("Sales will regenerate (forced).")

    # --- Clean outputs where safe (preserve your rule)
    # CSV and Delta must be regenerated every run
    # Parquet must NOT be deleted before packaging (keep existing behavior)
    if fmt != "parquet" and force_regenerate:
        _safe_clean_folder(sales_out_folder)

    sales_out_folder.mkdir(parents=True, exist_ok=True)

    # --- Returns effective config for THIS run
    cfg_for_run, returns_enabled_effective = _compute_returns_effective(cfg, sales_cfg)

    # --- Load active products pool
    active_product_np = _load_active_products(parquet_dims_p)

    # --- Bind runner-level globals
    _bind_runner_globals(skip_order_cols=skip_order_cols, active_product_np=active_product_np)

    # --- Partitioning normalization
    partition_enabled, partition_cols = _normalize_partitioning(sales_cfg)

    ctx = SalesRunContext(
        sales_cfg=sales_cfg,
        cfg=cfg,
        fact_out=fact_out_p,
        parquet_dims=parquet_dims_p,
        fmt=fmt,
        sales_out_folder=sales_out_folder,
        cfg_for_run=cfg_for_run,
        returns_enabled_effective=returns_enabled_effective,
        skip_order_cols=skip_order_cols,
        active_product_np=active_product_np,
        partition_enabled=partition_enabled,
        partition_cols=partition_cols,
    )

    # --- Step 1: Generate sales
    budget_acc = None
    inventory_acc = None
    wishlists_acc = None
    complaints_acc = None

    def _do_generate():
        nonlocal budget_acc
        nonlocal inventory_acc
        nonlocal wishlists_acc
        nonlocal complaints_acc
        result = generate_sales_fact(
            ctx.cfg_for_run,
            parquet_folder=str(ctx.parquet_dims),
            out_folder=str(ctx.sales_out_folder),
            total_rows=ctx.sales_cfg.total_rows,
            file_format=ctx.sales_cfg.file_format,
            # Parquet merge options
            merge_parquet=ctx.sales_cfg.merge_parquet,
            merged_file=ctx.sales_cfg.merged_file,
            # Performance / execution
            row_group_size=ctx.sales_cfg.row_group_size,
            compression=ctx.sales_cfg.compression,
            chunk_size=ctx.sales_cfg.chunk_size,
            workers=ctx.sales_cfg.workers,
            # Partitioning / delta
            partition_enabled=ctx.partition_enabled,
            partition_cols=ctx.partition_cols,
            delta_output_folder=str(ctx.sales_out_folder),
            skip_order_cols=ctx.skip_order_cols,
            return_manifest=True,
        )
        if isinstance(result, tuple):
            # result layout: (chunk_files, row_count, budget_acc?, inventory_acc?, wishlists_acc?, complaints_acc?)
            if len(result) >= 6:
                _chunk_files, _row_count, budget_acc, inventory_acc, wishlists_acc, complaints_acc = result[:6]
            elif len(result) >= 5:
                _chunk_files, _row_count, budget_acc, inventory_acc, wishlists_acc = result[:5]
            elif len(result) >= 4:
                _chunk_files, _row_count, budget_acc, inventory_acc = result[:4]
            elif len(result) >= 3:
                _chunk_files, _row_count, budget_acc = result[:3]

    _run_step("Generating Sales", _do_generate)

    # --- Step 2: Budget generation (optional, must run before packaging)
    if budget_acc is not None and budget_acc.has_data:
        from src.facts.budget.runner import run_budget_pipeline

        def _do_budget():
            run_budget_pipeline(
                accumulator=budget_acc,
                parquet_dims=ctx.parquet_dims,
                fact_out=ctx.fact_out,
                cfg=ctx.cfg,
                file_format=ctx.fmt,
            )

        _run_step("Generating Budget", _do_budget)

    # --- Step 2b: Inventory snapshot generation (optional, must run before packaging)
    if inventory_acc is not None and inventory_acc.has_data:
        from src.facts.inventory.runner import run_inventory_pipeline

        def _do_inventory():
            run_inventory_pipeline(
                accumulator=inventory_acc,
                parquet_dims=ctx.parquet_dims,
                fact_out=ctx.fact_out,
                cfg=ctx.cfg,
                file_format=ctx.fmt,
                workers=ctx.sales_cfg.workers,
            )

        _run_step("Generating Inventory Snapshots", _do_inventory)

    # --- Step 2c: Wishlists (optional, runs after sales to use purchase data)
    if wishlists_acc is not None and wishlists_acc.has_data:
        from src.facts.wishlists.runner import run_wishlist_pipeline

        def _do_wishlists():
            run_wishlist_pipeline(
                accumulator=wishlists_acc,
                parquet_dims=ctx.parquet_dims,
                cfg=ctx.cfg,
                file_format=ctx.fmt,
            )

        _run_step("Generating Customer Wishlists", _do_wishlists)

    # --- Step 2d: Complaints (optional, runs after sales to use order data)
    if complaints_acc is not None and complaints_acc.has_data:
        from src.facts.complaints.runner import run_complaints_pipeline

        def _do_complaints():
            run_complaints_pipeline(
                accumulator=complaints_acc,
                parquet_dims=ctx.parquet_dims,
                fact_out=ctx.fact_out,
                cfg=ctx.cfg,
                file_format=ctx.fmt,
            )

        _run_step("Generating Fact Complaints", _do_complaints)

    # --- Step 3: Package output + PBIP
    final_folder_result = None

    def _do_package():
        nonlocal final_folder_result
        final_folder_result = package_output(ctx.cfg_for_run, ctx.sales_cfg, ctx.parquet_dims, ctx.fact_out)

        # PBIP templates now live under:
        #   samples/powerbi/templates/{csv|parquet}/{Sales|Orders|Sales and Orders}
        # deltaparquet intentionally skips PBIP.
        maybe_attach_pbip_project(
            final_folder=final_folder_result,
            file_format=ctx.sales_cfg.file_format,
            sales_output=ctx.sales_cfg.sales_output,
        )

    _run_step("Packaging Output", _do_package)

    # --- Step 4: Quality report (optional)
    if report and final_folder_result is not None:
        from src.engine.quality_report import generate_quality_report

        def _do_report():
            generate_quality_report(final_folder_result, cfg=ctx.cfg)

        _run_step("Quality Report", _do_report)

    return None