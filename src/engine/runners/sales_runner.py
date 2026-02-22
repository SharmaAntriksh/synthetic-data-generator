from __future__ import annotations

import time
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from src.utils.logging_utils import stage, info, done
from src.engine.packaging import package_output
from src.engine.powerbi_packaging import maybe_attach_pbip_project
from src.facts.sales.sales import generate_sales_fact
from src.facts.sales.sales_logic import bind_globals, State


# =========================================================
# Helpers
# =========================================================

def _require_key(d: Dict[str, Any], key: str, ctx: str) -> Any:
    if key not in d:
        raise RuntimeError(f"Missing required config key: {ctx}.{key}")
    return d[key]


def _normalize_file_format(sales_cfg: Dict[str, Any]) -> str:
    fmt = str(_require_key(sales_cfg, "file_format", "sales")).strip().lower()
    if fmt not in {"csv", "parquet", "deltaparquet"}:
        raise RuntimeError("sales.file_format must be one of: csv | parquet | deltaparquet")
    return fmt


def _resolve_sales_out_folder(fact_out: Path, fmt: str) -> Path:
    # Preserve your existing folder conventions
    if fmt == "csv":
        return fact_out / "csv"
    if fmt == "parquet":
        return fact_out / "parquet"
    return fact_out / "sales"  # deltaparquet


def _safe_clean_folder(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _compute_returns_effective(cfg: Dict[str, Any], sales_cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    Apply your rule:
      - returns.enabled can be requested at top-level cfg
      - but if sales_output == 'sales' AND skip_order_cols == True -> warn and skip returns (do not fail)
    Returns (cfg_for_run, returns_enabled_effective)
    """
    returns_cfg = cfg.get("returns", {}) if isinstance(cfg, dict) else {}
    requested = bool(returns_cfg.get("enabled", False))

    sales_output = str(_require_key(sales_cfg, "sales_output", "sales")).strip().lower()
    skip_order_cols = bool(_require_key(sales_cfg, "skip_order_cols", "sales"))

    effective = requested
    if requested and sales_output == "sales" and skip_order_cols:
        info(
            "WARNING: returns.enabled=true but sales_output='sales' with skip_order_cols=true "
            "removes order identifiers. SalesReturn will be skipped. "
            "Set skip_order_cols=false or use sales_output='sales_order'/'both' to generate returns."
        )
        effective = False

    # If we changed effective value, copy cfg and override returns.enabled so downstream sees consistent truth
    if isinstance(cfg, dict) and (effective != requested):
        cfg_for_run = dict(cfg)
        rr = dict(cfg.get("returns", {}) or {})
        rr["enabled"] = effective
        cfg_for_run["returns"] = rr
        return cfg_for_run, effective

    return cfg, effective


def _load_active_products(parquet_dims: Path) -> Any:
    """
    Loads active products pool: (ProductKey, UnitPrice, UnitCost)
    Raises clear errors if file/columns missing or no active products.
    """
    products_path = parquet_dims / "products.parquet"
    if not products_path.exists():
        raise RuntimeError(f"Missing products parquet: {products_path}")

    wanted_cols = ["ProductKey", "IsActiveInSales", "UnitPrice", "UnitCost"]
    try:
        products_df = pd.read_parquet(products_path, columns=wanted_cols)
    except Exception as e:
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


def _normalize_partitioning(sales_cfg: Dict[str, Any]) -> Tuple[bool, Optional[list]]:
    partition_enabled = bool(sales_cfg.get("partition_enabled", False))
    if not partition_enabled:
        return False, None

    partition_cols = sales_cfg.get("partition_cols")
    if partition_cols is None:
        partition_cols = ["Year", "Month"]
    return True, partition_cols


def _run_step(label: str, fn) -> None:
    stage(label)
    t0 = time.time()
    fn()
    done(f"{label} completed in {time.time() - t0:.1f}s")


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

def run_sales_pipeline(sales_cfg, fact_out, parquet_dims, cfg, *, force_regenerate: bool = True):
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
    sales_out_folder = _resolve_sales_out_folder(fact_out_p, fmt)

    # --- Validate critical config early (same semantics, clearer errors)
    _require_key(sales_cfg, "skip_order_cols", "sales")
    _require_key(sales_cfg, "sales_output", "sales")
    _require_key(sales_cfg, "total_rows", "sales")

    skip_order_cols = bool(sales_cfg["skip_order_cols"])

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
    def _do_generate():
        generate_sales_fact(
            ctx.cfg_for_run,
            parquet_folder=str(ctx.parquet_dims),
            out_folder=str(ctx.sales_out_folder),
            total_rows=ctx.sales_cfg["total_rows"],
            file_format=ctx.sales_cfg["file_format"],
            # Parquet merge options
            merge_parquet=ctx.sales_cfg.get("merge_parquet", False),
            merged_file=ctx.sales_cfg.get("merged_file", "sales.parquet"),
            # Performance / execution
            row_group_size=ctx.sales_cfg.get("row_group_size", 2_000_000),
            compression=ctx.sales_cfg.get("compression", "snappy"),
            chunk_size=ctx.sales_cfg.get("chunk_size", 1_000_000),
            workers=ctx.sales_cfg.get("workers"),
            # Partitioning / delta
            partition_enabled=ctx.partition_enabled,
            partition_cols=ctx.partition_cols,
            delta_output_folder=str(ctx.sales_out_folder),
            skip_order_cols=ctx.skip_order_cols,
        )

    _run_step("Generating Sales", _do_generate)

    # --- Step 2: Package output + PBIP
    def _do_package():
        final_folder = package_output(ctx.cfg_for_run, ctx.sales_cfg, ctx.parquet_dims, ctx.fact_out)

        # PBIP templates now live under:
        #   samples/powerbi/templates/{csv|parquet}/{Sales|Orders|Sales and Orders}
        # deltaparquet intentionally skips PBIP.
        maybe_attach_pbip_project(
            final_folder=final_folder,
            file_format=ctx.sales_cfg["file_format"],
            sales_output=ctx.sales_cfg["sales_output"],
        )

    _run_step("Packaging Output", _do_package)

    return None