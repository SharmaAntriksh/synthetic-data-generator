from __future__ import annotations

import time
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pyarrow.parquet as pq

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


def _require_keys(d: Dict[str, Any], keys: tuple[str, ...], ctx: str) -> None:
    missing = [k for k in keys if k not in d]
    if missing:
        raise RuntimeError(f"Missing required config keys: {', '.join(f'{ctx}.{k}' for k in missing)}")


def _normalize_file_format(sales_cfg: Dict[str, Any]) -> str:
    fmt = str(_require_key(sales_cfg, "file_format", "sales")).strip().lower()
    if fmt not in {"csv", "parquet", "deltaparquet"}:
        raise RuntimeError("sales.file_format must be one of: csv | parquet | deltaparquet")
    return fmt


_SALES_OUT_FOLDERS = {"csv": "csv", "parquet": "parquet"}


def _resolve_sales_out_folder(fact_out: Path, fmt: str) -> Path:
    return fact_out / _SALES_OUT_FOLDERS.get(fmt, "sales")


def _safe_clean_folder(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _compute_returns_effective(cfg: Dict[str, Any], sales_cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """
    Apply the rule:
      - returns.enabled can be requested at top-level cfg
      - but if sales_output == 'sales' AND skip_order_cols == True -> warn and skip returns
    Returns (cfg_for_run, returns_enabled_effective)
    """
    returns_cfg = cfg.get("returns", {}) if isinstance(cfg, dict) else {}
    requested = bool(returns_cfg.get("enabled", False))

    sales_output = str(sales_cfg["sales_output"]).strip().lower()
    skip_order_cols = bool(sales_cfg["skip_order_cols"])

    effective = requested
    if requested and sales_output == "sales" and skip_order_cols:
        info(
            "WARNING: returns.enabled=true but sales_output='sales' with skip_order_cols=true "
            "removes order identifiers. SalesReturn will be skipped. "
            "Set skip_order_cols=false or use sales_output='sales_order'/'both' to generate returns."
        )
        effective = False

    if effective == requested:
        return cfg, effective

    cfg_for_run = {**cfg, "returns": {**(cfg.get("returns") or {}), "enabled": effective}}
    return cfg_for_run, effective


def _load_active_products(parquet_dims: Path) -> Any:
    """
    Loads active products pool: (ProductKey, UnitPrice, UnitCost)
    Uses PyArrow directly to avoid pandas overhead.
    """
    products_path = parquet_dims / "products.parquet"
    if not products_path.exists():
        raise RuntimeError(f"Missing products parquet: {products_path}")

    wanted_cols = ["ProductKey", "IsActiveInSales", "UnitPrice", "UnitCost"]
    try:
        table = pq.read_table(str(products_path), columns=wanted_cols)
    except Exception as e:
        raise RuntimeError(
            f"Failed reading {products_path} with columns {wanted_cols}. "
            f"Underlying error: {type(e).__name__}: {e}"
        ) from e

    missing = [c for c in wanted_cols if c not in table.schema.names]
    if missing:
        raise RuntimeError(f"products.parquet missing required columns: {missing}. Found: {table.schema.names}")

    import pyarrow.compute as pc
    mask = pc.equal(table.column("IsActiveInSales"), 1)
    active = table.filter(mask).select(["ProductKey", "UnitPrice", "UnitCost"])

    if active.num_rows == 0:
        raise RuntimeError("No active products found for sales generation (IsActiveInSales == 1)")

    return active.to_pandas().to_numpy()


def _bind_runner_globals(skip_order_cols: bool, active_product_np: Any) -> None:
    """Bind runner-level globals (keep the contract)."""
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


def _run_step(label: str, fn) -> Any:
    stage(label)
    t0 = time.perf_counter()
    result = fn()
    done(f"{label} completed in {time.perf_counter() - t0:.1f}s")
    return result


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

    # --- Validate critical config early
    _require_keys(sales_cfg, ("skip_order_cols", "sales_output", "total_rows"), "sales")

    skip_order_cols = bool(sales_cfg["skip_order_cols"])

    if force_regenerate:
        info("Sales will regenerate (forced).")

    # --- Clean outputs where safe
    # CSV and Delta must be regenerated every run; Parquet must NOT be deleted before packaging
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

    # --- Step 1: Generate sales
    def _do_generate():
        return generate_sales_fact(
            cfg_for_run,
            parquet_folder=str(parquet_dims_p),
            out_folder=str(sales_out_folder),
            total_rows=sales_cfg["total_rows"],
            file_format=sales_cfg["file_format"],
            merge_parquet=sales_cfg.get("merge_parquet", False),
            merged_file=sales_cfg.get("merged_file", "sales.parquet"),
            row_group_size=sales_cfg.get("row_group_size", 2_000_000),
            compression=sales_cfg.get("compression", "snappy"),
            chunk_size=sales_cfg.get("chunk_size", 1_000_000),
            workers=sales_cfg.get("workers"),
            partition_enabled=partition_enabled,
            partition_cols=partition_cols,
            delta_output_folder=str(sales_out_folder),
            skip_order_cols=skip_order_cols,
            return_manifest=True,
        )

    result = _run_step("Generating Sales", _do_generate)

    budget_acc = None
    if isinstance(result, tuple) and len(result) >= 3:
        _, _, budget_acc = result

    # --- Step 2: Budget generation (optional, must run before packaging)
    if budget_acc is not None and budget_acc.has_data:
        from src.facts.budget.runner import run_budget_pipeline

        _run_step(
            "Generating Budget",
            lambda: run_budget_pipeline(
                accumulator=budget_acc,
                parquet_dims=parquet_dims_p,
                fact_out=fact_out_p,
                cfg=cfg,
                file_format=fmt,
            ),
        )

    # --- Step 3: Package output + PBIP
    def _do_package():
        final_folder = package_output(cfg_for_run, sales_cfg, parquet_dims_p, fact_out_p)
        maybe_attach_pbip_project(
            final_folder=final_folder,
            file_format=sales_cfg["file_format"],
            sales_output=sales_cfg["sales_output"],
        )

    _run_step("Packaging Output", _do_package)

    return None
