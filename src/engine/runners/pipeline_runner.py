from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Any, Dict, Iterable, Optional, Set, Tuple

from src.engine.config.config_loader import load_config, load_config_file
from src.engine.runners.dimensions_runner import generate_dimensions
from src.engine.runners.sales_runner import run_sales_pipeline
from src.utils.logging_utils import info, fail, fmt_sec


# ----------------------------
# Public API
# ----------------------------

@dataclass(frozen=True)
class PipelineOverrides:
    # Sales / output
    file_format: Optional[str] = None           # csv | parquet | deltaparquet | delta
    sales_rows: Optional[int] = None            # sales.total_rows
    workers: Optional[int] = None               # sales.workers
    chunk_size: Optional[int] = None            # sales.chunk_size
    skip_order_cols: Optional[bool] = None      # sales.skip_order_cols
    row_group_size: Optional[int] = None        # sales.row_group_size (parquet/deltaparquet only)

    # Global dates (defaults.dates)
    start_date: Optional[str] = None            # YYYY-MM-DD
    end_date: Optional[str] = None              # YYYY-MM-DD

    # Dimension sizes
    customers: Optional[int] = None             # customers.total_customers
    stores: Optional[int] = None                # stores.num_stores
    products: Optional[int] = None              # products.num_products
    promotions: Optional[int] = None            # “total” promotions, distributed across buckets when possible


def run_pipeline(
    *,
    config_path: str = "config.yaml",
    models_config_path: str = "models.yaml",
    only: Optional[str] = None,                 # None | "dimensions" | "sales"
    clean: bool = False,
    dry_run: bool = False,
    regen_dimensions: Optional[Iterable[str]] = None,
    overrides: Optional[PipelineOverrides] = None,
) -> Dict[str, Any]:
    """
    Shared pipeline runner (CLI + Streamlit UI friendly).

    - Loads config + models config
    - Applies overrides (sales, global dates, dimension sizes)
    - Forces FX dimension to follow global dates
    - Runs dimensions and/or sales pipelines
    - Optionally resets scratch fact output (packaging.reset_scratch_fact_out)
    - Optionally cleans scratch fact output at end (packaging.clean_scratch_fact_out)

    Returns a small run summary dict. Raises on failures.
    """
    if only not in (None, "dimensions", "sales"):
        raise ValueError("only must be one of: None, 'dimensions', 'sales'")

    overrides = _normalize_overrides(overrides or PipelineOverrides())
    force_regenerate: Set[str] = set(regen_dimensions) if regen_dimensions else set()

    start_ts = time.time()

    try:
        # ----------------------------
        # Load configs
        # ----------------------------
        cfg_raw = load_config(config_path)
        if "sales" not in cfg_raw or not isinstance(cfg_raw["sales"], dict):
            fail("Config must contain a 'sales' section")
            raise RuntimeError("Missing 'sales' section")

        models_raw = load_config_file(models_config_path)
        if "models" not in models_raw or not isinstance(models_raw["models"], dict):
            fail("models.yaml must contain a top-level 'models' section")
            raise RuntimeError("Missing top-level 'models' section")
        models_cfg: Dict[str, Any] = models_raw["models"]

        # Work on a shallow-copied cfg so repeated Streamlit runs remain sane
        cfg = dict(cfg_raw)
        cfg["sales"] = dict(cfg_raw["sales"])
        sales_cfg: Dict[str, Any] = cfg["sales"]

        # Attach run spec paths (useful for downstream packaging/metadata)
        cfg["config_yaml_path"] = str(_resolve_input_path(config_path))
        cfg["model_yaml_path"] = str(_resolve_input_path(models_config_path))
        info(f"Attached run spec paths: config={cfg['config_yaml_path']} model={cfg['model_yaml_path']}")

        # ----------------------------
        # Apply overrides (copy-on-write)
        # ----------------------------
        cfg, sales_cfg = _apply_overrides(cfg, sales_cfg, overrides)

        # FX always follows global dates
        cfg = _force_fx_to_global_dates(cfg)

        # ----------------------------
        # Dry run
        # ----------------------------
        if dry_run:
            info("Dry run enabled. Resolved configuration:")
            pprint(cfg)
            return {
                "ok": True,
                "dry_run": True,
                "only": only,
                "force_regenerate": sorted(force_regenerate),
                "config_yaml_path": cfg.get("config_yaml_path"),
                "model_yaml_path": cfg.get("model_yaml_path"),
                "elapsed_sec": time.time() - start_ts,
            }

        # ----------------------------
        # Optional clean (final output root)
        # ----------------------------
        if clean:
            _clean_final_outputs(cfg)

        # ----------------------------
        # Resolve required paths
        # ----------------------------
        parquet_dims, fact_out = _resolve_required_paths(sales_cfg)

        parquet_dims.mkdir(parents=True, exist_ok=True)
        fact_out.mkdir(parents=True, exist_ok=True)

        # ----------------------------
        # Scratch output handling (reset/cleanup)
        # ----------------------------
        packaging_cfg = cfg.get("packaging", {}) if isinstance(cfg, dict) else {}
        reset_scratch = bool(packaging_cfg.get("reset_scratch_fact_out", True))
        clean_scratch = bool(packaging_cfg.get("clean_scratch_fact_out", True))

        if reset_scratch:
            info(f"Resetting fact output folder: {fact_out}")
            if fact_out.exists():
                shutil.rmtree(fact_out, ignore_errors=True)
            fact_out.mkdir(parents=True, exist_ok=True)
        else:
            info(f"Keeping existing fact_out folder (packaging.reset_scratch_fact_out=false): {fact_out}")
            fact_out.mkdir(parents=True, exist_ok=True)

        # ----------------------------
        # Attach models config to runtime state (ONLY if sales will run)
        # ----------------------------
        if only != "dimensions":
            from src.facts.sales.sales_logic import State
            State.models_cfg = models_cfg

        # ----------------------------
        # Run pipelines
        # ----------------------------
        info("Starting full pipeline.")
        dim_summary = None

        if only != "sales":
            dim_summary = generate_dimensions(cfg, parquet_dims, force_regenerate=force_regenerate)

        if only != "dimensions":
            run_sales_pipeline(sales_cfg, fact_out, parquet_dims, cfg)

        # ----------------------------
        # Final cleanup (scratch)
        # ----------------------------
        if clean_scratch:
            info(f"Cleaning scratch fact_out folder: {fact_out}")
            shutil.rmtree(fact_out, ignore_errors=True)
        else:
            info(f"Keeping scratch fact_out folder (packaging.clean_scratch_fact_out=false): {fact_out}")

        elapsed = time.time() - start_ts
        info(f"All pipelines completed in {fmt_sec(elapsed)}.")

        return {
            "ok": True,
            "dry_run": False,
            "only": only,
            "force_regenerate": sorted(force_regenerate),
            "parquet_dims": str(parquet_dims),
            "fact_out_scratch": str(fact_out),
            "config_yaml_path": cfg.get("config_yaml_path"),
            "model_yaml_path": cfg.get("model_yaml_path"),
            "reset_scratch_fact_out": reset_scratch,
            "clean_scratch_fact_out": clean_scratch,
            "dimensions": dim_summary,
            "elapsed_sec": elapsed,
        }

    except Exception as ex:
        fail(str(ex))
        raise


# ----------------------------
# Internals
# ----------------------------

def _normalize_overrides(overrides: PipelineOverrides) -> PipelineOverrides:
    # Normalize format alias
    ff = overrides.file_format
    if ff is not None and str(ff).strip().lower() == "delta":
        ff = "deltaparquet"
    if ff is not None:
        ff = str(ff).strip().lower()
    return PipelineOverrides(**{**overrides.__dict__, "file_format": ff})


def _apply_overrides(
    cfg: Dict[str, Any],
    sales_cfg: Dict[str, Any],
    overrides: PipelineOverrides,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Apply overrides in a copy-on-write style:
      - cfg and cfg['sales'] are already shallow-copied by caller
      - this function only mutates the copies
    """
    # Sales overrides
    if overrides.file_format:
        sales_cfg["file_format"] = overrides.file_format

    if overrides.sales_rows is not None:
        sales_cfg["total_rows"] = int(overrides.sales_rows)

    if overrides.workers is not None:
        sales_cfg["workers"] = int(overrides.workers)

    if overrides.chunk_size is not None:
        sales_cfg["chunk_size"] = int(overrides.chunk_size)

    if overrides.skip_order_cols is not None:
        sales_cfg["skip_order_cols"] = bool(overrides.skip_order_cols)

    if overrides.row_group_size is not None:
        fmt = str(sales_cfg.get("file_format", "")).lower()
        if fmt not in ("parquet", "deltaparquet"):
            fail("--row-group-size is only valid for parquet or deltaparquet output")
            raise ValueError("row_group_size only valid for parquet/deltaparquet")
        sales_cfg["row_group_size"] = int(overrides.row_group_size)

    # Global dates overrides
    _ensure_defaults_dates(cfg)
    if overrides.start_date:
        cfg["defaults"]["dates"]["start"] = overrides.start_date
    if overrides.end_date:
        cfg["defaults"]["dates"]["end"] = overrides.end_date

    # Dimension size overrides
    if overrides.customers is not None:
        cfg["customers"] = dict(cfg.get("customers", {}) or {})
        cfg["customers"]["total_customers"] = int(overrides.customers)

    if overrides.stores is not None:
        cfg["stores"] = dict(cfg.get("stores", {}) or {})
        n = int(overrides.stores)
        cfg["stores"]["num_stores"] = n
        cfg["stores"]["total_stores"] = n  # back-compat

    if overrides.products is not None:
        cfg["products"] = dict(cfg.get("products", {}) or {})
        cfg["products"]["num_products"] = int(overrides.products)

    if overrides.promotions is not None:
        cfg["promotions"] = dict(cfg.get("promotions", {}) or {})
        _apply_promotions_total(cfg["promotions"], int(overrides.promotions))

    return cfg, sales_cfg


def _force_fx_to_global_dates(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure exchange_rates follows injected global dates.
    """
    if "exchange_rates" in cfg and isinstance(cfg["exchange_rates"], dict):
        fx_cfg = dict(cfg["exchange_rates"])
        fx_cfg["use_global_dates"] = True
        fx_cfg.pop("dates", None)
        cfg["exchange_rates"] = fx_cfg
    return cfg


def _clean_final_outputs(cfg: Dict[str, Any]) -> None:
    info("Cleaning final output folders before run.")
    gen_root = (
        cfg.get("generated_datasets_root")
        or cfg.get("final_output_folder")
        or cfg.get("final_output_root")
    )
    if gen_root:
        shutil.rmtree(gen_root, ignore_errors=True)


def _resolve_required_paths(sales_cfg: Dict[str, Any]) -> Tuple[Path, Path]:
    if "parquet_folder" not in sales_cfg or "out_folder" not in sales_cfg:
        fail("sales.parquet_folder and sales.out_folder must be set in config")
        raise RuntimeError("Missing sales.parquet_folder/out_folder")

    parquet_dims = Path(sales_cfg["parquet_folder"]).resolve()
    fact_out = Path(sales_cfg["out_folder"]).resolve()
    return parquet_dims, fact_out


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

    # Find repo root by locating ".../src" in this file path and taking its parent
    here = Path(__file__).resolve()
    src_dir = None
    for parent in here.parents:
        if parent.name == "src":
            src_dir = parent
            break
    repo_root = src_dir.parent if src_dir is not None else here.parents[3]  # fallback

    return (repo_root / raw).resolve()


def _ensure_defaults_dates(cfg: dict) -> None:
    """
    Ensure cfg has canonical defaults.dates dict for overrides.
    Safety guard so overrides never write into _defaults.
    """
    cfg.setdefault("defaults", {})
    cfg["defaults"].setdefault("dates", {})
    cfg["defaults"]["dates"].setdefault("start", None)
    cfg["defaults"]["dates"].setdefault("end", None)


def _apply_promotions_total(promotions_cfg: Dict[str, Any], total: int) -> None:
    """
    Config has three promotion buckets:
      - num_seasonal
      - num_clearance
      - num_limited

    If those exist, scale them proportionally to match `total`.
    Otherwise, store a back-compat 'total_promotions' key.
    """
    total = max(0, int(total))

    keys = ("num_seasonal", "num_clearance", "num_limited")
    if all(k in promotions_cfg and isinstance(promotions_cfg[k], (int, float)) for k in keys):
        current = sum(int(promotions_cfg[k]) for k in keys)
        if current <= 0:
            base = [1, 1, 1]
            current = 3
        else:
            base = [int(promotions_cfg[k]) for k in keys]

        scaled = [b * total / current for b in base]
        floors = [int(x) for x in scaled]
        remainder = total - sum(floors)

        fracs = [(i, scaled[i] - floors[i]) for i in range(len(keys))]
        fracs.sort(key=lambda t: t[1], reverse=True)

        for i in range(remainder):
            floors[fracs[i % len(keys)][0]] += 1

        promotions_cfg["num_seasonal"] = floors[0]
        promotions_cfg["num_clearance"] = floors[1]
        promotions_cfg["num_limited"] = floors[2]
        promotions_cfg["total_promotions"] = total
    else:
        promotions_cfg["total_promotions"] = total