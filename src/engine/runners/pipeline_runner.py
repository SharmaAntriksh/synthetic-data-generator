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
    - Clears scratch fact output at end

    Returns a small run summary dict. Raises on failures.
    """
    if only not in (None, "dimensions", "sales"):
        raise ValueError("only must be one of: None, 'dimensions', 'sales'")

    overrides = overrides or PipelineOverrides()
    force_regenerate: Set[str] = set(regen_dimensions) if regen_dimensions else set()

    # Normalize format alias
    file_format = overrides.file_format
    if file_format is not None and str(file_format).strip().lower() == "delta":
        file_format = "deltaparquet"
    overrides = PipelineOverrides(**{**overrides.__dict__, "file_format": file_format})

    start_ts = time.time()

    try:
        # ----------------------------
        # Load configs
        # ----------------------------
        cfg = load_config(config_path)
        if "sales" not in cfg or not isinstance(cfg["sales"], dict):
            fail("Config must contain a 'sales' section")
            raise RuntimeError("Missing 'sales' section")

        sales_cfg: Dict[str, Any] = cfg["sales"]

        models_raw = load_config_file(models_config_path)
        if "models" not in models_raw or not isinstance(models_raw["models"], dict):
            fail("models.yaml must contain a top-level 'models' section")
            raise RuntimeError("Missing top-level 'models' section")
        models_cfg: Dict[str, Any] = models_raw["models"]

        # Attach run spec paths (useful for downstream packaging/metadata)
        cfg["config_yaml_path"] = str(_resolve_input_path(config_path))
        cfg["model_yaml_path"] = str(_resolve_input_path(models_config_path))
        info(f"Attached run spec paths: config={cfg['config_yaml_path']} model={cfg['model_yaml_path']}")

        # ----------------------------
        # Apply overrides (sales)
        # ----------------------------
        if overrides.file_format:
            sales_cfg["file_format"] = str(overrides.file_format).lower()

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

        # ----------------------------
        # Apply overrides (global dates)
        # ----------------------------
        _ensure_defaults_dates(cfg)
        if overrides.start_date:
            cfg["defaults"]["dates"]["start"] = overrides.start_date
        if overrides.end_date:
            cfg["defaults"]["dates"]["end"] = overrides.end_date

        # ----------------------------
        # FX always follows global dates
        # ----------------------------
        if "exchange_rates" in cfg and isinstance(cfg["exchange_rates"], dict):
            fx_cfg = cfg["exchange_rates"]
            fx_cfg["use_global_dates"] = True
            # Remove any local override dates to ensure it uses injected global dates
            fx_cfg.pop("dates", None)

        # ----------------------------
        # Dimension size overrides
        # ----------------------------
        if overrides.customers is not None:
            cfg.setdefault("customers", {})
            cfg["customers"]["total_customers"] = int(overrides.customers)

        if overrides.stores is not None:
            cfg.setdefault("stores", {})
            n = int(overrides.stores)
            # Preferred key (matches config.yaml)
            cfg["stores"]["num_stores"] = n
            # Back-compat alias (some code/older configs may look here)
            cfg["stores"]["total_stores"] = n

        if overrides.products is not None:
            cfg.setdefault("products", {})
            cfg["products"]["num_products"] = int(overrides.products)

        if overrides.promotions is not None:
            cfg.setdefault("promotions", {})
            _apply_promotions_total(cfg["promotions"], int(overrides.promotions))

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
                "elapsed_sec": time.time() - start_ts,
            }

        # ----------------------------
        # Attach models config to runtime state
        # ----------------------------
        from src.facts.sales.sales_logic import State
        State.models_cfg = models_cfg

        # ----------------------------
        # Optional clean (final output root)
        # ----------------------------
        if clean:
            info("Cleaning final output folders before run.")
            # CLI used generated_datasets_root, while config.yaml uses final_output_folder.
            gen_root = (
                cfg.get("generated_datasets_root")
                or cfg.get("final_output_folder")
                or cfg.get("final_output_root")
            )
            if gen_root:
                shutil.rmtree(gen_root, ignore_errors=True)

        # ----------------------------
        # Resolve required paths
        # ----------------------------
        if "parquet_folder" not in sales_cfg or "out_folder" not in sales_cfg:
            fail("sales.parquet_folder and sales.out_folder must be set in config")
            raise RuntimeError("Missing sales.parquet_folder/out_folder")

        parquet_dims = Path(sales_cfg["parquet_folder"]).resolve()
        fact_out = Path(sales_cfg["out_folder"]).resolve()

        parquet_dims.mkdir(parents=True, exist_ok=True)
        fact_out.mkdir(parents=True, exist_ok=True)

        # Hard reset scratch fact output (as per CLI behavior) — configurable
        packaging_cfg = cfg.get("packaging", {}) if isinstance(cfg, dict) else {}
        reset_scratch = bool(packaging_cfg.get("reset_scratch_fact_out", True))

        if reset_scratch:
            info(f"Resetting fact output folder: {fact_out}")
            if fact_out.exists():
                shutil.rmtree(fact_out, ignore_errors=True)
            fact_out.mkdir(parents=True, exist_ok=True)
        else:
            info(f"Keeping existing fact_out folder (packaging.reset_scratch_fact_out=false): {fact_out}")
            fact_out.mkdir(parents=True, exist_ok=True)

        # ----------------------------
        # Run pipelines
        # ----------------------------
        info("Starting full pipeline.")
        if only != "sales":
            generate_dimensions(cfg, parquet_dims, force_regenerate=force_regenerate)

        if only != "dimensions":
            run_sales_pipeline(sales_cfg, fact_out, parquet_dims, cfg)

        # ----------------------------
        # Final cleanup (scratch) — configurable
        # ----------------------------
        packaging_cfg = cfg.get("packaging", {}) if isinstance(cfg, dict) else {}
        clean_scratch = bool(packaging_cfg.get("clean_scratch_fact_out", True))

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
            "elapsed_sec": elapsed,
        }

    except Exception as ex:
        # Mirror CLI behavior: log and re-raise
        fail(str(ex))
        raise


# ----------------------------
# Internals
# ----------------------------

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

    repo_root = Path(__file__).resolve().parents[2]  # ./src/engine/runners -> repo root
    repo_candidate = (repo_root / raw).resolve()
    return repo_candidate  # may or may not exist; caller can decide


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
            # default split if current is invalid
            base = [1, 1, 1]
            current = 3
        else:
            base = [int(promotions_cfg[k]) for k in keys]

        # Scale and distribute rounding remainder deterministically
        scaled = [b * total / current for b in base]
        floors = [int(x) for x in scaled]
        remainder = total - sum(floors)

        # Add remainder to largest fractional parts first
        fracs = [(i, scaled[i] - floors[i]) for i in range(len(keys))]
        fracs.sort(key=lambda t: t[1], reverse=True)

        for i in range(remainder):
            floors[fracs[i % len(keys)][0]] += 1

        promotions_cfg["num_seasonal"] = floors[0]
        promotions_cfg["num_clearance"] = floors[1]
        promotions_cfg["num_limited"] = floors[2]

        # Back-compat
        promotions_cfg["total_promotions"] = total
    else:
        promotions_cfg["total_promotions"] = total
