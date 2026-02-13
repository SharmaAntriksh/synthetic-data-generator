from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import pandas as pd


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def load_config(path: str | Path = "config.yaml") -> dict:
    """
    Load a config file (YAML preferred, JSON supported) and normalize safely.

    Goals:
    - Support config.yaml / config.yml / config.json
    - Canonicalize defaults vs _defaults (both supported)
    - Validate defaults.dates.start/end
    - Normalize sales config without deleting user keys
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    cfg = _load_any(path)

    # Canonicalize defaults / _defaults and validate timeline
    cfg = normalize_defaults(cfg)

    # Normalize sections (only those we can validate safely)
    if "sales" in cfg:
        if not isinstance(cfg["sales"], dict):
            raise KeyError("Invalid 'sales' section in config (expected mapping)")
        cfg["sales"] = normalize_sales_config(cfg["sales"])

    # Customers section is validated in customers dimension generator,
    # but we keep a light sanity pass for common keys.
    if "customers" in cfg and not isinstance(cfg["customers"], dict):
        raise KeyError("Invalid 'customers' section in config (expected mapping)")

    return cfg


# ------------------------------------------------------------
# Loaders
# ------------------------------------------------------------

def _load_any(path: Path) -> dict:
    ext = path.suffix.lower()

    if ext in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except Exception as e:
            raise RuntimeError("PyYAML is required to load .yaml/.yml config files") from e

        with path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if not isinstance(cfg, dict):
            raise ValueError("Top-level YAML config must be a mapping/object")
        return cfg

    if ext == ".json":
        with path.open("r", encoding="utf-8") as f:
            cfg = json.load(f) or {}
        if not isinstance(cfg, dict):
            raise ValueError("Top-level JSON config must be an object")
        return cfg

    # If user passes no extension, attempt YAML then JSON
    # (keep this permissive for CLI convenience)
    with path.open("r", encoding="utf-8") as f:
        raw = f.read()

    # Try YAML first
    try:
        import yaml  # type: ignore
        cfg = yaml.safe_load(raw) or {}
        if isinstance(cfg, dict):
            return cfg
    except Exception:
        pass

    # Try JSON
    try:
        cfg = json.loads(raw) or {}
        if isinstance(cfg, dict):
            return cfg
    except Exception as e:
        raise ValueError(f"Unable to parse config file as YAML or JSON: {path}") from e

    raise ValueError(f"Unsupported or invalid config format: {path}")


# ------------------------------------------------------------
# Defaults normalization (dates)
# ------------------------------------------------------------

def normalize_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Canonicalize defaults:
      - Accept 'defaults' or '_defaults'
      - Ensure cfg['defaults'] exists (canonical)
      - Validate defaults.dates.start/end exist and are parseable
    """
    cfg = dict(cfg)

    # Accept either defaults or _defaults
    defaults_section = cfg.get("defaults")
    underscore_defaults = cfg.get("_defaults")

    if defaults_section is None and underscore_defaults is not None:
        cfg["defaults"] = underscore_defaults
        defaults_section = cfg["defaults"]

    if defaults_section is None:
        raise KeyError("Missing 'defaults' (or '_defaults') section in config")

    if not isinstance(defaults_section, dict):
        raise KeyError("Invalid defaults section in config (expected mapping)")

    dates = defaults_section.get("dates")
    if not isinstance(dates, dict):
        raise KeyError("Missing or invalid defaults.dates section in config")

    start = dates.get("start")
    end = dates.get("end")
    if not start or not end:
        raise KeyError("Missing defaults.dates.start or defaults.dates.end in config")

    # Validate dates are parseable and sane
    try:
        start_ts = pd.to_datetime(start).normalize()
        end_ts = pd.to_datetime(end).normalize()
    except Exception as e:
        raise ValueError("Missing or invalid defaults.dates.start/end in config.yaml") from e

    if pd.isna(start_ts) or pd.isna(end_ts) or start_ts >= end_ts:
        raise ValueError("Invalid defaults.dates range (start must be < end)")

    # Store canonical string values (keep user's format stable-ish)
    cfg_defaults = dict(cfg["defaults"])
    cfg_dates = dict(cfg_defaults["dates"])
    cfg_dates["start"] = str(start_ts.date())
    cfg_dates["end"] = str(end_ts.date())
    cfg_defaults["dates"] = cfg_dates
    cfg["defaults"] = cfg_defaults

    return cfg


def get_global_dates(cfg: Dict[str, Any]) -> Dict[str, str]:
    """
    Return canonical defaults.dates dict with string ISO dates.
    """
    defaults_section = cfg.get("defaults") or cfg.get("_defaults")
    if not isinstance(defaults_section, dict):
        raise KeyError("Missing defaults section")
    dates = defaults_section.get("dates")
    if not isinstance(dates, dict):
        raise KeyError("Missing defaults.dates section")
    if not dates.get("start") or not dates.get("end"):
        raise KeyError("Missing defaults.dates.start/end")
    return {"start": dates["start"], "end": dates["end"]}


# ------------------------------------------------------------
# Sales config normalization
# ------------------------------------------------------------

def normalize_sales_config(sales_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize sales configuration without destructive mutation.

    Rules:
    - file_format is required and normalized to lowercase
    - total_rows is required
    - skip_order_cols is required (explicit; no implicit defaults)
    - parquet_folder / out_folder are NOT required here (runners may supply them),
      but prepare_paths() can still use them if present.
    - CSV mode marks parquet-only keys as ignored (but does not delete them)
    - deltaparquet mode marks merge_parquet/merged_file as ignored (they don't apply)
    """
    sales_cfg = dict(sales_cfg)

    file_format = sales_cfg.get("file_format")
    if not file_format:
        raise KeyError("sales.file_format is required")

    file_format = str(file_format).lower()
    sales_cfg["file_format"] = file_format

    _validate_required_keys(
        sales_cfg,
        section="sales",
        required=("total_rows", "skip_order_cols"),
    )

    # Normalize core numeric options if present
    sales_cfg["total_rows"] = int(sales_cfg["total_rows"])
    sales_cfg["skip_order_cols"] = bool(sales_cfg["skip_order_cols"])

    if "chunk_size" in sales_cfg:
        sales_cfg["chunk_size"] = int(sales_cfg["chunk_size"])
    if "row_group_size" in sales_cfg:
        sales_cfg["row_group_size"] = int(sales_cfg["row_group_size"])
    if "workers" in sales_cfg and sales_cfg["workers"] is not None:
        sales_cfg["workers"] = int(sales_cfg["workers"])
        
    # Support nested config: sales.partitioning.{enabled, columns}
    # Map nested -> flat (but do NOT override explicit flat keys)
    part = sales_cfg.get("partitioning")
    if isinstance(part, dict):
        if "partition_enabled" not in sales_cfg and part.get("enabled") is not None:
            sales_cfg["partition_enabled"] = bool(part.get("enabled"))

        if "partition_cols" not in sales_cfg and part.get("columns") is not None:
            cols = part.get("columns")
            if cols is None:
                sales_cfg["partition_cols"] = None
            else:
                if not isinstance(cols, list):
                    raise ValueError("sales.partitioning.columns must be a list of column names")
                sales_cfg["partition_cols"] = [str(c) for c in cols]

    # Partition defaults for delta/parquet pipelines
    if "partition_enabled" in sales_cfg:
        sales_cfg["partition_enabled"] = bool(sales_cfg["partition_enabled"])
    if "partition_cols" in sales_cfg and sales_cfg["partition_cols"] is not None:
        if not isinstance(sales_cfg["partition_cols"], list):
            raise ValueError("sales.partition_cols must be a list of column names")
        sales_cfg["partition_cols"] = [str(c) for c in sales_cfg["partition_cols"]]

    # Track ignored keys for transparency (do not delete)
    sales_cfg.setdefault("_ignored_keys", [])

    if file_format == "csv":
        sales_cfg["_ignored_keys"].extend(
            k for k in _PARQUET_ONLY_KEYS if k in sales_cfg
        )
        sales_cfg["_ignored_keys"].extend(
            k for k in _DELTA_ONLY_KEYS if k in sales_cfg
        )

    elif file_format == "parquet":
        # Delta-only keys ignored in parquet mode
        sales_cfg["_ignored_keys"].extend(
            k for k in _DELTA_ONLY_KEYS if k in sales_cfg
        )

    elif file_format == "deltaparquet":
        # Parquet merge options don't apply to Delta output folder
        sales_cfg["_ignored_keys"].extend(
            k for k in _PARQUET_MERGE_ONLY_KEYS if k in sales_cfg
        )

        # partition defaults if not set
        sales_cfg.setdefault("partition_enabled", True)
        sales_cfg.setdefault("partition_cols", ["Year", "Month"])

    else:
        raise ValueError("sales.file_format must be one of: csv, parquet, deltaparquet")

    # De-duplicate ignored keys
    sales_cfg["_ignored_keys"] = sorted(set(sales_cfg["_ignored_keys"]))

    return sales_cfg


# ------------------------------------------------------------
# Validation helpers
# ------------------------------------------------------------

def _validate_required_keys(
    cfg: Dict[str, Any],
    section: str,
    required: Iterable[str],
) -> None:
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"Missing required keys in '{section}' config: {missing}")


# ------------------------------------------------------------
# Path preparation (optional utility)
# ------------------------------------------------------------

def prepare_paths(
    cfg: Dict[str, Any],
    parquet_folder_override: Optional[str | Path] = None,
    out_folder_override: Optional[str | Path] = None,
) -> Tuple[Path, Path]:
    """
    Prepare and validate filesystem paths for pipeline runners.

    Preferred usage now:
      - Runners/CLI pass explicit paths
    Backward compatibility:
      - If overrides are None, uses sales.parquet_folder / sales.out_folder if present
    """
    sales_cfg = cfg.get("sales") or {}
    if not isinstance(sales_cfg, dict):
        raise KeyError("Invalid sales section in config")

    parquet_folder = parquet_folder_override or sales_cfg.get("parquet_folder")
    out_folder = out_folder_override or sales_cfg.get("out_folder")

    if not parquet_folder or not out_folder:
        raise KeyError(
            "Missing parquet/out folders. Provide overrides or set sales.parquet_folder and sales.out_folder."
        )

    parquet_dims = Path(parquet_folder).resolve()
    fact_out = Path(out_folder).resolve()

    parquet_dims.mkdir(parents=True, exist_ok=True)
    fact_out.mkdir(parents=True, exist_ok=True)

    return parquet_dims, fact_out


# ------------------------------------------------------------
# Constants
# ------------------------------------------------------------

# Options that only make sense for parquet chunk output (not CSV/Delta)
_PARQUET_ONLY_KEYS = {
    "row_group_size",
    "compression",
    "chunk_size",
    "workers",
}

# Options that only apply to merging parquet chunks
_PARQUET_MERGE_ONLY_KEYS = {
    "merge_parquet",
    "merged_file",
}

# Options that only make sense for delta output/partitioning
_DELTA_ONLY_KEYS = {
    "partition_enabled",
    "partition_cols",
    "delta_output_folder",
}
