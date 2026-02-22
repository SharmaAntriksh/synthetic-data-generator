from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

import pandas as pd

# ------------------------------------------------------------
# Section normalization registry (single source of truth)
# ------------------------------------------------------------

SectionNormalizer = Callable[[Dict[str, Any]], Dict[str, Any]]

# Sections that must be mappings if present but do not have a normalizer.
# Keep this small; normalizer keys are automatically treated as mapping sections.
_SECTION_MAPPING_ONLY_KEYS: set[str] = {
    "customers",
}

# Sections that get normalized if present.
# Add new table/config section normalizers here (pipeline loader will pick them up automatically).
_SECTION_NORMALIZERS: Dict[str, SectionNormalizer] = {}

# For config files that intentionally don't have defaults (e.g. models.yaml),
# we keep normalization deliberately narrow to avoid surprising validation failures.
_CONFIG_FILE_NORMALIZE_KEYS: set[str] = {
    "customer_segments",
}

# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------

def load_pipeline_config(path: str | Path = "config.yaml") -> dict:
    """
    Load the *pipeline* config (config.yaml) and normalize safely.

    This loader REQUIRES defaults (or _defaults) to exist because the pipeline
    depends on a global date window for lifecycle-aware generation.
    """
    return _load_and_normalize(
        path=path,
        require_defaults=True,
        normalize_keys=None,  # normalize all registered sections
    )


def load_config(path: str | Path = "config.yaml") -> dict:
    """
    Backward-compatible alias of load_pipeline_config().
    """
    return load_pipeline_config(path)


def load_config_file(path: str | Path) -> dict:
    """
    Load a config file (YAML/JSON) that intentionally does NOT contain defaults,
    e.g. models.yaml (top-level 'models' only).

    This applies only very safe, non-default-dependent transforms:
      - tuning -> models mapping (if present)
      - customer_segments normalization (if present)
    """
    return _load_and_normalize(
        path=path,
        require_defaults=False,
        normalize_keys=_CONFIG_FILE_NORMALIZE_KEYS,
    )


# ------------------------------------------------------------
# High-level acquisition tuning
# ------------------------------------------------------------

def apply_acquisition_tuning(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Translate high-level acquisition tuning knobs into detailed model parameters.

    Expected config shape:
      tuning:
        acquisition_intensity: 0..1   (higher => more new customers)
        acquisition_smoothness: 0..1  (higher => flatter / fewer dead months)
        acquisition_cycles: 0..1      (higher => stronger multi-year waves)

    Notes:
      - Does NOT require pipeline defaults, so safe for models.yaml too.
      - Mutates only the 'models' subtree and only when 'tuning' exists.
    """
    tuning = cfg.get("tuning")
    if not isinstance(tuning, dict):
        return cfg

    cfg = dict(cfg)  # shallow copy
    models = cfg.get("models")
    if models is None:
        models = {}
        cfg["models"] = models
    if not isinstance(models, dict):
        raise KeyError("Invalid 'models' section in config (expected mapping)")

    discovery = models.setdefault("customer_discovery", {})
    participation = models.setdefault("customer_participation", {})
    cycles = participation.setdefault("cycles", {})

    def _clamp01(x: float) -> float:
        return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

    intensity = _clamp01(float(tuning.get("acquisition_intensity", 0.5)))
    smoothness = _clamp01(float(tuning.get("acquisition_smoothness", 0.5)))
    cycle_strength = _clamp01(float(tuning.get("acquisition_cycles", 0.3)))

    # Discovery mapping
    discovery["orders_per_new_customer"] = int(round(40 - 30 * intensity))  # 40..10
    discovery["min_new_customers_per_month"] = int(round(30 + 200 * smoothness))  # 30..230
    discovery.setdefault("max_fraction_per_month", 0.03)
    discovery["seasonal_amplitude"] = 0.0

    # Participation mapping
    cycles.setdefault("enabled", True)
    cycles["amplitude"] = round(0.05 + 0.30 * cycle_strength, 3)  # 0.05..0.35

    return cfg


# ------------------------------------------------------------
# Load + normalize (shared implementation)
# ------------------------------------------------------------

def _load_and_normalize(
    path: str | Path,
    *,
    require_defaults: bool,
    normalize_keys: Optional[set[str]],
) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    cfg: Dict[str, Any] = _load_any(path)

    # Safe, defaults-independent mapping of high-level tuning knobs (if present)
    cfg = apply_acquisition_tuning(cfg)

    # Pipeline config requires defaults for global date window
    if require_defaults:
        cfg = normalize_defaults(cfg)

    # Normalize/validate known sections
    cfg = _normalize_sections(cfg, normalize_keys=normalize_keys)

    return cfg


def _normalize_sections(cfg: Dict[str, Any], *, normalize_keys: Optional[set[str]]) -> Dict[str, Any]:
    """Validate section shapes and apply per-section normalizers.

    - Always validates known mapping sections if present.
    - Applies normalizers for sections in `normalize_keys` (or all registered normalizers if None).
    """
    cfg = dict(cfg)

    # Validate mapping sections (mapping-only keys + all normalizer keys)
    mapping_keys = set(_SECTION_MAPPING_ONLY_KEYS) | set(_SECTION_NORMALIZERS.keys())
    for key in mapping_keys:
        if key in cfg and not isinstance(cfg[key], dict):
            raise KeyError(f"Invalid '{key}' section in config (expected mapping)")

    # Decide which normalizers to run
    if normalize_keys is None:
        keys_to_normalize = set(_SECTION_NORMALIZERS.keys())
    else:
        keys_to_normalize = set(normalize_keys) & set(_SECTION_NORMALIZERS.keys())

    for key in keys_to_normalize:
        if key in cfg:
            cfg[key] = _SECTION_NORMALIZERS[key](cfg[key])  # type: ignore[arg-type]

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
    with path.open("r", encoding="utf-8") as f:
        raw = f.read()

    try:
        import yaml  # type: ignore
        cfg = yaml.safe_load(raw) or {}
        if isinstance(cfg, dict):
            return cfg
    except Exception:
        pass

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

    try:
        start_ts = pd.to_datetime(start).normalize()
        end_ts = pd.to_datetime(end).normalize()
    except Exception as e:
        raise ValueError("Missing or invalid defaults.dates.start/end in config.yaml") from e

    if pd.isna(start_ts) or pd.isna(end_ts) or start_ts >= end_ts:
        raise ValueError("Invalid defaults.dates range (start must be < end)")

    cfg_defaults = dict(cfg["defaults"])
    cfg_dates = dict(cfg_defaults["dates"])
    cfg_dates["start"] = str(start_ts.date())
    cfg_dates["end"] = str(end_ts.date())
    cfg_defaults["dates"] = cfg_dates
    cfg["defaults"] = cfg_defaults

    return cfg


def get_global_dates(cfg: Dict[str, Any]) -> Dict[str, str]:
    """Return canonical defaults.dates dict with string ISO dates."""
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

    sales_cfg["total_rows"] = int(sales_cfg["total_rows"])
    sales_cfg["skip_order_cols"] = bool(sales_cfg["skip_order_cols"])

    if "chunk_size" in sales_cfg and sales_cfg["chunk_size"] is not None:
        sales_cfg["chunk_size"] = int(sales_cfg["chunk_size"])
    if "row_group_size" in sales_cfg and sales_cfg["row_group_size"] is not None:
        sales_cfg["row_group_size"] = int(sales_cfg["row_group_size"])
    if "workers" in sales_cfg and sales_cfg["workers"] is not None:
        sales_cfg["workers"] = int(sales_cfg["workers"])

    # Support nested config: sales.partitioning.{enabled, columns}
    part = sales_cfg.get("partitioning")
    if isinstance(part, dict):
        if "partition_enabled" not in sales_cfg and part.get("enabled") is not None:
            sales_cfg["partition_enabled"] = bool(part.get("enabled"))

        # Use `"columns" in part` so explicit null is respected
        if "partition_cols" not in sales_cfg and "columns" in part:
            cols = part.get("columns")
            if cols is None:
                sales_cfg["partition_cols"] = None
            else:
                if not isinstance(cols, list):
                    raise ValueError("sales.partitioning.columns must be a list of column names")
                sales_cfg["partition_cols"] = [str(c) for c in cols]

    # Normalize / validate flat keys
    if "partition_enabled" in sales_cfg:
        sales_cfg["partition_enabled"] = bool(sales_cfg["partition_enabled"])
        if sales_cfg["partition_enabled"] is False:
            sales_cfg["partition_cols"] = []

    if "partition_cols" in sales_cfg and sales_cfg["partition_cols"] is not None:
        if not isinstance(sales_cfg["partition_cols"], list):
            raise ValueError("sales.partition_cols must be a list of column names")
        sales_cfg["partition_cols"] = [str(c) for c in sales_cfg["partition_cols"]]

    sales_cfg.setdefault("_ignored_keys", [])

    if file_format == "csv":
        sales_cfg["_ignored_keys"].extend(k for k in _PARQUET_ONLY_KEYS if k in sales_cfg)
        sales_cfg["_ignored_keys"].extend(k for k in _DELTA_ONLY_KEYS if k in sales_cfg)

    elif file_format == "parquet":
        sales_cfg["_ignored_keys"].extend(k for k in _DELTA_ONLY_KEYS if k in sales_cfg)

    elif file_format == "deltaparquet":
        sales_cfg["_ignored_keys"].extend(k for k in _PARQUET_MERGE_ONLY_KEYS if k in sales_cfg)
        sales_cfg.setdefault("partition_enabled", True)
        sales_cfg.setdefault("partition_cols", ["Year", "Month"])

    else:
        raise ValueError("sales.file_format must be one of: csv, parquet, deltaparquet")

    sales_cfg["_ignored_keys"] = sorted(set(sales_cfg["_ignored_keys"]))
    return sales_cfg


# ------------------------------------------------------------
# Customer segments normalization
# ------------------------------------------------------------

_VALID_SEG_GRAINS = {"month", "day"}

def normalize_customer_segments_config(seg_cfg: Dict[str, Any]) -> Dict[str, Any]:
    seg_cfg = dict(seg_cfg)

    seg_cfg.setdefault("enabled", True)
    seg_cfg["enabled"] = bool(seg_cfg["enabled"])
    if not seg_cfg["enabled"]:
        return seg_cfg

    # NEW: deterministic seed (top-level)
    seg_cfg.setdefault("seed", 123)
    if seg_cfg["seed"] is not None:
        seg_cfg["seed"] = int(seg_cfg["seed"])

    seg_cfg.setdefault("segment_count", 12)
    seg_cfg.setdefault("segments_per_customer_min", 1)
    seg_cfg.setdefault("segments_per_customer_max", 4)

    seg_cfg["segment_count"] = int(seg_cfg["segment_count"])
    seg_cfg["segments_per_customer_min"] = int(seg_cfg["segments_per_customer_min"])
    seg_cfg["segments_per_customer_max"] = int(seg_cfg["segments_per_customer_max"])

    # ... keep your existing validations ...

    seg_cfg.setdefault("include_score", True)
    seg_cfg.setdefault("include_primary_flag", True)

    # NEW: auto-enable validity if validity block exists (prevents “silent ignore”)
    if "include_validity" not in seg_cfg:
        seg_cfg["include_validity"] = ("validity" in seg_cfg)

    seg_cfg["include_score"] = bool(seg_cfg["include_score"])
    seg_cfg["include_primary_flag"] = bool(seg_cfg["include_primary_flag"])
    seg_cfg["include_validity"] = bool(seg_cfg["include_validity"])

    # validity block
    validity = seg_cfg.get("validity") or {}
    if seg_cfg["include_validity"]:
        if not isinstance(validity, dict):
            raise KeyError("customer_segments.validity must be a mapping/object")

        grain = str(validity.get("grain", "month")).lower()
        if grain not in _VALID_SEG_GRAINS:
            raise ValueError("customer_segments.validity.grain must be one of: month, day")

        churn = float(validity.get("churn_rate_qtr", 0.08))
        if churn < 0.0 or churn > 1.0:
            raise ValueError("customer_segments.validity.churn_rate_qtr must be between 0 and 1")

        new_months = int(validity.get("new_customer_months", 2))
        if new_months < 0:
            raise ValueError("customer_segments.validity.new_customer_months must be >= 0")

        seg_cfg["validity"] = {
            "grain": grain,
            "churn_rate_qtr": churn,
            "new_customer_months": new_months,
        }
    else:
        # NEW: if user supplied validity but include_validity is off, record it as ignored
        if "validity" in seg_cfg:
            seg_cfg.setdefault("_ignored_keys", [])
            seg_cfg["_ignored_keys"].append("validity")
            seg_cfg["_ignored_keys"] = sorted(set(seg_cfg["_ignored_keys"]))

    override = seg_cfg.get("override") or {}
    if not isinstance(override, dict):
        raise KeyError("customer_segments.override must be a mapping/object")
    override.setdefault("seed", None)
    override.setdefault("dates", {})
    override.setdefault("paths", {})

    # NEW: normalize override.seed type if present
    if override["seed"] is not None:
        override["seed"] = int(override["seed"])

    if not isinstance(override["dates"], dict):
        raise KeyError("customer_segments.override.dates must be a mapping/object")
    if not isinstance(override["paths"], dict):
        raise KeyError("customer_segments.override.paths must be a mapping/object")
    seg_cfg["override"] = override

    return seg_cfg


# ------------------------------------------------------------
# Validation helpers
# ------------------------------------------------------------

def _validate_required_keys(cfg: Dict[str, Any], section: str, required: Iterable[str]) -> None:
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

_PARQUET_ONLY_KEYS = {"row_group_size", "compression", "chunk_size", "workers"}
_PARQUET_MERGE_ONLY_KEYS = {"merge_parquet", "merged_file"}
_DELTA_ONLY_KEYS = {"partition_enabled", "partition_cols", "delta_output_folder"}


# ------------------------------------------------------------
# Register section normalizers (keep at bottom so functions exist)
# ------------------------------------------------------------

_SECTION_NORMALIZERS.update(
    {
        "sales": normalize_sales_config,
        "customer_segments": normalize_customer_segments_config,
    }
)
