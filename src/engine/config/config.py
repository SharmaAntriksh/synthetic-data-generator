from __future__ import annotations

import json
from datetime import date as _date
from datetime import datetime as _datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple
from src.dimensions.geography import normalize_geography_config

# ---------------------------------------------------------------------------
# Lazy YAML import (resolved once, avoids per-call try/except overhead)
# ---------------------------------------------------------------------------

try:
    import yaml as _yaml  # type: ignore
except ImportError:
    _yaml = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Format-specific key sets (defined before normalizers that reference them)
# ---------------------------------------------------------------------------

_PARQUET_ONLY_KEYS: frozenset[str] = frozenset(
    {"row_group_size", "compression", "chunk_size", "workers"}
)
_PARQUET_MERGE_ONLY_KEYS: frozenset[str] = frozenset(
    {"merge_parquet", "merged_file"}
)
_DELTA_ONLY_KEYS: frozenset[str] = frozenset(
    {"partition_enabled", "partition_cols", "delta_output_folder"}
)

_VALID_FILE_FORMATS: frozenset[str] = frozenset(
    {"csv", "parquet", "deltaparquet"}
)

_VALID_SEG_GRAINS: frozenset[str] = frozenset({"month", "day"})

# ---------------------------------------------------------------------------
# Section normalization registry (single source of truth)
# ---------------------------------------------------------------------------

SectionNormalizer = Callable[[Dict[str, Any]], Dict[str, Any]]

# Sections that must be mappings if present but do not have a normalizer.
# Keep this small; normalizer keys are automatically treated as mapping sections.
_SECTION_MAPPING_ONLY_KEYS: frozenset[str] = frozenset({"customers"})

# For config files that intentionally don't have defaults (e.g. models.yaml),
# we keep normalization deliberately narrow to avoid surprising validation failures.
_CONFIG_FILE_NORMALIZE_KEYS: frozenset[str] = frozenset({"customer_segments"})

# Populated inline after normalizer functions are defined (see bottom of file).
_SECTION_NORMALIZERS: Dict[str, SectionNormalizer] = {}


# ---------------------------------------------------------------------------
# Lightweight helpers
# ---------------------------------------------------------------------------

def _clamp01(x: float) -> float:
    """Clamp *x* to the [0.0, 1.0] interval."""
    return max(0.0, min(1.0, float(x)))


def _parse_date(value: Any, label: str) -> _date:
    """Parse a date value into a :class:`datetime.date`.

    Accepts ``datetime.date``, ``datetime.datetime``, and common ISO-style
    strings (``YYYY-MM-DD``, ``YYYY/MM/DD``).  Raises :class:`ValueError`
    with a clear message on failure.
    """
    if isinstance(value, _datetime):
        return value.date()
    if isinstance(value, _date):
        return value

    text = str(value).strip()
    # Try ISO first (fast path), then slash-separated variant
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            return _datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse {label} date '{value}' (expected YYYY-MM-DD)")


def _coerce_optional_int(cfg: Dict[str, Any], key: str) -> None:
    """Cast *cfg[key]* to ``int`` in place if it is present and non-None."""
    if key in cfg and cfg[key] is not None:
        cfg[key] = int(cfg[key])


def _coerce_optional_positive_int(
    cfg: Dict[str, Any], key: str, section: str
) -> None:
    """Cast and validate *cfg[key]* as a positive int if present and non-None."""
    if key in cfg and cfg[key] is not None:
        val = int(cfg[key])
        if val <= 0:
            raise ValueError(f"{section}.{key} must be a positive integer, got {val}")
        cfg[key] = val


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_pipeline_config(path: str | Path = "config.yaml") -> dict:
    """Load the *pipeline* config (config.yaml) and normalize safely.

    This loader REQUIRES defaults (or _defaults) to exist because the pipeline
    depends on a global date window for lifecycle-aware generation.
    """
    return _load_and_normalize(
        path=path,
        require_defaults=True,
        normalize_keys=None,  # normalize all registered sections
    )


def load_config(path: str | Path = "config.yaml") -> dict:
    """Backward-compatible alias of :func:`load_pipeline_config`."""
    return load_pipeline_config(path)


def load_config_file(path: str | Path) -> dict:
    """Load a config file (YAML/JSON) that intentionally does NOT contain defaults,
    e.g. models.yaml (top-level ``models`` only).

    This applies only very safe, non-default-dependent transforms:
      - tuning → models mapping (if present)
      - customer_segments normalization (if present)
    """
    return _load_and_normalize(
        path=path,
        require_defaults=False,
        normalize_keys=_CONFIG_FILE_NORMALIZE_KEYS,
    )


# ---------------------------------------------------------------------------
# High-level acquisition tuning
# ---------------------------------------------------------------------------

def apply_acquisition_tuning(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """No-op: the tuning block has been replaced by models.customers.

    Kept for backward compatibility with external callers.
    If an old-style ``tuning`` block is present it is silently ignored.
    """
    return cfg


# ---------------------------------------------------------------------------
# Load + normalize (shared implementation)
# ---------------------------------------------------------------------------

def _load_and_normalize(
    path: str | Path,
    *,
    require_defaults: bool,
    normalize_keys: Optional[frozenset[str] | set[str]],
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


def _normalize_sections(
    cfg: Dict[str, Any],
    *,
    normalize_keys: Optional[frozenset[str] | set[str]],
) -> Dict[str, Any]:
    """Validate section shapes and apply per-section normalizers.

    - Always validates known mapping sections if present.
    - Applies normalizers for sections in *normalize_keys* (or all registered
      normalizers if ``None``).
    """
    cfg = dict(cfg)

    # Validate mapping sections (mapping-only keys + all normalizer keys)
    mapping_keys = _SECTION_MAPPING_ONLY_KEYS | _SECTION_NORMALIZERS.keys()
    for key in mapping_keys:
        if key in cfg and not isinstance(cfg[key], dict):
            raise KeyError(f"Invalid '{key}' section in config (expected mapping)")

    # Decide which normalizers to run
    if normalize_keys is None:
        keys_to_normalize = set(_SECTION_NORMALIZERS.keys())
    else:
        keys_to_normalize = set(normalize_keys) & _SECTION_NORMALIZERS.keys()

    for key in keys_to_normalize:
        if key in cfg:
            cfg[key] = _SECTION_NORMALIZERS[key](cfg[key])  # type: ignore[arg-type]

    return cfg


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_any(path: Path) -> dict:
    """Load a YAML or JSON config file.

    When the extension is known (``.yaml``, ``.yml``, ``.json``) the matching
    parser is used directly.  For unknown extensions the file is tried as YAML
    first (structural errors are **not** swallowed) then as JSON.
    """
    ext = path.suffix.lower()

    if ext in (".yaml", ".yml"):
        return _load_yaml(path)

    if ext == ".json":
        return _load_json(path)

    # Unknown extension – try YAML then JSON, but only swallow *format* errors
    # so that genuine structural problems (bad indentation etc.) surface.
    raw = path.read_text(encoding="utf-8")

    if _yaml is not None:
        try:
            cfg = _yaml.safe_load(raw) or {}
            if isinstance(cfg, dict):
                return cfg
        except _yaml.YAMLError:
            # Content is not valid YAML – fall through to JSON attempt.
            pass

    try:
        cfg = json.loads(raw) or {}
        if isinstance(cfg, dict):
            return cfg
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(
            f"Unable to parse config file as YAML or JSON: {path}"
        ) from exc

    raise ValueError(f"Unsupported or invalid config format: {path}")


def _load_yaml(path: Path) -> dict:
    if _yaml is None:
        raise RuntimeError(
            "PyYAML is required to load .yaml/.yml config files"
        )
    with path.open("r", encoding="utf-8") as fh:
        cfg = _yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Top-level YAML config must be a mapping/object")
    return cfg


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        cfg = json.load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Top-level JSON config must be an object")
    return cfg


# ---------------------------------------------------------------------------
# Defaults normalization (dates)
# ---------------------------------------------------------------------------

def normalize_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Canonicalize defaults:

    - Accept ``defaults`` or ``_defaults``
    - Ensure ``cfg["defaults"]`` exists (canonical)
    - Remove stale ``_defaults`` alias to prevent duplication
    - Validate defaults.dates.start/end exist and are parseable
    """
    cfg = dict(cfg)

    defaults_section = cfg.get("defaults")
    underscore_defaults = cfg.get("_defaults")

    if defaults_section is None and underscore_defaults is not None:
        defaults_section = underscore_defaults

    # Clean up the underscore alias so only the canonical key survives
    cfg.pop("_defaults", None)

    if defaults_section is None:
        raise KeyError("Missing 'defaults' (or '_defaults') section in config")
    if not isinstance(defaults_section, dict):
        raise KeyError("Invalid defaults section in config (expected mapping)")

    dates = defaults_section.get("dates")
    if not isinstance(dates, dict):
        raise KeyError("Missing or invalid defaults.dates section in config")

    raw_start = dates.get("start")
    raw_end = dates.get("end")
    if not raw_start or not raw_end:
        raise KeyError("Missing defaults.dates.start or defaults.dates.end in config")

    try:
        start_date = _parse_date(raw_start, "defaults.dates.start")
        end_date = _parse_date(raw_end, "defaults.dates.end")
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "Missing or invalid defaults.dates.start/end in config.yaml"
        ) from exc

    if start_date >= end_date:
        raise ValueError("Invalid defaults.dates range (start must be < end)")

    # Rebuild with canonical ISO strings
    cfg_defaults = dict(defaults_section)
    cfg_defaults["dates"] = {
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
    }
    cfg["defaults"] = cfg_defaults

    return cfg


def get_global_dates(cfg: Dict[str, Any]) -> Dict[str, str]:
    """Return canonical defaults.dates dict with string ISO dates.

    Normalizes the raw values through :func:`_parse_date` so the caller
    always gets consistent ``YYYY-MM-DD`` strings regardless of whether
    :func:`normalize_defaults` was called earlier.
    """
    defaults_section = cfg.get("defaults") or cfg.get("_defaults")
    if not isinstance(defaults_section, dict):
        raise KeyError("Missing defaults section")

    dates = defaults_section.get("dates")
    if not isinstance(dates, dict):
        raise KeyError("Missing defaults.dates section")

    raw_start = dates.get("start")
    raw_end = dates.get("end")
    if not raw_start or not raw_end:
        raise KeyError("Missing defaults.dates.start/end")

    return {
        "start": _parse_date(raw_start, "defaults.dates.start").isoformat(),
        "end": _parse_date(raw_end, "defaults.dates.end").isoformat(),
    }


# ---------------------------------------------------------------------------
# Sales config normalization
# ---------------------------------------------------------------------------

def normalize_sales_config(sales_cfg: Dict[str, Any]) -> Dict[str, Any]:
    sales_cfg = dict(sales_cfg)

    # --- file_format ---
    file_format = sales_cfg.get("file_format")
    if not file_format:
        raise KeyError("sales.file_format is required")
    file_format = str(file_format).lower()
    if file_format not in _VALID_FILE_FORMATS:
        raise ValueError(
            f"sales.file_format must be one of: {', '.join(sorted(_VALID_FILE_FORMATS))}"
        )
    sales_cfg["file_format"] = file_format

    # --- required scalar keys ---
    _validate_required_keys(
        sales_cfg, section="sales", required=("total_rows", "skip_order_cols"),
    )
    total_rows = int(sales_cfg["total_rows"])
    if total_rows <= 0:
        raise ValueError(f"sales.total_rows must be a positive integer, got {total_rows}")
    sales_cfg["total_rows"] = total_rows
    sales_cfg["skip_order_cols"] = bool(sales_cfg["skip_order_cols"])

    # --- optional numeric keys ---
    _coerce_optional_positive_int(sales_cfg, "chunk_size", "sales")
    _coerce_optional_positive_int(sales_cfg, "row_group_size", "sales")
    _coerce_optional_positive_int(sales_cfg, "workers", "sales")

    # --- nested partitioning block → flat keys ---
    part = sales_cfg.get("partitioning")
    if isinstance(part, dict):
        if "partition_enabled" not in sales_cfg and part.get("enabled") is not None:
            sales_cfg["partition_enabled"] = bool(part["enabled"])

        # Use `"columns" in part` so explicit null is respected
        if "partition_cols" not in sales_cfg and "columns" in part:
            cols = part.get("columns")
            if cols is None:
                sales_cfg["partition_cols"] = None
            else:
                if not isinstance(cols, list):
                    raise ValueError(
                        "sales.partitioning.columns must be a list of column names"
                    )
                sales_cfg["partition_cols"] = [str(c) for c in cols]

    # --- normalize / validate flat partitioning keys ---
    if "partition_enabled" in sales_cfg:
        sales_cfg["partition_enabled"] = bool(sales_cfg["partition_enabled"])
        if sales_cfg["partition_enabled"] is False:
            sales_cfg["partition_cols"] = []

    if "partition_cols" in sales_cfg and sales_cfg["partition_cols"] is not None:
        if not isinstance(sales_cfg["partition_cols"], list):
            raise ValueError("sales.partition_cols must be a list of column names")
        sales_cfg["partition_cols"] = [str(c) for c in sales_cfg["partition_cols"]]

    # --- ignored keys per format (built as a set, sorted once) ---
    ignored: set[str] = set(sales_cfg.get("_ignored_keys") or [])

    if file_format == "csv":
        ignored.update(k for k in _PARQUET_ONLY_KEYS if k in sales_cfg)
        ignored.update(k for k in _DELTA_ONLY_KEYS if k in sales_cfg)

    elif file_format == "parquet":
        ignored.update(k for k in _DELTA_ONLY_KEYS if k in sales_cfg)

    elif file_format == "deltaparquet":
        ignored.update(k for k in _PARQUET_MERGE_ONLY_KEYS if k in sales_cfg)
        sales_cfg.setdefault("partition_enabled", True)
        sales_cfg.setdefault("partition_cols", ["Year", "Month"])

    sales_cfg["_ignored_keys"] = sorted(ignored)
    return sales_cfg


# ---------------------------------------------------------------------------
# Customer segments normalization
# ---------------------------------------------------------------------------

def normalize_customer_segments_config(seg_cfg: Dict[str, Any]) -> Dict[str, Any]:
    seg_cfg = dict(seg_cfg)

    seg_cfg.setdefault("enabled", True)
    seg_cfg["enabled"] = bool(seg_cfg["enabled"])
    if not seg_cfg["enabled"]:
        return seg_cfg

    # --- deterministic seed (top-level) ---
    seg_cfg.setdefault("seed", 123)
    if seg_cfg["seed"] is not None:
        seg_cfg["seed"] = int(seg_cfg["seed"])

    # --- segment counts with range validation ---
    seg_cfg.setdefault("segment_count", 12)
    seg_cfg.setdefault("segments_per_customer_min", 1)
    seg_cfg.setdefault("segments_per_customer_max", 4)

    seg_count = int(seg_cfg["segment_count"])
    seg_min = int(seg_cfg["segments_per_customer_min"])
    seg_max = int(seg_cfg["segments_per_customer_max"])

    if seg_count <= 0:
        raise ValueError(
            f"customer_segments.segment_count must be > 0, got {seg_count}"
        )
    if seg_min < 1:
        raise ValueError(
            f"customer_segments.segments_per_customer_min must be >= 1, got {seg_min}"
        )
    if seg_max < seg_min:
        raise ValueError(
            f"customer_segments.segments_per_customer_max ({seg_max}) "
            f"must be >= segments_per_customer_min ({seg_min})"
        )
    if seg_max > seg_count:
        raise ValueError(
            f"customer_segments.segments_per_customer_max ({seg_max}) "
            f"must be <= segment_count ({seg_count})"
        )

    seg_cfg["segment_count"] = seg_count
    seg_cfg["segments_per_customer_min"] = seg_min
    seg_cfg["segments_per_customer_max"] = seg_max

    # --- boolean flags ---
    seg_cfg.setdefault("include_score", True)
    seg_cfg.setdefault("include_primary_flag", True)

    # Auto-enable validity if a validity block exists (prevents "silent ignore")
    if "include_validity" not in seg_cfg:
        seg_cfg["include_validity"] = ("validity" in seg_cfg)

    seg_cfg["include_score"] = bool(seg_cfg["include_score"])
    seg_cfg["include_primary_flag"] = bool(seg_cfg["include_primary_flag"])
    seg_cfg["include_validity"] = bool(seg_cfg["include_validity"])

    # --- validity block ---
    validity = seg_cfg.get("validity") or {}
    if seg_cfg["include_validity"]:
        if not isinstance(validity, dict):
            raise KeyError("customer_segments.validity must be a mapping/object")

        grain = str(validity.get("grain", "month")).lower()
        if grain not in _VALID_SEG_GRAINS:
            raise ValueError(
                f"customer_segments.validity.grain must be one of: "
                f"{', '.join(sorted(_VALID_SEG_GRAINS))}"
            )

        churn = float(validity.get("churn_rate_qtr", 0.08))
        if not 0.0 <= churn <= 1.0:
            raise ValueError(
                "customer_segments.validity.churn_rate_qtr must be between 0 and 1"
            )

        new_months = int(validity.get("new_customer_months", 2))
        if new_months < 0:
            raise ValueError(
                "customer_segments.validity.new_customer_months must be >= 0"
            )

        seg_cfg["validity"] = {
            "grain": grain,
            "churn_rate_qtr": churn,
            "new_customer_months": new_months,
        }
    else:
        # If user supplied validity but include_validity is off, record it as ignored
        if "validity" in seg_cfg:
            ignored: set[str] = set(seg_cfg.get("_ignored_keys") or [])
            ignored.add("validity")
            seg_cfg["_ignored_keys"] = sorted(ignored)

    # --- override block ---
    override = seg_cfg.get("override") or {}
    if not isinstance(override, dict):
        raise KeyError("customer_segments.override must be a mapping/object")
    override.setdefault("seed", None)
    override.setdefault("dates", {})
    override.setdefault("paths", {})

    if override["seed"] is not None:
        override["seed"] = int(override["seed"])

    if not isinstance(override["dates"], dict):
        raise KeyError("customer_segments.override.dates must be a mapping/object")
    if not isinstance(override["paths"], dict):
        raise KeyError("customer_segments.override.paths must be a mapping/object")
    seg_cfg["override"] = override

    return seg_cfg


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_required_keys(
    cfg: Dict[str, Any], section: str, required: Iterable[str]
) -> None:
    missing = [k for k in required if k not in cfg]
    if missing:
        raise KeyError(f"Missing required keys in '{section}' config: {missing}")


# ---------------------------------------------------------------------------
# Path preparation (optional utility)
# ---------------------------------------------------------------------------

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
            "Missing parquet/out folders. Provide overrides or set "
            "sales.parquet_folder and sales.out_folder."
        )

    parquet_dims = Path(parquet_folder).resolve()
    fact_out = Path(out_folder).resolve()

    parquet_dims.mkdir(parents=True, exist_ok=True)
    fact_out.mkdir(parents=True, exist_ok=True)

    return parquet_dims, fact_out


# ---------------------------------------------------------------------------
# Register section normalizers
# ---------------------------------------------------------------------------

_SECTION_NORMALIZERS.update(
    {
        "sales": normalize_sales_config,
        "customer_segments": normalize_customer_segments_config,
        "geography": normalize_geography_config,
    }
)
