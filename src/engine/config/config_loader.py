from __future__ import annotations

import copy
import json
import yaml
from pathlib import Path
from typing import Dict, Any


# ============================================================
# Public API
# ============================================================

def load_config_file(path: str | Path) -> Dict[str, Any]:
    """
    Load a config file (YAML or JSON).
    Format is detected by extension, with a safe fallback.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    ext = path.suffix.lower()

    with path.open("r", encoding="utf-8") as f:
        if ext in (".yaml", ".yml"):
            return yaml.safe_load(f)
        if ext == ".json":
            return json.load(f)

        # Fallback auto-detection
        text = f.read().strip()
        try:
            return yaml.safe_load(text)
        except Exception:
            return json.loads(text)


def load_config(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve all module configs.

    Precedence:
        defaults  →  module section  →  override

    Output:
        {
            <module>: resolved_config,
            ...
            "_defaults": defaults
        }
    """
    if not isinstance(raw_cfg, dict):
        raise TypeError("Config root must be a dict")

    defaults = raw_cfg.get("defaults", {})
    resolved: Dict[str, Any] = {}

    for section_name, section_cfg in raw_cfg.items():
        if section_name == "defaults":
            continue

        # passthrough for non-dict sections
        if not isinstance(section_cfg, dict):
            resolved[section_name] = section_cfg
            continue

        resolved[section_name] = resolve_section(
            section_name=section_name,
            section_cfg=section_cfg,
            defaults=defaults,
        )

    resolved["_defaults"] = defaults
    return resolved


# ============================================================
# Section resolution
# ============================================================

def resolve_section(
    *,
    section_name: str,
    section_cfg: Dict[str, Any],
    defaults: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Resolve a single config section using strict, predictable rules.
    """
    section = copy.deepcopy(section_cfg)
    override = section.pop("override", {})

    out = _base_from_defaults(defaults)

    # --------------------------------------------------------
    # Merge section-level values (except reserved keys)
    # --------------------------------------------------------
    for key, value in section.items():
        if key not in _RESERVED_KEYS:
            out[key] = value

    # --------------------------------------------------------
    # Section-specific validation
    # --------------------------------------------------------
    if "active_ratio" in out:
        if section_name not in {"customers", "products"}:
            raise ValueError(
                f"'active_ratio' is not supported for section '{section_name}'"
            )

        try:
            ratio = float(out["active_ratio"])
        except Exception:
            raise ValueError("active_ratio must be a number")

        if not 0 < ratio <= 1:
            raise ValueError("active_ratio must be in the range (0, 1]")

    # --------------------------------------------------------
    # Section-specific logic
    # --------------------------------------------------------
    if section_name == "promotions":
        _apply_promotions_dates(out, section)

    # --------------------------------------------------------
    # Apply overrides
    # --------------------------------------------------------
    _apply_overrides(
        out=out,
        override=override,
        section_name=section_name,
    )

    return out


# ============================================================
# Helpers
# ============================================================

def _base_from_defaults(defaults: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create the base structure every module inherits.
    """
    return {
        "seed": defaults.get("seed"),
        "dates": copy.deepcopy(defaults.get("dates", {})),
        "paths": copy.deepcopy(defaults.get("paths", {})),
    }


def _apply_promotions_dates(out: Dict[str, Any], section: Dict[str, Any]) -> None:
    """
    Promotions-specific handling for date_ranges.
    """
    date_ranges = section.get("date_ranges")
    if date_ranges:
        out["date_ranges"] = date_ranges
    else:
        out["date_ranges"] = [{
            "start": out["dates"]["start"],
            "end": out["dates"]["end"],
        }]


def _apply_overrides(
    *,
    out: Dict[str, Any],
    override: Dict[str, Any],
    section_name: str,
) -> None:
    """
    Apply override rules in a controlled, explicit way.
    """
    # Dates override
    if isinstance(override.get("dates"), dict):
        if section_name == "exchange_rates" and out.get("use_global_dates"):
            pass  # explicitly ignored
        else:
            out["dates"] = {**out["dates"], **override["dates"]}

    # Seed override
    if override.get("seed") is not None:
        out["seed"] = override["seed"]

    # Paths override
    if isinstance(override.get("paths"), dict):
        out["paths"] = {**out["paths"], **override["paths"]}


# ============================================================
# Constants
# ============================================================

_RESERVED_KEYS = {"dates", "paths", "override"}
