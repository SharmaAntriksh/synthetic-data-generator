"""
web/shared_state.py -- Shared mutable state and helpers for the web API.

All route modules import from here so that every router sees the same
config dicts, job state, and helper functions.
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# ---------------------------------------------------------------------------
# Resolve repo root
# ---------------------------------------------------------------------------

def _find_repo_root() -> Path:
    d = Path(__file__).resolve().parent
    for _ in range(10):
        if (d / "main.py").exists():
            return d
        d = d.parent
    return Path.cwd()


REPO_ROOT = _find_repo_root()

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Config normalization (reuse pipeline's normalizers)
# ---------------------------------------------------------------------------

try:
    from src.engine.config.config import load_config as _load_pipeline_config
    _HAS_NORMALIZER = True
except ImportError:
    _HAS_NORMALIZER = False

# ---------------------------------------------------------------------------
# ANSI regex
# ---------------------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

_config_path = REPO_ROOT / "config.yaml"
_models_path = REPO_ROOT / "models.yaml"


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

def _load_yaml(p: Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _base_config() -> dict:
    if _HAS_NORMALIZER:
        try:
            return _load_pipeline_config(str(_config_path))
        except (KeyError, ValueError, OSError, TypeError):
            pass
    return _load_yaml(_config_path)


# ---------------------------------------------------------------------------
# Config state  (mutable module-level singletons, guarded by _cfg_lock)
# ---------------------------------------------------------------------------

_cfg_lock = threading.Lock()

_cfg: Dict[str, Any] = _base_config()
_cfg_disk_yaml: str = _config_path.read_text(encoding="utf-8") if _config_path.exists() else ""
_models_cfg: Dict[str, Any] = _load_yaml(_models_path)
_models_yaml_text: str = _models_path.read_text(encoding="utf-8") if _models_path.exists() else ""

# ---------------------------------------------------------------------------
# Preset logic
# ---------------------------------------------------------------------------

try:
    from .presets import PRESETS, apply_preset, build_presets_by_sales
except ImportError:
    PRESETS = {}
    apply_preset = None
    build_presets_by_sales = None

# ---------------------------------------------------------------------------
# Job state
# ---------------------------------------------------------------------------

_job_lock = threading.Lock()
_current_job: Optional[Dict[str, Any]] = None

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _g(d: dict, *keys, default=None):
    """Nested dict get."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
    return cur if cur is not None else default


def _promo_total(promos: dict) -> int:
    keys = ("num_seasonal", "num_clearance", "num_limited")
    if all(k in promos for k in keys):
        return sum(int(promos.get(k, 0) or 0) for k in keys)
    return int(promos.get("total_promotions", 0) or 0)


def _set_promotions_total(promos: dict, total: int):
    """Distribute total across buckets proportionally."""
    total = max(0, int(total))
    keys = ["num_seasonal", "num_clearance", "num_limited"]
    if all(k in promos for k in keys):
        cur = [int(promos.get(k, 0) or 0) for k in keys]
        s = sum(cur) or 3
        base = cur if sum(cur) > 0 else [1, 1, 1]
        scaled = [b * total / s for b in base]
        floors = [int(x) for x in scaled]
        remainder = total - sum(floors)
        fracs = sorted(range(3), key=lambda i: scaled[i] - floors[i], reverse=True)
        for i in range(remainder):
            floors[fracs[i % 3]] += 1
        for i, k in enumerate(keys):
            promos[k] = floors[i]
    else:
        promos["total_promotions"] = total


def normalize_config_yaml(parsed: dict) -> dict:
    """Run the pipeline normalizer on a parsed YAML dict, return normalized config."""
    global _HAS_NORMALIZER
    if _HAS_NORMALIZER:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False, encoding="utf-8"
            ) as tmp:
                yaml.safe_dump(parsed, tmp, sort_keys=False)
                tmp_path = tmp.name
            result = _load_pipeline_config(tmp_path)
            return result
        except Exception as exc:
            import logging
            logging.getLogger(__name__).warning(
                "Config normalization failed, using raw parsed config: %s", exc
            )
            return parsed
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
    return parsed
