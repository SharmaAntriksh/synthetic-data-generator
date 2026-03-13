"""Standardized config resolution helpers.

Config Precedence (applies to ALL generators):
    1. ``override.{key}``     -- per-section test/override block
    2. ``{section}.{key}``    -- explicit section-level value
    3. ``defaults.{key}``     -- global defaults (``_defaults`` as fallback)
    4. Hardcoded fallback     -- module-level constant

Seed Precedence (applies to ALL generators):
    1. ``{section}.override.seed``
    2. ``{section}.seed``
    3. ``defaults.seed``
    4. Fallback (typically 42)

Date Precedence (applies to ALL generators):
    1. ``{section}.override.dates.{start,end}``
    2. ``defaults.dates.{start,end}``  (or ``_defaults.dates``)
    3. Raise error (dates are required)

These helpers wrap :mod:`src.utils.config_helpers` to enforce the above
precedence uniformly, so new dimension/fact generators can reuse them
without reimplementing the chain.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import pandas as pd

from src.utils.config_helpers import as_dict, int_or


def _getattr_or_get(obj: Any, key: str) -> Any:
    """Read *key* from a Pydantic model (getattr) or dict (.get)."""
    if obj is None:
        return None
    if hasattr(obj, key):
        return getattr(obj, key, None)
    if isinstance(obj, dict):
        return obj.get(key)
    return None


def _is_random_mode(cfg: Any) -> bool:
    """Return True when ``defaults.random`` is truthy."""
    defaults = _getattr_or_get(cfg, "defaults")
    if defaults is None:
        return False
    val = _getattr_or_get(defaults, "random")
    return bool(val) if val is not None else False


def resolve_seed(
    cfg: Any,
    section_cfg: Any = None,
    *,
    fallback: int = 42,
) -> int:
    """Resolve seed using the standard precedence chain.

    override.seed -> section.seed -> defaults.seed -> fallback

    When ``defaults.random`` is ``True``, generates a one-time random
    seed from OS entropy so every run produces different output while
    keeping downstream code (which does ``int(seed)``) unchanged.

    Works with both Pydantic models (attribute access) and plain dicts.
    """
    if _is_random_mode(cfg):
        import numpy as np
        return int(np.random.default_rng(None).integers(1, 1 << 31))

    # 1. override.seed
    if section_cfg is not None:
        override = _getattr_or_get(section_cfg, "override")
        ov_dict = as_dict(override) if override is not None else {}
        seed = ov_dict.get("seed") if isinstance(ov_dict, dict) else _getattr_or_get(override, "seed")
        if seed is not None:
            return int_or(seed, fallback)

        # 2. section.seed
        seed = _getattr_or_get(section_cfg, "seed")
        if seed is not None:
            return int_or(seed, fallback)

    # 3. defaults.seed
    defaults = _getattr_or_get(cfg, "defaults")
    if defaults is not None:
        seed = _getattr_or_get(defaults, "seed")
        if seed is not None:
            return int_or(seed, fallback)

    # 3b. _defaults.seed (legacy normalizer key)
    _defaults = _getattr_or_get(cfg, "_defaults")
    if _defaults is not None:
        seed = _getattr_or_get(_defaults, "seed")
        if seed is not None:
            return int_or(seed, fallback)

    return fallback


def resolve_dates(
    cfg: Dict[str, Any],
    section_cfg: Dict[str, Any],
    *,
    allow_override: bool = True,
    section_name: str = "",
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Resolve date window using the standard precedence chain.

    override.dates.{start,end} -> defaults.dates.{start,end}

    Raises ``KeyError`` if dates cannot be resolved.
    """
    from src.utils.config_helpers import parse_global_dates
    return parse_global_dates(
        cfg,
        section_cfg,
        allow_override=allow_override,
        dimension_name=section_name,
    )
