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


def resolve_seed(
    cfg: Dict[str, Any],
    section_cfg: Dict[str, Any],
    *,
    fallback: int = 42,
) -> int:
    """Resolve seed using the standard precedence chain.

    override.seed -> section.seed -> defaults.seed -> fallback
    """
    override = as_dict(section_cfg.get("override"))
    seed = override.get("seed")
    if seed is None:
        seed = section_cfg.get("seed")
    if seed is None:
        seed = as_dict(cfg.get("defaults")).get("seed")
    if seed is None:
        seed = as_dict(cfg.get("_defaults")).get("seed")
    return int_or(seed, fallback)


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
