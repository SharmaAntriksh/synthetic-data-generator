"""Shared config merge utilities.

Used by both ``customer_profiles.py`` and ``trend_presets.py`` for
merging Pydantic model configs with dict overrides while preserving
only explicitly-set YAML values.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def pydantic_to_explicit_dict(obj) -> dict[str, Any]:
    """Convert a Pydantic model to a dict of only explicitly-set fields.

    Pydantic models expose ALL fields (including defaults) when iterated.
    This strips defaults so that only YAML-explicit values act as overrides,
    preventing Pydantic's zero-defaults from clobbering profile/preset values.
    """
    fields_set = getattr(obj, "model_fields_set", None)
    if fields_set is None:
        return dict(obj) if isinstance(obj, Mapping) else {}

    out: dict[str, Any] = {}
    for key in fields_set:
        val = getattr(obj, key)
        if hasattr(val, "model_fields_set"):
            out[key] = pydantic_to_explicit_dict(val)
        elif isinstance(val, Mapping):
            out[key] = dict(val)
        else:
            out[key] = val
    return out


def deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* on top of *base*. Overrides win."""
    merged = dict(base)
    for k, v in overrides.items():
        if isinstance(v, Mapping) and isinstance(merged.get(k), Mapping):
            merged[k] = deep_merge(dict(merged[k]), dict(v))
        else:
            merged[k] = v
    return merged
