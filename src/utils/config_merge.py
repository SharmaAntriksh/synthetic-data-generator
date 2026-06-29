"""Shared config merge utilities.

Used by ``trend_presets.py`` and other config resolvers for merging
Pydantic model configs with dict overrides while preserving only
explicitly-set YAML values.
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
    """Recursively merge *overrides* on top of *base*. Overrides win.

    The result never aliases nested dicts from *base*: untouched nested
    mappings are copied so mutating the merged result can't reach back into
    *base* (or vice versa).
    """
    merged: dict = {}
    for k, v in base.items():
        merged[k] = dict(v) if isinstance(v, Mapping) else v
    for k, v in overrides.items():
        if isinstance(v, Mapping) and isinstance(merged.get(k), Mapping):
            merged[k] = deep_merge(dict(merged[k]), dict(v))
        else:
            merged[k] = v
    return merged
