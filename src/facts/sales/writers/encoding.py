from __future__ import annotations

from typing import List

from .constants import DICT_EXCLUDE, REQUIRED_PRICING_COLS
from .utils import _arrow


def _schema_dict_cols(schema) -> List[str]:
    """
    Dictionary-encode only string/binary-like columns (except exclusions).
    """
    pa, _, _ = _arrow()
    out: List[str] = []
    for f in schema:
        if f.name in DICT_EXCLUDE:
            continue
        t = f.type
        if (
            pa.types.is_string(t)
            or pa.types.is_large_string(t)
            or pa.types.is_binary(t)
            or pa.types.is_large_binary(t)
        ):
            out.append(f.name)
    return out


def _validate_required(schema) -> None:
    names = set(schema.names)
    missing = REQUIRED_PRICING_COLS - names
    if missing:
        raise RuntimeError(f"Missing required pricing columns: {sorted(missing)}")
