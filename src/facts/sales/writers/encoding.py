from __future__ import annotations

from typing import Iterable, List, Optional, Set

from .constants import DICT_EXCLUDE, REQUIRED_PRICING_COLS
from .utils import _arrow
from ..output_paths import TABLE_SALES, TABLE_SALES_ORDER_DETAIL


def required_pricing_cols_for_table(table_name: str | None) -> Set[str]:
    """
    Pricing cols are required only for line-grain tables.

    Back-compat:
      - If table_name is None, keep old strict behavior (require pricing cols).
    """
    if table_name is None:
        return set(REQUIRED_PRICING_COLS)
    if table_name in {TABLE_SALES, TABLE_SALES_ORDER_DETAIL}:
        return set(REQUIRED_PRICING_COLS)
    return set()


def _validate_required(schema, *, table_name: str | None = None) -> None:
    """
    Legacy name kept for compatibility with existing imports.
    """
    required = required_pricing_cols_for_table(table_name)
    if not required:
        return

    names = set(schema.names)
    missing = required - names
    if missing:
        raise RuntimeError(f"Missing required pricing columns: {sorted(missing)}")


def _schema_dict_cols(
    schema,
    exclude: Optional[Iterable[str]] = None,
    *,
    table_name: str | None = None,
) -> List[str]:
    """
    Legacy name kept for compatibility with existing imports.

    Dictionary-encode only string/binary-like columns (except exclusions).
    """
    _validate_required(schema, table_name=table_name)

    pa, _, _ = _arrow()

    exclude_set = set(DICT_EXCLUDE)
    if exclude:
        exclude_set |= set(exclude)

    out: List[str] = []
    for f in schema:
        if f.name in exclude_set:
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
