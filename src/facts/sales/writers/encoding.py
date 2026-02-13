from __future__ import annotations

from typing import Iterable, List, Optional, Set

from .constants import DICT_EXCLUDE, REQUIRED_PRICING_COLS
from ..output_paths import TABLE_SALES, TABLE_SALES_ORDER_DETAIL

from src.facts.common.writers.encoding import (
    schema_dict_cols as _common_schema_dict_cols,
    validate_required_columns,
)


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
    validate_required_columns(schema, required, what="required pricing columns")


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

    exclude_set = set(DICT_EXCLUDE)
    if exclude:
        exclude_set |= set(exclude)

    return _common_schema_dict_cols(schema, exclude=exclude_set)
