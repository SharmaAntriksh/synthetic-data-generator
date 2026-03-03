from __future__ import annotations

from typing import Iterable, List, Optional, Set

from ..output_paths import TABLE_SALES, TABLE_SALES_ORDER_DETAIL

# ===============================================================
# Sales policy/constants
# ===============================================================

# Columns we never dictionary-encode.
# frozenset: immutable, so callers can use it directly without copying.
DICT_EXCLUDE: frozenset[str] = frozenset({"SalesOrderNumber", "CustomerKey"})

# Columns that must always exist in Sales (line-grain)
REQUIRED_PRICING_COLS: frozenset[str] = frozenset({
    "UnitPrice",
    "NetPrice",
    "UnitCost",
    "DiscountAmount",
})


def validate_required_columns(
    schema,
    required: Iterable[str] | Set[str],
    *,
    what: str = "required columns",
) -> None:
    req = frozenset(required) if required else frozenset()
    if not req:
        return

    names = frozenset(getattr(schema, "names", None) or ())
    missing = req - names
    if missing:
        raise RuntimeError(f"Missing {what}: {sorted(missing)}")


def schema_dict_cols(schema, exclude: Optional[Iterable[str]] = None) -> List[str]:
    """
    Mechanics only:
      - dictionary-encode string/binary-like columns
      - skip excluded columns
    """
    from .utils import _arrow
    pa, _, _ = _arrow()

    exclude_set = frozenset(exclude) if exclude is not None else frozenset()
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


# ===============================================================
# Sales-specific encoding policy
# ===============================================================

def required_pricing_cols_for_table(table_name: str | None) -> frozenset[str]:
    """
    Pricing cols are required only for line-grain tables.

    Back-compat:
      - If table_name is None, keep old strict behavior (require pricing cols).
    """
    if table_name is None or table_name in {TABLE_SALES, TABLE_SALES_ORDER_DETAIL}:
        return REQUIRED_PRICING_COLS
    return frozenset()


def _validate_required(schema, *, table_name: str | None = None) -> None:
    """Legacy name kept for compatibility with existing imports."""
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

    NOTE: This no longer calls _validate_required internally.
    Callers that need validation should call _validate_required separately
    before calling this function.  This avoids the double-validation issue
    where merge_parquet_files would validate twice per merge.
    """
    # Avoid copying DICT_EXCLUDE when there are no extra exclusions (common path).
    exclude_set: frozenset[str] | set[str] = (
        DICT_EXCLUDE | frozenset(exclude) if exclude else DICT_EXCLUDE
    )
    return schema_dict_cols(schema, exclude=exclude_set)
