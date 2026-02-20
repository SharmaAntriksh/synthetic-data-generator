from __future__ import annotations

from typing import Iterable, List, Optional, Set

from ..output_paths import TABLE_SALES, TABLE_SALES_ORDER_DETAIL

# ===============================================================
# Sales policy/constants
# ===============================================================

# Columns we never dictionary-encode
DICT_EXCLUDE = {"SalesOrderNumber", "CustomerKey"}

# Columns that must always exist in Sales (line-grain)
REQUIRED_PRICING_COLS = {
    "UnitPrice",
    "NetPrice",
    "UnitCost",
    "DiscountAmount",
}


def validate_required_columns(
    schema,
    required: Iterable[str] | Set[str],
    *,
    what: str = "required columns",
) -> None:
    req = set(required or [])
    if not req:
        return

    names = set(getattr(schema, "names", []) or [])
    missing = req - names
    if missing:
        raise RuntimeError(f"Missing {what}: {sorted(missing)}")


def schema_dict_cols(schema, exclude: Optional[Iterable[str]] = None) -> List[str]:
    """
    Mechanics only:
      - dictionary-encode string/binary-like columns
      - skip excluded columns
    """
    try:
        import pyarrow as pa  # type: ignore
    except Exception as e:
        raise RuntimeError("pyarrow is required for dictionary-encoding decisions") from e

    exclude_set = set(exclude or [])
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
    """Legacy name kept for compatibility with existing imports."""
    required = required_pricing_cols_for_table(table_name)
    validate_required_columns(schema, required, what="required pricing columns")


def _schema_dict_cols(
    schema,
    exclude: Optional[Iterable[str]] = None,
    *,
    table_name: str | None = None,
) -> List[str]:
    """Legacy name kept for compatibility with existing imports."""
    _validate_required(schema, table_name=table_name)

    exclude_set = set(DICT_EXCLUDE)
    if exclude:
        exclude_set |= set(exclude)

    return schema_dict_cols(schema, exclude=exclude_set)