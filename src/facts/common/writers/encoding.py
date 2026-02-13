from __future__ import annotations

from typing import Iterable, List, Optional, Set


def validate_required_columns(
    schema,
    required: Iterable[str] | Set[str],
    *,
    what: str = "required columns",
) -> None:
    """
    Generic required-column validation.
    Policy (which columns are required) must live in the fact module.
    """
    req = set(required or [])
    if not req:
        return

    names = set(getattr(schema, "names", []) or [])
    missing = req - names
    if missing:
        raise RuntimeError(f"Missing {what}: {sorted(missing)}")


def schema_dict_cols(
    schema,
    exclude: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Generic dictionary-encoding candidate columns for Parquet writers.

    Mechanics only:
      - dictionary-encode string/binary-like columns
      - skip excluded columns

    Policy stays outside:
      - required-column enforcement
      - table-specific exclusions
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


__all__ = ["validate_required_columns", "schema_dict_cols"]
