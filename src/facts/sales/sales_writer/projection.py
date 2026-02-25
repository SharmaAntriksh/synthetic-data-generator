from __future__ import annotations

from typing import Any


def project_table_to_schema(table: Any, schema: Any, *, cast_safe: bool = False) -> Any:
    """
    Project/align an Arrow table to the canonical schema:
      - reorder columns
      - add missing columns as typed nulls
      - cast columns to canonical types when safe/possible
    """
    try:
        import pyarrow as pa  # type: ignore
        import pyarrow.compute as pc  # type: ignore
    except Exception as e:
        raise RuntimeError("pyarrow is required for schema projection") from e

    if table.schema == schema:
        return table

    n = table.num_rows
    have = set(table.schema.names)

    arrays = []
    for field in schema:
        name = field.name

        if name not in have:
            arrays.append(pa.nulls(n, type=field.type))
            continue

        col = table[name]  # ChunkedArray
        if col.type != field.type:
            try:
                col = pc.cast(col, field.type, safe=bool(cast_safe))
            except Exception as ex:
                raise RuntimeError(
                    f"Failed to cast column '{name}' from {col.type} to {field.type}: {ex}"
                ) from ex

        arrays.append(col)

    return pa.Table.from_arrays(arrays, schema=schema)


def _project_table_to_schema(table: Any, schema: Any) -> Any:
    """Backward-compatible alias (many callers expect underscore name)."""
    return project_table_to_schema(table, schema, cast_safe=False)