from __future__ import annotations

from .utils import _arrow


def _project_table_to_schema(table, schema):
    """
    Project/align a table to the canonical schema:
      - reorder columns
      - add missing columns as typed nulls
      - cast columns to canonical types when safe
    """
    pa, pc, _ = _arrow()

    # Fast path
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
                col = pc.cast(col, field.type, safe=False)
            except Exception as ex:
                raise RuntimeError(
                    f"Failed to cast column '{name}' from {col.type} to {field.type}: {ex}"
                ) from ex
        arrays.append(col)

    return pa.Table.from_arrays(arrays, schema=schema)
