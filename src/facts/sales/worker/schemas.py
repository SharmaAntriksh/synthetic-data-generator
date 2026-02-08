from __future__ import annotations

from typing import List, Optional, Set

import pyarrow as pa


def schema_dict_cols(schema: pa.Schema, exclude: Optional[Set[str]] = None) -> List[str]:
    """
    Dictionary encode only string-ish columns (excluding some IDs).

    Matches prior behavior from sales_worker._schema_dict_cols().
    """
    exclude = exclude or set()

    out: List[str] = []
    for f in schema:
        if f.name in exclude:
            continue
        t = f.type
        if pa.types.is_string(t) or pa.types.is_large_string(t):
            out.append(f.name)
    return out
