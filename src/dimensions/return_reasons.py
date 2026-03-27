from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from src.defaults import RETURN_REASONS as _CANONICAL_REASONS
from src.exceptions import DimensionError
from src.utils.logging_utils import done, skip
from src.versioning.version_store import should_regenerate, save_version


# Derived from defaults.py (single source of truth), re-exported as (key, label, category) tuples
RETURN_REASONS: list[tuple[int, str, str]] = [
    (k, lbl, cat) for k, lbl, cat, _w in _CANONICAL_REASONS
]

RETURN_REASON_SCHEMA = pa.schema(
    [
        pa.field("ReturnReasonKey", pa.int32()),
        pa.field("ReturnReason", pa.string()),
        pa.field("ReturnReasonCategory", pa.string()),
    ]
)


@dataclass(frozen=True)
class ReturnReason:
    key: int
    reason: str
    category: str


def _parse_reason_item(item: Any, idx: int) -> ReturnReason:
    """
    Acceptable forms:
      - (key, reason, category)
      - {"key":1, "label":"Defective", "category":"Quality"}  (or reason/ReturnReason)
      - {"ReturnReasonKey":1, "ReturnReason":"Defective", "ReturnReasonCategory":"Quality"}
      - "Some reason"  (auto key/category)
    """
    if isinstance(item, ReturnReason):
        return item

    if isinstance(item, tuple) and len(item) == 3:
        k, r, c = item
        return ReturnReason(int(k), str(r), str(c))

    if isinstance(item, Mapping):
        # Be permissive with field names
        k = item.get("key", item.get("ReturnReasonKey", idx))
        r = item.get("label", item.get("reason", item.get("ReturnReason", f"Reason {idx}")))
        c = item.get("category", item.get("ReturnReasonCategory", "Other"))
        return ReturnReason(int(k), str(r), str(c))

    # Fallback: treat as label string
    return ReturnReason(int(idx), str(item), "Other")


def _normalize_reasons(reasons: Iterable[Any]) -> list[ReturnReason]:
    out: list[ReturnReason] = []
    seen: set[int] = set()

    for i, item in enumerate(reasons, start=1):
        rr = _parse_reason_item(item, i)
        if rr.key in seen:
            raise DimensionError(f"Duplicate ReturnReasonKey detected: {rr.key}")
        if not rr.reason or not rr.reason.strip():
            raise DimensionError(f"Empty ReturnReason for key={rr.key}")
        if not rr.category or not rr.category.strip():
            rr = ReturnReason(rr.key, rr.reason, "Other")

        seen.add(rr.key)
        out.append(rr)

    out.sort(key=lambda x: x.key)
    return out


def _extract_reasons_from_cfg(cfg: Mapping[str, Any]) -> Optional[Sequence[Any]]:
    """
    Optional future-proofing: if a runner passes cfg here, support overrides.

    Supported locations (first match wins):
      - cfg["returns"]["reasons"]
      - cfg["models"]["returns"]["reasons"]
      - cfg["models_cfg"]["models"]["returns"]["reasons"]
    """

    def _get(path: Sequence[str]) -> Any:
        cur: Any = cfg
        for p in path:
            if isinstance(cur, dict):
                if p not in cur:
                    return None
                cur = cur[p]
            elif isinstance(cur, Mapping):
                val = getattr(cur, p, None)
                if val is None:
                    return None
                cur = val
            else:
                return None
        return cur

    for path in (
        ("returns", "reasons"),
        ("models", "returns", "reasons"),
        ("models_cfg", "models", "returns", "reasons"),
    ):
        v = _get(path)
        if isinstance(v, Sequence) and not isinstance(v, (str, bytes)) and len(v) > 0:
            return v

    return None


def build_return_reason_dimension(
    cfg: Optional[Mapping[str, Any]] = None,
    reasons: Optional[Sequence[Any]] = None,
) -> pa.Table:
    """
    Build ReturnReason dimension as a pyarrow.Table.

    Backward compatible:
      - Existing callers can call build_return_reason_dimension() with no args.

    Optional overrides:
      - pass `reasons=[...]` directly
      - or pass `cfg` containing returns/models returns reasons.
    """
    if reasons is None and cfg is not None:
        reasons = _extract_reasons_from_cfg(cfg)

    if reasons is None:
        reasons = RETURN_REASONS

    rr = _normalize_reasons(reasons)

    keys = pa.array([x.key for x in rr], type=pa.int64())
    labels = pa.array([x.reason for x in rr], type=pa.string())
    cats = pa.array([x.category for x in rr], type=pa.string())

    return pa.Table.from_arrays(
        [keys, labels, cats],
        schema=RETURN_REASON_SCHEMA,
    )


def run_return_reasons(cfg: Mapping[str, Any], parquet_dims_folder: Path) -> None:
    """
    Writes return_reason.parquet to parquet_dims_folder.
    Skips when up-to-date (version_store check via should_regenerate/save_version).
    """
    parquet_dims_folder = Path(parquet_dims_folder)
    parquet_dims_folder.mkdir(parents=True, exist_ok=True)

    dim_cfg = getattr(cfg, "return_reason", None) if isinstance(cfg, Mapping) else None
    dim_cfg = dim_cfg if isinstance(dim_cfg, Mapping) else {}
    # Determine reasons + stable expected_config
    raw_reasons = _extract_reasons_from_cfg(cfg) if isinstance(cfg, Mapping) else None
    if raw_reasons is None:
        raw_reasons = RETURN_REASONS

    normalized = _normalize_reasons(raw_reasons)

    expected_config = {
        "schema_version": 1,
        "schema": "return_reason(v1)",
        # JSON-stable structures (lists), not tuples
        "reasons": [[r.key, r.reason, r.category] for r in normalized],
    }

    out_path = parquet_dims_folder / "return_reason.parquet"

    if not should_regenerate("return_reason", expected_config, out_path):
        skip("Return Reason up-to-date")
        return

    table = build_return_reason_dimension(reasons=raw_reasons)
    pq.write_table(table, out_path)

    save_version("return_reason", expected_config, out_path)
    done("Generating Return Reason completed")
