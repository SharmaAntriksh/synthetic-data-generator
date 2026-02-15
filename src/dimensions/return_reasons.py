from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from src.engine.dimension_loader import load_dimension
from src.utils.logging_utils import done, skip
from src.versioning.version_store import save_version

# ---------------------------------------------------------------------
# Canonical default reasons
# ---------------------------------------------------------------------
# Keep this constant (tuple format) for compatibility with existing code.
RETURN_REASONS: list[tuple[int, str, str]] = [
    (1, "Defective", "Quality"),
    (2, "Damaged in shipping", "Logistics"),
    (3, "Wrong item", "Fulfillment"),
    (4, "Not as described", "Customer"),
    (5, "No longer needed", "Customer"),
    (6, "Late delivery", "Logistics"),
    (7, "Better price found", "Customer"),
    (8, "Other", "Other"),
]

RETURN_REASON_SCHEMA = pa.schema(
    [
        pa.field("ReturnReasonKey", pa.int64()),
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
            raise ValueError(f"Duplicate ReturnReasonKey detected: {rr.key}")
        if not rr.reason or not rr.reason.strip():
            raise ValueError(f"Empty ReturnReason for key={rr.key}")
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
            if not isinstance(cur, Mapping) or p not in cur:
                return None
            cur = cur[p]
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
    Dimension runner entrypoint (matches other dimension modules).

    Behavior:
      - Writes return_reason.parquet to parquet_dims_folder
      - Skips when up-to-date unless forced via cfg["return_reason"]["_force_regenerate"] == True
      - "Up-to-date" is determined by dimension_loader.load_dimension using expected_config.
      - Persists expected_config via save_version(...) so future runs can skip.
    """
    parquet_dims_folder = Path(parquet_dims_folder)
    parquet_dims_folder.mkdir(parents=True, exist_ok=True)

    # Forced regeneration flag is passed by dimensions_runner via _cfg_for_dimension(...)
    dim_cfg = cfg.get("return_reason") if isinstance(cfg, Mapping) else None
    dim_cfg = dim_cfg if isinstance(dim_cfg, Mapping) else {}
    forced = bool(dim_cfg.get("_force_regenerate", False))

    # Determine reasons + stable expected_config for up-to-date checks
    raw_reasons = _extract_reasons_from_cfg(cfg) if isinstance(cfg, Mapping) else None
    if raw_reasons is None:
        raw_reasons = RETURN_REASONS

    normalized = _normalize_reasons(raw_reasons)

    # IMPORTANT: use JSON-stable structures (lists), not tuples, so version comparisons are stable.
    expected_config = {
        "reasons": [[r.key, r.reason, r.category] for r in normalized],
        "schema": "ReturnReason(v1)",
    }

    if not forced:
        _, changed = load_dimension("ReturnReason", parquet_dims_folder, expected_config)
        if not changed:
            skip("ReturnReason up-to-date; skipping.")
            return

    table = build_return_reason_dimension(reasons=raw_reasons)
    out_path = parquet_dims_folder / "return_reason.parquet"
    pq.write_table(table, out_path)

    # Your version_store.save_version requires output_path in this repo version.
    # Use the same folder load_dimension checks against.
    save_version("ReturnReason", expected_config, parquet_dims_folder)

    done("Generating ReturnReason completed")
