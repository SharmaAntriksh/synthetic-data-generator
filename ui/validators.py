# ui/validators.py
import os
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple


def cpu_count_safe() -> int:
    return os.cpu_count() or 1


def _get(d: Dict[str, Any], path: List[str]) -> Tuple[Any, bool]:
    cur: Any = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None, False
        cur = cur[k]
    return cur, True


def _parse_date(v: Any) -> Optional[date]:
    """
    Accept:
      - 'YYYY-MM-DD' string
      - datetime.date
      - datetime.datetime
    Return: datetime.date or None if invalid.
    """
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    if isinstance(v, str):
        try:
            return date.fromisoformat(v)
        except ValueError:
            return None
    return None


def _as_int(v: Any) -> Optional[int]:
    if v is None or isinstance(v, bool):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _as_bool(v: Any) -> Optional[bool]:
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    # be conservative: don't coerce strings here; UI should pass bools
    return None


def validate(cfg: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Returns (errors, warnings)
    UI-safe: never throws for missing keys; reports errors instead.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # -----------------------------
    # Dates
    # -----------------------------
    dates, ok = _get(cfg, ["defaults", "dates"])
    if not ok or not isinstance(dates, dict):
        errors.append("Missing defaults.dates in config.")
        return errors, warnings

    start_raw = dates.get("start")
    end_raw = dates.get("end")

    start = _parse_date(start_raw)
    end = _parse_date(end_raw)

    if start is None or end is None:
        errors.append("Start date and end date must be set as ISO dates (YYYY-MM-DD).")
    elif end < start:
        errors.append("End date must be after start date.")

    # -----------------------------
    # Sales basics
    # -----------------------------
    sales, ok = _get(cfg, ["sales"])
    if not ok or not isinstance(sales, dict):
        errors.append("Missing sales section in config.")
        return errors, warnings

    total_rows = _as_int(sales.get("total_rows"))
    if total_rows is None:
        errors.append("sales.total_rows must be an integer.")
    elif total_rows <= 0:
        errors.append("Total rows must be greater than zero.")

    chunk_size = _as_int(sales.get("chunk_size"))
    if chunk_size is None:
        errors.append("sales.chunk_size must be an integer.")
    elif chunk_size <= 0:
        errors.append("Chunk size must be greater than zero.")
    elif total_rows is not None and chunk_size > total_rows:
        warnings.append("Chunk size exceeds total rows.")

    file_format = str(sales.get("file_format", "")).strip().lower()
    if file_format == "delta":
        # Allow the alias but guide toward canonical value
        warnings.append('Output format "delta" is treated as "deltaparquet".')
        file_format = "deltaparquet"

    allowed_formats = {"csv", "parquet", "deltaparquet"}
    if file_format not in allowed_formats:
        errors.append("sales.file_format must be one of: csv, parquet, deltaparquet.")

    # CSV warning threshold (keep your existing heuristic)
    if file_format == "csv" and total_rows is not None and total_rows > 5_000_000:
        warnings.append("Large CSV outputs can be slow and very large.")

    # -----------------------------
    # Workers
    # -----------------------------
    workers_raw = sales.get("workers")
    workers = _as_int(workers_raw)

    # Allow null (auto-detect behavior mentioned in config)
    if workers_raw is not None and workers is None:
        errors.append("sales.workers must be an integer or null.")
    elif workers is not None:
        if workers <= 0:
            errors.append("Workers must be greater than zero (or null for auto-detect).")
        elif workers > cpu_count_safe():
            warnings.append("Workers exceed CPU cores.")

    # -----------------------------
    # Parquet / delta-parquet specifics
    # -----------------------------
    row_group_size_raw = sales.get("row_group_size")
    row_group_size = _as_int(row_group_size_raw)

    # Only validate row_group_size when the format can actually use it.
    if file_format in ("parquet", "deltaparquet"):
        if row_group_size_raw is not None:
            if row_group_size is None or row_group_size <= 0:
                errors.append("sales.row_group_size must be a positive integer.")
            elif chunk_size is not None and row_group_size > chunk_size:
                warnings.append("row_group_size exceeds chunk_size (may reduce write efficiency).")
    # CSV mode: ignore row_group_size entirely (config may still carry it)

    # -----------------------------
    # Required output paths (quick sanity)
    # -----------------------------
    if not sales.get("parquet_folder"):
        errors.append("sales.parquet_folder must be set.")
    if not sales.get("out_folder"):
        errors.append("sales.out_folder must be set.")

    # -----------------------------
    # Minor type sanity (non-fatal)
    # -----------------------------
    skip_order_cols = _as_bool(sales.get("skip_order_cols"))
    if sales.get("skip_order_cols") is not None and skip_order_cols is None:
        warnings.append("sales.skip_order_cols should be a boolean (true/false).")

    return errors, warnings
