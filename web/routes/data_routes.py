"""
web/routes/data_routes.py -- Dataset listing and data preview endpoints.

Supports CSV, Parquet, and Delta Lake formats with paginated row preview.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from web.shared_state import REPO_ROOT

router = APIRouter(prefix="/api", tags=["data"])

_DATASETS_DIR = REPO_ROOT / "generated_datasets"

# ---------------------------------------------------------------------------
# CSV row count cache (avoids re-scanning files on every pagination click)
# Key: file path string → (mtime, row_count)
# ---------------------------------------------------------------------------

_csv_row_cache: Dict[str, tuple[float, int]] = {}
_csv_row_cache_lock = threading.Lock()


def _csv_row_count(path: Path) -> int:
    """Return the row count (excluding header) for a CSV file, cached by mtime."""
    key = str(path)
    mtime = path.stat().st_mtime
    with _csv_row_cache_lock:
        cached = _csv_row_cache.get(key)
        if cached and cached[0] == mtime:
            return cached[1]
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        count = sum(1 for _ in fh) - 1  # subtract header
    with _csv_row_cache_lock:
        _csv_row_cache[key] = (mtime, count)
    return count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_format(dataset_path: Path) -> str:
    """Detect the output format of a dataset folder."""
    dims = dataset_path / "dimensions"
    facts = dataset_path / "facts"
    # Check for delta logs first
    for sub in (dims, facts):
        if sub.is_dir():
            for child in sub.iterdir():
                if child.is_dir() and (child / "_delta_log").is_dir():
                    return "delta"
    # Check file extensions
    for sub in (dims, facts):
        if sub.is_dir():
            for f in sub.iterdir():
                if f.suffix == ".parquet":
                    return "parquet"
                if f.suffix == ".csv":
                    return "csv"
            # Check one level deeper (facts have subfolders)
            for child in sub.iterdir():
                if child.is_dir():
                    for f in child.iterdir():
                        if f.suffix == ".parquet":
                            return "parquet"
                        if f.suffix == ".csv":
                            return "csv"
    return "unknown"


def _discover_tables(dataset_path: Path, fmt: str) -> List[Dict[str, Any]]:
    """Discover all tables in a dataset folder."""
    tables = []

    for category in ("dimensions", "facts"):
        cat_path = dataset_path / category
        if not cat_path.is_dir():
            continue

        if fmt == "delta":
            # Delta tables are directories with _delta_log
            for child in sorted(cat_path.iterdir()):
                if child.is_dir() and (child / "_delta_log").is_dir():
                    tables.append({
                        "name": child.name,
                        "category": category,
                        "format": "delta",
                        "path": str(child.relative_to(dataset_path)),
                    })
        elif fmt == "parquet":
            # Parquet: single files directly in dims/facts
            for f in sorted(cat_path.iterdir()):
                if f.suffix == ".parquet":
                    tables.append({
                        "name": f.stem,
                        "category": category,
                        "format": "parquet",
                        "path": str(f.relative_to(dataset_path)),
                    })
        else:
            # CSV: files directly in dims, or in subfolders for facts
            for entry in sorted(cat_path.iterdir()):
                if entry.suffix == ".csv":
                    tables.append({
                        "name": entry.stem,
                        "category": category,
                        "format": "csv",
                        "path": str(entry.relative_to(dataset_path)),
                    })
                elif entry.is_dir():
                    csv_files = sorted(entry.glob("*.csv"))
                    if csv_files:
                        tables.append({
                            "name": entry.name,
                            "category": category,
                            "format": "csv",
                            "path": str(csv_files[0].relative_to(dataset_path)),
                        })

    return tables


def _read_preview(dataset_path: Path, table: Dict[str, Any],
                  offset: int, limit: int) -> Dict[str, Any]:
    """Read a slice of rows from a table."""
    fmt = table["format"]
    full_path = dataset_path / table["path"]

    if fmt == "csv":
        # For chunked CSVs, find all chunks in the parent dir
        parent = full_path.parent
        csv_files = sorted(parent.glob("*.csv")) if parent.name != "dimensions" else [full_path]
        if not csv_files:
            csv_files = [full_path]

        # Read just the columns/header from first file
        header_df = pd.read_csv(csv_files[0], nrows=0)
        columns = header_df.columns.tolist()

        # Count total rows across all chunks (cached by mtime)
        chunk_counts = [_csv_row_count(f) for f in csv_files]
        total_rows = sum(chunk_counts)

        # Read the requested slice
        rows_skipped = 0
        collected = []
        for f, chunk_rows in zip(csv_files, chunk_counts):

            if rows_skipped + chunk_rows <= offset:
                rows_skipped += chunk_rows
                continue

            skip_in_file = max(0, offset - rows_skipped)
            need = limit - len(collected)
            df = pd.read_csv(f, skiprows=range(1, skip_in_file + 1), nrows=need)
            collected.append(df)
            rows_skipped += chunk_rows

            if len(collected) > 0 and sum(len(d) for d in collected) >= limit:
                break

        if collected:
            result_df = pd.concat(collected, ignore_index=True).head(limit)
        else:
            result_df = pd.DataFrame(columns=columns)

    elif fmt == "parquet":
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(str(full_path))
        total_rows = pf.metadata.num_rows
        columns = pf.schema_arrow.names

        # Read only the needed row groups
        rows_seen = 0
        collected = []
        for rg_idx in range(pf.metadata.num_row_groups):
            rg_rows = pf.metadata.row_group(rg_idx).num_rows
            if rows_seen + rg_rows <= offset:
                rows_seen += rg_rows
                continue

            rg_table = pf.read_row_group(rg_idx)
            rg_df = rg_table.to_pandas()
            skip_in_rg = max(0, offset - rows_seen)
            rg_df = rg_df.iloc[skip_in_rg:]
            collected.append(rg_df)
            rows_seen += rg_rows

            if sum(len(d) for d in collected) >= limit:
                break

        if collected:
            result_df = pd.concat(collected, ignore_index=True).head(limit)
        else:
            result_df = pd.DataFrame(columns=columns)

    elif fmt == "delta":
        from deltalake import DeltaTable
        dt = DeltaTable(str(full_path))
        ds = dt.to_pyarrow_dataset()
        total_rows = dt.to_pyarrow_table().num_rows
        columns = dt.schema().to_pyarrow().names

        # Read a slice via scanner
        scanner = ds.scanner(columns=columns)
        batches = scanner.to_batches()
        rows_seen = 0
        collected = []
        for batch in batches:
            batch_len = batch.num_rows
            if rows_seen + batch_len <= offset:
                rows_seen += batch_len
                continue

            batch_df = batch.to_pandas()
            skip_in_batch = max(0, offset - rows_seen)
            batch_df = batch_df.iloc[skip_in_batch:]
            collected.append(batch_df)
            rows_seen += batch_len

            if sum(len(d) for d in collected) >= limit:
                break

        if collected:
            result_df = pd.concat(collected, ignore_index=True).head(limit)
        else:
            result_df = pd.DataFrame(columns=columns)
    else:
        raise HTTPException(400, f"Unsupported format: {fmt}")

    # Convert to JSON-safe types
    for col in result_df.columns:
        if result_df[col].dtype.kind in ("M",):  # datetime
            result_df[col] = result_df[col].astype(str)
        elif result_df[col].dtype.kind in ("m",):  # timedelta
            result_df[col] = result_df[col].astype(str)

    return {
        "columns": columns,
        "rows": result_df.fillna("").values.tolist(),
        "total_rows": total_rows,
        "offset": offset,
        "limit": limit,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/datasets")
def list_datasets():
    """List all generated datasets with metadata."""
    if not _DATASETS_DIR.is_dir():
        return {"datasets": []}

    datasets = []
    for entry in sorted(_DATASETS_DIR.iterdir(), reverse=True):
        if not entry.is_dir():
            continue
        # Skip hidden/system folders
        if entry.name.startswith(".") or entry.name.startswith("_"):
            continue

        fmt = _detect_format(entry)
        tables = _discover_tables(entry, fmt)

        # Calculate total size
        total_size = 0
        for root, _dirs, files in os.walk(entry):
            for f in files:
                total_size += os.path.getsize(os.path.join(root, f))

        datasets.append({
            "name": entry.name,
            "format": fmt,
            "table_count": len(tables),
            "size_mb": round(total_size / (1024 * 1024), 1),
            "has_config": (entry / "config" / "config.yaml").exists(),
        })

    return {"datasets": datasets}


@router.get("/datasets/{folder}/tables")
def list_tables(folder: str):
    """List all tables in a specific dataset."""
    dataset_path = _DATASETS_DIR / folder
    if not dataset_path.is_dir():
        raise HTTPException(404, f"Dataset not found: {folder}")

    # Security: ensure the resolved path is within generated_datasets
    try:
        dataset_path.resolve().relative_to(_DATASETS_DIR.resolve())
    except ValueError:
        raise HTTPException(400, "Invalid dataset path")

    fmt = _detect_format(dataset_path)
    tables = _discover_tables(dataset_path, fmt)

    return {"folder": folder, "format": fmt, "tables": tables}


@router.get("/datasets/{folder}/tables/{table}/preview")
def preview_table(
    folder: str,
    table: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
):
    """Preview rows from a table with pagination."""
    dataset_path = _DATASETS_DIR / folder
    if not dataset_path.is_dir():
        raise HTTPException(404, f"Dataset not found: {folder}")

    try:
        dataset_path.resolve().relative_to(_DATASETS_DIR.resolve())
    except ValueError:
        raise HTTPException(400, "Invalid dataset path")

    fmt = _detect_format(dataset_path)
    tables = _discover_tables(dataset_path, fmt)

    # Find the requested table
    table_info = None
    for t in tables:
        if t["name"] == table:
            table_info = t
            break

    if table_info is None:
        raise HTTPException(404, f"Table not found: {table}")

    try:
        preview = _read_preview(dataset_path, table_info, offset, limit)
    except Exception as exc:
        raise HTTPException(500, f"Failed to read table: {exc}")

    return {
        "folder": folder,
        "table": table,
        "category": table_info["category"],
        "format": table_info["format"],
        **preview,
    }
