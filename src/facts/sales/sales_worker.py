"""
Single-file Sales worker module (source of truth).

This file intentionally combines:
  - src/facts/sales/sales_worker.py
  - src/facts/common/worker/{init,task,pool,chunk_io,return_builder}.py

Goal: during the ongoing refactor, keep all Sales worker mechanics + shared worker
utilities in one place so you don't have to navigate many small files. You can
split again later once the design for adding new fact tables stabilizes.

Backwards compatible imports (keep stable):
  from src.facts.sales.sales_worker import init_sales_worker, _worker_task
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union, Callable, Iterator, Iterable

import numpy as np
import pyarrow as pa

from .sales_logic import chunk_builder
from .sales_logic.globals import State, bind_globals
from .output_paths import (
    OutputPaths,
    TABLE_SALES,
    TABLE_SALES_ORDER_DETAIL,
    TABLE_SALES_ORDER_HEADER,
)

# Optional: returns table constant may or may not exist depending on branch/version.
try:
    from .output_paths import TABLE_SALES_RETURN  # type: ignore
except Exception:
    TABLE_SALES_RETURN = None  # type: ignore


# ============================================================================
# common/worker/init.py (inlined)
# ============================================================================
from typing import Any, Optional

import numpy as np


def build_buckets_from_key(key: Any) -> list[np.ndarray]:
    """
    Build index buckets for a dense-ish non-negative integer key.

    Returns:
        buckets[k] = np.ndarray of row indices i where key[i] == k

    Notes:
      - Keys are coerced to int64.
      - Keys must be non-negative (0..K). Sparse keys are supported but allocate up to max(key)+1.
      - Stable grouping (uses mergesort).
    """
    key_np = np.asarray(key, dtype=np.int64)
    if key_np.size == 0:
        return []

    if key_np.min() < 0:
        raise RuntimeError("Key values must be non-negative integers")

    max_k = int(key_np.max())
    K = max_k + 1

    order = np.argsort(key_np, kind="mergesort")
    k_sorted = key_np[order]

    starts = np.flatnonzero(np.r_[True, k_sorted[1:] != k_sorted[:-1]])
    ends = np.r_[starts[1:], k_sorted.size]

    buckets: list[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(K)]
    for s, e in zip(starts, ends):
        k = int(k_sorted[int(s)])
        buckets[k] = order[int(s) : int(e)].astype(np.int64, copy=False)

    return buckets


def int_or(v: Any, default: int) -> int:
    """Parse int with a safe default for None/empty/invalid."""
    try:
        if v is None or v == "":
            return int(default)
        return int(v)
    except Exception:
        return int(default)


def float_or(v: Any, default: float) -> float:
    """Parse float with a safe default for None/empty/invalid."""
    try:
        if v is None or v == "":
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def str_or(v: Any, default: str) -> str:
    """Parse/normalize string with a safe default for None/empty."""
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def as_int64(x: Any) -> np.ndarray:
    """Coerce array-like to int64 ndarray."""
    return np.asarray(x, dtype=np.int64)


def as_f64(x: Any) -> np.ndarray:
    """Coerce array-like to float64 ndarray."""
    return np.asarray(x, dtype=np.float64)


def dense_map(mapping: Optional[dict]) -> Optional[np.ndarray]:
    """
    Build a dense lookup array: arr[key] -> value, missing -> -1.

    - Keys/values coerced to int64.
    - Returns None for empty/invalid mapping.
    """
    if not mapping:
        return None

    keys = np.fromiter((int(k) for k in mapping.keys()), dtype=np.int64)
    vals = np.fromiter((int(v) for v in mapping.values()), dtype=np.int64)

    if keys.size == 0:
        return None

    max_key = int(keys.max())
    if max_key < 0:
        return None

    arr = np.full(max_key + 1, -1, dtype=np.int64)
    arr[keys] = vals
    return arr


def infer_T_from_date_pool(date_pool: Any) -> int:
    """
    Infer number of unique months in a date pool.

    Returned T is count of unique numpy datetime64[M] buckets.
    """
    dp = np.asarray(date_pool, dtype="datetime64[D]")
    months = dp.astype("datetime64[M]")
    return int(np.unique(months).size)


# ------------------------------------------------------------------
# Backward-compatible aliases (Sales worker init historically used _*)
# ------------------------------------------------------------------
def _build_buckets_from_brand_key(brand_key: Any) -> list[np.ndarray]:
    return build_buckets_from_key(brand_key)


def _int_or(v: Any, default: int) -> int:
    return int_or(v, default)


def _float_or(v: Any, default: float) -> float:
    return float_or(v, default)


def _str_or(v: Any, default: str) -> str:
    return str_or(v, default)


def _as_int64(x: Any) -> np.ndarray:
    return as_int64(x)


def _as_f64(x: Any) -> np.ndarray:
    return as_f64(x)


def _dense_map(mapping: Optional[dict]) -> Optional[np.ndarray]:
    return dense_map(mapping)


def _infer_T_from_date_pool(date_pool: Any) -> int:
    return infer_T_from_date_pool(date_pool)


# ============================================================================
# common/worker/task.py (inlined)
# ============================================================================
import os
from typing import Any, Callable, Dict, Iterable, List, Sequence, Tuple, Union

import pyarrow as pa



Task = Tuple[int, int, Any]  # (idx, batch_size, seed)
TaskArgs = Union[Task, Sequence[Task]]


def normalize_tasks(args: TaskArgs) -> Tuple[List[Task], bool]:
    """
    Supports:
      - single task: (idx, batch_size, seed)
      - batched tasks: [(idx, batch_size, seed), ...]
    Returns: (tasks, single)
    """
    if isinstance(args, tuple):
        # single task
        if len(args) != 3:
            raise ValueError(f"Task tuple must be (idx,batch_size,seed), got len={len(args)}")
        return [args], True

    tasks = list(args)
    return tasks, False


def derive_chunk_seed(seed: Any, idx: int, *, stride: int = 10_000) -> int:
    """
    Deterministic per-chunk seed derivation.
    """
    base_seed = int_or(seed, 0)
    return int(base_seed) + int(idx) * int(stride)


def write_table_by_format(
    *,
    file_format: str,
    output_paths: Any,
    table_name: str,
    idx: int,
    table: pa.Table,
    write_csv_fn: Callable[[pa.Table, str], None],
    write_parquet_fn: Callable[[pa.Table, str], None],
) -> Union[str, Dict[str, Any]]:
    """
    Generic writer shim used by worker tasks.

    Requires output_paths to provide:
      - delta_part_path(table_name, idx) -> str   (deltaparquet only)
      - chunk_path(table_name, idx, ext) -> str   (csv/parquet)

    Returns:
      - csv/parquet: full path (str)
      - deltaparquet: {"part": basename, "rows": n}
    """
    ff = (file_format or "").strip().lower()
    if ff == "deltaparquet":
        if not hasattr(output_paths, "delta_part_path"):
            raise RuntimeError("output_paths must implement delta_part_path() for deltaparquet")
        path = output_paths.delta_part_path(table_name, int(idx))
        write_parquet_fn(table, path)
        return {"part": os.path.basename(path), "rows": table.num_rows}

    if ff == "csv":
        if not hasattr(output_paths, "chunk_path"):
            raise RuntimeError("output_paths must implement chunk_path() for csv")
        path = output_paths.chunk_path(table_name, int(idx), "csv")
        write_csv_fn(table, path)
        return path

    # parquet (default)
    if not hasattr(output_paths, "chunk_path"):
        raise RuntimeError("output_paths must implement chunk_path() for parquet")
    path = output_paths.chunk_path(table_name, int(idx), "parquet")
    write_parquet_fn(table, path)
    return path


# ============================================================================
# common/worker/pool.py (inlined)
# ============================================================================
from dataclasses import dataclass
from multiprocessing import Pool
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence, Tuple


@dataclass(frozen=True)
class PoolRunSpec:
    processes: int
    chunksize: int = 1
    maxtasksperchild: Optional[int] = None
    label: str = ""


def iter_imap_unordered(
    *,
    tasks: Sequence[Any],
    task_fn: Callable[[Any], Any],
    spec: PoolRunSpec,
    initializer: Optional[Callable[..., Any]] = None,
    initargs: Tuple[Any, ...] = (),
) -> Iterator[Any]:
    """
    Generic multiprocessing runner.

    - 'tasks' must be pickleable (dict/tuple/list of primitives is ideal)
    - 'task_fn' and 'initializer' must be top-level functions (importable) for Windows spawn
    - yields results from Pool.imap_unordered(task_fn, tasks)

    This is mechanics only: no assumptions about the task payload or return type.
    """
    if spec.processes <= 0:
        raise ValueError("spec.processes must be >= 1")

    with Pool(
        processes=spec.processes,
        initializer=initializer,
        initargs=initargs,
        maxtasksperchild=spec.maxtasksperchild,
    ) as pool:
        # NOTE: Pool.imap_unordered chunksize is an int >= 1
        for result in pool.imap_unordered(task_fn, tasks, chunksize=max(1, int(spec.chunksize))):
            yield result


# ============================================================================
# common/worker/chunk_io.py (inlined)
# ============================================================================
import os
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import pyarrow as pa


# -------------------------
# Lazy imports
# -------------------------
def _pa_compute():
    import pyarrow.compute as pc  # type: ignore
    return pc


def _pa_csv():
    import pyarrow.csv as pacsv  # type: ignore
    return pacsv


def _pa_parquet():
    import pyarrow.parquet as pq  # type: ignore
    return pq


# -------------------------
# Config + hooks
# -------------------------
@dataclass(frozen=True)
class ChunkIOConfig:
    compression: str = "snappy"
    row_group_size: int = 1_000_000
    write_statistics: bool = True


# Hook types (policy injected by fact modules)
EnsureColsFn = Callable[[pa.Table], pa.Table]
CsvPostprocessFn = Callable[[pa.Table], pa.Table]


# -------------------------
# Generic helpers
# -------------------------
def add_year_month_from_date(
    table: pa.Table,
    *,
    date_cols: Sequence[str] = ("OrderDate", "DeliveryDate"),
    year_col: str = "Year",
    month_col: str = "Month",
    year_type: pa.DataType = pa.int16(),
    month_type: pa.DataType = pa.int16(),
) -> pa.Table:
    """
    Generic helper: derive Year/Month from the first available date column.
    Useful for delta partition columns.
    """
    date_col = None
    for c in date_cols:
        if c in table.schema.names:
            date_col = c
            break
    if not date_col:
        return table

    pc = _pa_compute()
    year = pc.cast(pc.year(table[date_col]), year_type)
    month = pc.cast(pc.month(table[date_col]), month_type)

    if year_col not in table.schema.names:
        table = table.append_column(year_col, year)
    if month_col not in table.schema.names:
        table = table.append_column(month_col, month)
    return table


def normalize_to_schema(
    table: pa.Table,
    expected: pa.Schema,
    *,
    table_name: Optional[str] = None,
    ensure_cols: Optional[Sequence[str]] = None,
    ensure_cols_fn: Optional[EnsureColsFn] = None,
) -> pa.Table:
    """
    Strict normalization:
      - optionally ensure/derive columns (e.g., Year/Month) if expected schema wants them
      - require exact column-name match (set equality)
      - reorder + cast columns exactly to expected

    Policy is external:
      - which columns to ensure
      - how to derive them (ensure_cols_fn)
    """
    expected = expected.remove_metadata()

    # Optional auto-add / derive of partition columns
    if ensure_cols and ensure_cols_fn:
        wants = all(c in expected.names for c in ensure_cols)
        missing_any = any(c not in table.schema.names for c in ensure_cols)
        if wants and missing_any:
            table = ensure_cols_fn(table)

    got_names = set(table.schema.names)
    exp_names = set(expected.names)
    if got_names != exp_names:
        missing = sorted(exp_names - got_names)
        extra = sorted(got_names - exp_names)
        raise RuntimeError(
            "Schema mismatch in writer.\n"
            f"Table: {table_name or 'unknown'}\n"
            f"Missing: {missing}\n"
            f"Extra: {extra}\n\n"
            f"Expected:\n{expected}\n\nGot:\n{table.schema}"
        )

    # Reorder + cast columns exactly
    pc = _pa_compute()
    arrays = []
    for field in expected:
        col = table[field.name]
        if col.type != field.type:
            try:
                col = pc.cast(col, field.type, safe=False)
            except Exception as ex:
                raise RuntimeError(
                    f"[{table_name or 'unknown'}] Failed cast '{field.name}': {col.type} -> {field.type}: {ex}"
                ) from ex
        arrays.append(col)

    return pa.Table.from_arrays(arrays, schema=expected)


def write_parquet_table(
    table: pa.Table,
    path: str,
    *,
    expected_schema: pa.Schema,
    cfg: ChunkIOConfig,
    use_dictionary: Optional[Sequence[str]] = None,
    table_name: Optional[str] = None,
    ensure_cols: Optional[Sequence[str]] = None,
    ensure_cols_fn: Optional[EnsureColsFn] = None,
) -> None:
    pq = _pa_parquet()

    table = normalize_to_schema(
        table,
        expected_schema,
        table_name=table_name,
        ensure_cols=ensure_cols,
        ensure_cols_fn=ensure_cols_fn,
    )

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    writer = pq.ParquetWriter(
        path,
        table.schema,
        compression=cfg.compression,
        use_dictionary=list(use_dictionary) if use_dictionary else False,
        write_statistics=bool(cfg.write_statistics),
    )
    try:
        writer.write_table(table, row_group_size=int(max(1, cfg.row_group_size)))
    finally:
        writer.close()


def write_csv_table(
    table: pa.Table,
    path: str,
    *,
    expected_schema: pa.Schema,
    table_name: Optional[str] = None,
    ensure_cols: Optional[Sequence[str]] = None,
    ensure_cols_fn: Optional[EnsureColsFn] = None,
    postprocess: Optional[CsvPostprocessFn] = None,
) -> None:
    pacsv = _pa_csv()

    table = normalize_to_schema(
        table,
        expected_schema,
        table_name=table_name,
        ensure_cols=ensure_cols,
        ensure_cols_fn=ensure_cols_fn,
    )

    if postprocess is not None:
        table = postprocess(table)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    pacsv.write_csv(
        table,
        path,
        write_options=pacsv.WriteOptions(include_header=True, quoting_style="none"),
    )


# ============================================================================
# common/worker/return_builder.py (inlined)
# ============================================================================
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pyarrow as pa


@dataclass(frozen=True)
class ReturnsConfig:
    enabled: bool = False
    return_rate: float = 0.0  # probability per SalesOrderDetail line
    max_lag_days: int = 60    # return date = delivery date + lag_days
    reason_keys: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8)
    reason_probs: Sequence[float] = (0.20, 0.12, 0.14, 0.10, 0.14, 0.08, 0.07, 0.15)


def _as_np_i64(x) -> np.ndarray:
    arr = np.asarray(x)
    return arr if arr.dtype == np.int64 else arr.astype(np.int64, copy=False)


def _as_np_f64(x) -> np.ndarray:
    arr = np.asarray(x)
    return arr if arr.dtype == np.float64 else arr.astype(np.float64, copy=False)


def _col_np(table: pa.Table, name: str) -> np.ndarray:
    if name not in table.column_names:
        raise RuntimeError(f"Returns builder: missing required column {name!r}")
    return table[name].combine_chunks().to_numpy(zero_copy_only=False)


def build_sales_returns_from_detail(
    detail: pa.Table,
    *,
    chunk_seed: int,
    cfg: ReturnsConfig,
) -> pa.Table:
    """
    Build a SalesReturn-like fact table from SalesOrderDetail lines.

    Deterministic per chunk (seeded by chunk_seed). Returns are sampled per line.

    Required input columns (from raw detail table):
      SalesOrderNumber, SalesOrderLineNumber, CustomerKey, ProductKey, StoreKey,
      PromotionKey, CurrencyKey, DeliveryDate, Quantity, UnitPrice, DiscountAmount,
      NetPrice, UnitCost
    """
    if not cfg.enabled or cfg.return_rate <= 0:
        return pa.table({})[:0]

    n = int(detail.num_rows)
    if n <= 0:
        return pa.table({})[:0]

    rng = np.random.default_rng(int(chunk_seed) ^ 0xA5A5_F00D)

    mask = rng.random(n) < float(cfg.return_rate)
    if not bool(mask.any()):
        return pa.table({})[:0]

    so = _as_np_i64(_col_np(detail, "SalesOrderNumber"))[mask]
    line = _as_np_i64(_col_np(detail, "SalesOrderLineNumber"))[mask]
    cust = _as_np_i64(_col_np(detail, "CustomerKey"))[mask]
    prod = _as_np_i64(_col_np(detail, "ProductKey"))[mask]
    store = _as_np_i64(_col_np(detail, "StoreKey"))[mask]
    promo = _as_np_i64(_col_np(detail, "PromotionKey"))[mask]
    curr = _as_np_i64(_col_np(detail, "CurrencyKey"))[mask]

    delivery = _col_np(detail, "DeliveryDate")[mask].astype("datetime64[D]", copy=False)

    qty = _as_np_i64(_col_np(detail, "Quantity"))[mask]
    if np.any(qty <= 0):
        keep = qty > 0
        so, line, cust, prod, store, promo, curr = (
            so[keep], line[keep], cust[keep], prod[keep], store[keep], promo[keep], curr[keep]
        )
        delivery, qty = delivery[keep], qty[keep]
        if qty.size == 0:
            return pa.table({})[:0]

    unit_price = _as_np_f64(_col_np(detail, "UnitPrice"))[mask][: qty.size]
    disc_amt = _as_np_f64(_col_np(detail, "DiscountAmount"))[mask][: qty.size]
    net_price = _as_np_f64(_col_np(detail, "NetPrice"))[mask][: qty.size]
    unit_cost = _as_np_f64(_col_np(detail, "UnitCost"))[mask][: qty.size]

    # ReturnQuantity: integer in [1, qty]
    u = rng.random(qty.size)
    ret_qty = (np.floor(u * qty).astype(np.int64) + 1).clip(1, qty)

    frac = ret_qty.astype(np.float64) / qty.astype(np.float64)

    max_lag = max(1, int(cfg.max_lag_days))
    lag = rng.integers(1, max_lag + 1, size=qty.size, dtype=np.int32).astype("timedelta64[D]")
    ret_date = (delivery + lag).astype("datetime64[D]", copy=False)

    reason_keys = _as_np_i64(list(cfg.reason_keys))
    probs = _as_np_f64(list(cfg.reason_probs))

    if reason_keys.size == 0:
        reason_keys = np.array([1], dtype=np.int64)
        probs = np.array([1.0], dtype=np.float64)

    if probs.size != reason_keys.size:
        raise RuntimeError("ReturnsConfig reason_probs must match reason_keys length")

    s = float(probs.sum())
    probs = (probs / s) if s > 0 else np.full_like(probs, 1.0 / probs.size, dtype=np.float64)

    reason = rng.choice(reason_keys, size=qty.size, p=probs).astype(np.int64)

    return pa.table(
        {
            "SalesOrderNumber": so,
            "SalesOrderLineNumber": line,
            "CustomerKey": cust,
            "ProductKey": prod,
            "StoreKey": store,
            "PromotionKey": promo,
            "CurrencyKey": curr,
            "ReturnDate": ret_date,
            "ReturnReasonKey": reason,
            "ReturnQuantity": ret_qty,
            "ReturnUnitPrice": unit_price,
            "ReturnDiscountAmount": disc_amt * frac,
            "ReturnNetPrice": net_price * frac,
            "ReturnUnitCost": unit_cost,
        }
    )


# ============================================================================
# sales_worker (sales-specific logic; previously inlined sections)
# ============================================================================
# ===============================================================
# schemas.py (inlined)
# ===============================================================
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


# ===============================================================
# returns_builder.py (inlined from common/worker/return_builder.py)
# ===============================================================
# NOTE: see ReturnsConfig + build_sales_returns_from_detail above.


# ===============================================================
# header_builder.py (inlined)
# ===============================================================
def build_header_from_detail(detail: pa.Table) -> pa.Table:
    """
    Build SalesOrderHeader (ORDER-GRAIN) from SalesOrderDetail.

    Output columns:
      - SalesOrderNumber
      - CustomerKey
      - OrderDate
      - IsOrderDelayed   (1 if any line is delayed)

    NOTE:
    StoreKey/PromotionKey/CurrencyKey/DueDate can vary per line, so they are not
    included in the header.
    """
    gb = detail.group_by(["SalesOrderNumber"])

    out = gb.aggregate(
        [
            ("CustomerKey", "min"),
            ("OrderDate", "min"),
            ("IsOrderDelayed", "max"),
        ]
    )

    rename_map = {
        "CustomerKey_min": "CustomerKey",
        "OrderDate_min": "OrderDate",
        "IsOrderDelayed_max": "IsOrderDelayed",
    }

    cols = []
    names = []
    for name in out.schema.names:
        cols.append(out[name])
        names.append(rename_map.get(name, name))

    return pa.Table.from_arrays(cols, names=names)


# ===============================================================
# chunk_io.py (inlined)
# ===============================================================
def _expected_schema(table_name: Optional[str]) -> pa.Schema:
    schema_by_table = getattr(State, "schema_by_table", None)
    if table_name and isinstance(schema_by_table, dict) and table_name in schema_by_table:
        return schema_by_table[table_name]
    return State.sales_schema


def _dict_cols(table_name: Optional[str]) -> list[str]:
    m = getattr(State, "parquet_dict_cols_by_table", None)
    if table_name and isinstance(m, dict) and table_name in m:
        return list(m[table_name])
    return list(getattr(State, "parquet_dict_cols", []))


def _schema_needs_year_month(expected: pa.Schema) -> bool:
    names = set(expected.names)
    return ("Year" in names) and ("Month" in names)


def _derive_year_month_from_int_order_date(order_date: pa.ChunkedArray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Supports two common integer encodings:
      1) YYYYMMDD (e.g. 20240131)
      2) epoch-days (days since 1970-01-01), typically small (~ 10k-30k for modern years)
    """
    x = order_date.combine_chunks().to_numpy(zero_copy_only=False)
    x = np.asarray(x)

    if x.dtype.kind not in {"i", "u"}:
        raise RuntimeError(f"OrderDate integer derivation expected int dtype, got {x.dtype}")

    xi = x.astype(np.int64, copy=False)

    if xi.size == 0:
        return xi.astype(np.int16), xi.astype(np.int16)

    if np.any(xi == np.iinfo(np.int64).min):
        raise RuntimeError("OrderDate contains nulls; cannot derive Year/Month")

    mx = int(np.max(xi))
    mn = int(np.min(xi))

    if 19_000_000 <= mx <= 210_012_31 and 19_000_000 <= mn <= 210_012_31:
        year = (xi // 10_000).astype(np.int16, copy=False)
        month = ((xi // 100) % 100).astype(np.int16, copy=False)
        return year, month

    if -100_000 <= mn <= 200_000 and -100_000 <= mx <= 200_000:
        epoch = np.datetime64("1970-01-01", "D")
        dt = (epoch + xi.astype("timedelta64[D]")).astype("datetime64[D]", copy=False)

        year = (dt.astype("datetime64[Y]").astype(np.int32) + 1970).astype(np.int16)
        months = dt.astype("datetime64[M]").astype(np.int32)
        month = ((months % 12) + 1).astype(np.int16)
        return year, month

    raise RuntimeError(
        f"OrderDate integer format not recognized for Year/Month derivation; min={mn} max={mx}"
    )


def _ensure_year_month_if_needed_for_table(
    table: pa.Table,
    *,
    table_name: str,
    expected_schema: pa.Schema,
) -> pa.Table:
    if ("Year" not in expected_schema.names) or ("Month" not in expected_schema.names):
        return table

    if ("Year" in table.column_names) and ("Month" in table.column_names):
        return table

    policy = getattr(State, "date_cols_by_table", {}) or {}
    candidates = policy.get(table_name) or ["DeliveryDate", "OrderDate"]

    usable: list[str] = []
    for c in candidates:
        if c not in table.column_names:
            continue
        t = table.schema.field(c).type
        if pa.types.is_date32(t) or pa.types.is_date64(t) or pa.types.is_timestamp(t):
            usable.append(c)

    if usable:
        return add_year_month_from_date(table, date_cols=tuple(usable))

    if "OrderDate" in table.column_names and pa.types.is_integer(table.schema.field("OrderDate").type):
        year, month = _derive_year_month_from_int_order_date(table["OrderDate"])
        table = table.append_column("Year", pa.array(year, type=pa.int16()))
        table = table.append_column("Month", pa.array(month, type=pa.int16()))
        return table

    raise RuntimeError(
        f"Cannot derive Year/Month for table={table_name}: no usable date column among {candidates}"
    )


def _csv_postprocess_sales(table: pa.Table) -> pa.Table:
    """
    Preserve existing CSV behavior: ensure null-safe int8 for IsOrderDelayed.
    """
    try:
        import pyarrow.compute as pc  # type: ignore
    except Exception as e:
        raise RuntimeError("pyarrow.compute is required for CSV postprocess") from e

    if "IsOrderDelayed" in table.column_names:
        idx = table.schema.get_field_index("IsOrderDelayed")
        table = table.set_column(
            idx,
            "IsOrderDelayed",
            pc.cast(pc.fill_null(table["IsOrderDelayed"], 0), pa.int8()),
        )
    return table


def _write_parquet_table(table: pa.Table, path: str, *, table_name: Optional[str] = None) -> None:
    tn = table_name or "Sales"
    expected = _expected_schema(tn)

    cfg = ChunkIOConfig(
        compression=getattr(State, "compression", "snappy"),
        row_group_size=int(getattr(State, "row_group_size", 1_000_000)),
        write_statistics=bool(getattr(State, "write_statistics", True)),
    )

    need_ym = _schema_needs_year_month(expected)

    ensure_fn = (
        (lambda t: _ensure_year_month_if_needed_for_table(t, table_name=tn, expected_schema=expected))
        if need_ym
        else None
    )

    write_parquet_table(
        table,
        path,
        expected_schema=expected,
        cfg=cfg,
        use_dictionary=_dict_cols(tn),
        table_name=tn,
        ensure_cols=("Year", "Month") if need_ym else (),
        ensure_cols_fn=ensure_fn,
    )


def _write_csv(table: pa.Table, path: str, *, table_name: Optional[str] = None) -> None:
    tn = table_name or "Sales"
    expected = _expected_schema(tn)
    need_ym = _schema_needs_year_month(expected)

    ensure_fn = (
        (lambda t: _ensure_year_month_if_needed_for_table(t, table_name=tn, expected_schema=expected))
        if need_ym
        else None
    )

    write_csv_table(
        table,
        path,
        expected_schema=expected,
        table_name=tn,
        ensure_cols=("Year", "Month") if need_ym else (),
        ensure_cols_fn=ensure_fn,
        postprocess=_csv_postprocess_sales,
    )


# ===============================================================
# init.py (inlined)
# ===============================================================
def _build_brand_prob_by_month_rotate_winner(
    rng: np.random.Generator,
    *,
    T: int,
    B: int,
    winner_boost: float = 2.5,
    noise_sd: float = 0.15,
    min_share: float = 0.02,
    year_len_months: int = 12,
) -> np.ndarray:
    """
    Build a (T x B) probability matrix over brands by month.
    Each year segment rotates a "winner" brand, giving it a boost.
    Ensures each brand has at least `min_share`, then renormalizes.
    """
    if T <= 0 or B <= 0:
        raise RuntimeError(f"Invalid T/B for brand_prob_by_month: T={T}, B={B}")

    year_len = max(1, int(year_len_months))
    winner_boost = float(winner_boost)
    noise_sd = float(noise_sd)
    min_share = float(min_share)

    base = np.ones(B, dtype=np.float64) / float(B)
    out = np.empty((T, B), dtype=np.float64)

    for t in range(T):
        year_idx = t // year_len
        winner = year_idx % B

        v = base.copy()
        v[winner] *= winner_boost

        if noise_sd > 0:
            v *= np.exp(rng.normal(loc=0.0, scale=noise_sd, size=B))

        if min_share > 0:
            v = np.maximum(v, min_share)

        s = float(v.sum())
        if s <= 0:
            v = base
            s = float(v.sum())

        out[t] = v / s

    return out


def init_sales_worker(worker_cfg: dict):
    """
    Initialize immutable worker state.
    Runs exactly once per worker process.

    Lifecycle-aware contract:
      - Binds customer arrays needed by chunk_builder:
        customer_keys, customer_is_active_in_sales, customer_start_month, customer_end_month,
        optional customer_base_weight
    """
    try:
        product_np = worker_cfg["product_np"]
        product_brand_key = worker_cfg.get("product_brand_key")
        store_keys = worker_cfg["store_keys"]

        store_to_geo = worker_cfg.get("store_to_geo")  # optional dict
        geo_to_currency = worker_cfg.get("geo_to_currency")  # optional dict

        promo_keys_all = worker_cfg["promo_keys_all"]
        promo_pct_all = worker_cfg["promo_pct_all"]
        promo_start_all = worker_cfg["promo_start_all"]
        promo_end_all = worker_cfg["promo_end_all"]

        customer_keys = worker_cfg["customer_keys"]
        customer_is_active_in_sales = worker_cfg.get("customer_is_active_in_sales")
        customer_start_month = worker_cfg.get("customer_start_month")
        customer_end_month = worker_cfg.get("customer_end_month")
        customer_base_weight = worker_cfg.get("customer_base_weight")

        date_pool = worker_cfg["date_pool"]
        date_prob = worker_cfg["date_prob"]

        op = worker_cfg["output_paths"]
        if not isinstance(op, dict):
            raise RuntimeError("output_paths must be a dict")

        file_format = worker_cfg.get("file_format") or op.get("file_format")
        out_folder = worker_cfg.get("out_folder") or op.get("out_folder")

        row_group_size = int(worker_cfg.get("row_group_size", 2_000_000))
        compression = str(worker_cfg.get("compression", "snappy"))

        no_discount_key = worker_cfg["no_discount_key"]
        delta_output_folder = worker_cfg.get("delta_output_folder") or op.get("delta_output_folder")
        merged_file = worker_cfg.get("merged_file") or op.get("merged_file")

        write_delta = worker_cfg.get("write_delta", False)

        skip_order_cols = worker_cfg["skip_order_cols"]

        sales_output = _str_or(worker_cfg.get("sales_output"), "sales").lower()
        if sales_output not in {"sales", "sales_order", "both"}:
            raise RuntimeError(f"Invalid sales_output: {sales_output}")

        # Preserve user's intent for the Sales table, even if we force order cols on
        # to generate Header/Detail.
        skip_order_cols_requested = bool(worker_cfg.get("skip_order_cols_requested", skip_order_cols))

        # Returns (optional)
        returns_enabled = bool(worker_cfg.get("returns_enabled", False))
        returns_rate = float(worker_cfg.get("returns_rate", 0.0))
        returns_max_lag_days = int(worker_cfg.get("returns_max_lag_days", 60))
        returns_reason_keys = worker_cfg.get("returns_reason_keys")
        returns_reason_probs = worker_cfg.get("returns_reason_probs")

        # Effective behavior:
        # - Order tables require order keys, so chunk_builder must output them.
        if sales_output in {"sales_order", "both"}:
            skip_order_cols = False

        partition_enabled = worker_cfg.get("partition_enabled", False)
        partition_cols = worker_cfg.get("partition_cols") or []
        models_cfg = worker_cfg.get("models_cfg")

        parquet_dict_exclude = worker_cfg.get("parquet_dict_exclude")
        parquet_dict_cols = worker_cfg.get("parquet_dict_cols")
        write_pyarrow = worker_cfg.get("write_pyarrow", True)

    except KeyError as e:
        raise RuntimeError(f"Missing worker_cfg key: {e}") from e

    product_np = np.asarray(product_np)

    brand_to_row_idx = None
    if product_brand_key is not None:
        product_brand_key = _as_int64(product_brand_key)
        if product_brand_key.shape[0] != product_np.shape[0]:
            raise RuntimeError("product_brand_key must align with product_np row count")
        brand_to_row_idx = _build_buckets_from_brand_key(product_brand_key)

    store_keys = _as_int64(store_keys)

    promo_keys_all = _as_int64(promo_keys_all)
    promo_pct_all = _as_f64(promo_pct_all)
    promo_start_all = np.asarray(promo_start_all, dtype="datetime64[D]")
    promo_end_all = np.asarray(promo_end_all, dtype="datetime64[D]")

    customer_keys = _as_int64(customer_keys)

    if customer_is_active_in_sales is not None:
        customer_is_active_in_sales = _as_int64(customer_is_active_in_sales)
        if customer_is_active_in_sales.shape[0] != customer_keys.shape[0]:
            raise RuntimeError("customer_is_active_in_sales must align with customer_keys length")

    if customer_start_month is not None:
        customer_start_month = _as_int64(customer_start_month)
        if customer_start_month.shape[0] != customer_keys.shape[0]:
            raise RuntimeError("customer_start_month must align with customer_keys length")

    if customer_end_month is not None:
        customer_end_month = _as_int64(customer_end_month)
        if customer_end_month.shape[0] != customer_keys.shape[0]:
            raise RuntimeError("customer_end_month must align with customer_keys length")

    if customer_base_weight is not None:
        customer_base_weight = np.asarray(customer_base_weight, dtype=np.float64)
        if customer_base_weight.shape[0] != customer_keys.shape[0]:
            raise RuntimeError("customer_base_weight must align with customer_keys length")

    brand_prob_by_month = None
    if models_cfg and isinstance(models_cfg, dict):
        models_root = models_cfg.get("models") if isinstance(models_cfg.get("models"), dict) else models_cfg
        brand_cfg = None
        if isinstance(models_root, dict):
            brand_cfg = models_root.get("brand_popularity")
        if brand_cfg:
            T = _infer_T_from_date_pool(date_pool)
            B = int(product_brand_key.max()) + 1 if product_brand_key is not None and product_brand_key.size else 0
            seed = _int_or(brand_cfg.get("seed") if isinstance(brand_cfg, dict) else None, 1234)
            winner_boost = _float_or(brand_cfg.get("winner_boost") if isinstance(brand_cfg, dict) else None, 2.5)
            noise_sd = _float_or(brand_cfg.get("noise_sd") if isinstance(brand_cfg, dict) else None, 0.15)
            min_share = _float_or(brand_cfg.get("min_share") if isinstance(brand_cfg, dict) else None, 0.02)
            year_len = _int_or(brand_cfg.get("year_len_months") if isinstance(brand_cfg, dict) else None, 12)

            rng_bp = np.random.default_rng(int(seed))
            brand_prob_by_month = _build_brand_prob_by_month_rotate_winner(
                rng_bp,
                T=T,
                B=B,
                winner_boost=winner_boost,
                noise_sd=noise_sd,
                min_share=min_share,
                year_len_months=year_len,
            )

    store_to_geo_arr = _dense_map(store_to_geo) if isinstance(store_to_geo, dict) else None
    geo_to_currency_arr = _dense_map(geo_to_currency) if isinstance(geo_to_currency, dict) else None

    output_paths = OutputPaths(
        out_folder=out_folder,
        delta_output_folder=delta_output_folder,
        file_format=file_format,
        merged_file=merged_file,
    )

    tables: list[str] = []
    if sales_output in {"sales", "both"}:
        tables.append(TABLE_SALES)
    if sales_output in {"sales_order", "both"}:
        tables += [TABLE_SALES_ORDER_DETAIL, TABLE_SALES_ORDER_HEADER]
    if returns_enabled and TABLE_SALES_RETURN is not None:
        tables.append(TABLE_SALES_RETURN)

    for t in tables:
        output_paths.ensure_dirs(t)

    # Base fields shared by Sales and SalesOrderDetail/OrderHeader
    base_fields = [
        pa.field("CustomerKey", pa.int64()),
        pa.field("ProductKey", pa.int64()),
        pa.field("StoreKey", pa.int64()),
        pa.field("PromotionKey", pa.int64()),
        pa.field("CurrencyKey", pa.int64()),
        pa.field("OrderDate", pa.date32()),
        pa.field("DueDate", pa.date32()),
        pa.field("DeliveryDate", pa.date32()),
        pa.field("Quantity", pa.int64()),
        pa.field("NetPrice", pa.float64()),
        pa.field("UnitCost", pa.float64()),
        pa.field("UnitPrice", pa.float64()),
        pa.field("DiscountAmount", pa.float64()),
        pa.field("DeliveryStatus", pa.string()),
        pa.field("IsOrderDelayed", pa.int8()),
    ]

    order_fields = [
        pa.field("SalesOrderNumber", pa.int64()),
        pa.field("SalesOrderLineNumber", pa.int64()),
    ]

    delta_fields = [
        pa.field("Year", pa.int16()),
        pa.field("Month", pa.int16()),
    ]

    schema_no_order = pa.schema(base_fields)
    schema_with_order = pa.schema(order_fields + base_fields)

    schema_no_order_delta = pa.schema(base_fields + delta_fields)
    schema_with_order_delta = pa.schema(order_fields + base_fields + delta_fields)

    # Generation schema must match effective skip_order_cols
    # Output schema must match user intent (skip_order_cols_requested)
    if file_format == "deltaparquet":
        sales_schema_gen = schema_no_order_delta if skip_order_cols else schema_with_order_delta
        sales_schema_out = schema_no_order_delta if skip_order_cols_requested else schema_with_order_delta
    else:
        sales_schema_gen = schema_no_order if skip_order_cols else schema_with_order
        sales_schema_out = schema_no_order if skip_order_cols_requested else schema_with_order

    detail_fields = [
        pa.field("SalesOrderNumber", pa.int64()),
        pa.field("SalesOrderLineNumber", pa.int64()),
        pa.field("ProductKey", pa.int64()),
        pa.field("StoreKey", pa.int64()),
        pa.field("PromotionKey", pa.int64()),
        pa.field("CurrencyKey", pa.int64()),
        pa.field("DueDate", pa.date32()),
        pa.field("DeliveryDate", pa.date32()),
        pa.field("Quantity", pa.int64()),
        pa.field("NetPrice", pa.float64()),
        pa.field("UnitCost", pa.float64()),
        pa.field("UnitPrice", pa.float64()),
        pa.field("DiscountAmount", pa.float64()),
        pa.field("DeliveryStatus", pa.string()),
    ]
    detail_schema = pa.schema(detail_fields + delta_fields) if file_format == "deltaparquet" else pa.schema(detail_fields)

    header_fields = [
        pa.field("SalesOrderNumber", pa.int64()),
        pa.field("CustomerKey", pa.int64()),
        pa.field("OrderDate", pa.date32()),
        pa.field("IsOrderDelayed", pa.int8()),
    ]
    header_schema = pa.schema(header_fields + delta_fields) if file_format == "deltaparquet" else pa.schema(header_fields)

    schema_by_table: dict[str, pa.Schema] = {
        TABLE_SALES: sales_schema_out,
        TABLE_SALES_ORDER_DETAIL: detail_schema,
        TABLE_SALES_ORDER_HEADER: header_schema,
    }

    if returns_enabled:
        if TABLE_SALES_RETURN is None:
            raise RuntimeError(
                "returns_enabled=True but TABLE_SALES_RETURN is not defined in output_paths.py"
            )

        return_fields = [
            pa.field("SalesOrderNumber", pa.int64()),
            pa.field("SalesOrderLineNumber", pa.int64()),
            pa.field("CustomerKey", pa.int64()),
            pa.field("ProductKey", pa.int64()),
            pa.field("StoreKey", pa.int64()),
            pa.field("PromotionKey", pa.int64()),
            pa.field("CurrencyKey", pa.int64()),
            pa.field("ReturnDate", pa.date32()),
            pa.field("ReturnReasonKey", pa.int64()),
            pa.field("ReturnQuantity", pa.int64()),
            pa.field("ReturnUnitPrice", pa.float64()),
            pa.field("ReturnDiscountAmount", pa.float64()),
            pa.field("ReturnNetPrice", pa.float64()),
            pa.field("ReturnUnitCost", pa.float64()),
        ]
        return_schema = pa.schema(return_fields + delta_fields) if file_format == "deltaparquet" else pa.schema(return_fields)
        schema_by_table[TABLE_SALES_RETURN] = return_schema

    date_cols_by_table: dict[str, list[str]] = {
        TABLE_SALES: ["OrderDate", "DeliveryDate"],
        TABLE_SALES_ORDER_DETAIL: ["DeliveryDate", "OrderDate"],
        TABLE_SALES_ORDER_HEADER: ["OrderDate"],
    }
    if returns_enabled and TABLE_SALES_RETURN is not None:
        date_cols_by_table[TABLE_SALES_RETURN] = ["ReturnDate", "DeliveryDate", "OrderDate"]

    # Optional: allow models.yaml override (non-blocking)
    if models_cfg and isinstance(models_cfg, dict):
        models_root = models_cfg.get("models") if isinstance(models_cfg.get("models"), dict) else models_cfg
        overrides = None
        if isinstance(models_root, dict) and isinstance(models_root.get("returns"), dict):
            overrides = models_root["returns"].get("date_cols_by_table")

        if isinstance(overrides, dict):
            for k, v in overrides.items():
                if isinstance(k, str) and isinstance(v, (list, tuple)) and v:
                    date_cols_by_table[k] = [str(x) for x in v]

    parquet_dict_exclude = set(parquet_dict_exclude) if parquet_dict_exclude else {"SalesOrderNumber", "CustomerKey"}
    parquet_dict_cols_by_table = {t: schema_dict_cols(s, parquet_dict_exclude) for t, s in schema_by_table.items()}
    parquet_dict_cols = parquet_dict_cols_by_table[TABLE_SALES]  # back-compat

    bind_globals(
        {
            # core arrays
            "product_np": product_np,
            "brand_to_row_idx": brand_to_row_idx,
            "product_brand_key": product_brand_key,
            "brand_prob_by_month": brand_prob_by_month,
            "store_keys": store_keys,
            "promo_keys_all": promo_keys_all,
            "promo_pct_all": promo_pct_all,
            "promo_start_all": promo_start_all,
            "promo_end_all": promo_end_all,
            "customer_keys": customer_keys,
            "customer_is_active_in_sales": customer_is_active_in_sales,
            "customer_start_month": customer_start_month,
            "customer_end_month": customer_end_month,
            "customer_base_weight": customer_base_weight,
            # fast lookup arrays
            "store_to_geo_arr": store_to_geo_arr,
            "geo_to_currency_arr": geo_to_currency_arr,
            # dates
            "date_pool": date_pool,
            "date_prob": date_prob,
            # output config
            "file_format": file_format,
            "out_folder": out_folder,
            "row_group_size": int(max(1, row_group_size)),
            "compression": compression,
            "output_paths": output_paths,
            "sales_output": sales_output,
            # preserve user intent for Sales, even if we force order cols for order tables
            "skip_order_cols_requested": bool(skip_order_cols_requested),
            "skip_order_cols": bool(skip_order_cols),
            # delta
            "delta_output_folder": os.path.normpath(delta_output_folder) if delta_output_folder else None,
            "write_delta": bool(write_delta),
            # behavior
            "no_discount_key": no_discount_key,
            "partition_enabled": bool(partition_enabled),
            "partition_cols": list(partition_cols),
            "models_cfg": models_cfg,
            "write_pyarrow": bool(write_pyarrow),
            # schemas
            "date_cols_by_table": date_cols_by_table,
            "schema_no_order": schema_no_order,
            "schema_with_order": schema_with_order,
            "schema_no_order_delta": schema_no_order_delta,
            "schema_with_order_delta": schema_with_order_delta,
            "sales_schema": sales_schema_gen,  # used by chunk_builder
            "sales_schema_out": sales_schema_out,  # optional (debug / future use)
            "schema_by_table": schema_by_table,
            "parquet_dict_cols_by_table": parquet_dict_cols_by_table,
            # parquet tuning
            "parquet_dict_exclude": parquet_dict_exclude,
            "parquet_dict_cols": parquet_dict_cols,
            # returns
            "returns_enabled": bool(returns_enabled),
            "returns_rate": float(returns_rate),
            "returns_max_lag_days": int(returns_max_lag_days),
            "returns_reason_keys": returns_reason_keys,
            "returns_reason_probs": returns_reason_probs,
        }
    )


# ===============================================================
# task.py (inlined)
# ===============================================================
_DROP_ORDER_COLS = {"SalesOrderNumber", "SalesOrderLineNumber"}


def _drop_order_cols_for_sales(table: pa.Table) -> pa.Table:
    keep = [n for n in table.schema.names if n not in _DROP_ORDER_COLS]
    return table.select(keep)


def _partition_cols() -> set[str]:
    cols = getattr(State, "partition_cols", None)
    if isinstance(cols, (list, tuple)) and cols:
        return {str(c) for c in cols}
    return {"Year", "Month"}


def _project_for_table(table_name: str, table: pa.Table) -> pa.Table:
    expected = State.schema_by_table[table_name]
    part_cols = _partition_cols()
    cols = [n for n in expected.names if n not in part_cols]

    got = set(table.schema.names)
    exp = set(cols)

    missing = sorted(exp - got)
    if missing:
        raise RuntimeError(
            f"Cannot project {table_name}: missing columns {missing}. "
            f"Available columns: {table.schema.names}"
        )

    return table.select(cols)


def _write_table(table_name: str, idx: int, table: pa.Table) -> Union[str, Dict[str, Any]]:
    return write_table_by_format(
        file_format=State.file_format,
        output_paths=State.output_paths,
        table_name=table_name,
        idx=int(idx),
        table=table,
        write_csv_fn=lambda t, p: _write_csv(t, p, table_name=table_name),
        write_parquet_fn=lambda t, p: _write_parquet_table(t, p, table_name=table_name),
    )


def _mode() -> str:
    return str(getattr(State, "sales_output", "sales") or "sales").strip().lower()


def _task_require_cols(table: pa.Table, cols: Sequence[str], *, ctx: str) -> None:
    missing = sorted(set(cols).difference(table.schema.names))
    if missing:
        raise RuntimeError(f"{ctx} missing columns: {missing}. Available: {table.schema.names}")


def _as_list(v: Any, default: Sequence[Any]) -> list[Any]:
    if v is None:
        return list(default)
    if isinstance(v, list):
        return v
    if isinstance(v, tuple):
        return list(v)
    if isinstance(v, (str, bytes)):
        return [v]
    tolist = getattr(v, "tolist", None)
    if callable(tolist):
        try:
            out = tolist()
            if isinstance(out, list):
                return out
        except Exception:
            pass
    return [v]


def _maybe_build_returns(source_table: pa.Table, *, chunk_seed: int) -> Optional[pa.Table]:
    if not bool(getattr(State, "returns_enabled", False)):
        return None

    if TABLE_SALES_RETURN is None:
        raise RuntimeError(
            "returns_enabled=True but TABLE_SALES_RETURN is not defined in output_paths.py"
        )

    mode = _mode()
    if mode not in {"sales", "sales_order", "both"}:
        return None

    _task_require_cols(
        source_table,
        [
            "SalesOrderNumber",
            "SalesOrderLineNumber",
            "CustomerKey",
            "ProductKey",
            "StoreKey",
            "PromotionKey",
            "CurrencyKey",
            "DeliveryDate",
            "Quantity",
            "UnitPrice",
            "DiscountAmount",
            "NetPrice",
            "UnitCost",
        ],
        ctx="SalesReturn build requires",
    )

    rate_raw = getattr(State, "returns_rate", None)
    max_lag_raw = getattr(State, "returns_max_lag_days", None)
    reason_keys_raw = getattr(State, "returns_reason_keys", None)
    reason_probs_raw = getattr(State, "returns_reason_probs", None)

    cfg = ReturnsConfig(
        enabled=True,
        return_rate=float(rate_raw) if rate_raw is not None else 0.0,
        max_lag_days=int(max_lag_raw) if max_lag_raw is not None else 60,
        reason_keys=_as_list(reason_keys_raw, default=[1]),
        reason_probs=_as_list(reason_probs_raw, default=[1.0]),
    )

    returns_seed = int(chunk_seed) ^ 0x5A5A_1234

    returns_table = build_sales_returns_from_detail(
        source_table,
        chunk_seed=int(returns_seed),
        cfg=cfg,
    )

    # common builder returns an empty table when no rows are sampled
    return returns_table if int(returns_table.num_rows) > 0 else None


def _worker_task(args):
    """
    Supports:
      - single task: (idx, batch_size, seed)
      - batched tasks: [(idx, batch_size, seed), ...]
    """
    tasks, single = normalize_tasks(args)
    results = []

    for idx, batch_size, seed in tasks:
        idx_i = int(idx)
        batch_i = int(batch_size)

        chunk_seed = derive_chunk_seed(seed, idx_i, stride=10_000)

        detail_table = chunk_builder.build_chunk_table(
            batch_i,
            int(chunk_seed),
            no_discount_key=State.no_discount_key,
            chunk_idx=idx_i,
            chunk_capacity_orders=int(getattr(State, "chunk_size", batch_i)),
        )
        if not isinstance(detail_table, pa.Table):
            raise TypeError("chunk_builder must return pyarrow.Table")

        mode = _mode()

        if mode in {"sales_order", "both"}:
            _task_require_cols(
                detail_table,
                ["SalesOrderNumber", "SalesOrderLineNumber"],
                ctx=f"sales_output={mode} requires",
            )

            _task_require_cols(
                detail_table,
                ["SalesOrderNumber", "CustomerKey", "OrderDate", "IsOrderDelayed"],
                ctx="Header build requires",
            )

        if mode == "sales":
            sales_table = detail_table
            if bool(getattr(State, "skip_order_cols_requested", False)):
                sales_table = _drop_order_cols_for_sales(sales_table)

            returns_table = _maybe_build_returns(detail_table, chunk_seed=int(chunk_seed))
            if returns_table is None:
                results.append(_write_table(TABLE_SALES, idx_i, sales_table))
                continue

            out: Dict[str, Any] = {}
            out[TABLE_SALES] = _write_table(TABLE_SALES, idx_i, sales_table)

            returns_out = _project_for_table(TABLE_SALES_RETURN, returns_table)  # type: ignore[arg-type]
            out[TABLE_SALES_RETURN] = _write_table(TABLE_SALES_RETURN, idx_i, returns_out)  # type: ignore[arg-type]

            results.append(out)
            continue

        out: Dict[str, Any] = {}

        if mode == "both":
            sales_table = detail_table
            if bool(getattr(State, "skip_order_cols_requested", False)):
                sales_table = _drop_order_cols_for_sales(sales_table)

            sales_out = _project_for_table(TABLE_SALES, sales_table)
            out[TABLE_SALES] = _write_table(TABLE_SALES, idx_i, sales_out)

        header_table = build_header_from_detail(detail_table)

        detail_out = _project_for_table(TABLE_SALES_ORDER_DETAIL, detail_table)
        header_out = _project_for_table(TABLE_SALES_ORDER_HEADER, header_table)

        out[TABLE_SALES_ORDER_DETAIL] = _write_table(TABLE_SALES_ORDER_DETAIL, idx_i, detail_out)
        out[TABLE_SALES_ORDER_HEADER] = _write_table(TABLE_SALES_ORDER_HEADER, idx_i, header_out)

        returns_table = _maybe_build_returns(detail_table, chunk_seed=int(chunk_seed))
        if returns_table is not None:
            returns_out = _project_for_table(TABLE_SALES_RETURN, returns_table)  # type: ignore[arg-type]
            out[TABLE_SALES_RETURN] = _write_table(TABLE_SALES_RETURN, idx_i, returns_out)  # type: ignore[arg-type]

        results.append(out)

    return results[0] if single else results

# ============================================================================
# Public exports
# ============================================================================
__all__ = [
    # Sales worker entrypoints
    "init_sales_worker",
    "_worker_task",

    # Common worker utilities (now hosted here as source-of-truth)
    "ChunkIOConfig",
    "add_year_month_from_date",
    "normalize_to_schema",
    "write_parquet_table",
    "write_csv_table",

    "PoolRunSpec",
    "iter_imap_unordered",

    "normalize_tasks",
    "derive_chunk_seed",
    "write_table_by_format",

    "ReturnsConfig",
    "build_sales_returns_from_detail",

    "build_buckets_from_key",
    "int_or",
    "float_or",
    "str_or",
    "as_int64",
    "as_f64",
    "dense_map",
    "infer_T_from_date_pool",

    # legacy/back-compat names used across older worker code
    "_build_buckets_from_brand_key",
    "_int_or",
    "_float_or",
    "_str_or",
    "_as_int64",
    "_as_f64",
    "_dense_map",
    "_infer_T_from_date_pool",

    # sales-specific helper
    "schema_dict_cols",
]
