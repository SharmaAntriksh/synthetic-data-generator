from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Callable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

try:
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pq = None  # type: ignore

from ..sales_logic import State, build_chunk_table
from ..sales_logic.columns import (
    RETAIL_HOUR_W, DIGITAL_HOUR_W, BUSINESS_HOUR_W, ASSISTED_HOUR_W,
    _normalize_prob as _normalize_hour_prob,
    _PROFILE_FROM_GROUP,
)
from ..output_paths import TABLE_SALES, TABLE_SALES_ORDER_DETAIL, TABLE_SALES_ORDER_HEADER

try:
    from ..output_paths import TABLE_SALES_RETURN  # type: ignore
except Exception:  # pragma: no cover
    TABLE_SALES_RETURN = None  # type: ignore

from .init import int_or
from .io import _write_csv, _write_parquet_table
from .returns_builder import ReturnsConfig, RETURNS_REQUIRED_DETAIL_COLS, build_sales_returns_from_detail

# Budget streaming aggregation (lazy import to avoid hard dependency)
try:
    from src.facts.budget.micro_agg import micro_aggregate_sales, micro_aggregate_returns
    _BUDGET_AGG_AVAILABLE = True
except ImportError:
    _BUDGET_AGG_AVAILABLE = False

try:
    from src.facts.inventory.micro_agg import micro_aggregate_inventory
    _INVENTORY_AGG_AVAILABLE = True
except ImportError:
    _INVENTORY_AGG_AVAILABLE = False

try:
    from src.facts.wishlists.micro_agg import micro_aggregate_wishlists
    _WISHLISTS_AGG_AVAILABLE = True
except ImportError:
    _WISHLISTS_AGG_AVAILABLE = False


Task = Tuple[int, int, Any]  # (idx, batch_size, seed)
TaskArgs = Union[Task, Sequence[Task]]


# ---------------------------------------------------------------------------
# Shared order encoding — computed once per chunk, reused by all _ensure_* fns
# ---------------------------------------------------------------------------

class _OrderEncoding:
    """Pre-computed SalesOrderNumber dictionary encoding + first-row indices.

    Eliminates 3x redundant pc.dictionary_encode + np.minimum.at calls
    across _ensure_sales_channel_key, _ensure_time_key, _ensure_salesperson.
    """
    __slots__ = ("enc", "n_orders", "order_inv", "first_row")

    def __init__(self, enc: pa.DictionaryArray, n_orders: int,
                 order_inv: np.ndarray, first_row: np.ndarray):
        self.enc = enc
        self.n_orders = n_orders
        self.order_inv = order_inv
        self.first_row = first_row


def _encode_orders(table: pa.Table) -> Optional[_OrderEncoding]:
    """Encode SalesOrderNumber once; returns None if column absent."""
    if "SalesOrderNumber" not in table.column_names:
        return None

    order_col = table["SalesOrderNumber"]
    if isinstance(order_col, pa.ChunkedArray):
        order_col = order_col.combine_chunks()

    enc = pc.dictionary_encode(order_col)
    n_orders = len(enc.dictionary)
    order_inv = np.asarray(enc.indices.to_numpy(zero_copy_only=False), dtype=np.int64)

    # First-row index per order (vectorized, single pass)
    pos = np.arange(order_inv.size, dtype=np.int64)
    first_row = np.full(n_orders, order_inv.size, dtype=np.int64)
    np.minimum.at(first_row, order_inv, pos)
    first_row[first_row == order_inv.size] = 0

    return _OrderEncoding(enc=enc, n_orders=n_orders,
                          order_inv=order_inv, first_row=first_row)


def normalize_tasks(args: TaskArgs) -> Tuple[List[Task], bool]:
    if isinstance(args, tuple):
        if len(args) != 3:
            raise ValueError(f"Task tuple must be (idx,batch_size,seed), got len={len(args)}")
        return [args], True
    return list(args), False


def derive_chunk_seed(seed: Any, idx: int, *, stride: int = 10_000) -> int:
    base_seed = int_or(seed, 0)
    return base_seed + idx * stride


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
    ff = (file_format or "").strip().lower()
    if ff == "deltaparquet":
        if not hasattr(output_paths, "delta_part_path"):
            raise RuntimeError("output_paths must implement delta_part_path() for deltaparquet")
        path = output_paths.delta_part_path(table_name, idx)
        write_parquet_fn(table, path)
        return {"part": os.path.basename(path), "rows": table.num_rows}

    if ff == "csv":
        if not hasattr(output_paths, "chunk_path"):
            raise RuntimeError("output_paths must implement chunk_path() for csv")
        path = output_paths.chunk_path(table_name, idx, "csv")
        write_csv_fn(table, path)
        return path

    if not hasattr(output_paths, "chunk_path"):
        raise RuntimeError("output_paths must implement chunk_path() for parquet")
    path = output_paths.chunk_path(table_name, idx, "parquet")
    write_parquet_fn(table, path)
    return path


_DROP_ORDER_COLS = {"SalesOrderNumber", "SalesOrderLineNumber"}


def _drop_order_cols_for_sales(table: pa.Table) -> pa.Table:
    to_drop = _DROP_ORDER_COLS.intersection(table.schema.names)
    if not to_drop:
        return table
    return table.drop_columns(list(to_drop))


# Cached partition cols to avoid repeated State attribute lookups.
_cached_partition_cols: Optional[set[str]] = None


def _partition_cols() -> set[str]:
    global _cached_partition_cols
    if _cached_partition_cols is not None:
        return _cached_partition_cols
    cols = getattr(State, "partition_cols", None)
    if isinstance(cols, (list, tuple)) and cols:
        _cached_partition_cols = {str(c) for c in cols}
    else:
        _cached_partition_cols = {"Year", "Month"}
    return _cached_partition_cols


# Per-table projection column lists, cached to avoid recomputation.
_projection_cache: Dict[str, List[str]] = {}


def _project_for_table(table_name: str, table: pa.Table) -> pa.Table:
    cols = _projection_cache.get(table_name)
    if cols is None:
        expected = State.schema_by_table[table_name]
        part_cols = _partition_cols()
        cols = [n for n in expected.names if n not in part_cols]
        _projection_cache[table_name] = cols

    got = table.column_names
    if len(cols) == len(got) and all(a == b for a, b in zip(cols, got)):
        return table

    got_set = set(got)
    missing = [c for c in cols if c not in got_set]
    if missing:
        raise RuntimeError(
            f"Cannot project {table_name}: missing columns {sorted(missing)}. "
            f"Available columns: {got}"
        )
    return table.select(cols)


def _write_table(table_name: str, idx: int, table: pa.Table) -> Union[str, Dict[str, Any]]:
    return write_table_by_format(
        file_format=State.file_format,
        output_paths=State.output_paths,
        table_name=table_name,
        idx=idx,
        table=table,
        write_csv_fn=lambda t, p: _write_csv(t, p, table_name=table_name),
        write_parquet_fn=lambda t, p: _write_parquet_table(t, p, table_name=table_name, is_chunk=True),
    )


def _mode() -> str:
    return str(getattr(State, "sales_output", "sales") or "sales").strip().lower()


def _task_require_cols(table: pa.Table, cols: Sequence[str], *, ctx: str) -> None:
    got = set(table.column_names)
    missing = sorted(set(cols) - got)
    if missing:
        raise RuntimeError(f"{ctx} missing columns: {missing}. Available: {sorted(got)}")


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
        except (TypeError, ValueError):
            pass
    return [v]


def _build_returns_config() -> Optional[ReturnsConfig]:
    """Build ReturnsConfig from State once, return None if returns disabled."""
    if not bool(getattr(State, "returns_enabled", False)):
        return None
    if TABLE_SALES_RETURN is None:
        raise RuntimeError("returns_enabled=True but TABLE_SALES_RETURN is not defined in output_paths.py")
    return ReturnsConfig(
        enabled=True,
        return_rate=float(getattr(State, "returns_rate", 0.0) or 0.0),
        min_lag_days=int(getattr(State, "returns_min_lag_days", 0) or 0),
        max_lag_days=int(getattr(State, "returns_max_lag_days", 60) or 60),
        reason_keys=_as_list(getattr(State, "returns_reason_keys", None), default=[1]),
        reason_probs=_as_list(getattr(State, "returns_reason_probs", None), default=[1.0]),
    )


def _maybe_build_returns(
    source_table: pa.Table, *, chunk_seed: int, mode: str, returns_cfg: Optional[ReturnsConfig]
) -> Optional[pa.Table]:
    if returns_cfg is None:
        return None
    if mode not in {"sales", "sales_order", "both"}:
        return None

    _task_require_cols(source_table, RETURNS_REQUIRED_DETAIL_COLS, ctx="SalesReturn build requires")

    returns_seed = chunk_seed ^ 0x5A5A_1234
    returns_table = build_sales_returns_from_detail(source_table, chunk_seed=returns_seed, cfg=returns_cfg)
    return returns_table if returns_table.num_rows > 0 else None


# -----------------------------
# SalesChannelKey + TimeKey
# -----------------------------

def _dims_parquet_folder() -> Optional[Path]:
    for attr in ("parquet_folder_p", "parquet_folder", "dimensions_parquet_folder", "dims_parquet_folder"):
        v = getattr(State, attr, None)
        if v:
            return Path(v)
    return None


def _sales_channels_spec() -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (keys:int16[], p:float64[]) for sampling SalesChannelKey.
    Tries sales_channels.parquet; falls back to [1..5].
    Cached on State.
    """
    cached = getattr(State, "_sales_channel_spec", None)
    if cached is not None:
        return cached

    keys: Optional[np.ndarray] = None

    folder = _dims_parquet_folder()
    if pq is not None and folder is not None:
        pth = folder / "sales_channels.parquet"
        if pth.exists():
            tab = pq.read_table(pth, columns=["SalesChannelKey"])
            arr = np.asarray(tab["SalesChannelKey"].to_numpy(), dtype=np.int16)
            arr = arr[arr != np.int16(0)]  # don't sample Unknown
            if arr.size:
                keys = arr

    if keys is None or keys.size == 0:
        keys = np.array([1, 2, 3, 4, 5], dtype=np.int16)

    p = np.full(keys.shape[0], 1.0 / keys.shape[0], dtype=np.float64)
    State._sales_channel_spec = (keys, p)
    return State._sales_channel_spec


def _ensure_sales_channel_key_on_lines(
    table: pa.Table, *, seed: int,
    order_enc: Optional[_OrderEncoding] = None,
) -> pa.Table:
    """Add SalesChannelKey to a line-level table; constant within SalesOrderNumber if present."""
    if "SalesChannelKey" in table.column_names:
        return table

    keys, p = _sales_channels_spec()
    rng = np.random.default_rng(seed)

    if order_enc is not None:
        per_order = rng.choice(keys, size=order_enc.n_orders, p=p).astype(np.int16, copy=False)
        per_order_arr = pa.array(per_order, type=pa.int16())
        col = pc.take(per_order_arr, order_enc.enc.indices)
    elif "SalesOrderNumber" in table.column_names:
        order_col = table["SalesOrderNumber"]
        if isinstance(order_col, pa.ChunkedArray):
            order_col = order_col.combine_chunks()

        enc = pc.dictionary_encode(order_col)
        n_orders = len(enc.dictionary)

        per_order = rng.choice(keys, size=n_orders, p=p).astype(np.int16, copy=False)
        per_order_arr = pa.array(per_order, type=pa.int16())
        col = pc.take(per_order_arr, enc.indices)
    else:
        per_row = rng.choice(keys, size=table.num_rows, p=p).astype(np.int16, copy=False)
        col = pa.array(per_row, type=pa.int16())

    return table.append_column("SalesChannelKey", col)


# Pre-normalized hour weight arrays (sourced from columns.py — single source of truth).
_HOUR_WEIGHTS_NORM: list[np.ndarray] = [
    _normalize_hour_prob(RETAIL_HOUR_W),
    _normalize_hour_prob(DIGITAL_HOUR_W),
    _normalize_hour_prob(BUSINESS_HOUR_W),
    _normalize_hour_prob(ASSISTED_HOUR_W),
]


def _sample_hour_weighted_minute(rng: np.random.Generator, size: int, hour_p: np.ndarray) -> np.ndarray:
    """Sample minute-of-day values given pre-normalized hour probabilities."""
    if size <= 0:
        return np.empty(0, dtype=np.int16)
    hours = rng.choice(24, size=size, p=hour_p).astype(np.int32)
    mins = rng.integers(0, 60, size=size, dtype=np.int32)
    return (hours * 60 + mins).astype(np.int16, copy=False)


def _sample_time_keys_by_profile(
    rng: np.random.Generator,
    prof: np.ndarray,
    size: int,
) -> np.ndarray:
    """Sample minute-of-day TimeKey values based on channel profile codes (0-3)."""
    out = np.empty(size, dtype=np.int16)
    for code, weights_p in enumerate(_HOUR_WEIGHTS_NORM):
        mask = prof == code
        n = mask.sum()
        if n:
            out[mask] = _sample_hour_weighted_minute(rng, int(n), weights_p)
    return out


def _profile_lut_from_dim() -> Optional[np.ndarray]:
    """
    profile_lut[key] -> profile_code:
      0 Retail, 1 Digital, 2 Business, 3 Assisted
    Cached on State.
    """
    cached = getattr(State, "_sales_channel_profile_lut", None)
    if cached is not None:
        return cached

    folder = _dims_parquet_folder()
    if pq is None or folder is None:
        return None

    pth = folder / "sales_channels.parquet"
    if not pth.exists():
        return None

    tab = pq.read_table(pth, columns=["SalesChannelKey", "ChannelGroup"])
    keys = np.asarray(tab["SalesChannelKey"].to_numpy(), dtype=np.int64)
    groups = np.asarray(tab["ChannelGroup"].to_numpy(), dtype=object)

    if keys.size == 0:
        return None

    maxk = int(keys.max())
    lut = np.full(maxk + 1, 1, dtype=np.int8)  # default Digital

    for k, g in zip(keys, groups):
        if k < 0:
            continue
        lut[int(k)] = np.int8(_PROFILE_FROM_GROUP.get((g or "").strip().lower(), 1))

    State._sales_channel_profile_lut = lut
    return lut


def _combine_if_chunked(col: pa.Array) -> pa.Array:
    """Combine a ChunkedArray into a single array; pass-through otherwise."""
    if isinstance(col, pa.ChunkedArray):
        return col.combine_chunks()
    return col


def _ensure_time_key_on_lines(
    table: pa.Table, *, seed: int,
    order_enc: Optional[_OrderEncoding] = None,
) -> pa.Table:
    """Ensure TimeKey exists and is constant within SalesOrderNumber (if present)."""
    rng = np.random.default_rng(seed)

    profile_lut = _profile_lut_from_dim()
    has_channel = "SalesChannelKey" in table.column_names and profile_lut is not None

    if order_enc is not None:
        enc = order_enc.enc
        n_orders = order_enc.n_orders
        first = order_enc.first_row

        if "TimeKey" in table.column_names:
            tc_np = np.asarray(
                _combine_if_chunked(table["TimeKey"]).to_numpy(zero_copy_only=False),
                dtype=np.int16,
            )
            per_order_arr = pa.array(tc_np[first], type=pa.int16())
            time_col = pc.take(per_order_arr, enc.indices)
            idx = table.schema.get_field_index("TimeKey")
            return table.set_column(idx, "TimeKey", time_col)

        if has_channel:
            sc_np = np.asarray(
                _combine_if_chunked(table["SalesChannelKey"]).to_numpy(zero_copy_only=False),
                dtype=np.int16,
            )
            per_order_sc = sc_np[first]
            prof = profile_lut[np.clip(per_order_sc.astype(np.int64), 0, profile_lut.shape[0] - 1)]
            per_order_time = _sample_time_keys_by_profile(rng, prof, n_orders)
            per_order_arr = pa.array(per_order_time, type=pa.int16())
            time_col = pc.take(per_order_arr, enc.indices)
        else:
            per_order = rng.integers(0, 1440, size=n_orders, dtype=np.int16)
            per_order_arr = pa.array(per_order, type=pa.int16())
            time_col = pc.take(per_order_arr, enc.indices)

        return table.append_column("TimeKey", time_col)

    # --- Fallback: no pre-computed encoding ---

    if "SalesOrderNumber" in table.column_names:
        order_col = _combine_if_chunked(table["SalesOrderNumber"])
        enc = pc.dictionary_encode(order_col)
        n_orders = len(enc.dictionary)

        inv = np.asarray(enc.indices.to_numpy(zero_copy_only=False), dtype=np.int64)
        pos = np.arange(inv.size, dtype=np.int64)
        first = np.full(n_orders, inv.size, dtype=np.int64)
        np.minimum.at(first, inv, pos)
        first[first == inv.size] = 0

        if "TimeKey" in table.column_names:
            tc_np = np.asarray(
                _combine_if_chunked(table["TimeKey"]).to_numpy(zero_copy_only=False),
                dtype=np.int16,
            )
            per_order_arr = pa.array(tc_np[first], type=pa.int16())
            time_col = pc.take(per_order_arr, enc.indices)
            idx = table.schema.get_field_index("TimeKey")
            return table.set_column(idx, "TimeKey", time_col)

        if has_channel:
            sc_np = np.asarray(
                _combine_if_chunked(table["SalesChannelKey"]).to_numpy(zero_copy_only=False),
                dtype=np.int16,
            )
            per_order_sc = sc_np[first]
            prof = profile_lut[np.clip(per_order_sc.astype(np.int64), 0, profile_lut.shape[0] - 1)]
            per_order_time = _sample_time_keys_by_profile(rng, prof, n_orders)
            per_order_arr = pa.array(per_order_time, type=pa.int16())
            time_col = pc.take(per_order_arr, enc.indices)
        else:
            per_order = rng.integers(0, 1440, size=n_orders, dtype=np.int16)
            per_order_arr = pa.array(per_order, type=pa.int16())
            time_col = pc.take(per_order_arr, enc.indices)

        return table.append_column("TimeKey", time_col)

    # No SalesOrderNumber: leave existing TimeKey alone; otherwise sample per row
    if "TimeKey" in table.column_names:
        return table

    if has_channel:
        sc_np = np.asarray(
            _combine_if_chunked(table["SalesChannelKey"]).to_numpy(zero_copy_only=False),
            dtype=np.int16,
        )
        prof = profile_lut[np.clip(sc_np.astype(np.int64), 0, profile_lut.shape[0] - 1)]
        out = _sample_time_keys_by_profile(rng, prof, table.num_rows)
        time_col = pa.array(out, type=pa.int16())
    else:
        per_row = rng.integers(0, 1440, size=table.num_rows, dtype=np.int16)
        time_col = pa.array(per_row, type=pa.int16())

    return table.append_column("TimeKey", time_col)


def _ensure_salesperson_employee_key_effective(
    table: pa.Table, *, seed: int,
    order_enc: Optional[_OrderEncoding] = None,
) -> pa.Table:
    """
    Ensure SalesPersonEmployeeKey respects effective-dated EmployeeStoreAssignments.

    Resolves per unique (StoreKey, OrderDate) pair, then batch-samples and broadcasts.
    When SalesOrderNumber exists, employee is constant within each order.

    If no assignment map is available (State.salesperson_effective_by_store),
    the table is returned unchanged.
    """
    eff = getattr(State, "salesperson_effective_by_store", None)
    if not eff:
        return table

    col_names = table.column_names
    if "SalesPersonEmployeeKey" not in col_names:
        return table
    if "StoreKey" not in col_names or "OrderDate" not in col_names:
        return table

    store = table.column("StoreKey").to_numpy(zero_copy_only=False).astype("int64", copy=False)
    od = table.column("OrderDate").to_numpy(zero_copy_only=False)
    try:
        odD = od.astype("datetime64[D]")
    except Exception:
        return table

    rng = np.random.default_rng(seed)

    out_emp = table.column("SalesPersonEmployeeKey").to_numpy(zero_copy_only=False).astype("int32", copy=True)

    has_orders = order_enc is not None or "SalesOrderNumber" in col_names

    if has_orders:
        if order_enc is not None:
            n_orders = order_enc.n_orders
            order_inv = order_enc.order_inv
            first = order_enc.first_row
        else:
            order_col = _combine_if_chunked(table["SalesOrderNumber"])
            enc = pc.dictionary_encode(order_col)
            n_orders = len(enc.dictionary)
            order_inv = np.asarray(enc.indices.to_numpy(zero_copy_only=False), dtype=np.int64)

            first = np.full(n_orders, order_inv.size, dtype=np.int64)
            pos = np.arange(order_inv.size, dtype=np.int64)
            np.minimum.at(first, order_inv, pos)
            first[first == order_inv.size] = 0

        ord_store = store[first]
        ord_date = odD[first]
    else:
        ord_store = store
        ord_date = odD
        n_orders = len(store)
        order_inv = None

    day_i64 = ord_date.astype("datetime64[D]").astype("int64")
    day_min = int(day_i64.min())
    day_range = int(day_i64.max()) - day_min + 1

    sd_key = ord_store * day_range + (day_i64 - day_min)

    sd_uniq, sd_inv = np.unique(sd_key, return_inverse=True)
    n_pairs = sd_uniq.size

    pair_store = (sd_uniq // day_range).astype(np.int64)
    pair_day64 = (sd_uniq % day_range + day_min)

    pair_counts = np.bincount(sd_inv, minlength=n_pairs).astype(np.int64)

    ord_emp = np.full(n_orders, -1, dtype=np.int64)

    pair_order = np.argsort(sd_inv, kind="mergesort")
    pair_starts = np.zeros(n_pairs + 1, dtype=np.int64)
    np.cumsum(pair_counts, out=pair_starts[1:])

    # Pre-fetch store keys that actually appear to skip dict lookups for missing stores.
    unique_stores = np.unique(pair_store)

    # Build a local cache of filtered (non-manager) employee pools per store to avoid
    # re-filtering the same store's pool across many dates.
    _store_pool_cache: Dict[int, Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {}

    for i in range(n_pairs):
        sk = int(pair_store[i])

        # Use cached per-store pool lookup
        if sk not in _store_pool_cache:
            rec = eff.get(sk)
            if rec is None:
                _store_pool_cache[sk] = None
            else:
                emp_arr, startD, endD, w_arr = rec
                # Pre-filter manager keys once per store
                non_mgr = (emp_arr < 30_000_000) | (emp_arr >= 40_000_000)
                if not non_mgr.any():
                    _store_pool_cache[sk] = None
                else:
                    _store_pool_cache[sk] = (
                        np.asarray(emp_arr[non_mgr], dtype=np.int64),
                        startD[non_mgr],
                        endD[non_mgr],
                        np.asarray(w_arr[non_mgr], dtype=np.float64),
                    )

        pool = _store_pool_cache[sk]
        if pool is None:
            continue

        emp2, startD2, endD2, w2 = pool
        d = np.datetime64(int(pair_day64[i]), "D")

        ok = (startD2 <= d) & (d <= endD2)
        if not ok.any():
            continue

        emp_ok = emp2[ok]
        w_ok = w2[ok]

        count = int(pair_counts[i])
        sw = w_ok.sum()

        if sw <= 1e-12:
            picked = emp_ok[rng.integers(0, emp_ok.size, size=count)]
        else:
            picked = emp_ok[rng.choice(emp_ok.size, size=count, p=w_ok / sw)]

        s, e = int(pair_starts[i]), int(pair_starts[i + 1])
        ord_emp[pair_order[s:e]] = picked

    if has_orders:
        row_emp = ord_emp[order_inv]
        valid = row_emp >= 0
        out_emp[valid] = row_emp[valid]
    else:
        valid = ord_emp >= 0
        out_emp[valid] = ord_emp[valid]

    idx = table.schema.get_field_index("SalesPersonEmployeeKey")
    return table.set_column(idx, "SalesPersonEmployeeKey", pa.array(out_emp, type=pa.int32()))


def build_header_from_detail(detail: pa.Table, *, validate_invariants: bool = True) -> pa.Table:
    inv_cols = ["CustomerKey", "StoreKey", "SalesPersonEmployeeKey", "OrderDate"]

    col_names = detail.column_names
    if "PromotionKey" in col_names:
        inv_cols.append("PromotionKey")
    if "CurrencyKey" in col_names:
        inv_cols.append("CurrencyKey")
    if "SalesChannelKey" in col_names:
        inv_cols.append("SalesChannelKey")
    if "TimeKey" in col_names:
        inv_cols.append("TimeKey")

    gb = detail.group_by(["SalesOrderNumber"])

    if validate_invariants:
        aggs = []
        for c in inv_cols:
            aggs.append((c, "min"))
            aggs.append((c, "max"))
        aggs.append(("IsOrderDelayed", "max"))

        out = gb.aggregate(aggs)

        bad = None
        for c in inv_cols:
            neq = pc.not_equal(out[f"{c}_min"], out[f"{c}_max"])
            bad = neq if bad is None else pc.or_(bad, neq)

        if bad is not None and bool(pc.any(bad).as_py()):
            bad_out = out.filter(bad).slice(0, 5)

            def _py(name: str):
                return bad_out[name].to_pylist() if name in bad_out.column_names else None

            parts = [f"SalesOrderNumber(s)={_py('SalesOrderNumber')}"]
            for c in inv_cols:
                parts.append(f"{c}_min={_py(f'{c}_min')}")
                parts.append(f"{c}_max={_py(f'{c}_max')}")

            raise RuntimeError(
                "Invalid SalesOrderNumber invariants: a SalesOrderNumber maps to multiple values. "
                + " | ".join(parts)
            )

        cols, names = [], []

        def _add(src: str, dst: str):
            if src in out.column_names:
                cols.append(out[src])
                names.append(dst)

        _add("SalesOrderNumber", "SalesOrderNumber")
        _add("CustomerKey_min", "CustomerKey")
        _add("StoreKey_min", "StoreKey")
        _add("SalesPersonEmployeeKey_min", "SalesPersonEmployeeKey")
        _add("PromotionKey_min", "PromotionKey")
        _add("CurrencyKey_min", "CurrencyKey")
        _add("SalesChannelKey_min", "SalesChannelKey")
        _add("OrderDate_min", "OrderDate")
        _add("TimeKey_min", "TimeKey")
        _add("IsOrderDelayed_max", "IsOrderDelayed")

        return pa.Table.from_arrays(cols, names=names)

    # Fast path: MIN only for invariants (halves group-by work)
    aggs = [(c, "min") for c in inv_cols]
    aggs.append(("IsOrderDelayed", "max"))

    out = gb.aggregate(aggs)

    cols, names = [], []

    def _add(src: str, dst: str):
        if src in out.column_names:
            cols.append(out[src])
            names.append(dst)

    _add("SalesOrderNumber", "SalesOrderNumber")
    _add("CustomerKey_min", "CustomerKey")
    _add("StoreKey_min", "StoreKey")
    _add("SalesPersonEmployeeKey_min", "SalesPersonEmployeeKey")
    _add("PromotionKey_min", "PromotionKey")
    _add("CurrencyKey_min", "CurrencyKey")
    _add("SalesChannelKey_min", "SalesChannelKey")
    _add("OrderDate_min", "OrderDate")
    _add("TimeKey_min", "TimeKey")
    _add("IsOrderDelayed_max", "IsOrderDelayed")

    return pa.Table.from_arrays(cols, names=names)


def _inventory_enabled() -> bool:
    return (
        _INVENTORY_AGG_AVAILABLE
        and bool(getattr(State, "inventory_enabled", False))
    )


def _maybe_inventory_agg(detail_table: pa.Table) -> Optional[Dict[str, Any]]:
    if not _inventory_enabled():
        return None
    return micro_aggregate_inventory(detail_table)


def _wishlists_enabled() -> bool:
    return (
        _WISHLISTS_AGG_AVAILABLE
        and bool(getattr(State, "wishlists_enabled", False))
    )


def _maybe_wishlists_agg(detail_table: pa.Table) -> Optional[Dict[str, Any]]:
    if not _wishlists_enabled():
        return None
    return micro_aggregate_wishlists(detail_table)


def _budget_enabled() -> bool:
    """Check if budget streaming aggregation is active for this worker."""
    return (
        _BUDGET_AGG_AVAILABLE
        and bool(getattr(State, "budget_enabled", False))
        and getattr(State, "budget_store_to_country", None) is not None
        and getattr(State, "budget_product_to_cat", None) is not None
    )


def _maybe_budget_agg(detail_table: pa.Table) -> Optional[Dict[str, Any]]:
    """Compute budget micro-aggregate from a detail chunk. Returns None if disabled."""
    if not _budget_enabled():
        return None
    if "SalesChannelKey" not in detail_table.column_names:
        return None
    return micro_aggregate_sales(
        detail_table,
        store_to_country=State.budget_store_to_country,
        product_to_cat=State.budget_product_to_cat,
    )


def _maybe_returns_agg(
    returns_table: Optional[pa.Table],
    detail_table: pa.Table,
) -> Optional[Dict[str, Any]]:
    """Compute returns micro-aggregate from a returns chunk. Returns None if disabled."""
    if not _budget_enabled():
        return None
    if returns_table is None or returns_table.num_rows == 0:
        return None
    return micro_aggregate_returns(
        returns_table, detail_table,
        store_to_country=State.budget_store_to_country,
        product_to_cat=State.budget_product_to_cat,
    )


def _attach_budget(
    result: Any,
    budget_agg: Optional[Dict[str, Any]],
    returns_agg: Optional[Dict[str, Any]],
    table_name_fallback: str = TABLE_SALES,
    inventory_agg: Optional[Dict[str, Any]] = None,
    wishlists_agg: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Attach budget, inventory, and wishlists micro-aggregates to a worker result.

    Normalizes bare str/dict results into a dict so the main process can
    pop _budget_agg / _returns_agg / _inventory_agg / _wishlists_agg
    without breaking existing result handling.

    If all aggs are None, returns the original result untouched
    to preserve backward compatibility.
    """
    if budget_agg is None and returns_agg is None and inventory_agg is None and wishlists_agg is None:
        return result

    if isinstance(result, str):
        result = {table_name_fallback: result}

    if not isinstance(result, dict):
        return result

    if budget_agg is not None:
        result["_budget_agg"] = budget_agg
    if returns_agg is not None:
        result["_returns_agg"] = returns_agg
    if inventory_agg is not None:
        result["_inventory_agg"] = inventory_agg
    if wishlists_agg is not None:
        result["_wishlists_agg"] = wishlists_agg

    return result


def _worker_task(args):
    tasks, single = normalize_tasks(args)
    results = []

    State.validate(["chunk_size"])

    # --- Hoist loop-invariant reads from State ---
    validate_header = bool(getattr(State, "validate_header_invariants", True))
    cap_orders = int(getattr(State, "chunk_size", 0) or 0)
    if cap_orders <= 0:
        raise RuntimeError(f"State.chunk_size must be > 0, got {cap_orders}")

    no_discount_key = State.no_discount_key
    file_format = State.file_format
    output_paths = State.output_paths
    skip_order_cols_flag = bool(getattr(State, "skip_order_cols_requested", False))

    mode = _mode()
    returns_cfg = _build_returns_config()
    do_budget = _budget_enabled()
    do_inventory = _inventory_enabled()
    do_wishlists = _wishlists_enabled()

    # Pre-validate column requirements for sales_order/both modes once
    so_require = None
    header_need_set = None
    if mode in {"sales_order", "both"}:
        so_require = {"SalesOrderNumber", "SalesOrderLineNumber",
                      "CustomerKey", "StoreKey", "SalesPersonEmployeeKey",
                      "OrderDate", "IsOrderDelayed"}
        expected_header = State.schema_by_table[TABLE_SALES_ORDER_HEADER]
        header_need_set = {"StoreKey", "SalesPersonEmployeeKey", "SalesChannelKey",
                           "TimeKey", "PromotionKey", "CurrencyKey"} & set(expected_header.names)

    for idx, batch_size, seed in tasks:
        idx_i = int(idx)
        batch_i = int(batch_size)

        if cap_orders < batch_i:
            raise RuntimeError(
                f"State.chunk_size={cap_orders} < batch_i={batch_i}; "
                "chunk_size must be the constant order-id stride and >= the maximum batch size."
            )

        chunk_seed = derive_chunk_seed(seed, idx_i, stride=10_000)

        detail_table = build_chunk_table(
            batch_i,
            chunk_seed,
            no_discount_key=no_discount_key,
            chunk_idx=idx_i,
            chunk_capacity_orders=cap_orders,
        )
        order_enc = _encode_orders(detail_table)

        detail_table = _ensure_sales_channel_key_on_lines(
            detail_table, seed=chunk_seed ^ 0xA11CE, order_enc=order_enc)
        detail_table = _ensure_time_key_on_lines(
            detail_table, seed=chunk_seed ^ 0xC0FFEE, order_enc=order_enc)
        detail_table = _ensure_salesperson_employee_key_effective(
            detail_table, seed=chunk_seed ^ 0xE1E1_1337, order_enc=order_enc)

        if not isinstance(detail_table, pa.Table):
            raise TypeError("chunk_builder must return pyarrow.Table")

        budget_agg = _maybe_budget_agg(detail_table) if do_budget else None
        inventory_agg = _maybe_inventory_agg(detail_table) if do_inventory else None
        wishlists_agg = _maybe_wishlists_agg(detail_table) if do_wishlists else None

        if mode in {"sales_order", "both"}:
            got = set(detail_table.column_names)
            missing = sorted(so_require - got)
            if missing:
                raise RuntimeError(
                    f"sales_output={mode} / Header build missing columns: {missing}. "
                    f"Available: {detail_table.column_names}")

            header_missing = sorted(header_need_set - got)
            if header_missing:
                raise RuntimeError(
                    f"Header build missing columns: {header_missing}. "
                    f"Available: {detail_table.column_names}")

        if mode == "sales":
            sales_table = detail_table
            if skip_order_cols_flag:
                sales_table = _drop_order_cols_for_sales(sales_table)

            returns_table = _maybe_build_returns(
                detail_table, chunk_seed=chunk_seed, mode=mode, returns_cfg=returns_cfg)
            sales_out = _project_for_table(TABLE_SALES, sales_table)

            if returns_table is None:
                result = _write_table(TABLE_SALES, idx_i, sales_out)
                results.append(_attach_budget(result, budget_agg, None, TABLE_SALES, inventory_agg=inventory_agg, wishlists_agg=wishlists_agg))
                continue

            out: Dict[str, Any] = {TABLE_SALES: _write_table(TABLE_SALES, idx_i, sales_out)}
            returns_out = _project_for_table(TABLE_SALES_RETURN, returns_table)  # type: ignore[arg-type]
            out[TABLE_SALES_RETURN] = _write_table(TABLE_SALES_RETURN, idx_i, returns_out)  # type: ignore[arg-type]
            returns_agg = _maybe_returns_agg(returns_table, detail_table) if do_budget else None
            results.append(_attach_budget(out, budget_agg, returns_agg, inventory_agg=inventory_agg, wishlists_agg=wishlists_agg))
            continue

        out: Dict[str, Any] = {}

        if mode == "both":
            sales_table = detail_table
            if skip_order_cols_flag:
                sales_table = _drop_order_cols_for_sales(sales_table)
            out[TABLE_SALES] = _write_table(TABLE_SALES, idx_i, _project_for_table(TABLE_SALES, sales_table))

        header_table = build_header_from_detail(
            detail_table, validate_invariants=validate_header)

        out[TABLE_SALES_ORDER_DETAIL] = _write_table(
            TABLE_SALES_ORDER_DETAIL, idx_i, _project_for_table(TABLE_SALES_ORDER_DETAIL, detail_table)
        )
        out[TABLE_SALES_ORDER_HEADER] = _write_table(
            TABLE_SALES_ORDER_HEADER, idx_i, _project_for_table(TABLE_SALES_ORDER_HEADER, header_table)
        )

        returns_table = _maybe_build_returns(
            detail_table, chunk_seed=chunk_seed, mode=mode, returns_cfg=returns_cfg)
        if returns_table is not None:
            out[TABLE_SALES_RETURN] = _write_table(
                TABLE_SALES_RETURN, idx_i, _project_for_table(TABLE_SALES_RETURN, returns_table)  # type: ignore[arg-type]
            )

        returns_agg = _maybe_returns_agg(returns_table, detail_table) if do_budget else None
        results.append(_attach_budget(out, budget_agg, returns_agg, inventory_agg=inventory_agg, wishlists_agg=wishlists_agg))

    return results[0] if single else results
