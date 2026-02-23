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
from ..output_paths import TABLE_SALES, TABLE_SALES_ORDER_DETAIL, TABLE_SALES_ORDER_HEADER

try:
    from ..output_paths import TABLE_SALES_RETURN  # type: ignore
except Exception:  # pragma: no cover
    TABLE_SALES_RETURN = None  # type: ignore

from .init import int_or
from .io import _write_csv, _write_parquet_table
from .returns_builder import ReturnsConfig, RETURNS_REQUIRED_DETAIL_COLS, build_sales_returns_from_detail

Task = Tuple[int, int, Any]  # (idx, batch_size, seed)
TaskArgs = Union[Task, Sequence[Task]]


def normalize_tasks(args: TaskArgs) -> Tuple[List[Task], bool]:
    if isinstance(args, tuple):
        if len(args) != 3:
            raise ValueError(f"Task tuple must be (idx,batch_size,seed), got len={len(args)}")
        return [args], True
    return list(args), False


def derive_chunk_seed(seed: Any, idx: int, *, stride: int = 10_000) -> int:
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

    if not hasattr(output_paths, "chunk_path"):
        raise RuntimeError("output_paths must implement chunk_path() for parquet")
    path = output_paths.chunk_path(table_name, int(idx), "parquet")
    write_parquet_fn(table, path)
    return path


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
    missing = sorted(set(cols) - got)
    if missing:
        raise RuntimeError(
            f"Cannot project {table_name}: missing columns {missing}. Available columns: {table.schema.names}"
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
        raise RuntimeError("returns_enabled=True but TABLE_SALES_RETURN is not defined in output_paths.py")

    mode = _mode()
    if mode not in {"sales", "sales_order", "both"}:
        return None

    _task_require_cols(source_table, RETURNS_REQUIRED_DETAIL_COLS, ctx="SalesReturn build requires")

    cfg = ReturnsConfig(
        enabled=True,
        return_rate=float(getattr(State, "returns_rate", 0.0) or 0.0),
        max_lag_days=int(getattr(State, "returns_max_lag_days", 60) or 60),
        reason_keys=_as_list(getattr(State, "returns_reason_keys", None), default=[1]),
        reason_probs=_as_list(getattr(State, "returns_reason_probs", None), default=[1.0]),
    )

    returns_seed = int(chunk_seed) ^ 0x5A5A_1234
    returns_table = build_sales_returns_from_detail(source_table, chunk_seed=int(returns_seed), cfg=cfg)
    return returns_table if int(returns_table.num_rows) > 0 else None


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
            arr = arr[arr != np.int16(0)]  # don’t sample Unknown
            if arr.size:
                keys = arr

    if keys is None or keys.size == 0:
        # fallback matches your current lookup dim
        keys = np.array([1, 2, 3, 4, 5], dtype=np.int16)

    p = np.full(keys.shape[0], 1.0 / keys.shape[0], dtype=np.float64)
    State._sales_channel_spec = (keys, p)
    return State._sales_channel_spec


def _ensure_sales_channel_key_on_lines(table: pa.Table, *, seed: int) -> pa.Table:
    """Add SalesChannelKey to a line-level table; constant within SalesOrderNumber if present."""
    if "SalesChannelKey" in table.column_names:
        return table

    keys, p = _sales_channels_spec()
    rng = np.random.default_rng(seed)

    if "SalesOrderNumber" in table.column_names:
        order_col = table["SalesOrderNumber"]
        if isinstance(order_col, pa.ChunkedArray):
            order_col = order_col.combine_chunks()

        enc = pc.dictionary_encode(order_col)  # DictionaryArray
        n_orders = len(enc.dictionary)

        per_order = rng.choice(keys, size=n_orders, p=p).astype(np.int16, copy=False)
        per_order_arr = pa.array(per_order, type=pa.int16())
        col = pc.take(per_order_arr, enc.indices)
    else:
        per_row = rng.choice(keys, size=table.num_rows, p=p).astype(np.int16, copy=False)
        col = pa.array(per_row, type=pa.int16())

    return table.append_column("SalesChannelKey", col)


# Hour weights per channel group profile (Retail/Digital/Business/Assisted)
_RETAIL_HOUR_W = np.array(
    [0.002, 0.001, 0.001, 0.001, 0.002, 0.004,
     0.010, 0.020, 0.050, 0.080, 0.100, 0.110,
     0.110, 0.105, 0.095, 0.090, 0.085, 0.090,
     0.100, 0.095, 0.070, 0.040, 0.015, 0.006],
    dtype=np.float64,
)
_DIGITAL_HOUR_W = np.array(
    [0.030, 0.025, 0.020, 0.020, 0.022, 0.026,
     0.030, 0.040, 0.050, 0.055, 0.060, 0.065,
     0.070, 0.070, 0.065, 0.060, 0.060, 0.065,
     0.080, 0.090, 0.090, 0.070, 0.050, 0.040],
    dtype=np.float64,
)
_BUSINESS_HOUR_W = np.array(
    [0.001, 0.001, 0.001, 0.001, 0.001, 0.002,
     0.006, 0.020, 0.060, 0.090, 0.110, 0.120,
     0.120, 0.110, 0.100, 0.090, 0.070, 0.040,
     0.020, 0.010, 0.006, 0.003, 0.002, 0.001],
    dtype=np.float64,
)
_ASSISTED_HOUR_W = np.array(
    [0.002, 0.001, 0.001, 0.001, 0.001, 0.003,
     0.010, 0.030, 0.060, 0.090, 0.110, 0.120,
     0.120, 0.110, 0.100, 0.090, 0.070, 0.040,
     0.020, 0.010, 0.006, 0.004, 0.003, 0.002],
    dtype=np.float64,
)


def _normalize_prob(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    s = float(w.sum())
    if s <= 1e-12:
        return np.full(w.shape[0], 1.0 / w.shape[0], dtype=np.float64)
    return w / s


def _sample_hour_weighted_minute(rng: np.random.Generator, size: int, hour_w: np.ndarray) -> np.ndarray:
    size = int(size)
    if size <= 0:
        return np.empty(0, dtype=np.int16)
    p = _normalize_prob(hour_w)
    hours = rng.choice(24, size=size, p=p).astype(np.int32)
    mins = rng.integers(0, 60, size=size, dtype=np.int32)
    return (hours * 60 + mins).astype(np.int16, copy=False)


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

    def prof_for_group(g: str) -> int:
        gg = (g or "").strip().lower()
        if gg == "physical":
            return 0
        if gg == "digital":
            return 1
        if gg == "business":
            return 2
        if gg == "assisted":
            return 3
        return 1

    for k, g in zip(keys, groups):
        if k < 0:
            continue
        lut[int(k)] = np.int8(prof_for_group(str(g)))

    State._sales_channel_profile_lut = lut
    return lut


def _ensure_time_key_on_lines(table: pa.Table, *, seed: int) -> pa.Table:
    """Add TimeKey to a line-level table; constant within SalesOrderNumber if present."""
    if "TimeKey" in table.column_names:
        return table

    rng = np.random.default_rng(seed)

    profile_lut = _profile_lut_from_dim()
    has_channel = "SalesChannelKey" in table.column_names and profile_lut is not None

    if "SalesOrderNumber" in table.column_names:
        order_col = table["SalesOrderNumber"]
        if isinstance(order_col, pa.ChunkedArray):
            order_col = order_col.combine_chunks()
        enc = pc.dictionary_encode(order_col)
        n_orders = len(enc.dictionary)

        if has_channel:
            sc_col = table["SalesChannelKey"]
            if isinstance(sc_col, pa.ChunkedArray):
                sc_col = sc_col.combine_chunks()
            sc_np = np.asarray(sc_col.to_numpy(zero_copy_only=False), dtype=np.int16)

            inv = np.asarray(enc.indices.to_numpy(zero_copy_only=False), dtype=np.int64)
            first = np.full(n_orders, -1, dtype=np.int64)
            for i in range(inv.shape[0]):
                j = inv[i]
                if first[j] == -1:
                    first[j] = i
            per_order_sc = sc_np[first]
            # map channel -> profile
            prof = profile_lut[np.clip(per_order_sc.astype(np.int64), 0, profile_lut.shape[0] - 1)]

            per_order_time = np.empty(n_orders, dtype=np.int16)
            m0 = prof == 0
            if m0.any():
                per_order_time[m0] = _sample_hour_weighted_minute(rng, int(m0.sum()), _RETAIL_HOUR_W)
            m1 = prof == 1
            if m1.any():
                per_order_time[m1] = _sample_hour_weighted_minute(rng, int(m1.sum()), _DIGITAL_HOUR_W)
            m2 = prof == 2
            if m2.any():
                per_order_time[m2] = _sample_hour_weighted_minute(rng, int(m2.sum()), _BUSINESS_HOUR_W)
            m3 = prof == 3
            if m3.any():
                per_order_time[m3] = _sample_hour_weighted_minute(rng, int(m3.sum()), _ASSISTED_HOUR_W)

            per_order_arr = pa.array(per_order_time, type=pa.int16())
            time_col = pc.take(per_order_arr, enc.indices)
        else:
            per_order = rng.integers(0, 1440, size=n_orders, dtype=np.int32).astype(np.int16, copy=False)
            per_order_arr = pa.array(per_order, type=pa.int16())
            time_col = pc.take(per_order_arr, enc.indices)
    else:
        if has_channel:
            sc_col = table["SalesChannelKey"]
            if isinstance(sc_col, pa.ChunkedArray):
                sc_col = sc_col.combine_chunks()
            sc_np = np.asarray(sc_col.to_numpy(zero_copy_only=False), dtype=np.int16)
            prof = profile_lut[np.clip(sc_np.astype(np.int64), 0, profile_lut.shape[0] - 1)]
            out = np.empty(table.num_rows, dtype=np.int16)

            m0 = prof == 0
            if m0.any():
                out[m0] = _sample_hour_weighted_minute(rng, int(m0.sum()), _RETAIL_HOUR_W)
            m1 = prof == 1
            if m1.any():
                out[m1] = _sample_hour_weighted_minute(rng, int(m1.sum()), _DIGITAL_HOUR_W)
            m2 = prof == 2
            if m2.any():
                out[m2] = _sample_hour_weighted_minute(rng, int(m2.sum()), _BUSINESS_HOUR_W)
            m3 = prof == 3
            if m3.any():
                out[m3] = _sample_hour_weighted_minute(rng, int(m3.sum()), _ASSISTED_HOUR_W)

            time_col = pa.array(out, type=pa.int16())
        else:
            per_row = rng.integers(0, 1440, size=table.num_rows, dtype=np.int32).astype(np.int16, copy=False)
            time_col = pa.array(per_row, type=pa.int16())

    return table.append_column("TimeKey", time_col)


def build_header_from_detail(detail: pa.Table) -> pa.Table:
    gb = detail.group_by(["SalesOrderNumber"])

    # Aggregate MIN+MAX for invariants we expect to be constant within an order
    aggs = [
        ("CustomerKey", "min"), ("CustomerKey", "max"),
        ("StoreKey", "min"), ("StoreKey", "max"),
        ("SalesPersonEmployeeKey", "min"), ("SalesPersonEmployeeKey", "max"),
        ("OrderDate", "min"), ("OrderDate", "max"),
        ("IsOrderDelayed", "max"),
    ]
    if "SalesChannelKey" in detail.column_names:
        aggs += [("SalesChannelKey", "min"), ("SalesChannelKey", "max")]
    if "TimeKey" in detail.column_names:
        aggs += [("TimeKey", "min"), ("TimeKey", "max")]

    out = gb.aggregate(aggs)

    # Validate invariants: for each order, min == max for constant fields
    bad = pc.not_equal(out["CustomerKey_min"], out["CustomerKey_max"])
    bad = pc.or_(bad, pc.not_equal(out["StoreKey_min"], out["StoreKey_max"]))
    bad = pc.or_(bad, pc.not_equal(out["SalesPersonEmployeeKey_min"], out["SalesPersonEmployeeKey_max"]))
    bad = pc.or_(bad, pc.not_equal(out["OrderDate_min"], out["OrderDate_max"]))

    if "SalesChannelKey_min" in out.column_names:
        bad = pc.or_(bad, pc.not_equal(out["SalesChannelKey_min"], out["SalesChannelKey_max"]))
    if "TimeKey_min" in out.column_names:
        bad = pc.or_(bad, pc.not_equal(out["TimeKey_min"], out["TimeKey_max"]))

    if bool(pc.any(bad).as_py()):
        bad_idx = pc.indices_nonzero(bad)
        # take first few offenders for error message
        k = min(5, bad_idx.length())
        take_idx = bad_idx.slice(0, k)

        so = pc.take(out["SalesOrderNumber"], take_idx).to_pylist()
        ck_min = pc.take(out["CustomerKey_min"], take_idx).to_pylist()
        ck_max = pc.take(out["CustomerKey_max"], take_idx).to_pylist()

        raise RuntimeError(
            "Invalid SalesOrderNumber invariants: a SalesOrderNumber maps to multiple values "
            f"(example SalesOrderNumber(s)={so}, CustomerKey_min={ck_min}, CustomerKey_max={ck_max}). "
            "This indicates overlapping order-id ranges across chunks OR duplicated/misaligned rows during order expansion."
        )

    # Build final header columns (use *_min for constant fields, max for IsOrderDelayed)
    cols, names = [], []

    def _add(src: str, dst: str):
        if src in out.column_names:
            cols.append(out[src])
            names.append(dst)

    _add("SalesOrderNumber", "SalesOrderNumber")
    _add("CustomerKey_min", "CustomerKey")
    _add("StoreKey_min", "StoreKey")
    _add("SalesPersonEmployeeKey_min", "SalesPersonEmployeeKey")
    _add("OrderDate_min", "OrderDate")
    _add("IsOrderDelayed_max", "IsOrderDelayed")
    _add("SalesChannelKey_min", "SalesChannelKey")
    _add("TimeKey_min", "TimeKey")

    return pa.Table.from_arrays(cols, names=names)


def _worker_task(args):
    tasks, single = normalize_tasks(args)
    results = []

    # chunk_size MUST be constant across chunks; never fall back to per-task batch size
    State.validate(["chunk_size"])
    cap_orders = int(getattr(State, "chunk_size", 0) or 0)
    if cap_orders <= 0:
        raise RuntimeError(f"State.chunk_size must be > 0, got {cap_orders}")

    for idx, batch_size, seed in tasks:
        idx_i = int(idx)
        batch_i = int(batch_size)

        chunk_seed = derive_chunk_seed(seed, idx_i, stride=10_000)

        detail_table = build_chunk_table(
            batch_i,
            int(chunk_seed),
            no_discount_key=State.no_discount_key,
            chunk_idx=idx_i,
            chunk_capacity_orders=cap_orders,   # constant stride across ALL chunks
        )

        # Ensure FKs exist even if Sales schema hasn’t been updated yet
        detail_table = _ensure_sales_channel_key_on_lines(detail_table, seed=int(chunk_seed) ^ 0xA11CE)
        detail_table = _ensure_time_key_on_lines(detail_table, seed=int(chunk_seed) ^ 0xC0FFEE)

        if not isinstance(detail_table, pa.Table):
            raise TypeError("chunk_builder must return pyarrow.Table")

        mode = _mode()

        if mode in {"sales_order", "both"}:
            _task_require_cols(detail_table, ["SalesOrderNumber", "SalesOrderLineNumber"], ctx=f"sales_output={mode} requires")
            _task_require_cols(detail_table, ["SalesOrderNumber", "CustomerKey", "StoreKey", "SalesPersonEmployeeKey", "OrderDate", "IsOrderDelayed"], ctx="Header build requires")

            expected_header = State.schema_by_table[TABLE_SALES_ORDER_HEADER]
            if "StoreKey" in expected_header.names:
                _task_require_cols(detail_table, ["StoreKey"], ctx="Header build requires StoreKey")
            if "SalesPersonEmployeeKey" in expected_header.names:
                _task_require_cols(detail_table, ["SalesPersonEmployeeKey"], ctx="Header build requires SalesPersonEmployeeKey")
            if "SalesChannelKey" in expected_header.names:
                _task_require_cols(detail_table, ["SalesChannelKey"], ctx="Header build requires SalesChannelKey")
            if "TimeKey" in expected_header.names:
                _task_require_cols(detail_table, ["TimeKey"], ctx="Header build requires TimeKey")

        if mode == "sales":
            sales_table = detail_table
            if bool(getattr(State, "skip_order_cols_requested", False)):
                sales_table = _drop_order_cols_for_sales(sales_table)

            returns_table = _maybe_build_returns(detail_table, chunk_seed=int(idx_i))
            sales_out = _project_for_table(TABLE_SALES, sales_table)

            if returns_table is None:
                results.append(_write_table(TABLE_SALES, idx_i, sales_out))
                continue

            out: Dict[str, Any] = {TABLE_SALES: _write_table(TABLE_SALES, idx_i, sales_out)}
            returns_out = _project_for_table(TABLE_SALES_RETURN, returns_table)  # type: ignore[arg-type]
            out[TABLE_SALES_RETURN] = _write_table(TABLE_SALES_RETURN, idx_i, returns_out)  # type: ignore[arg-type]
            results.append(out)
            continue

        out: Dict[str, Any] = {}

        if mode == "both":
            sales_table = detail_table
            if bool(getattr(State, "skip_order_cols_requested", False)):
                sales_table = _drop_order_cols_for_sales(sales_table)
            out[TABLE_SALES] = _write_table(TABLE_SALES, idx_i, _project_for_table(TABLE_SALES, sales_table))

        header_table = build_header_from_detail(detail_table)

        out[TABLE_SALES_ORDER_DETAIL] = _write_table(
            TABLE_SALES_ORDER_DETAIL, idx_i, _project_for_table(TABLE_SALES_ORDER_DETAIL, detail_table)
        )
        out[TABLE_SALES_ORDER_HEADER] = _write_table(
            TABLE_SALES_ORDER_HEADER, idx_i, _project_for_table(TABLE_SALES_ORDER_HEADER, header_table)
        )

        returns_table = _maybe_build_returns(detail_table, chunk_seed=int(idx_i))
        if returns_table is not None:
            out[TABLE_SALES_RETURN] = _write_table(
                TABLE_SALES_RETURN, idx_i, _project_for_table(TABLE_SALES_RETURN, returns_table)  # type: ignore[arg-type]
            )

        results.append(out)

    return results[0] if single else results