from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pyarrow as pa


# Columns required from SalesOrderDetail to produce a returns fact table.
RETURNS_REQUIRED_DETAIL_COLS: tuple[str, ...] = (
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
)

# Output schema for the SalesReturn fact table.
#
# Notes on semantics:
# - ReturnUnitPrice / ReturnUnitCost are per-unit values copied from the original sales line.
# - ReturnDiscountAmount / ReturnNetPrice are *line amounts* prorated by returned quantity.
RETURNS_SCHEMA = pa.schema(
    [
        ("SalesOrderNumber", pa.int64()),
        ("SalesOrderLineNumber", pa.int64()),
        ("CustomerKey", pa.int64()),
        ("ProductKey", pa.int64()),
        ("StoreKey", pa.int64()),
        ("PromotionKey", pa.int64()),
        ("CurrencyKey", pa.int64()),
        ("ReturnDate", pa.date32()),
        ("ReturnReasonKey", pa.int64()),
        ("ReturnQuantity", pa.int64()),
        ("ReturnUnitPrice", pa.float64()),
        ("ReturnDiscountAmount", pa.float64()),
        ("ReturnNetPrice", pa.float64()),
        ("ReturnUnitCost", pa.float64()),
        ("ReturnEventKey", pa.int64()),
    ]
)


@dataclass(frozen=True)
class ReturnsConfig:
    enabled: bool = False
    return_rate: float = 0.0  # probability per SalesOrderDetail line
    max_lag_days: int = 60    # return date = delivery date + lag_days (allows 0 for same-day)
    reason_keys: Sequence[int] = (1, 2, 3, 4, 5, 6, 7, 8)
    reason_probs: Sequence[float] = (0.20, 0.12, 0.14, 0.10, 0.14, 0.08, 0.07, 0.15)


def _empty_returns_table() -> pa.Table:
    arrays = [pa.array([], type=f.type) for f in RETURNS_SCHEMA]
    return pa.Table.from_arrays(arrays, schema=RETURNS_SCHEMA)


def _as_np_i64(x) -> np.ndarray:
    arr = np.asarray(x)
    return arr if arr.dtype == np.int64 else arr.astype(np.int64, copy=False)


def _as_np_f64(x) -> np.ndarray:
    arr = np.asarray(x)
    return arr if arr.dtype == np.float64 else arr.astype(np.float64, copy=False)


def _ensure_required_columns(detail: pa.Table) -> None:
    missing = [c for c in RETURNS_REQUIRED_DETAIL_COLS if c not in detail.column_names]
    if missing:
        raise RuntimeError(f"Returns builder: missing required column(s): {missing!r}")


def _col_np(table: pa.Table, name: str) -> np.ndarray:
    """
    Convert a table column to a NumPy array (materializing if required).
    """
    if name not in table.column_names:
        raise RuntimeError(f"Returns builder: missing required column {name!r}")

    col = table[name]
    # table[name] is typically a ChunkedArray.
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
        # combine_chunks() returns an Array (or a ChunkedArray with 1 chunk depending on version).
        if isinstance(col, pa.ChunkedArray):
            col = col.chunk(0)

    return col.to_numpy(zero_copy_only=False)


def _to_np_date_days(x: np.ndarray) -> np.ndarray:
    """
    Normalize input to numpy datetime64[D].

    Handles:
    - numpy datetime64 (any unit)
    - object arrays of python date/datetime
    - strings parsable by numpy
    """
    if np.issubdtype(x.dtype, np.datetime64):
        return x.astype("datetime64[D]", copy=False)

    # object/strings -> let numpy parse; this may raise if unparseable.
    return np.asarray(x, dtype="datetime64[D]")


def _validate_cfg(cfg: ReturnsConfig) -> tuple[float, int, np.ndarray, np.ndarray]:
    """
    Returns:
      (return_rate, max_lag_days, reason_keys_i64, reason_probs_f64_normalized)
    """
    if not isinstance(cfg.enabled, bool):
        raise RuntimeError("ReturnsConfig.enabled must be a bool.")

    # return_rate validation
    rr = float(cfg.return_rate)
    if not np.isfinite(rr):
        raise RuntimeError("ReturnsConfig.return_rate must be finite.")
    if rr < 0.0 or rr > 1.0:
        raise RuntimeError("ReturnsConfig.return_rate must be in [0, 1].")

    # max_lag_days validation (allow 0)
    max_lag = int(cfg.max_lag_days)
    if max_lag < 0:
        raise RuntimeError("ReturnsConfig.max_lag_days must be >= 0.")

    # reasons
    rk = _as_np_i64(list(cfg.reason_keys))
    rp = _as_np_f64(list(cfg.reason_probs))

    if rk.size == 0:
        rk = np.array([1], dtype=np.int64)
        rp = np.array([1.0], dtype=np.float64)

    if rp.size != rk.size:
        raise RuntimeError("ReturnsConfig.reason_probs must match reason_keys length.")

    if not np.all(np.isfinite(rp)):
        raise RuntimeError("ReturnsConfig.reason_probs must be finite.")
    if np.any(rp < 0):
        raise RuntimeError("ReturnsConfig.reason_probs must be >= 0.")

    s = float(rp.sum())
    if s <= 0.0:
        rp = np.full_like(rp, 1.0 / rp.size, dtype=np.float64)
    else:
        rp = rp / s

    return rr, max_lag, rk, rp


def build_sales_returns_from_detail(
    detail: pa.Table,
    *,
    chunk_seed: int,
    cfg: ReturnsConfig,
) -> pa.Table:
    """
    Build SalesReturn event rows from SalesOrderDetail lines.

    Determinism:
      - Uses a RNG seeded from chunk_seed, so the output is deterministic per chunk.

    Keys:
      - ReturnEventKey packs a 32-bit chunk id and 32-bit ordinal:
          (chunk_seed_u32 << 32) | ordinal_u32
        This is unique as long as chunk_seed is unique across the dataset/run.
    """
    if not cfg.enabled:
        return _empty_returns_table()

    rr, max_lag, reason_keys, reason_probs = _validate_cfg(cfg)

    if rr <= 0.0:
        return _empty_returns_table()

    n = int(detail.num_rows)
    if n <= 0:
        return _empty_returns_table()

    _ensure_required_columns(detail)

    # Avoid repeated combine_chunks() work per-column.
    detail_cc = detail.combine_chunks()

    rng = np.random.default_rng(int(chunk_seed) ^ 0xA5A5_F00D)

    qty_all = _as_np_i64(_col_np(detail_cc, "Quantity"))
    if qty_all.size != n:
        raise RuntimeError("Returns builder: unexpected Quantity length mismatch.")

    # Select candidate return lines.
    mask = (rng.random(n) < rr) & (qty_all > 0)
    if not bool(mask.any()):
        return _empty_returns_table()

    # Extract masked arrays (alignment-safe).
    so = _as_np_i64(_col_np(detail_cc, "SalesOrderNumber"))[mask]
    line = _as_np_i64(_col_np(detail_cc, "SalesOrderLineNumber"))[mask]
    cust = _as_np_i64(_col_np(detail_cc, "CustomerKey"))[mask]
    prod = _as_np_i64(_col_np(detail_cc, "ProductKey"))[mask]
    store = _as_np_i64(_col_np(detail_cc, "StoreKey"))[mask]
    promo = _as_np_i64(_col_np(detail_cc, "PromotionKey"))[mask]
    curr = _as_np_i64(_col_np(detail_cc, "CurrencyKey"))[mask]

    delivery_raw = _col_np(detail_cc, "DeliveryDate")[mask]
    delivery = _to_np_date_days(delivery_raw)

    qty = qty_all[mask]

    unit_price = _as_np_f64(_col_np(detail_cc, "UnitPrice"))[mask]
    disc_amt = _as_np_f64(_col_np(detail_cc, "DiscountAmount"))[mask]
    net_price = _as_np_f64(_col_np(detail_cc, "NetPrice"))[mask]
    unit_cost = _as_np_f64(_col_np(detail_cc, "UnitCost"))[mask]

    m = int(qty.size)
    if m == 0:
        return _empty_returns_table()
    if m >= (1 << 32):
        raise RuntimeError("SalesReturn chunk too large for ReturnEventKey packing (>= 2^32 rows).")

    # ReturnQuantity: integer in [1, qty]
    u = rng.random(m)
    ret_qty = (np.floor(u * qty).astype(np.int64) + 1).clip(1, qty)

    frac = ret_qty.astype(np.float64) / qty.astype(np.float64)

    # Lag is in [0, max_lag] days (0 allowed for same-day returns).
    if max_lag == 0:
        lag_days = np.zeros(m, dtype=np.int32)
    else:
        lag_days = rng.integers(0, max_lag + 1, size=m, dtype=np.int32)

    lag = lag_days.astype("timedelta64[D]")
    ret_date = (delivery + lag).astype("datetime64[D]", copy=False)

    reason = rng.choice(reason_keys, size=m, p=reason_probs).astype(np.int64)

    # ReturnEventKey: (chunk_seed_u32 << 32) | ordinal
    seed32 = np.int64(int(chunk_seed) & 0xFFFF_FFFF)
    ordinal = np.arange(m, dtype=np.int64)  # < 2^32 by check above
    return_event_key = (seed32 << 32) | ordinal
    return_net_price = np.round(net_price * frac, 2).astype(np.float64, copy=False)

    # Build table using explicit schema to keep empty/non-empty consistent.
    return pa.table(
        {
            "SalesOrderNumber": pa.array(so, type=pa.int64()),
            "SalesOrderLineNumber": pa.array(line, type=pa.int64()),
            "CustomerKey": pa.array(cust, type=pa.int64()),
            "ProductKey": pa.array(prod, type=pa.int64()),
            "StoreKey": pa.array(store, type=pa.int64()),
            "PromotionKey": pa.array(promo, type=pa.int64()),
            "CurrencyKey": pa.array(curr, type=pa.int64()),
            "ReturnDate": pa.array(ret_date, type=pa.date32()),
            "ReturnReasonKey": pa.array(reason, type=pa.int64()),
            "ReturnQuantity": pa.array(ret_qty, type=pa.int64()),
            "ReturnUnitPrice": pa.array(unit_price, type=pa.float64()),
            "ReturnDiscountAmount": pa.array(disc_amt * frac, type=pa.float64()),
            "ReturnNetPrice": pa.array(return_net_price, type=pa.float64()),
            "ReturnUnitCost": pa.array(unit_cost, type=pa.float64()),
            "ReturnEventKey": pa.array(return_event_key, type=pa.int64()),
        },
        schema=RETURNS_SCHEMA,
    )