from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Optional, Sequence

import numpy as np
import pyarrow as pa


@dataclass(frozen=True)
class ReturnsConfig:
    enabled: bool = True
    return_rate: float = 0.08
    max_lag_days: int = 60
    reason_keys: Sequence[int] = (1,)
    reason_probs: Sequence[float] = (1.0,)


def _require_cols(t: pa.Table, cols: Sequence[str], *, ctx: str) -> None:
    missing = [c for c in cols if c not in t.schema.names]
    if missing:
        raise RuntimeError(f"{ctx} missing columns: {missing}. Available: {t.schema.names}")


def _normalize_probs(keys: Sequence[int], probs: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    k = np.asarray(list(keys), dtype=np.int64)
    if k.size == 0:
        k = np.asarray([1], dtype=np.int64)

    p = np.asarray(list(probs), dtype=np.float64)
    if p.size != k.size or p.size == 0:
        p = np.ones(k.size, dtype=np.float64)

    s = float(p.sum())
    if s <= 0:
        p = np.ones(k.size, dtype=np.float64)
        s = float(p.sum())
    p = p / s
    return k, p


def _to_py_dates(col: pa.ChunkedArray | pa.Array) -> list:
    if isinstance(col, pa.ChunkedArray):
        return col.combine_chunks().to_pylist()
    return col.to_pylist()


def _to_np_f64(arr: pa.ChunkedArray | pa.Array) -> np.ndarray:
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    return np.asarray(arr.to_numpy(zero_copy_only=False), dtype=np.float64)


def _to_np_i64(arr: pa.ChunkedArray | pa.Array) -> np.ndarray:
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()
    return np.asarray(arr.to_numpy(zero_copy_only=False), dtype=np.int64)


def build_sales_returns_from_detail(
    detail: pa.Table,
    *,
    chunk_seed: int,
    cfg: ReturnsConfig,
) -> Optional[pa.Table]:
    """
    Build SalesReturn from a line-grain source table.

    Required input columns:
      SalesOrderNumber, SalesOrderLineNumber,
      CustomerKey, ProductKey, StoreKey, PromotionKey, CurrencyKey,
      OrderDate, DeliveryDate,
      Quantity, UnitPrice, DiscountAmount, NetPrice, UnitCost

    Output includes (at least):
      SalesReturnKey
      SalesOrderNumber, SalesOrderLineNumber
      CustomerKey, ProductKey, StoreKey, PromotionKey, CurrencyKey
      OrderDate, DeliveryDate, ReturnDate
      ReturnReasonKey
      ReturnQuantity
      ReturnUnitPrice, ReturnDiscountAmount, ReturnNetPrice, ReturnUnitCost
      ReturnAmount, ReturnCost
    """
    if not cfg.enabled or cfg.return_rate <= 0:
        return None

    n = int(detail.num_rows)
    if n <= 0:
        return None

    _require_cols(
        detail,
        [
            "SalesOrderNumber",
            "SalesOrderLineNumber",
            "CustomerKey",
            "ProductKey",
            "StoreKey",
            "PromotionKey",
            "CurrencyKey",
            "OrderDate",
            "DeliveryDate",
            "Quantity",
            "UnitPrice",
            "DiscountAmount",
            "NetPrice",
            "UnitCost",
        ],
        ctx="SalesReturn build requires",
    )

    rng = np.random.default_rng(int(chunk_seed))

    # Decide which lines return
    mask = rng.random(n) < float(cfg.return_rate)
    idxs = np.nonzero(mask)[0]
    k = int(idxs.size)
    if k == 0:
        return None

    # Take returned lines
    take_idx = pa.array(idxs.astype(np.int64), type=pa.int64())
    d = detail.take(take_idx)

    # Quantities (return 1..Quantity)
    qty = _to_np_i64(d["Quantity"])
    qty = np.where(qty < 1, 1, qty).astype(np.int64)
    return_qty = rng.integers(1, qty + 1, size=k, dtype=np.int64)

    # Reasons
    reason_keys, reason_probs = _normalize_probs(cfg.reason_keys, cfg.reason_probs)
    reason = rng.choice(reason_keys, size=k, replace=True, p=reason_probs).astype(np.int64)

    # ReturnDate = DeliveryDate + lag
    lag = rng.integers(0, int(cfg.max_lag_days) + 1, size=k, dtype=np.int64)
    delivery_dates = _to_py_dates(d["DeliveryDate"])
    return_dates = []
    for dd, lag_days in zip(delivery_dates, lag.tolist()):
        base = dd.date() if hasattr(dd, "date") else dd
        return_dates.append(base + timedelta(days=int(lag_days)))

    # Per-unit economics
    unit_price = _to_np_f64(d["UnitPrice"])
    unit_cost = _to_np_f64(d["UnitCost"])
    disc_total = _to_np_f64(d["DiscountAmount"])
    net_total = _to_np_f64(d["NetPrice"])

    qty_f = np.maximum(qty.astype(np.float64), 1.0)
    per_unit_disc = disc_total / qty_f
    per_unit_net = net_total / qty_f

    rq_f = return_qty.astype(np.float64)

    return_unit_price = unit_price
    return_unit_cost = unit_cost
    return_discount = per_unit_disc * rq_f
    return_net = per_unit_net * rq_f

    # Keep these consistent (ReturnAmount == ReturnNetPrice)
    return_amount = return_net
    return_cost = return_unit_cost * rq_f

    # Deterministic SalesReturnKey per chunk
    base = (np.int64(chunk_seed) & np.int64(0xFFFF_FFFF)) << np.int64(32)
    return_key = base + (np.arange(1, k + 1, dtype=np.int64))

    cols = {
        "SalesReturnKey": pa.array(return_key, type=pa.int64()),
        "SalesOrderNumber": d["SalesOrderNumber"],
        "SalesOrderLineNumber": d["SalesOrderLineNumber"],
        "CustomerKey": d["CustomerKey"],
        "ProductKey": d["ProductKey"],
        "StoreKey": d["StoreKey"],
        "PromotionKey": d["PromotionKey"],
        "CurrencyKey": d["CurrencyKey"],
        "OrderDate": d["OrderDate"],
        "DeliveryDate": d["DeliveryDate"],
        "ReturnDate": pa.array(return_dates, type=pa.date32()),
        "ReturnReasonKey": pa.array(reason, type=pa.int64()),
        "ReturnQuantity": pa.array(return_qty, type=pa.int64()),
        "ReturnUnitPrice": pa.array(return_unit_price, type=pa.float64()),
        "ReturnDiscountAmount": pa.array(return_discount, type=pa.float64()),
        "ReturnNetPrice": pa.array(return_net, type=pa.float64()),
        "ReturnUnitCost": pa.array(return_unit_cost, type=pa.float64()),
        "ReturnAmount": pa.array(return_amount, type=pa.float64()),
        "ReturnCost": pa.array(return_cost, type=pa.float64()),
    }

    return pa.table(cols)
