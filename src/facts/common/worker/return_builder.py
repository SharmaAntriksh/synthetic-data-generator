from __future__ import annotations

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
