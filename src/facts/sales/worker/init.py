from __future__ import annotations

import os
from typing import Any, Optional

import numpy as np
import pyarrow as pa

from ..sales_logic.globals import State, bind_globals
from .schemas import schema_dict_cols
from ..output_paths import OutputPaths
from ..output_paths import TABLE_SALES, TABLE_SALES_ORDER_DETAIL, TABLE_SALES_ORDER_HEADER


# ===============================================================
# Small utils (moved from sales_worker.py)
# ===============================================================

def _int_or(v: Any, default: int) -> int:
    try:
        if v is None or v == "":
            return int(default)
        return int(v)
    except Exception:
        return int(default)


def _str_or(v: Any, default: str) -> str:
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def _as_int64(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.int64)


def _as_f64(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def _dense_map(mapping: Optional[dict]) -> Optional[np.ndarray]:
    """
    Build dense lookup array: arr[key] -> value, missing -> -1.
    Vectorized (fast). Assumes keys are non-negative ints.
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


# ===============================================================
# Worker initializer (runs once per process)
# ===============================================================

def init_sales_worker(worker_cfg: dict):
    """
    Initialize immutable worker state.
    Runs exactly once per worker process.

    Lifecycle-aware contract:
      - Binds customer arrays needed by chunk_builder:
        customer_keys, customer_is_active_in_sales, customer_start_month, customer_end_month,
        optional customer_base_weight
    """

    # -----------------------------------------------------------
    # Extract config (explicit, fail-fast)
    # -----------------------------------------------------------
    try:
        product_np = worker_cfg["product_np"]
        store_keys = worker_cfg["store_keys"]

        promo_keys_all = worker_cfg["promo_keys_all"]
        promo_pct_all = worker_cfg["promo_pct_all"]
        promo_start_all = worker_cfg["promo_start_all"]
        promo_end_all = worker_cfg["promo_end_all"]

        # Backward compat: still accept 'customers' as a plain key array
        customers = worker_cfg.get("customers")

        # New contract (preferred)
        customer_keys = worker_cfg.get("customer_keys", customers)
        customer_is_active_in_sales = worker_cfg.get("customer_is_active_in_sales")
        customer_start_month = worker_cfg.get("customer_start_month")
        customer_end_month = worker_cfg.get("customer_end_month")
        customer_base_weight = worker_cfg.get("customer_base_weight")

        store_to_geo = worker_cfg["store_to_geo"]
        geo_to_currency = worker_cfg["geo_to_currency"]

        date_pool = worker_cfg["date_pool"]
        date_prob = worker_cfg["date_prob"]

        op = worker_cfg.get("output_paths") or {}

        file_format = worker_cfg.get("file_format") or op.get("file_format")
        out_folder = worker_cfg.get("out_folder") or op.get("out_folder")

        row_group_size = _int_or(worker_cfg.get("row_group_size"), 2_000_000)
        compression = _str_or(worker_cfg.get("compression"), "snappy")

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
        skip_order_cols_requested = bool(skip_order_cols)

        # Effective behavior: Order tables require order keys, so chunk_builder must output them.
        if sales_output in {"sales_order", "both"}:
            skip_order_cols = False

        partition_enabled = worker_cfg.get("partition_enabled", False)
        partition_cols = worker_cfg.get("partition_cols") or []
        models_cfg = worker_cfg.get("models_cfg")

        # Optional (passed by sales.py; keep for future compatibility)
        write_pyarrow = worker_cfg.get("write_pyarrow", True)

    except KeyError as e:
        raise RuntimeError(f"Missing worker config key: {e}") from None

    if not file_format:
        raise RuntimeError("Missing worker config key: 'file_format'")
    if not out_folder:
        raise RuntimeError("Missing worker config key: 'out_folder'")

    if skip_order_cols not in (True, False):
        raise RuntimeError("skip_order_cols must be a boolean")

    if customer_keys is None:
        raise RuntimeError("worker_cfg must include customer_keys or customers")

    output_paths = OutputPaths(
        file_format=file_format,
        out_folder=out_folder,
        merged_file=merged_file,
        delta_output_folder=delta_output_folder,
    )

    # -----------------------------------------------------------
    # Normalize arrays (dtype + shape checks)
    # -----------------------------------------------------------
    product_np = np.asarray(product_np)  # keep original dtype/shape
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
        customer_base_weight = _as_f64(customer_base_weight)
        if customer_base_weight.shape[0] != customer_keys.shape[0]:
            raise RuntimeError("customer_base_weight must align with customer_keys length")

    # -----------------------------------------------------------
    # Dense mapping helpers (fast lookup)
    # -----------------------------------------------------------
    store_to_geo_arr = _dense_map(store_to_geo) if isinstance(store_to_geo, dict) else None
    geo_to_currency_arr = _dense_map(geo_to_currency) if isinstance(geo_to_currency, dict) else None

    # -----------------------------------------------------------
    # Ensure output folders once
    # -----------------------------------------------------------
    tables = []
    if sales_output in {"sales", "both"}:
        tables.append(TABLE_SALES)
    if sales_output in {"sales_order", "both"}:
        tables += [TABLE_SALES_ORDER_DETAIL, TABLE_SALES_ORDER_HEADER]

    for t in tables:
        output_paths.ensure_dirs(t)

    # -----------------------------------------------------------
    # Canonical schemas
    #
    # CRITICAL RULE:
    # - Sales output must remain unchanged for sales_output="sales" and "both"
    # -----------------------------------------------------------

    # --- Sales (unchanged) ---
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
    ]

    order_fields = [
        pa.field("SalesOrderNumber", pa.int64()),
        pa.field("SalesOrderLineNumber", pa.int64()),
    ]

    delta_fields = [
        pa.field("Year", pa.int16()),
        pa.field("Month", pa.int8()),
    ]

    schema_no_order = pa.schema(base_fields)
    schema_with_order = pa.schema(order_fields + base_fields)

    schema_no_order_delta = pa.schema(base_fields + delta_fields)
    schema_with_order_delta = pa.schema(order_fields + base_fields + delta_fields)

    if file_format == "deltaparquet":
        sales_schema = schema_no_order_delta if skip_order_cols_requested else schema_with_order_delta
    else:
        sales_schema = schema_no_order if skip_order_cols_requested else schema_with_order

    # --- SalesOrderDetail (SLIM, conventional) ---
    # Conventional: header-level foreign keys + order dates live in Header, not Detail.
    # DeliveryDate can vary per line -> keep in Detail.
    detail_fields = [
        pa.field("SalesOrderNumber", pa.int64()),
        pa.field("SalesOrderLineNumber", pa.int64()),

        pa.field("ProductKey", pa.int64()),

        pa.field("DeliveryDate", pa.date32()),

        pa.field("Quantity", pa.int64()),
        pa.field("NetPrice", pa.float64()),
        pa.field("UnitCost", pa.float64()),
        pa.field("UnitPrice", pa.float64()),
        pa.field("DiscountAmount", pa.float64()),

        pa.field("DeliveryStatus", pa.string()),
        pa.field("IsOrderDelayed", pa.int8()),
    ]
    detail_schema = pa.schema(detail_fields + delta_fields) if file_format == "deltaparquet" else pa.schema(detail_fields)

    # --- SalesOrderHeader (SLIM, NO aggregates, NO DeliveryDate) ---
    header_fields = [
        pa.field("SalesOrderNumber", pa.int64()),

        pa.field("CustomerKey", pa.int64()),
        pa.field("StoreKey", pa.int64()),
        pa.field("PromotionKey", pa.int64()),
        pa.field("CurrencyKey", pa.int64()),

        pa.field("OrderDate", pa.date32()),
        pa.field("DueDate", pa.date32()),

        pa.field("IsOrderDelayed", pa.int8()),
    ]
    header_schema = pa.schema(header_fields + delta_fields) if file_format == "deltaparquet" else pa.schema(header_fields)

    schema_by_table = {
        TABLE_SALES: sales_schema,
        TABLE_SALES_ORDER_DETAIL: detail_schema,
        TABLE_SALES_ORDER_HEADER: header_schema,
    }

    # Dictionary encoding: only for strings; keep exclusions consistent
    parquet_dict_exclude = {"SalesOrderNumber", "CustomerKey"}
    parquet_dict_cols_by_table = {
        t: schema_dict_cols(s, parquet_dict_exclude) for t, s in schema_by_table.items()
    }

    # Back-compat (old code expects these names)
    parquet_dict_cols = parquet_dict_cols_by_table[TABLE_SALES]

    # -----------------------------------------------------------
    # Bind immutable globals (ONCE)
    # -----------------------------------------------------------
    bind_globals({
        # core data
        "product_np": product_np,
        "store_keys": store_keys,

        # Backward compat: keep 'customers' for any old codepaths
        "customers": customers if customers is not None else customer_keys,

        # New: lifecycle-aware customer arrays for chunk_builder
        "customer_keys": customer_keys,
        "customer_is_active_in_sales": customer_is_active_in_sales,
        "customer_start_month": customer_start_month,
        "customer_end_month": customer_end_month,
        "customer_base_weight": customer_base_weight,

        # promotions
        "promo_keys_all": promo_keys_all,
        "promo_pct_all": promo_pct_all,
        "promo_start_all": promo_start_all,
        "promo_end_all": promo_end_all,

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
        "schema_no_order": schema_no_order,
        "schema_with_order": schema_with_order,
        "schema_no_order_delta": schema_no_order_delta,
        "schema_with_order_delta": schema_with_order_delta,
        "sales_schema": sales_schema,
        "schema_by_table": schema_by_table,
        "parquet_dict_cols_by_table": parquet_dict_cols_by_table,

        # parquet tuning
        "parquet_dict_exclude": parquet_dict_exclude,
        "parquet_dict_cols": parquet_dict_cols,
    })

    State.seal()
