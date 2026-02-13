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
# Worker init helpers (shared across facts)
# ===============================================================
from src.facts.common.worker.init import (
    _build_buckets_from_brand_key,
    _int_or,
    _float_or,
    _str_or,
    _as_int64,
    _as_f64,
    _dense_map,
    _infer_T_from_date_pool,
)


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

    Each year segment rotates a "winner" brand, giving it a boost. Others get small noise.
    Ensures each brand has at least `min_share` share, then renormalizes.

    Returns:
        brand_prob_by_month: float64 array shape (T, B), rows sum to 1.
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

        # floor
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

    # -----------------------------------------------------------
    # Extract config (explicit, fail-fast)
    # -----------------------------------------------------------
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

        # Effective behavior: Order tables require order keys, so chunk_builder must output them.
        if sales_output in {"sales_order", "both"}:
            skip_order_cols = False

        partition_enabled = worker_cfg.get("partition_enabled", False)
        partition_cols = worker_cfg.get("partition_cols") or []
        models_cfg = worker_cfg.get("models_cfg")
        # models_cfg may be the *root* YAML dict with {"tuning": ..., "models": {...}}.

        parquet_dict_exclude = worker_cfg.get("parquet_dict_exclude")
        parquet_dict_cols = worker_cfg.get("parquet_dict_cols")
        write_pyarrow = worker_cfg.get("write_pyarrow", True)

    except KeyError as e:
        raise RuntimeError(f"Missing worker_cfg key: {e}") from e

    # -----------------------------------------------------------
    # Normalize arrays (dtype + shape checks)
    # -----------------------------------------------------------
    product_np = np.asarray(product_np)  # keep original dtype/shape

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

    # -----------------------------------------------------------
    # Optional brand popularity by month
    # -----------------------------------------------------------
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

    # -----------------------------------------------------------
    # Dense mapping helpers (fast lookup)
    # -----------------------------------------------------------
    store_to_geo_arr = _dense_map(store_to_geo) if isinstance(store_to_geo, dict) else None
    geo_to_currency_arr = _dense_map(geo_to_currency) if isinstance(geo_to_currency, dict) else None

    # -----------------------------------------------------------
    # Ensure output folders once
    # -----------------------------------------------------------
    output_paths = OutputPaths(
        out_folder=out_folder,
        delta_output_folder=delta_output_folder,
        file_format=file_format,
        merged_file=merged_file,
    )

    tables = []
    if sales_output in {"sales", "both"}:
        tables.append(TABLE_SALES)
    if sales_output in {"sales_order", "both"}:
        tables += [TABLE_SALES_ORDER_DETAIL, TABLE_SALES_ORDER_HEADER]

    for t in tables:
        output_paths.ensure_dirs(t)

    # -----------------------------------------------------------
    # Canonical schemas
    # -----------------------------------------------------------
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

    # Order fields (optional depending on skip_order_cols)
    order_fields = [
        pa.field("SalesOrderNumber", pa.int64()),
        pa.field("SalesOrderLineNumber", pa.int64()),
    ]

    # Delta fields (only for deltaparquet)
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

    # SalesOrderDetail schema
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

    # SalesOrderHeader schema
    header_fields = [
        pa.field("SalesOrderNumber", pa.int64()),
        pa.field("CustomerKey", pa.int64()),
        pa.field("OrderDate", pa.date32()),
        pa.field("IsOrderDelayed", pa.int8()),
    ]
    header_schema = pa.schema(header_fields + delta_fields) if file_format == "deltaparquet" else pa.schema(header_fields)

    schema_by_table = {
        TABLE_SALES: sales_schema_out,
        TABLE_SALES_ORDER_DETAIL: detail_schema,
        TABLE_SALES_ORDER_HEADER: header_schema,
    }

    parquet_dict_exclude = {"SalesOrderNumber", "CustomerKey"}
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
        }
    )