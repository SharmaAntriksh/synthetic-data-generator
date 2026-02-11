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
def _build_buckets_from_brand_key(brand_key: np.ndarray) -> list[np.ndarray]:
    """
    Returns buckets[b] = np.ndarray of product_np row indices belonging to brand b.
    Expects brand_key to be int64, non-negative. Dense (0..B-1) is ideal; sparse works but
    allocates up to max(brand_key)+1.
    """
    brand_key = np.asarray(brand_key, dtype=np.int64)
    if brand_key.size == 0:
        return []

    if brand_key.min() < 0:
        raise RuntimeError("product_brand_key must be non-negative ints")

    max_b = int(brand_key.max())
    B = max_b + 1

    order = np.argsort(brand_key, kind="mergesort")
    b_sorted = brand_key[order]

    starts = np.flatnonzero(np.r_[True, b_sorted[1:] != b_sorted[:-1]])
    ends = np.r_[starts[1:], b_sorted.size]

    buckets: list[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(B)]
    for s, e in zip(starts, ends):
        b = int(b_sorted[int(s)])
        buckets[b] = order[int(s):int(e)].astype(np.int64, copy=False)

    return buckets


def _int_or(v: Any, default: int) -> int:
    try:
        if v is None or v == "":
            return int(default)
        return int(v)
    except Exception:
        return int(default)


def _float_or(v: Any, default: float) -> float:
    try:
        if v is None or v == "":
            return float(default)
        return float(v)
    except Exception:
        return float(default)


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


def _infer_T_from_date_pool(date_pool: Any) -> int:
    """
    Infer number of unique months in date_pool.
    Returned T is the count of unique numpy datetime64[M] buckets.
    """
    dp = np.asarray(date_pool, dtype="datetime64[D]")
    months = dp.astype("datetime64[M]")
    # np.unique returns sorted unique values for datetime64
    return int(np.unique(months).size)


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
    Build (T, B) probabilities where each year rotates a "winner" brand (year % B),
    optionally with multiplicative lognormal noise, then normalize.

    This guarantees "no single brand stays #1 every year" assuming:
      - B > 1
      - winner_boost is meaningfully > 1
      - T spans multiple years (>= 24 months for obvious rotation)
    """
    T = int(max(1, T))
    B = int(max(1, B))

    W = np.ones((T, B), dtype=np.float64)

    # Rotate yearly winner
    if B > 1 and winner_boost and float(winner_boost) > 1.0:
        for t in range(T):
            year = int(t // max(1, int(year_len_months)))
            winner = int(year % B)
            W[t, winner] *= float(winner_boost)

    # Add mild noise (keeps it from looking too deterministic)
    if noise_sd and float(noise_sd) > 0:
        W *= np.exp(rng.normal(0.0, float(noise_sd), size=W.shape))

    # Floor shares to avoid permanent collapse
    if min_share and float(min_share) > 0:
        floor = float(min_share)
        W = np.maximum(W, floor)

    # Normalize each month
    row_sum = W.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum <= 0, 1.0, row_sum)
    return W / row_sum


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

        # brand sampling inputs (optional)
        product_brand_key = worker_cfg.get("product_brand_key")  # optional
        brand_prob_by_month = worker_cfg.get("brand_prob_by_month")  # optional

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
        skip_order_cols_requested = bool(worker_cfg.get("skip_order_cols_requested", skip_order_cols))

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
        customer_base_weight = _as_f64(customer_base_weight)
        if customer_base_weight.shape[0] != customer_keys.shape[0]:
            raise RuntimeError("customer_base_weight must align with customer_keys length")

    # -----------------------------------------------------------
    # Build brand_prob_by_month (optional, deterministic, run-once)
    #
    # This enables "top brand rotates by year" when chunk_builder samples by month offset.
    # If caller already provided brand_prob_by_month, we keep it as-is.
    # -----------------------------------------------------------
    if brand_prob_by_month is None and brand_to_row_idx is not None:
        brand_cfg = None
        if isinstance(models_cfg, dict):
            brand_cfg = models_cfg.get("brand_popularity")

        use_brand_popularity = bool(brand_cfg) and bool(brand_cfg.get("enabled", True)) if isinstance(brand_cfg, dict) else False

        if use_brand_popularity:
            B = int(len(brand_to_row_idx))
            T = _infer_T_from_date_pool(date_pool)

            seed = _int_or(brand_cfg.get("seed") if isinstance(brand_cfg, dict) else None, 123)
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
        pa.field("IsOrderDelayed", pa.int8()),
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

    # Generation schema (must match effective skip_order_cols)
    # Output schema for Sales table (must match user intent: skip_order_cols_requested)
    if file_format == "deltaparquet":
        sales_schema_gen = schema_no_order_delta if skip_order_cols else schema_with_order_delta
        sales_schema_out = schema_no_order_delta if skip_order_cols_requested else schema_with_order_delta
    else:
        sales_schema_gen = schema_no_order if skip_order_cols else schema_with_order
        sales_schema_out = schema_no_order if skip_order_cols_requested else schema_with_order

    # --- SalesOrderDetail (LINE-GRAIN; matches static_schemas) ---
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

    # --- SalesOrderHeader (ORDER-GRAIN; minimal) ---
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

    # Dictionary encoding: only for strings; keep exclusions consistent
    parquet_dict_exclude = {"SalesOrderNumber", "CustomerKey"}
    parquet_dict_cols_by_table = {t: schema_dict_cols(s, parquet_dict_exclude) for t, s in schema_by_table.items()}

    # Back-compat (old code expects these names)
    parquet_dict_cols = parquet_dict_cols_by_table[TABLE_SALES]

    # -----------------------------------------------------------
    # Bind immutable globals (ONCE)
    # -----------------------------------------------------------
    bind_globals(
        {
            # core data
            "product_np": product_np,
            "store_keys": store_keys,
            "product_brand_key": product_brand_key,
            "brand_to_row_idx": brand_to_row_idx,
            "brand_prob_by_month": brand_prob_by_month,
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
            "sales_schema": sales_schema_gen,  # used by chunk_builder
            "sales_schema_out": sales_schema_out,  # optional (debug / future use)
            "schema_by_table": schema_by_table,
            "parquet_dict_cols_by_table": parquet_dict_cols_by_table,
            # parquet tuning
            "parquet_dict_exclude": parquet_dict_exclude,
            "parquet_dict_cols": parquet_dict_cols,
        }
    )

    State.seal()
