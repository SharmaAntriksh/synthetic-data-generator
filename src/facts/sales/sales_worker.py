from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .sales_logic import chunk_builder
from .sales_logic.globals import State, bind_globals


# ===============================================================
# Small utils
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

    # Vectorize keys/vals
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


def _schema_dict_cols(schema: pa.Schema, exclude: set) -> List[str]:
    """
    Dictionary encode only string-ish columns (excluding some IDs).
    """
    out: List[str] = []
    for f in schema:
        if f.name in exclude:
            continue
        t = f.type
        if pa.types.is_string(t) or pa.types.is_large_string(t):
            out.append(f.name)
    return out


# Cache CSV modules (avoid importing each chunk)
_PACSV = None
_PC = None


def _pa_csv():
    global _PACSV
    if _PACSV is None:
        import pyarrow.csv as pacsv
        _PACSV = pacsv
    return _PACSV


def _pa_compute():
    global _PC
    if _PC is None:
        import pyarrow.compute as pc
        _PC = pc
    return _PC


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

        out_folder = worker_cfg["out_folder"]
        file_format = worker_cfg["file_format"]

        row_group_size = _int_or(worker_cfg.get("row_group_size"), 2_000_000)
        compression = _str_or(worker_cfg.get("compression"), "snappy")

        no_discount_key = worker_cfg["no_discount_key"]
        delta_output_folder = worker_cfg.get("delta_output_folder")
        write_delta = worker_cfg.get("write_delta", False)

        skip_order_cols = worker_cfg["skip_order_cols"]
        partition_enabled = worker_cfg.get("partition_enabled", False)
        partition_cols = worker_cfg.get("partition_cols") or []
        models_cfg = worker_cfg.get("models_cfg")

        # Optional (passed by sales.py; keep for future compatibility)
        write_pyarrow = worker_cfg.get("write_pyarrow", True)

    except KeyError as e:
        raise RuntimeError(f"Missing worker config key: {e}") from None

    if skip_order_cols not in (True, False):
        raise RuntimeError("skip_order_cols must be a boolean")

    if customer_keys is None:
        raise RuntimeError("worker_cfg must include customer_keys or customers")

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
    if file_format == "deltaparquet":
        if not delta_output_folder:
            raise RuntimeError("delta_output_folder is required when file_format=deltaparquet")
        os.makedirs(os.path.join(delta_output_folder, "_tmp_parts"), exist_ok=True)
    else:
        os.makedirs(out_folder, exist_ok=True)

    # -----------------------------------------------------------
    # Canonical schemas (NO inference, NO drift)
    # -----------------------------------------------------------
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

    # Canonical sales schema (single source of truth)
    if file_format == "deltaparquet":
        sales_schema = schema_with_order_delta if not skip_order_cols else schema_no_order_delta
    else:
        sales_schema = schema_with_order if not skip_order_cols else schema_no_order

    # Dictionary encoding: only for strings; keep exclusions consistent
    parquet_dict_exclude = {"SalesOrderNumber", "CustomerKey"}
    parquet_dict_cols = _schema_dict_cols(sales_schema, parquet_dict_exclude)

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

        # delta
        "delta_output_folder": os.path.normpath(delta_output_folder) if delta_output_folder else None,
        "write_delta": bool(write_delta),

        # behavior
        "no_discount_key": no_discount_key,
        "skip_order_cols": skip_order_cols,
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

        # parquet tuning
        "parquet_dict_exclude": parquet_dict_exclude,
        "parquet_dict_cols": parquet_dict_cols,
    })

    State.seal()


# ===============================================================
# Writers
# ===============================================================

def _assert_schema(table: pa.Table) -> None:
    schema = State.sales_schema
    if table.schema != schema:
        raise RuntimeError(
            "Schema mismatch in writer.\n"
            f"Expected:\n{schema}\n\nGot:\n{table.schema}"
        )


def _write_parquet_table(table: pa.Table, path: str) -> None:
    """
    Fast path: write whole table with row_group_size; ParquetWriter chunks internally.
    """
    _assert_schema(table)

    writer = pq.ParquetWriter(
        path,
        State.sales_schema,
        compression=State.compression,
        use_dictionary=State.parquet_dict_cols,  # only string columns
        write_statistics=True,
    )
    try:
        writer.write_table(table, row_group_size=State.row_group_size)
    finally:
        writer.close()


def _write_csv(table: pa.Table, path: str) -> None:
    """
    CSV output (mainly for debugging/smaller runs).
    """
    _assert_schema(table)

    pc = _pa_compute()
    pacsv = _pa_csv()

    # Ensure null-safe int8 for CSV
    if "IsOrderDelayed" in table.column_names:
        idx = table.schema.get_field_index("IsOrderDelayed")
        table = table.set_column(
            idx,
            "IsOrderDelayed",
            pc.cast(pc.fill_null(table["IsOrderDelayed"], 0), pa.int8()),
        )

    pacsv.write_csv(
        table,
        path,
        write_options=pacsv.WriteOptions(include_header=True, quoting_style="none"),
    )


# ===============================================================
# Worker task
# ===============================================================

def _worker_task(args):
    """
    Supports:
      - single task: (idx, batch_size, seed)
      - batched tasks: [(idx, batch_size, seed), ...]
    """
    if isinstance(args, tuple):
        tasks = [args]
        single = True
    else:
        tasks = list(args)
        single = False

    results = []

    for idx, batch_size, seed in tasks:
        base_seed = _int_or(seed, 0)
        chunk_seed = base_seed + int(idx) * 10_000

        table = chunk_builder.build_chunk_table(
            int(batch_size),
            int(chunk_seed),
            no_discount_key=State.no_discount_key,
        )

        if not isinstance(table, pa.Table):
            raise TypeError("chunk_builder must return pyarrow.Table")

        # DELTA (write tmp parquet parts; merge later in sales_writer.write_delta_partitioned)
        if State.file_format == "deltaparquet":
            name = f"delta_part_{idx:04d}.parquet"
            path = os.path.join(State.delta_output_folder, "_tmp_parts", name)
            _write_parquet_table(table, path)
            rows = table.num_rows
            results.append({"part": name, "rows": rows})
            continue

        # CSV
        if State.file_format == "csv":
            path = os.path.join(State.out_folder, f"sales_chunk{idx:04d}.csv")
            _write_csv(table, path)
            results.append(path)
            continue

        # PARQUET
        path = os.path.join(State.out_folder, f"sales_chunk{idx:04d}.parquet")
        _write_parquet_table(table, path)
        results.append(path)

    return results[0] if single else results
