import os
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .sales_logic import chunk_builder
from .sales_logic.globals import State, bind_globals


# ===============================================================
# Worker initializer (runs once per process)
# ===============================================================

def init_sales_worker(worker_cfg: dict):
    """
    Initialize immutable worker state.
    Runs exactly once per worker process.
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

        customers = worker_cfg["customers"]

        store_to_geo = worker_cfg["store_to_geo"]
        geo_to_currency = worker_cfg["geo_to_currency"]

        date_pool = worker_cfg["date_pool"]
        date_prob = worker_cfg["date_prob"]

        out_folder = worker_cfg["out_folder"]
        file_format = worker_cfg["file_format"]

        row_group_size = int(worker_cfg["row_group_size"])
        compression = worker_cfg["compression"]

        no_discount_key = worker_cfg["no_discount_key"]
        delta_output_folder = worker_cfg["delta_output_folder"]
        write_delta = worker_cfg["write_delta"]

        skip_order_cols = worker_cfg["skip_order_cols"]
        partition_enabled = worker_cfg["partition_enabled"]
        partition_cols = worker_cfg["partition_cols"]

    except KeyError as e:
        raise RuntimeError(f"Missing worker config key: {e}") from None

    if skip_order_cols not in (True, False):
        raise RuntimeError("skip_order_cols must be a boolean")

    # -----------------------------------------------------------
    # Dense mapping helpers (fast lookup)
    # -----------------------------------------------------------
    def _dense_map(mapping: dict | None):
        if not mapping:
            return None
        max_key = max(mapping.keys())
        arr = np.full(max_key + 1, -1, dtype=np.int64)
        for k, v in mapping.items():
            arr[int(k)] = int(v)
        return arr

    store_to_geo_arr = (
        _dense_map(store_to_geo) if isinstance(store_to_geo, dict) else None
    )
    geo_to_currency_arr = (
        _dense_map(geo_to_currency)
        if isinstance(geo_to_currency, dict)
        else None
    )

    # -----------------------------------------------------------
    # Ensure output folders once
    # -----------------------------------------------------------
    if file_format == "deltaparquet":
        os.makedirs(
            os.path.join(delta_output_folder, "_tmp_parts"),
            exist_ok=True,
        )
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

    # -----------------------------------------------------------
    # Canonical sales schema (single source of truth)
    # -----------------------------------------------------------
    if file_format == "deltaparquet":
        sales_schema = (
            schema_with_order_delta
            if not skip_order_cols
            else schema_no_order_delta
        )
    else:
        sales_schema = (
            schema_with_order
            if not skip_order_cols
            else schema_no_order
        )

    # -----------------------------------------------------------
    # Bind immutable globals (ONCE)
    # -----------------------------------------------------------
    bind_globals({
        # core data
        "product_np": product_np,
        "store_keys": store_keys,
        "customers": customers,

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
        "row_group_size": row_group_size,
        "compression": compression,

        # delta
        "delta_output_folder": os.path.normpath(delta_output_folder),
        "write_delta": write_delta,

        # behavior
        "no_discount_key": no_discount_key,
        "skip_order_cols": skip_order_cols,
        "partition_enabled": partition_enabled,
        "partition_cols": partition_cols,

        # schemas
        "schema_no_order": schema_no_order,
        "schema_with_order": schema_with_order,
        "schema_no_order_delta": schema_no_order_delta,
        "schema_with_order_delta": schema_with_order_delta,
        "sales_schema": sales_schema,

        # parquet tuning
        "parquet_dict_exclude": {"SalesOrderNumber", "CustomerKey"},
    })

    State.seal()


# ===============================================================
# Writers
# ===============================================================

def _write_parquet_batches(table: pa.Table, path: str):
    schema = State.sales_schema

    if table.schema != schema:
        raise RuntimeError(
            "Schema mismatch in parquet writer.\n"
            f"Expected:\n{schema}\n\nGot:\n{table.schema}"
        )

    dict_cols = [
        c for c in table.column_names
        if c not in State.parquet_dict_exclude
    ]

    writer = pq.ParquetWriter(
        path,
        schema,
        compression=State.compression,
        use_dictionary=dict_cols,
        write_statistics=True,
    )

    try:
        for batch in table.to_batches(
            max_chunksize=State.row_group_size
        ):
            writer.write_batch(batch)
    finally:
        writer.close()


def _write_csv(table: pa.Table, path: str):
    import pyarrow.compute as pc
    import pyarrow.csv as pacsv

    schema = State.sales_schema

    if table.schema != schema:
        raise RuntimeError(
            "Schema mismatch in CSV writer.\n"
            f"Expected:\n{schema}\n\nGot:\n{table.schema}"
        )

    # Ensure null-safe int8 for CSV
    if "IsOrderDelayed" in table.column_names:
        idx = table.schema.get_field_index("IsOrderDelayed")
        table = table.set_column(
            idx,
            "IsOrderDelayed",
            pc.cast(
                pc.fill_null(table["IsOrderDelayed"], 0),
                pa.int8(),
            ),
        )

    pacsv.write_csv(
        table,
        path,
        write_options=pacsv.WriteOptions(
            include_header=True,
            quoting_style="none",
        ),
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
        tasks = args
        single = False

    results = []

    for idx, batch_size, seed in tasks:
        base_seed = int(seed) if seed is not None else 0
        chunk_seed = base_seed + idx * 10_000

        table = chunk_builder.build_chunk_table(
            batch_size,
            chunk_seed,
            no_discount_key=State.no_discount_key,
        )

        if not isinstance(table, pa.Table):
            raise TypeError("chunk_builder must return pyarrow.Table")

        # DELTA
        if State.file_format == "deltaparquet":
            name = f"delta_part_{idx:04d}.parquet"
            path = os.path.join(
                State.delta_output_folder,
                "_tmp_parts",
                name,
            )
            _write_parquet_batches(table, path)
            rows = table.num_rows
            del table
            results.append({"part": name, "rows": rows})
            continue

        # CSV
        if State.file_format == "csv":
            path = os.path.join(
                State.out_folder,
                f"sales_chunk{idx:04d}.csv",
            )
            _write_csv(table, path)
            del table
            results.append(path)
            continue

        # PARQUET
        path = os.path.join(
            State.out_folder,
            f"sales_chunk{idx:04d}.parquet",
        )
        _write_parquet_batches(table, path)
        del table
        results.append(path)

    return results[0] if single else results
