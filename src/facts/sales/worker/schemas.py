from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Set

import pyarrow as pa

from ..output_paths import TABLE_SALES, TABLE_SALES_ORDER_DETAIL, TABLE_SALES_ORDER_HEADER

try:
    from ..output_paths import TABLE_SALES_RETURN  # type: ignore
except Exception:
    TABLE_SALES_RETURN = None  # type: ignore


def schema_dict_cols(schema: pa.Schema, exclude: Optional[Set[str]] = None) -> List[str]:
    """
    Return columns that are candidates for parquet dictionary encoding.
    Only string columns are included.
    """
    exclude = exclude or set()
    out: List[str] = []
    for f in schema:
        if f.name in exclude:
            continue
        if pa.types.is_string(f.type) or pa.types.is_large_string(f.type):
            out.append(f.name)
    return out


@dataclass(frozen=True)
class WorkerSchemaBundle:
    # Sales schemas
    sales_schema_gen: pa.Schema        # used by chunk builder; MUST match arrays it produces
    sales_schema_out: pa.Schema        # used for write/projection/merge; includes injected cols like TimeKey

    # Table schemas + date-col policy
    schema_by_table: dict[str, pa.Schema]
    date_cols_by_table: dict[str, list[str]]

    # Parquet dictionary encoding policy
    parquet_dict_exclude: set[str]
    parquet_dict_cols_by_table: dict[str, list[str]]
    parquet_dict_cols: list[str]

    # Convenience variants (OUTPUT variants)
    schema_no_order: pa.Schema
    schema_with_order: pa.Schema
    schema_no_order_delta: pa.Schema
    schema_with_order_delta: pa.Schema


def build_worker_schemas(
    *,
    file_format: str,
    skip_order_cols: bool,
    skip_order_cols_requested: bool,
    returns_enabled: bool,
    parquet_dict_exclude: Optional[Set[str]] = None,
    models_cfg: Optional[dict] = None,
) -> WorkerSchemaBundle:
    """
    Single source of truth for Sales / Order / Returns schemas & date-col policy.

    IMPORTANT:
      - sales_schema_gen is used by build_chunk_table (chunk builder). It must match the arrays that
        the builder currently emits. If you inject columns later in task.py (e.g., TimeKey), DO NOT
        add them to sales_schema_gen.
      - sales_schema_out is the "contract" for output chunks/merged files. It can include injected
        columns because projection/writing happens after injection.
    """

    ff = (file_format or "").strip().lower()
    is_delta = ff == "deltaparquet"

    # ---------------------------------------------------------------------
    # Sales schemas (GEN vs OUT)
    # ---------------------------------------------------------------------
    # GEN: what the chunk builder actually produces (NO TimeKey here)
    base_fields_gen = [
        pa.field("CustomerKey", pa.int64()),
        pa.field("ProductKey", pa.int64()),
        pa.field("StoreKey", pa.int64()),
        pa.field("SalesPersonEmployeeKey", pa.int64()),
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

    # OUT: what we want to write/merge/project (TimeKey injected later in task.py)
    base_fields_out = [
        pa.field("CustomerKey", pa.int64()),
        pa.field("ProductKey", pa.int64()),
        pa.field("StoreKey", pa.int64()),
        pa.field("SalesPersonEmployeeKey", pa.int64()),
        pa.field("PromotionKey", pa.int64()),
        pa.field("CurrencyKey", pa.int64()),
        pa.field("SalesChannelKey", pa.int16()),
        pa.field("TimeKey", pa.int16()),   # injected later; OUTPUT schema expects it
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
    delta_fields = [pa.field("Year", pa.int16()), pa.field("Month", pa.int16())]

    # GEN variants
    schema_no_order_gen = pa.schema(base_fields_gen)
    schema_with_order_gen = pa.schema(order_fields + base_fields_gen)
    schema_no_order_delta_gen = pa.schema(base_fields_gen + delta_fields)
    schema_with_order_delta_gen = pa.schema(order_fields + base_fields_gen + delta_fields)

    # OUT variants
    schema_no_order_out = pa.schema(base_fields_out)
    schema_with_order_out = pa.schema(order_fields + base_fields_out)
    schema_no_order_delta_out = pa.schema(base_fields_out + delta_fields)
    schema_with_order_delta_out = pa.schema(order_fields + base_fields_out + delta_fields)

    if is_delta:
        sales_schema_gen = schema_no_order_delta_gen if skip_order_cols else schema_with_order_delta_gen
        sales_schema_out = schema_no_order_delta_out if skip_order_cols_requested else schema_with_order_delta_out
        schema_no_order = schema_no_order_delta_out
        schema_with_order = schema_with_order_delta_out
        schema_no_order_delta = schema_no_order_delta_out
        schema_with_order_delta = schema_with_order_delta_out
    else:
        sales_schema_gen = schema_no_order_gen if skip_order_cols else schema_with_order_gen
        sales_schema_out = schema_no_order_out if skip_order_cols_requested else schema_with_order_out
        schema_no_order = schema_no_order_out
        schema_with_order = schema_with_order_out
        schema_no_order_delta = schema_no_order_delta_out
        schema_with_order_delta = schema_with_order_delta_out

    # ---------------------------------------------------------------------
    # SalesOrderDetail / SalesOrderHeader schemas (OUTPUT only)
    # ---------------------------------------------------------------------
    detail_fields = [
        pa.field("SalesOrderNumber", pa.int64()),
        pa.field("SalesOrderLineNumber", pa.int64()),
        pa.field("ProductKey", pa.int64()),
        pa.field("StoreKey", pa.int64()),
        pa.field("SalesPersonEmployeeKey", pa.int64()),  # line-level (multi-store orders)
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
    detail_schema = pa.schema(detail_fields + delta_fields) if is_delta else pa.schema(detail_fields)

    header_fields = [
        pa.field("SalesOrderNumber", pa.int64()),
        pa.field("CustomerKey", pa.int64()),
        pa.field("SalesChannelKey", pa.int16()),
        pa.field("OrderDate", pa.date32()),
        pa.field("TimeKey", pa.int16()),  # order-level time
        pa.field("IsOrderDelayed", pa.int8()),
    ]
    header_schema = pa.schema(header_fields + delta_fields) if is_delta else pa.schema(header_fields)

    schema_by_table: dict[str, pa.Schema] = {
        TABLE_SALES: sales_schema_out,
        TABLE_SALES_ORDER_DETAIL: detail_schema,
        TABLE_SALES_ORDER_HEADER: header_schema,
    }

    # ---------------------------------------------------------------------
    # Returns schema (thin)
    # ---------------------------------------------------------------------
    if returns_enabled:
        if TABLE_SALES_RETURN is None:
            raise RuntimeError("returns_enabled=True but TABLE_SALES_RETURN is not defined in output_paths.py")

        return_fields = [
            pa.field("SalesOrderNumber", pa.int64()),
            pa.field("SalesOrderLineNumber", pa.int64()),
            pa.field("ReturnDate", pa.date32()),
            pa.field("ReturnReasonKey", pa.int64()),
            pa.field("ReturnQuantity", pa.int64()),
            pa.field("ReturnNetPrice", pa.float64()),
        ]
        return_schema = pa.schema(return_fields + delta_fields) if is_delta else pa.schema(return_fields)
        schema_by_table[TABLE_SALES_RETURN] = return_schema

    # ---------------------------------------------------------------------
    # Date-column policy (must reference existing columns for each table)
    # ---------------------------------------------------------------------
    date_cols_by_table: dict[str, list[str]] = {
        TABLE_SALES: ["OrderDate", "DeliveryDate"],
        TABLE_SALES_ORDER_DETAIL: ["DueDate", "DeliveryDate"],  # FIX: no OrderDate in detail schema
        TABLE_SALES_ORDER_HEADER: ["OrderDate"],
    }
    if returns_enabled and TABLE_SALES_RETURN is not None:
        date_cols_by_table[TABLE_SALES_RETURN] = ["ReturnDate"]

    # Optional models.yaml override (preserve previous behavior)
    if models_cfg and isinstance(models_cfg, dict):
        models_root = models_cfg.get("models") if isinstance(models_cfg.get("models"), dict) else models_cfg
        overrides = None
        if isinstance(models_root, dict) and isinstance(models_root.get("returns"), dict):
            overrides = models_root["returns"].get("date_cols_by_table")
        if isinstance(overrides, dict):
            for k, v in overrides.items():
                if isinstance(k, str) and isinstance(v, (list, tuple)) and v:
                    date_cols_by_table[k] = [str(x) for x in v]

    # ---------------------------------------------------------------------
    # Parquet dictionary encoding policy
    # ---------------------------------------------------------------------
    pdx = set(parquet_dict_exclude) if parquet_dict_exclude else {"SalesOrderNumber", "CustomerKey"}
    parquet_dict_cols_by_table = {t: schema_dict_cols(s, pdx) for t, s in schema_by_table.items()}
    parquet_dict_cols = parquet_dict_cols_by_table[TABLE_SALES]

    return WorkerSchemaBundle(
        sales_schema_gen=sales_schema_gen,
        sales_schema_out=sales_schema_out,
        schema_by_table=schema_by_table,
        date_cols_by_table=date_cols_by_table,
        parquet_dict_exclude=pdx,
        parquet_dict_cols_by_table=parquet_dict_cols_by_table,
        parquet_dict_cols=parquet_dict_cols,
        schema_no_order=schema_no_order,
        schema_with_order=schema_with_order,
        schema_no_order_delta=schema_no_order_delta,
        schema_with_order_delta=schema_with_order_delta,
    )