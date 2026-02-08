from __future__ import annotations

import pandas as pd
import pyarrow as pa


def build_header_from_detail(detail: pa.Table) -> pa.Table:
    """
    Build SalesOrderHeader from SalesOrderDetail (detail).
    Assumes detail includes SalesOrderNumber + SalesOrderLineNumber.
    """
    df = detail.to_pandas(types_mapper=None)  # keep default dtypes

    # Compute line-level amounts (adjust names if your columns differ)
    df["GrossAmount"] = df["UnitPrice"] * df["Quantity"]
    df["NetAmount"] = df["NetPrice"] * df["Quantity"]
    df["TotalCost"] = df["UnitCost"] * df["Quantity"]

    grp = df.groupby("SalesOrderNumber", sort=False, dropna=False)

    header = grp.agg(
        CustomerKey=("CustomerKey", "first"),
        StoreKey=("StoreKey", "first"),
        PromotionKey=("PromotionKey", "first"),
        CurrencyKey=("CurrencyKey", "first"),
        OrderDate=("OrderDate", "first"),
        DueDate=("DueDate", "first"),
        DeliveryDate=("DeliveryDate", "first"),

        LineCount=("SalesOrderLineNumber", "count"),
        TotalQuantity=("Quantity", "sum"),
        GrossAmount=("GrossAmount", "sum"),
        NetAmount=("NetAmount", "sum"),
        TotalCost=("TotalCost", "sum"),
        TotalDiscount=("DiscountAmount", "sum"),

        IsOrderDelayed=("IsOrderDelayed", "max"),
    ).reset_index()

    # Convert back to Arrow
    return pa.Table.from_pandas(header, preserve_index=False)
