from __future__ import annotations

import pyarrow as pa


def build_header_from_detail(detail: pa.Table) -> pa.Table:
    """
    Build *slim* SalesOrderHeader from SalesOrderDetail (detail).

    IMPORTANT:
    - Do NOT use ordered aggregators like "first" in PyArrow group_by aggregate.
      They are not supported with multi-thread execution and will throw:
      "Using ordered aggregator in multiple threaded execution is not supported"

    This uses unordered aggregators ("min"/"max") which are thread-safe.
    Assumption: CustomerKey/StoreKey/PromotionKey/CurrencyKey/OrderDate/DueDate
    are constant per SalesOrderNumber.
    """
    gb = detail.group_by(["SalesOrderNumber"])

    out = gb.aggregate(
        [
            ("CustomerKey", "min"),
            ("StoreKey", "min"),
            ("PromotionKey", "min"),
            ("CurrencyKey", "min"),
            ("OrderDate", "min"),
            ("DueDate", "min"),
            ("IsOrderDelayed", "max"),
        ]
    )

    # group_by() returns names like "CustomerKey_min"
    rename_map = {
        "CustomerKey_min": "CustomerKey",
        "StoreKey_min": "StoreKey",
        "PromotionKey_min": "PromotionKey",
        "CurrencyKey_min": "CurrencyKey",
        "OrderDate_min": "OrderDate",
        "DueDate_min": "DueDate",
        "IsOrderDelayed_max": "IsOrderDelayed",
    }

    cols = []
    names = []
    for name in out.schema.names:
        cols.append(out[name])
        names.append(rename_map.get(name, name))

    return pa.Table.from_arrays(cols, names=names)
