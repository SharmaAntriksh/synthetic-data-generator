from __future__ import annotations

import pyarrow as pa


def build_header_from_detail(detail: pa.Table) -> pa.Table:
    """
    Build SalesOrderHeader (ORDER-GRAIN) from SalesOrderDetail.

    Output columns:
      - SalesOrderNumber
      - CustomerKey
      - OrderDate
      - IsOrderDelayed   (1 if any line is delayed)

    NOTE:
    StoreKey/PromotionKey/CurrencyKey/DueDate can vary per line, so they are not
    included in the header.
    """
    gb = detail.group_by(["SalesOrderNumber"])

    out = gb.aggregate(
        [
            ("CustomerKey", "min"),
            ("OrderDate", "min"),
            ("IsOrderDelayed", "max"),
        ]
    )

    rename_map = {
        "CustomerKey_min": "CustomerKey",
        "OrderDate_min": "OrderDate",
        "IsOrderDelayed_max": "IsOrderDelayed",
    }

    cols = []
    names = []
    for name in out.schema.names:
        cols.append(out[name])
        names.append(rename_map.get(name, name))

    return pa.Table.from_arrays(cols, names=names)
