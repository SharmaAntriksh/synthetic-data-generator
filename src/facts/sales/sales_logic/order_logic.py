import numpy as np


def build_orders(
    rng, n, skip_cols,
    date_pool, date_prob,
    customers, product_keys,
    _len_date_pool, _len_customers
):
    avg_lines = 2.0
    order_count = max(1, int(n / avg_lines))

    # ------------------------------------------------------------
    # Generate order-level data
    # ------------------------------------------------------------

    # Order dates
    od_idx = rng.choice(len(date_pool), size=order_count, p=date_prob)
    order_dates = date_pool[od_idx]

    # YYYYMMDD → int
    date_str = np.datetime_as_string(order_dates, unit="D")
    date_str = np.char.replace(date_str, "-", "")
    date_int = date_str.astype(np.int64)

    # Per-order random suffix (per order, not per chunk)
    suffix_int = rng.integers(0, 1_000_000_000, order_count, dtype=np.int64)

    # ✅ Numeric, per-order, globally unique SalesOrderNumber
    order_ids_int = date_int * 1_000_000_000 + suffix_int

    # One customer per order
    cust_idx = rng.integers(0, len(customers), order_count)
    order_customers = customers[cust_idx].astype(np.int64)

    # ------------------------------------------------------------
    # Lines per order
    # ------------------------------------------------------------

    lines_per_order = rng.choice(
        [1, 2, 3, 4, 5],
        order_count,
        p=[0.55, 0.25, 0.10, 0.06, 0.04]
    )

    expanded_len = lines_per_order.sum()

    starts = np.repeat(
        np.cumsum(lines_per_order) - lines_per_order,
        lines_per_order
    )
    line_num = (np.arange(expanded_len) - starts + 1).astype(np.int64)

    sales_order_num_int = np.repeat(order_ids_int, lines_per_order)
    customer_keys = np.repeat(order_customers, lines_per_order).astype(np.int64)
    order_dates_expanded = np.repeat(order_dates, lines_per_order)

    # ------------------------------------------------------------
    # Pad if needed (REPEAT existing rows, never invent orders)
    # ------------------------------------------------------------
    curr_len = len(sales_order_num_int)
    if curr_len < n:
        extra = n - curr_len

        sales_order_num_int = np.concatenate([
            sales_order_num_int,
            sales_order_num_int[:extra]
        ])
        line_num = np.concatenate([
            line_num,
            line_num[:extra]
        ])
        customer_keys = np.concatenate([
            customer_keys,
            customer_keys[:extra]
        ])
        order_dates_expanded = np.concatenate([
            order_dates_expanded,
            order_dates_expanded[:extra]
        ])

    # ------------------------------------------------------------
    # Trim to exactly n rows
    # ------------------------------------------------------------

    sales_order_num_int = sales_order_num_int[:n]
    line_num = line_num[:n]
    customer_keys = customer_keys[:n]
    order_dates_expanded = order_dates_expanded[:n]

    # ------------------------------------------------------------
    # Output (skip_cols controls schema ONLY)
    # ------------------------------------------------------------

    result = {
        "customer_keys": customer_keys,
        "order_dates": order_dates_expanded.astype("datetime64[D]")
    }

    if skip_cols:
        result["order_ids_str"] = np.full(n, "", dtype=object)
    else:
        result["order_ids_int"] = sales_order_num_int
        result["line_num"] = line_num
        # compatibility for remaining pipeline
        result["order_ids_str"] = sales_order_num_int.astype(str)

    return result
