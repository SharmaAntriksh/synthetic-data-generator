import numpy as np
from .globals import _fmt

def build_orders(
    rng, n, skip_cols,
    date_pool, date_prob,
    customers, product_keys,
    _len_date_pool, _len_customers
):
    avg_lines = 2.0
    order_count = max(1, int(n / avg_lines))

    # Generate order suffixes + order dates
    suffix_int = rng.integers(0, 999999, order_count, dtype=np.int64)
    od_idx = rng.choice(len(date_pool), size=order_count, p=date_prob)
    order_dates = date_pool[od_idx]

    # Convert order dates to YYYYMMDD integer
    date_str = np.datetime_as_string(order_dates, unit="D")
    date_str = np.char.replace(date_str, "-", "")
    date_int = date_str.astype(np.int64)

    # Build integer order ids
    order_ids_int = date_int * 1_000_000 + suffix_int

    # String IDs only when needed
    if skip_cols:
        order_ids_str = None
    else:
        suf = np.char.zfill(suffix_int.astype(str), 6)
        order_ids_str = np.char.add(date_str, suf)

    # Assign customers to orders
    cust_idx = rng.integers(0, len(customers), order_count)
    order_customers = customers[cust_idx].astype(np.int64)

    # Lines per order
    lines_per_order = rng.choice(
        [1, 2, 3, 4, 5],
        order_count,
        p=[0.55, 0.25, 0.10, 0.06, 0.04]
    )
    expanded_len = lines_per_order.sum()

    # Expand arrays
    order_idx = np.repeat(np.arange(order_count), lines_per_order)

    starts = np.repeat(np.cumsum(lines_per_order) - lines_per_order, lines_per_order)
    line_num = (np.arange(expanded_len) - starts + 1).astype(np.int64)

    sales_order_num_int = np.repeat(order_ids_int, lines_per_order)
    sales_order_num = (
        None if order_ids_str is None else np.repeat(order_ids_str, lines_per_order)
    )

    customer_keys = np.repeat(order_customers, lines_per_order).astype(np.int64)
    order_dates_expanded = np.repeat(order_dates, lines_per_order)

    # Pad if needed
    curr_len = len(sales_order_num_int)
    if curr_len < n:
        extra = n - curr_len

        ext_dates = date_pool[rng.choice(_len_date_pool, size=extra, p=date_prob)]
        ext_suffix_int = rng.integers(0, 999999, extra, dtype=np.int64)

        ext_dt_str = _fmt(ext_dates)
        ext_dt_int = ext_dt_str.astype(np.int64)
        ext_ids_int = ext_dt_int * 1_000_000 + ext_suffix_int

        if sales_order_num is not None:
            ext_suf = np.char.zfill(ext_suffix_int.astype(str), 6)
            ext_ids_str = np.char.add(ext_dt_str, ext_suf)
            sales_order_num = np.concatenate([sales_order_num, ext_ids_str])

        sales_order_num_int = np.concatenate([sales_order_num_int, ext_ids_int])
        line_num = np.concatenate([line_num, np.ones(extra, dtype=np.int64)])
        customer_keys = np.concatenate([
            customer_keys,
            customers[rng.integers(0, _len_customers, extra)]
        ])
        order_dates_expanded = np.concatenate([order_dates_expanded, ext_dates])

    # Trim to exactly n rows
    sales_order_num_int = sales_order_num_int[:n]
    if sales_order_num is not None:
        sales_order_num = sales_order_num[:n]
    line_num = line_num[:n]
    customer_keys = customer_keys[:n]
    order_dates_expanded = order_dates_expanded[:n]

    return {
        "order_ids_int": sales_order_num_int,
        "order_ids_str": sales_order_num,
        "line_num": line_num,
        "customer_keys": customer_keys,
        "order_dates": order_dates_expanded.astype("datetime64[D]")
    }
