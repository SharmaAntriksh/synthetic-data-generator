# sales_logic.py — optimized (Arrow-first, vectorized, memory-conscious)
# Compatible replacement for your existing logic. Returns pyarrow.Table when possible.
import numpy as np
import pandas as pd
import pyarrow as pa

# ============================================================
# GLOBAL WORKER STATE (injected via bind_globals)
# ============================================================
_G_skip_order_cols = None
_G_product_np = None
_G_customers = None
_G_date_pool = None
_G_date_prob = None
_G_store_keys = None
_G_promo_keys_all = None
_G_promo_pct_all = None
_G_promo_start_all = None
_G_promo_end_all = None
_G_store_to_geo_arr = None
_G_geo_to_currency_arr = None
_G_store_to_geo = None
_G_geo_to_currency = None


def bind_globals(gdict):
    """Multiprocessing worker initializer inserts globals."""
    globals().update(gdict)


# ============================================================
# CHUNK BUILDER
# ============================================================
def _build_chunk_table(n, seed, no_discount_key=1):
    """
    Build n synthetic sales rows.
    Returns a pyarrow.Table when pyarrow is available.
    """
    rng = np.random.default_rng(seed)
    skip_cols = _G_skip_order_cols

    # ---- references ----
    product_np = _G_product_np
    customers = _G_customers
    date_pool = _G_date_pool
    date_prob = _G_date_prob
    store_keys = _G_store_keys
    promo_keys_all = _G_promo_keys_all
    promo_pct_all = _G_promo_pct_all
    promo_start_all = _G_promo_start_all
    promo_end_all = _G_promo_end_all

    st2g_arr = _G_store_to_geo_arr
    g2c_arr = _G_geo_to_currency_arr
    store_to_geo = _G_store_to_geo
    geo_to_currency = _G_geo_to_currency

    # ---------------------------------------------------------
    # PRODUCTS
    # ---------------------------------------------------------
    prod_idx = rng.integers(0, len(product_np), size=n)
    prods = product_np[prod_idx]  # shape (n, cols)

    # ensure numeric dtypes (avoid Python objects)
    product_keys = prods[:, 0].astype(np.int64)
    unit_price = prods[:, 1].astype(np.float64)
    unit_cost = prods[:, 2].astype(np.float64)

    # ---------------------------------------------------------
    # STORE → GEO → CURRENCY (vectorized fast path)
    # ---------------------------------------------------------
    store_key_arr = store_keys[rng.integers(0, len(store_keys), size=n)].astype(np.int64)
    if st2g_arr is not None and g2c_arr is not None:
        geo_arr = st2g_arr[store_key_arr]
        currency_arr = g2c_arr[geo_arr].astype(np.int64)
    else:
        # fallback to mapping dicts (avoid list comprehensions in hot path where possible)
        geo_arr = np.fromiter((store_to_geo[s] for s in store_key_arr), dtype=np.int64, count=n)
        currency_arr = np.fromiter((geo_to_currency[g] for g in geo_arr), dtype=np.int64, count=n)

    # ---------------------------------------------------------
    # QUANTITY
    # ---------------------------------------------------------
    qty = np.clip(rng.poisson(3, n) + 1, 1, 10).astype(np.int64)

    # ---------------------------------------------------------
    # ORDER GROUPING (vectorized generation of order-level data)
    # ---------------------------------------------------------
    avg_lines = 2.0
    order_count = max(1, int(n / avg_lines))

    # suffix & order dates (vectorized)
    suffix = np.char.zfill(rng.integers(0, 999999, order_count).astype(str), 6)
    od_idx = rng.choice(len(date_pool), size=order_count, p=date_prob)
    order_dates = date_pool[od_idx]  # numpy datetime64 array

    # vectorized date string without Python loop
    date_str = np.datetime_as_string(order_dates, unit='D')  # 'YYYY-MM-DD'
    date_str = np.char.replace(date_str, "-", "")  # 'YYYYMMDD'
    order_ids_str = np.char.add(date_str, suffix)  # numpy char array
    # integer ID (use for grouping to avoid string-heavy operations)
    order_ids_int = order_ids_str.astype(np.int64)

    # customers
    cust_idx = rng.integers(0, len(customers), order_count)
    order_customers = customers[cust_idx].astype(np.int64)

    # lines per order
    lines_per_order = rng.choice([1, 2, 3, 4, 5], order_count, p=[0.55, 0.25, 0.10, 0.06, 0.04])
    expanded_len = lines_per_order.sum()
    order_idx = np.repeat(np.arange(order_count), lines_per_order)

    starts = np.repeat(np.cumsum(lines_per_order) - lines_per_order, lines_per_order)
    line_num = (np.arange(expanded_len) - starts + 1).astype(np.int64)

    # expand order-level arrays to line-level
    sales_order_num = np.repeat(order_ids_str, lines_per_order)           # strings (numpy char array)
    sales_order_num_int = np.repeat(order_ids_int, lines_per_order)      # integer IDs for grouping
    customer_keys = np.repeat(order_customers, lines_per_order).astype(np.int64)
    order_dates_expanded = np.repeat(order_dates, lines_per_order)

    # pad if needed to reach exactly n rows
    curr_len = len(sales_order_num)
    if curr_len < n:
        extra = n - curr_len
        ext_suf = np.char.zfill(rng.integers(0, 999999, extra).astype(str), 6)
        ext_dates = date_pool[rng.choice(len(date_pool), size=extra, p=date_prob)]

        ext_dt_str = np.datetime_as_string(ext_dates, unit='D')
        ext_dt_str = np.char.replace(ext_dt_str, "-", "")
        ext_ids_str = np.char.add(ext_dt_str, ext_suf)
        ext_ids_int = ext_ids_str.astype(np.int64)

        sales_order_num = np.concatenate([sales_order_num, ext_ids_str])
        sales_order_num_int = np.concatenate([sales_order_num_int, ext_ids_int])
        line_num = np.concatenate([line_num, np.ones(extra, dtype=np.int64)])
        customer_keys = np.concatenate([customer_keys, customers[rng.integers(0, len(customers), extra)]])
        order_dates_expanded = np.concatenate([order_dates_expanded, ext_dates])

    # trim/truncate exactly to n rows
    sales_order_num = sales_order_num[:n]
    sales_order_num_int = sales_order_num_int[:n].astype(np.int64)
    line_num = line_num[:n].astype(np.int64)
    customer_keys = customer_keys[:n].astype(np.int64)
    order_dates_expanded = order_dates_expanded[:n]

    od_np = order_dates_expanded.astype("datetime64[D]")

    # ---------------------------------------------------------
    # DELIVERY / DUE DATE LOGIC (vectorized)
    # ---------------------------------------------------------
    hash_vals = sales_order_num_int  # integer grouping key

    due_offset = (hash_vals % 5).astype(np.int64) + 3
    due_date_np = od_np + due_offset.astype("timedelta64[D]")

    line_seed = (product_keys + (hash_vals % 100)) % 100
    product_seed = (hash_vals + product_keys) % 100
    order_seed = (hash_vals % 100).astype(np.int64)

    base_offset = np.zeros(n, dtype=np.int64)
    mask_c = (60 <= order_seed) & (order_seed < 85) & (product_seed >= 60)
    if mask_c.any():
        base_offset[mask_c] = (line_seed[mask_c] % 4) + 1
    mask_d = order_seed >= 85
    if mask_d.any():
        base_offset[mask_d] = (product_seed[mask_d] % 5) + 2

    # early deliveries
    early_mask = rng.random(n) < 0.10
    early_days = rng.integers(1, 3, n)
    delivery_offset = base_offset.copy()
    delivery_offset[early_mask] = -early_days[early_mask]
    delivery_date_np = due_date_np + delivery_offset.astype("timedelta64[D]")

    # delivery_status as fixed-length numpy unicode array (avoids python object arrays)
    delivery_status = np.where(
        delivery_date_np < due_date_np, "Early Delivery",
        np.where(delivery_date_np > due_date_np, "Delayed", "On Time")
    ).astype(f'U15')  # small fixed-width unicode dtype

    # ---------------------------------------------------------
    # PROMOTIONS (keeps loop — cheap if promo list small)
    # ---------------------------------------------------------
    promo_keys = np.full(n, no_discount_key, dtype=np.int64)
    promo_pct = np.zeros(n, dtype=np.float64)

    if promo_keys_all is not None and promo_keys_all.size > 0:
        # promo_start_all / promo_end_all are expected to be numpy datetimes
        for pk, pct, start, end in zip(promo_keys_all, promo_pct_all, promo_start_all, promo_end_all):
            # mask is boolean vector over order dates (od_np)
            mask = (od_np >= start) & (od_np <= end)
            if mask.any():
                promo_keys[mask] = pk
                promo_pct[mask] = pct

    # ---------------------------------------------------------
    # DISCOUNTS
    # ---------------------------------------------------------
    promo_disc = unit_price * (promo_pct / 100.0)

    rnd_pct = rng.choice([0, 5, 10, 15, 20], n, p=[0.85, 0.06, 0.04, 0.03, 0.02])
    rnd_disc = unit_price * (rnd_pct / 100.0)

    discount_amt = np.maximum(promo_disc, rnd_disc)
    discount_amt *= rng.choice([0.90, 0.95, 1.00, 1.05, 1.10], n)
    # round to quarters (0.25)
    discount_amt = np.round(discount_amt * 4) / 4
    discount_amt = np.minimum(discount_amt, unit_price - 0.01)

    # ---------------------------------------------------------
    # ORDER DELAY FLAG (order-level → line-level) using int grouping
    # ---------------------------------------------------------
    delayed_line = (delivery_status == "Delayed").astype(np.int64)
    unique_ids, inv_idx = np.unique(sales_order_num_int, return_inverse=True)
    counts = np.bincount(inv_idx, weights=delayed_line, minlength=len(unique_ids))
    delayed_any = (counts > 0).astype(np.int8)
    is_order_delayed = delayed_any[inv_idx].astype(np.int8)

    # ---------------------------------------------------------
    # FINAL PRICE & COST TRANSFORM
    # ---------------------------------------------------------
    factor = rng.uniform(0.43, 0.61, size=n)
    final_unit_price = np.round(unit_price * factor, 2)
    final_unit_cost = np.round(unit_cost * factor, 2)
    final_discount_amt = np.round(discount_amt * factor, 2)
    final_net_price = np.round(final_unit_price - final_discount_amt, 2)
    final_net_price = np.clip(final_net_price, 0.01, None)

    # ---------------------------------------------------------
    # OUTPUT: PYARROW OR PANDAS (Arrow-first, minimizing copies)
    # ---------------------------------------------------------
    if pa is not None:
        cols = {
            "OrderDate": pa.array(od_np),
            "DueDate": pa.array(due_date_np.astype("datetime64[D]")),
            "DeliveryDate": pa.array(delivery_date_np.astype("datetime64[D]")),
            "StoreKey": pa.array(store_key_arr, pa.int64()),
            "ProductKey": pa.array(product_keys, pa.int64()),
            "PromotionKey": pa.array(promo_keys, pa.int64()),
            "CurrencyKey": pa.array(currency_arr, pa.int64()),
            "CustomerKey": pa.array(customer_keys, pa.int64()),
            "Quantity": pa.array(qty, pa.int64()),
            "NetPrice": pa.array(final_net_price, pa.float64()),
            "UnitCost": pa.array(final_unit_cost, pa.float64()),
            "UnitPrice": pa.array(final_unit_price, pa.float64()),
            "DiscountAmount": pa.array(final_discount_amt, pa.float64()),
            # delivery_status is a numpy unicode (fixed-width) — pass directly
            "DeliveryStatus": pa.array(delivery_status, pa.string()),
            "IsOrderDelayed": pa.array(is_order_delayed, pa.int8()),
        }

        if not skip_cols:
            cols["SalesOrderNumber"] = pa.array(sales_order_num, pa.string())
            cols["SalesOrderLineNumber"] = pa.array(line_num, pa.int64())

        return pa.table(cols)

    else:
        df = {
            "OrderDate": od_np,
            "DueDate": due_date_np.astype("datetime64[D]"),
            "DeliveryDate": delivery_date_np.astype("datetime64[D]"),
            "StoreKey": store_key_arr,
            "ProductKey": product_keys,
            "PromotionKey": promo_keys,
            "CurrencyKey": currency_arr,
            "CustomerKey": customer_keys,
            "Quantity": qty,
            "NetPrice": final_net_price,
            "UnitCost": final_unit_cost,
            "UnitPrice": final_unit_price,
            "DiscountAmount": final_discount_amt,
            "DeliveryStatus": delivery_status,
            "IsOrderDelayed": is_order_delayed,
        }

        if not skip_cols:
            df["SalesOrderNumber"] = sales_order_num.astype(str)
            df["SalesOrderLineNumber"] = line_num

        return pd.DataFrame(df)
