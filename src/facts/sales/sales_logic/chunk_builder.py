import numpy as np
import pandas as pd
import pyarrow as pa

from .globals import State, PA_AVAILABLE
from .order_logic import build_orders
from .date_logic import compute_dates
from .promo_logic import apply_promotions
from .price_logic import compute_prices


def build_chunk_table(n, seed, no_discount_key=1):
    """
    Build n synthetic sales rows.
    All shared state is read from `State`.
    """
    rng = np.random.default_rng(seed)

    # pull from State instead of _G_
    skip_cols = State.skip_order_cols
    product_np = State.product_np
    customers = State.customers
    date_pool = State.date_pool
    date_prob = State.date_prob
    store_keys = State.store_keys

    promo_keys_all = State.promo_keys_all
    promo_pct_all = State.promo_pct_all
    promo_start_all = State.promo_start_all
    promo_end_all = State.promo_end_all

    st2g_arr = State.store_to_geo_arr
    g2c_arr = State.geo_to_currency_arr
    store_to_geo = State.store_to_geo
    geo_to_currency = State.geo_to_currency

    file_format = State.file_format

    # Validate required globals
    if date_pool is None:
        raise RuntimeError("State.date_pool is None")
    if product_np is None:
        raise RuntimeError("State.product_np is None")
    if store_keys is None:
        raise RuntimeError("State.store_keys is None")

    _len_date_pool = len(date_pool)
    _len_customers = len(customers)
    _len_store_keys = len(store_keys)
    _len_products = len(product_np)

    # ------------------------------------------------------------
    # PRODUCTS
    # ------------------------------------------------------------
    prod_idx = rng.integers(0, _len_products, size=n)
    prods = product_np[prod_idx]

    product_keys = prods[:, 0]
    unit_price = prods[:, 1].astype(np.float64, copy=False)
    unit_cost = prods[:, 2].astype(np.float64, copy=False)

    # ------------------------------------------------------------
    # STORE → GEO → CURRENCY
    # ------------------------------------------------------------
    store_key_arr = store_keys[rng.integers(0, _len_store_keys, size=n)].astype(np.int64)

    if st2g_arr is not None and g2c_arr is not None:
        try:
            max_key = int(store_key_arr.max()) if store_key_arr.size else -1
            if (
                st2g_arr.ndim == 1 and g2c_arr.ndim == 1
                and max_key < st2g_arr.shape[0]
            ):
                geo_arr = st2g_arr[store_key_arr]
                currency_arr = g2c_arr[geo_arr].astype(np.int64, copy=False)
            else:
                raise IndexError
        except Exception:
            geo_arr = np.fromiter((store_to_geo.get(int(s), 0) for s in store_key_arr),
                                   dtype=np.int64, count=n)
            currency_arr = np.fromiter((geo_to_currency.get(int(g), 0) for g in geo_arr),
                                       dtype=np.int64, count=n)
    else:
        geo_arr = np.fromiter((store_to_geo.get(int(s), 0) for s in store_key_arr),
                              dtype=np.int64, count=n)
        currency_arr = np.fromiter((geo_to_currency.get(int(g), 0) for g in geo_arr),
                                   dtype=np.int64, count=n)

    # ------------------------------------------------------------
    # ORDERS
    # ------------------------------------------------------------
    orders = build_orders(
        rng=rng,
        n=n,
        skip_cols=skip_cols,
        date_pool=date_pool,
        date_prob=date_prob,
        customers=customers,
        product_keys=product_keys,
        _len_date_pool=_len_date_pool,
        _len_customers=_len_customers,
    )

    order_ids_int = orders["order_ids_int"]
    order_ids_str = orders["order_ids_str"]
    line_num = orders["line_num"]
    customer_keys = orders["customer_keys"]
    order_dates = orders["order_dates"]
    order_dates[0] = date_pool[0]
    order_dates[-1] = date_pool[-1]
    
    qty = np.clip(rng.poisson(3, n) + 1, 1, 4)

    # ------------------------------------------------------------
    # DATE LOGIC
    # ------------------------------------------------------------
    dates = compute_dates(
        rng=rng,
        n=n,
        product_keys=product_keys,
        order_ids_int=order_ids_int,
        order_dates=order_dates,
    )

    due_date = dates["due_date"]
    delivery_date = dates["delivery_date"]
    delivery_status = dates["delivery_status"]
    is_order_delayed = dates["is_order_delayed"]

    # ------------------------------------------------------------
    # PROMOTIONS
    # ------------------------------------------------------------
    promo_keys, promo_pct = apply_promotions(
        rng=rng,
        n=n,
        order_dates=order_dates,
        promo_keys_all=promo_keys_all,
        promo_pct_all=promo_pct_all,
        promo_start_all=promo_start_all,
        promo_end_all=promo_end_all,
        no_discount_key=no_discount_key,
    )

    # ------------------------------------------------------------
    # PRICE LOGIC
    # ------------------------------------------------------------
    price = compute_prices(
        rng=rng,
        n=n,
        unit_price=unit_price,
        unit_cost=unit_cost,
        promo_pct=promo_pct,
    )

    final_unit_price = price["final_unit_price"]
    final_unit_cost = price["final_unit_cost"]
    final_discount_amt = price["discount_amt"]
    final_net_price = price["final_net_price"]

    # ------------------------------------------------------------
    # OUTPUT — Arrow first
    # ------------------------------------------------------------
    if PA_AVAILABLE:
        cols = {}

        if not skip_cols:
            cols["SalesOrderNumber"] = pa.array(order_ids_str, pa.string())
            cols["SalesOrderLineNumber"] = pa.array(line_num, pa.int64())

        cols["CustomerKey"] = pa.array(customer_keys, pa.int64())
        cols["ProductKey"] = pa.array(product_keys, pa.int64())
        cols["StoreKey"] = pa.array(store_key_arr, pa.int64())
        cols["PromotionKey"] = pa.array(promo_keys, pa.int64())
        cols["CurrencyKey"] = pa.array(currency_arr, pa.int64())

        cols["OrderDate"] = pa.array(order_dates)
        cols["DueDate"] = pa.array(due_date)
        cols["DeliveryDate"] = pa.array(delivery_date)

        cols["Quantity"] = pa.array(qty, pa.int64())
        cols["NetPrice"] = pa.array(final_net_price, pa.float64())
        cols["UnitCost"] = pa.array(final_unit_cost, pa.float64())
        cols["UnitPrice"] = pa.array(final_unit_price, pa.float64())
        cols["DiscountAmount"] = pa.array(final_discount_amt, pa.float64())

        cols["DeliveryStatus"] = pa.array(delivery_status, pa.string())
        cols["IsOrderDelayed"] = pa.array(is_order_delayed, pa.int8())

        months = order_dates.astype("datetime64[M]").astype("int64")
        year_arr = (months // 12 + 1970).astype("int16")
        month_arr = (months % 12 + 1).astype("int8")

        if file_format == "deltaparquet":
            cols["Year"] = pa.array(year_arr, pa.int16())
            cols["Month"] = pa.array(month_arr, pa.int8())

        return pa.table(cols)

    # ------------------------------------------------------------
    # Pandas fallback
    # ------------------------------------------------------------
    df = {}

    if not skip_cols:
        df["SalesOrderNumber"] = order_ids_str.astype(str)
        df["SalesOrderLineNumber"] = line_num

    df["CustomerKey"] = customer_keys
    df["ProductKey"] = product_keys
    df["StoreKey"] = store_key_arr
    df["PromotionKey"] = promo_keys
    df["CurrencyKey"] = currency_arr

    df["OrderDate"] = order_dates
    df["DueDate"] = due_date
    df["DeliveryDate"] = delivery_date

    df["Quantity"] = qty
    df["NetPrice"] = final_net_price
    df["UnitCost"] = final_unit_cost
    df["UnitPrice"] = final_unit_price
    df["DiscountAmount"] = final_discount_amt

    df["DeliveryStatus"] = delivery_status
    df["IsOrderDelayed"] = is_order_delayed

    months = order_dates.astype("datetime64[M]").astype("int64")
    year_np = (months // 12 + 1970).astype("int16")
    month_np = (months % 12 + 1).astype("int8")

    if file_format == "deltaparquet":
        df["Year"] = year_np
        df["Month"] = month_np

    return pd.DataFrame(df)
