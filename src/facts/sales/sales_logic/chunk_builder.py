import numpy as np
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
    # STORE → GEO → CURRENCY (FAST, NO PYTHON FALLBACKS)
    # ------------------------------------------------------------
    # Generate StoreKey array
    store_key_arr: np.ndarray = store_keys[
        rng.integers(0, _len_store_keys, size=n)
    ].astype(np.int64, copy=False)

    # Dense mapping arrays MUST exist (built in init_sales_worker)
    if st2g_arr is None or g2c_arr is None:
        raise RuntimeError(
            "Dense store_to_geo_arr / geo_to_currency_arr not initialized. "
            "Check init_sales_worker."
        )

    # Bounds safety check (cheap, prevents silent corruption)
    max_store_key = int(store_key_arr.max()) if store_key_arr.size else -1
    if max_store_key >= st2g_arr.shape[0]:
        raise RuntimeError(
            f"StoreKey {max_store_key} exceeds store_to_geo_arr size "
            f"{st2g_arr.shape[0]}"
        )

    # Vectorized lookup (pure NumPy, fast)
    geo_arr = st2g_arr[store_key_arr]
    currency_arr = g2c_arr[geo_arr].astype(np.int64, copy=False)


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

    customer_keys = orders["customer_keys"]
    order_dates = orders["order_dates"]

    if not skip_cols:
        order_ids_int = orders["order_ids_int"]
        line_num = orders["line_num"]
    else:
        order_ids_int = None
        line_num = None

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
    # OUTPUT — Arrow ONLY (pre-sized, no Python dict)
    # ------------------------------------------------------------
    if not PA_AVAILABLE:
        raise RuntimeError("pyarrow is required")

    arrays = []
    fields = []

    def _add(name, arr, dtype):
        arrays.append(pa.array(arr, type=dtype, safe=False))
        fields.append(pa.field(name, dtype))

    if not skip_cols:
        _add("SalesOrderNumber", order_ids_int, pa.int64())
        _add("SalesOrderLineNumber", line_num, pa.int64())

    _add("CustomerKey", customer_keys, pa.int64())
    _add("ProductKey", product_keys, pa.int64())
    _add("StoreKey", store_key_arr, pa.int64())
    _add("PromotionKey", promo_keys, pa.int64())
    _add("CurrencyKey", currency_arr, pa.int64())

    _add("OrderDate", order_dates, pa.date32())
    _add("DueDate", due_date, pa.date32())
    _add("DeliveryDate", delivery_date, pa.date32())

    _add("Quantity", qty, pa.int64())
    _add("NetPrice", final_net_price, pa.float64())
    _add("UnitCost", final_unit_cost, pa.float64())
    _add("UnitPrice", final_unit_price, pa.float64())
    _add("DiscountAmount", final_discount_amt, pa.float64())

    _add("DeliveryStatus", delivery_status, pa.string())
    _add("IsOrderDelayed", is_order_delayed, pa.int8())

    months = order_dates.astype("datetime64[M]").astype("int64")
    year_arr = (months // 12 + 1970).astype("int16")
    month_arr = (months % 12 + 1).astype("int8")

    if file_format == "deltaparquet":
        months = order_dates.astype("datetime64[M]").astype("int64")
        year_arr = (months // 12 + 1970).astype("int16")
        month_arr = (months % 12 + 1).astype("int8")
        _add("Year", year_arr, pa.int16())
        _add("Month", month_arr, pa.int8())


    schema = pa.schema(fields)
    return pa.Table.from_arrays(arrays, schema=schema)
