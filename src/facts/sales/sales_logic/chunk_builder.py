import numpy as np
import pyarrow as pa

from .globals import State, PA_AVAILABLE
from .order_logic import build_orders
from .date_logic import compute_dates
from .promo_logic import apply_promotions
from .price_logic import compute_prices


def build_chunk_table(n: int, seed: int, no_discount_key: int = 1) -> pa.Table:
    """
    Build `n` synthetic sales rows.
    All shared, immutable state is read from `State`.
    """

    if not PA_AVAILABLE:
        raise RuntimeError("pyarrow is required")

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------
    # Pull immutable state ONCE (important for multiprocessing)
    # ------------------------------------------------------------
    skip_cols = State.skip_order_cols
    if skip_cols not in (True, False):
        raise RuntimeError("State.skip_order_cols must be a boolean")

    product_np = (
        State.active_product_np
        if hasattr(State, "active_product_np") and State.active_product_np is not None
        else State.product_np
    )

    customers = (
        State.active_customer_keys
        if hasattr(State, "active_customer_keys") and State.active_customer_keys is not None
        else State.customers
    )

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
    schema = State.sales_schema

    # ------------------------------------------------------------
    # Validation (fail fast)
    # ------------------------------------------------------------
    if date_pool is None:
        raise RuntimeError("State.date_pool is None")
    if product_np is None:
        raise RuntimeError("State.product_np is None")
    if store_keys is None:
        raise RuntimeError("State.store_keys is None")
    if st2g_arr is None or g2c_arr is None:
        raise RuntimeError(
            "Dense store_to_geo_arr / geo_to_currency_arr not initialized"
        )

    # Cache schema types once (big win)
    schema_types = {f.name: f.type for f in schema}

    # ------------------------------------------------------------
    # PRODUCTS
    # ------------------------------------------------------------
    prod_idx = rng.integers(0, len(product_np), size=n)
    prods = product_np[prod_idx]

    product_keys = prods[:, 0]
    unit_price = prods[:, 1].astype(np.float64, copy=False)
    unit_cost = prods[:, 2].astype(np.float64, copy=False)

    # ------------------------------------------------------------
    # STORE → GEO → CURRENCY
    # ------------------------------------------------------------
    store_key_arr = store_keys[
        rng.integers(0, len(store_keys), size=n)
    ]

    if store_key_arr.dtype != np.int64:
        store_key_arr = store_key_arr.astype(np.int64, copy=False)

    geo_arr = st2g_arr[store_key_arr]
    currency_arr = g2c_arr[geo_arr]

    if currency_arr.dtype != np.int64:
        currency_arr = currency_arr.astype(np.int64, copy=False)

    # ------------------------------------------------------------
    # ORDERS (ONLY if enabled)
    # ------------------------------------------------------------
    if not skip_cols:
        orders = build_orders(
            rng=rng,
            n=n,
            skip_cols=False,
            date_pool=date_pool,
            date_prob=date_prob,
            customers=customers,
            product_keys=product_keys,
            _len_date_pool=len(date_pool),
            _len_customers=len(customers),
        )

        customer_keys = orders["customer_keys"]
        order_dates = orders["order_dates"]
        order_ids_int = orders["order_ids_int"]
        line_num = orders["line_num"]

    else:
        customer_keys = customers[
            rng.integers(0, len(customers), size=n)
        ]
        order_dates = date_pool[
            rng.integers(0, len(date_pool), size=n)
        ]

        order_ids_int = None
        line_num = None

    # Edge pinning: guarantees boundary coverage
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

    # ------------------------------------------------------------
    # YEAR / MONTH (partitioning only)
    # ------------------------------------------------------------
    if file_format == "deltaparquet":
        months = order_dates.astype("datetime64[M]").astype("int64")
        year_arr = (months // 12 + 1970).astype("int16")
        month_arr = (months % 12 + 1).astype("int8")

    # ------------------------------------------------------------
    # Arrow output (schema-driven, deterministic)
    # ------------------------------------------------------------
    arrays = []

    def add(name, data):
        arrays.append(
            pa.array(
                data,
                type=schema_types[name],
                safe=False,
            )
        )

    # Order columns (conditional)
    if not skip_cols:
        add("SalesOrderNumber", order_ids_int)
        add("SalesOrderLineNumber", line_num)

    # Keys
    add("CustomerKey", customer_keys)
    add("ProductKey", product_keys)
    add("StoreKey", store_key_arr)
    add("PromotionKey", promo_keys)
    add("CurrencyKey", currency_arr)

    # Dates
    add("OrderDate", order_dates)
    add("DueDate", due_date)
    add("DeliveryDate", delivery_date)

    # Measures
    add("Quantity", qty)
    add("NetPrice", price["final_net_price"])
    add("UnitCost", price["final_unit_cost"])
    add("UnitPrice", price["final_unit_price"])
    add("DiscountAmount", price["discount_amt"])

    # Status
    add("DeliveryStatus", delivery_status)
    add("IsOrderDelayed", is_order_delayed)

    # Partitioning
    if file_format == "deltaparquet":
        add("Year", year_arr)
        add("Month", month_arr)

    return pa.Table.from_arrays(arrays, schema=schema)
