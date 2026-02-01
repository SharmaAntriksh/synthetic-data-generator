import numpy as np
import pyarrow as pa

from .globals import State, PA_AVAILABLE
from .order_logic import build_orders
from .date_logic import compute_dates
from .promo_logic import apply_promotions
from .price_logic import compute_prices

from .models.activity_model import apply_activity_thinning
from .models.quantity_model import build_quantity
from .models.customer_lifecycle import apply_customer_churn
from .models.pricing_pipeline import build_prices


def build_chunk_table(n: int, seed: int, no_discount_key: int = 1) -> pa.Table:
    """
    Build a chunk of synthetic sales data.

    Orchestrates:
    - order creation
    - customer lifecycle (growth & churn)
    - activity thinning (transactions)
    - quantity modeling
    - pricing pipeline
    - Arrow table output
    """

    if not PA_AVAILABLE:
        raise RuntimeError("pyarrow is required")

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------
    # IMMUTABLE STATE
    # ------------------------------------------------------------
    skip_cols = State.skip_order_cols

    product_np = (
        State.active_product_np
        if getattr(State, "active_product_np", None) is not None
        else State.product_np
    )

    customers_all = (
        State.active_customer_keys
        if getattr(State, "active_customer_keys", None) is not None
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

    schema = State.sales_schema
    file_format = State.file_format
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
    store_key_arr = store_keys[rng.integers(0, len(store_keys), size=n)]
    geo_arr = st2g_arr[store_key_arr]
    currency_arr = g2c_arr[geo_arr]

    # ------------------------------------------------------------
    # ORDERS
    # ------------------------------------------------------------
    if not skip_cols:
        orders = build_orders(
            rng=rng,
            n=n,
            skip_cols=False,
            date_pool=date_pool,
            date_prob=date_prob,
            customers=customers_all,
            product_keys=product_keys,
            _len_date_pool=len(date_pool),
            _len_customers=len(customers_all),
        )

        customer_keys = orders["customer_keys"]
        order_dates = orders["order_dates"]
        order_ids_int = orders["order_ids_int"]
        line_num = orders["line_num"]
    else:
        customer_keys = customers_all[rng.integers(0, len(customers_all), size=n)]
        order_dates = date_pool[rng.integers(0, len(date_pool), size=n)]
        order_ids_int = None
        line_num = None

    # Ensure full date coverage
    order_dates[0] = date_pool[0]
    order_dates[-1] = date_pool[-1]

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
    # CUSTOMER LIFECYCLE (growth + churn)
    # ------------------------------------------------------------
    customer_keys = apply_customer_churn(
        rng=rng,
        customer_keys=customer_keys,
        order_dates=order_dates,
        all_customers=customers_all,
        seed=seed,
    )

    # ------------------------------------------------------------
    # ACTIVITY THINNING (row count control)
    # ------------------------------------------------------------
    keep_mask = apply_activity_thinning(
        rng=rng,
        order_dates=order_dates,
    )

    def _f(x):
        return x[keep_mask]

    product_keys = _f(product_keys)
    unit_price = _f(unit_price)
    unit_cost = _f(unit_cost)
    store_key_arr = _f(store_key_arr)
    geo_arr = _f(geo_arr)
    currency_arr = _f(currency_arr)
    order_dates = _f(order_dates)
    customer_keys = _f(customer_keys)

    if not skip_cols:
        order_ids_int = _f(order_ids_int)
        line_num = _f(line_num)

    n = int(keep_mask.sum())

    # ------------------------------------------------------------
    # QUANTITY
    # ------------------------------------------------------------
    qty = build_quantity(rng, order_dates)

    # ------------------------------------------------------------
    # BASE PRICING (AFTER THINNING)
    # ------------------------------------------------------------
    price = compute_prices(
        rng=rng,
        n=n,
        unit_price=unit_price,
        unit_cost=unit_cost,
        promo_pct=promo_pct[keep_mask],
    )

    # ------------------------------------------------------------
    # PRICING PIPELINE
    # ------------------------------------------------------------
    price = build_prices(
        rng=rng,
        order_dates=order_dates,
        qty=qty,
        price=price,
    )

    # ------------------------------------------------------------
    # ARROW OUTPUT
    # ------------------------------------------------------------
    arrays = []

    def add(name, data):
        arrays.append(pa.array(data, type=schema_types[name], safe=False))

    if not skip_cols:
        add("SalesOrderNumber", order_ids_int)
        add("SalesOrderLineNumber", line_num)

    add("CustomerKey", customer_keys)
    add("ProductKey", product_keys)
    add("StoreKey", store_key_arr)
    add("PromotionKey", promo_keys[keep_mask])
    add("CurrencyKey", currency_arr)

    add("OrderDate", order_dates)
    add("DueDate", dates["due_date"][keep_mask])
    add("DeliveryDate", dates["delivery_date"][keep_mask])

    add("Quantity", qty)
    add("NetPrice", price["final_net_price"])
    add("UnitCost", price["final_unit_cost"])
    add("UnitPrice", price["final_unit_price"])
    add("DiscountAmount", price["discount_amt"])

    add("DeliveryStatus", dates["delivery_status"][keep_mask])
    add("IsOrderDelayed", dates["is_order_delayed"][keep_mask])

    if file_format == "deltaparquet":
        months_int = order_dates.astype("datetime64[M]").astype("int64")
        add("Year", (months_int // 12 + 1970).astype("int16"))
        add("Month", (months_int % 12 + 1).astype("int8"))

    return pa.Table.from_arrays(arrays, schema=schema)
