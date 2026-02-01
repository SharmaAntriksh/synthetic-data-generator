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
    Deterministic, smooth, and Power BI–friendly.
    """

    if not PA_AVAILABLE:
        raise RuntimeError("pyarrow is required")

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------
    # Pull immutable state ONCE
    # ------------------------------------------------------------
    skip_cols = State.skip_order_cols
    product_np = (
        State.active_product_np
        if getattr(State, "active_product_np", None) is not None
        else State.product_np
    )
    customers = (
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
        customer_keys = customers[rng.integers(0, len(customers), size=n)]
        order_dates = date_pool[rng.integers(0, len(date_pool), size=n)]
        order_ids_int = None
        line_num = None

    # Ensure coverage
    order_dates[0] = date_pool[0]
    order_dates[-1] = date_pool[-1]

    # ------------------------------------------------------------
    # QUANTITY
    # ------------------------------------------------------------
    qty = np.clip(rng.poisson(2.8, n) + 1, 1, 4)

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
    # BASE PRICE COMPUTATION
    # ------------------------------------------------------------
    price = compute_prices(
        rng=rng,
        n=n,
        unit_price=unit_price,
        unit_cost=unit_cost,
        promo_pct=promo_pct,
    )

    # ------------------------------------------------------------
    # EARLY-RAMP (cold start only)
    # ------------------------------------------------------------
    months_since_start = (
        order_dates.astype("datetime64[M]").astype(int)
        - order_dates.astype("datetime64[M]").min().astype(int)
    )
    ramp = np.clip(months_since_start / 9.0, 0.6, 1.0)

    price["final_unit_price"] *= ramp
    price["discount_amt"] *= ramp
    price["final_net_price"] = (
        price["final_unit_price"] - price["discount_amt"]
    )

    # ------------------------------------------------------------
    # LONG-TERM GROWTH (trend backbone)
    # ------------------------------------------------------------
    base_year = order_dates.astype("datetime64[Y]").min().astype(int)
    trend_year_idx = (
        order_dates.astype("datetime64[Y]").astype(int) - base_year
    )

    inflation = (1.0 + 0.04) ** trend_year_idx

    price["final_unit_price"] *= inflation
    price["discount_amt"] *= inflation
    price["final_net_price"] = (
        price["final_unit_price"] - price["discount_amt"]
    )

    # ------------------------------------------------------------
    # MONTHLY AGGREGATION
    # ------------------------------------------------------------
    order_months = order_dates.astype("datetime64[M]")
    months, inv = np.unique(order_months, return_inverse=True)

    monthly_sales = np.zeros(len(months), dtype=np.float64)
    for i in range(len(months)):
        mask = inv == i
        monthly_sales[i] = np.sum(qty[mask] * price["final_net_price"][mask])

    # ------------------------------------------------------------
    # EMA SMOOTHING (defines the visible trend)
    # ------------------------------------------------------------
    alpha = 0.47
    smoothed = monthly_sales.copy()
    for i in range(1, len(smoothed)):
        smoothed[i] = alpha * monthly_sales[i] + (1 - alpha) * smoothed[i - 1]

    scale = smoothed / np.maximum(monthly_sales, 1.0)

    price["final_unit_price"] *= scale[inv]
    price["discount_amt"] *= scale[inv]
    price["final_net_price"] = (
        price["final_unit_price"] - price["discount_amt"]
    )

    # ------------------------------------------------------------
    # RESIDUAL SEASONALITY (texture only)
    # ------------------------------------------------------------
    month_idx = order_months.astype(int) % 12
    season_year_idx = (
        order_months.astype("datetime64[Y]").astype(int)
        - order_months.astype("datetime64[Y]").min().astype(int)
    )

    phase_shift = (season_year_idx % 5) * (np.pi / 10)
    raw = np.sin(2 * np.pi * (month_idx - 1) / 12 + phase_shift)

    seasonality = 1.0 + 0.025 * np.tanh(1.2 * raw)

    n_years = season_year_idx.max() + 1
    year_jitter = 1.0 + rng.normal(0, 0.015, size=n_years)[season_year_idx]
    seasonality *= year_jitter

    price["final_unit_price"] *= seasonality
    price["final_net_price"] = (
        price["final_unit_price"] - price["discount_amt"]
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
    add("PromotionKey", promo_keys)
    add("CurrencyKey", currency_arr)

    add("OrderDate", order_dates)
    add("DueDate", dates["due_date"])
    add("DeliveryDate", dates["delivery_date"])

    add("Quantity", qty)
    add("NetPrice", price["final_net_price"])
    add("UnitCost", price["final_unit_cost"])
    add("UnitPrice", price["final_unit_price"])
    add("DiscountAmount", price["discount_amt"])

    add("DeliveryStatus", dates["delivery_status"])
    add("IsOrderDelayed", dates["is_order_delayed"])

    if file_format == "deltaparquet":
        months_int = order_dates.astype("datetime64[M]").astype("int64")
        add("Year", (months_int // 12 + 1970).astype("int16"))
        add("Month", (months_int % 12 + 1).astype("int8"))

    return pa.Table.from_arrays(arrays, schema=schema)
