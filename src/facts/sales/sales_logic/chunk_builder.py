import numpy as np
import pyarrow as pa

from .globals import State, PA_AVAILABLE
from .order_logic import build_orders
from .date_logic import compute_dates
from .promo_logic import apply_promotions
from .price_logic import compute_prices
from src.facts.sales.demand_engine import build_demand_timeline


def build_chunk_table(n: int, seed: int, no_discount_key: int = 1) -> pa.Table:
    """
    Build `n` synthetic sales rows.
    All shared, immutable state is read from `State`.
    """

    if not PA_AVAILABLE:
        raise RuntimeError("pyarrow is required")

    rng = np.random.default_rng(seed)

    # ------------------------------------------------------------
    # Pull immutable state ONCE (multiprocessing-safe)
    # ------------------------------------------------------------
    skip_cols = State.skip_order_cols
    if skip_cols not in (True, False):
        raise RuntimeError("State.skip_order_cols must be boolean")

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

    file_format = State.file_format
    schema = State.sales_schema
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
    # YEAR-SPECIFIC CALENDAR DISTORTION (break repeating patterns)
    # ------------------------------------------------------------
    years = date_pool.astype("datetime64[Y]").astype(int)
    unique_years = np.unique(years)

    # Each year has its own calendar "personality"
    year_factor = {
        y: rng.normal(1.0, 0.20) for y in unique_years
    }

    date_year_factor = np.array(
        [year_factor[y] for y in years],
        dtype=np.float64,
    )

    date_prob_adj = date_prob * date_year_factor
    date_prob_adj /= date_prob_adj.sum()

    order_dates = np.empty(n, dtype="datetime64[D]")

    rows_left = n
    start = 0

    for y in unique_years:
        mask = years == y
        year_dates = date_pool[mask]
        year_probs = date_prob_adj[mask].copy()

        # ------------------------------------------------------------
        # YEAR-SPECIFIC MONTH STRENGTH (break fixed peak months)
        # ------------------------------------------------------------
        months = year_dates.astype("datetime64[M]").astype(int) % 12

        # Random business seasonality per year
        month_strength = rng.lognormal(mean=0.0, sigma=0.45, size=12)

        # ------------------------------------------------------------
        # Smooth month strength (adjacent months correlated)
        # ------------------------------------------------------------
        alpha = rng.uniform(0.50, 0.70)   # main month weight
        beta = (1.0 - alpha) / 2.0        # neighbor weights

        month_strength = (
            alpha * month_strength
            + beta * np.roll(month_strength, 1)
            + beta * np.roll(month_strength, -1)
        )

        year_probs *= month_strength[months]

        # ---- Normalize
        year_probs /= year_probs.sum()

        # ---- How many rows this year gets
        year_weight = rng.lognormal(mean=0.0, sigma=0.35) * rng.normal(1.0, 0.15)
        k = min(
            rows_left,
            int(n * year_weight / len(unique_years))
        )
        if y == unique_years[-1]:
            k = rows_left

        # ---- Sample dates FOR THIS YEAR ONLY
        order_dates[start:start+k] = rng.choice(
            year_dates,
            size=k,
            p=year_probs,
        )

        start += k
        rows_left -= k

    rng.shuffle(order_dates)

    order_months = order_dates.astype("datetime64[M]")

    # ------------------------------------------------------------
    # DEMAND ENGINE (stateful, month-level)
    # ------------------------------------------------------------
    demand_timeline = build_demand_timeline(rng, order_months)

    demand_multiplier = np.array(
        [demand_timeline[m].demand_multiplier for m in order_months],
        dtype=np.float64,
    )

    price_pressure = np.array(
        [demand_timeline[m].price_pressure for m in order_months],
        dtype=np.float64,
    )

    promo_intensity = np.array(
        [demand_timeline[m].promo_intensity for m in order_months],
        dtype=np.float64,
    )

    # ------------------------------------------------------------
    # DAILY SHOCKS (discrete events)
    # ------------------------------------------------------------
    daily_shock = np.ones(n, dtype=np.float64)
    shock_draw = rng.random(n)

    # Strong bad days
    bad_mask = shock_draw < 0.06
    daily_shock[bad_mask] *= rng.uniform(0.55, 0.80, size=bad_mask.sum())

    # Rare hot days
    hot_mask = shock_draw > 0.97
    daily_shock[hot_mask] *= rng.uniform(1.25, 1.60, size=hot_mask.sum())

    # ------------------------------------------------------------
    # DAILY SHOCK PERSISTENCE (DAY-GROUPED)
    # ------------------------------------------------------------
    # Group rows by actual calendar day
    _, day_idx = np.unique(order_dates, return_inverse=True)

    for d in np.unique(day_idx):
        mask = day_idx == d

        # Some days are globally "bad"
        if rng.random() < 0.12:
            daily_shock[mask] *= rng.uniform(0.70, 0.90)

    # ------------------------------------------------------------
    # MONTHLY CAPACITY WITH PERSISTENCE (CLUSTERED BAD / GOOD MONTHS)
    # ------------------------------------------------------------
    unique_months = np.unique(order_months)

    monthly_cap = {}
    momentum = 0.0     # carries regime across months

    for m in unique_months:
        # Random shock this month
        shock = rng.normal(0.0, 0.30)

        # Momentum carries forward (0.6–0.8 is realistic)
        momentum = 0.75 * momentum + shock

        # Convert momentum to capacity
        cap = 1.0 + momentum

        # Hard business limits
        cap = np.clip(cap, 0.40, 1.70)

        monthly_cap[m] = cap

    capacity_multiplier = np.array(
        [monthly_cap[m] for m in order_months],
        dtype=np.float64,
    )

    # ------------------------------------------------------------
    # FINAL VOLUME MULTIPLIER (clipped safety rail)
    # ------------------------------------------------------------
    volume_multiplier = np.clip(
        demand_multiplier * daily_shock * capacity_multiplier,
        0.25,
        2.5,
    )

    # ------------------------------------------------------------
    # BUILD ORDERS (NOW VOLUME ACTUALLY MOVES)
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
            volume_multiplier=volume_multiplier,
        )

        customer_keys = orders["customer_keys"]
        order_ids_int = orders["order_ids_int"]
        line_num = orders["line_num"]

    else:
        customer_keys = customers[
            rng.integers(0, len(customers), size=n)
        ]
        order_ids_int = None
        line_num = None

    # ------------------------------------------------------------
    # QUANTITY
    # ------------------------------------------------------------
    qty = np.clip(rng.poisson(3, n) + 1, 1, 5)

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
        promo_intensity=promo_intensity,
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
        price_pressure=price_pressure,
    )

    # ------------------------------------------------------------
    # PARTITIONING
    # ------------------------------------------------------------
    if file_format == "deltaparquet":
        months = order_dates.astype("datetime64[M]").astype("int64")
        year_arr = (months // 12 + 1970).astype("int16")
        month_arr = (months % 12 + 1).astype("int8")

    # ------------------------------------------------------------
    # ARROW OUTPUT (SCHEMA-DRIVEN)
    # ------------------------------------------------------------
    arrays = []

    def add(name, data):
        arrays.append(
            pa.array(data, type=schema_types[name], safe=False)
        )

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
        add("Year", year_arr)
        add("Month", month_arr)

    return pa.Table.from_arrays(arrays, schema=schema)
