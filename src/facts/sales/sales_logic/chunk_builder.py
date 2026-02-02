import numpy as np
import pyarrow as pa

from .globals import State, PA_AVAILABLE
from .order_logic import build_orders
from .date_logic import compute_dates
from .promo_logic import apply_promotions
from .price_logic import compute_prices

from .models.activity_model import apply_activity_thinning
from .models.quantity_model import build_quantity
from .models.pricing_pipeline import build_prices
from .models.customer_lifecycle import build_active_customer_pool


def build_chunk_table(n: int, seed: int, no_discount_key: int = 1) -> pa.Table:
    """
    Build a chunk of synthetic sales data.

    Design:
    - Lifecycle controls which customers are ACTIVE per month
    - Discovery controls when ACTIVE customers place their FIRST order
    - Orders are generated only from eligible customers
    """

    if not PA_AVAILABLE:
        raise RuntimeError("pyarrow is required")

    rng = np.random.default_rng(seed)
    skip_cols = State.skip_order_cols

    # ------------------------------------------------------------
    # STATIC STATE
    # ------------------------------------------------------------
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
    schema_types = {f.name: f.type for f in schema}
    file_format = State.file_format

    # ------------------------------------------------------------
    # CUSTOMER LIFECYCLE (month → active mask)
    # ------------------------------------------------------------
    months_int = date_pool.astype("datetime64[M]").astype("int64")
    start_month = int(months_int.min())
    end_month = int(months_int.max())

    active_by_month = build_active_customer_pool(
        all_customers=customers_all,
        start_month=0,
        end_month=end_month - start_month,
        seed=seed,
    )

    # ------------------------------------------------------------
    # SPLIT ROW BUDGET ACROSS MONTHS (scaled by active customers)
    # ------------------------------------------------------------
    active_counts = np.array(
        [mask.sum() for mask in active_by_month.values()],
        dtype=np.float64,
    )

    active_weights = active_counts / active_counts.sum()

    rows_per_month = np.maximum(
        1,
        (active_weights * n).astype(int),
    )

    # ------------------------------------------------------------
    # DISCOVERY STATE (NEW)
    # ------------------------------------------------------------
    seen_customers = set()

    disc_cfg = State.models_cfg.get("customer_discovery", {})

    base_discovery_rate = disc_cfg.get("base_discovery_rate", 0.12)
    seasonal_amp = disc_cfg.get("seasonal_amplitude", 0.35)
    seasonal_period = disc_cfg.get("seasonal_period_months", 24)

    min_p = disc_cfg.get("min_discovery_rate", 0.02)
    max_p = disc_cfg.get("max_discovery_rate", 0.60)

    # ------------------------------------------------------------
    # COLLECT PER-MONTH RESULTS
    # ------------------------------------------------------------
    tables = []

    for m_offset, m_rows in enumerate(rows_per_month):
        mask = active_by_month.get(m_offset)
        if mask is None or not mask.any():
            continue

        active_customers = customers_all[mask]
        if len(active_customers) == 0:
            continue

        # --------------------------------------------------------
        # PROBABILISTIC DISCOVERY (Option A)
        # --------------------------------------------------------
        undiscovered = np.array(
            [c for c in active_customers if c not in seen_customers],
            dtype=active_customers.dtype,
        )

        if len(undiscovered) > 0:
            cycle = np.sin(2 * np.pi * m_offset / seasonal_period)
            p_discovery = base_discovery_rate * (1.0 + seasonal_amp * cycle)
            p_discovery = np.clip(p_discovery, min_p, max_p)

            discover_n = max(1, int(len(undiscovered) * p_discovery))
            newly_discovered = rng.choice(
                undiscovered,
                size=min(discover_n, len(undiscovered)),
                replace=False,
            )
        else:
            newly_discovered = np.empty(0, dtype=active_customers.dtype)

        # customers allowed to place orders this month
        eligible_customers = np.concatenate(
            [
                np.array(list(seen_customers), dtype=active_customers.dtype),
                newly_discovered,
            ]
        )

        if len(eligible_customers) == 0:
            continue

        # --------------------------------------------------------
        # PRODUCTS
        # --------------------------------------------------------
        prod_idx = rng.integers(0, len(product_np), size=m_rows)
        prods = product_np[prod_idx]

        product_keys = prods[:, 0]
        unit_price = prods[:, 1].astype(np.float64, copy=False)
        unit_cost = prods[:, 2].astype(np.float64, copy=False)

        # --------------------------------------------------------
        # STORE → GEO → CURRENCY
        # --------------------------------------------------------
        store_key_arr = store_keys[
            rng.integers(0, len(store_keys), size=m_rows)
        ]
        geo_arr = st2g_arr[store_key_arr]
        currency_arr = g2c_arr[geo_arr]

        # --------------------------------------------------------
        # ORDERS (restricted to eligible customers)
        # --------------------------------------------------------
        if not skip_cols:
            orders = build_orders(
                rng=rng,
                n=m_rows,
                skip_cols=False,
                date_pool=date_pool,
                date_prob=date_prob,
                customers=eligible_customers,
                product_keys=product_keys,
                _len_date_pool=len(date_pool),
                _len_customers=len(eligible_customers),
            )

            customer_keys = orders["customer_keys"]
            order_dates = orders["order_dates"]
            order_ids_int = orders["order_ids_int"]
            line_num = orders["line_num"]
        else:
            customer_keys = eligible_customers[
                rng.integers(0, len(eligible_customers), size=m_rows)
            ]
            order_dates = date_pool[
                rng.integers(0, len(date_pool), size=m_rows)
            ]
            order_ids_int = None
            line_num = None

        # --------------------------------------------------------
        # DATE LOGIC
        # --------------------------------------------------------
        dates = compute_dates(
            rng=rng,
            n=len(customer_keys),
            product_keys=product_keys,
            order_ids_int=order_ids_int,
            order_dates=order_dates,
        )

        # --------------------------------------------------------
        # PROMOTIONS
        # --------------------------------------------------------
        promo_keys, promo_pct = apply_promotions(
            rng=rng,
            n=len(customer_keys),
            order_dates=order_dates,
            promo_keys_all=promo_keys_all,
            promo_pct_all=promo_pct_all,
            promo_start_all=promo_start_all,
            promo_end_all=promo_end_all,
            no_discount_key=no_discount_key,
        )

        # --------------------------------------------------------
        # ACTIVITY THINNING
        # --------------------------------------------------------
        keep_mask = apply_activity_thinning(
            rng=rng,
            order_dates=order_dates,
        )

        if not keep_mask.any():
            continue

        def _f(x):
            return x[keep_mask]

        customer_keys = _f(customer_keys)
        product_keys = _f(product_keys)
        unit_price = _f(unit_price)
        unit_cost = _f(unit_cost)
        store_key_arr = _f(store_key_arr)
        geo_arr = _f(geo_arr)
        currency_arr = _f(currency_arr)
        order_dates = _f(order_dates)

        if not skip_cols:
            order_ids_int = _f(order_ids_int)
            line_num = _f(line_num)

        # --------------------------------------------------------
        # UPDATE DISCOVERY STATE (IMPORTANT)
        # --------------------------------------------------------
        seen_customers.update(customer_keys.tolist())

        # --------------------------------------------------------
        # QUANTITY + PRICING
        # --------------------------------------------------------
        qty = build_quantity(rng, order_dates)

        price = compute_prices(
            rng=rng,
            n=len(customer_keys),
            unit_price=unit_price,
            unit_cost=unit_cost,
            promo_pct=promo_pct[keep_mask],
        )

        price = build_prices(
            rng=rng,
            order_dates=order_dates,
            qty=qty,
            price=price,
        )

        # --------------------------------------------------------
        # BUILD ARROW TABLE
        # --------------------------------------------------------
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
            m_int = order_dates.astype("datetime64[M]").astype("int64")
            add("Year", (m_int // 12 + 1970).astype("int16"))
            add("Month", (m_int % 12 + 1).astype("int8"))

        tables.append(pa.Table.from_arrays(arrays, schema=schema))

    if not tables:
        return pa.Table.from_arrays([], schema=schema)

    return pa.concat_tables(tables)
