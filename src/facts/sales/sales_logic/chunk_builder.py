import math
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

from .month_planing import build_rows_per_month
from .customer_sampling import (
    _normalize_end_month,
    _eligible_customer_mask_for_month,
    _participation_distinct_target,
    _sample_customers,
)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _get_state_attr(*names, default=None):
    for n in names:
        if hasattr(State, n):
            v = getattr(State, n)
            if v is not None:
                return v
    return default


def _build_month_slices(date_pool: np.ndarray) -> dict:
    """
    Build a mapping: month_offset -> indices in date_pool belonging to that month.
    month_offset is 0..T-1 where 0 is min month in date_pool.
    """
    months_int = date_pool.astype("datetime64[M]").astype("int64")
    min_m = int(months_int.min())
    max_m = int(months_int.max())
    T = (max_m - min_m) + 1

    month_slices = {}
    for m in range(T):
        m_int = min_m + m
        idx = np.nonzero(months_int == m_int)[0]
        month_slices[m] = idx

    return month_slices


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def build_chunk_table(n: int, seed: int, no_discount_key: int = 1) -> pa.Table:
    """
    Build a chunk of synthetic sales data.

    Guarantees:
    - Customer lifecycle controls eligibility (IsActiveInSales + Start/End month)
    - Month loop actually generates orders within that month (date_pool sliced)
    - Optional discovery forcing (persistable across chunks via State.seen_customers)
    - All per-order arrays remain perfectly aligned
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

    # Customer dimension arrays (new contract)
    customer_keys = _get_state_attr("customer_keys", "customers")
    if customer_keys is None:
        raise RuntimeError("State must provide customer_keys/customers")

    customer_keys = np.asarray(customer_keys)

    is_active_in_sales = _get_state_attr("customer_is_active_in_sales", "is_active_in_sales")
    if is_active_in_sales is None:
        # backward compat: if State.active_customer_keys exists, treat those as active
        active_keys = getattr(State, "active_customer_keys", None)
        if active_keys is not None:
            # Build mask by assuming customer keys are 1..N
            is_active_in_sales = np.zeros(customer_keys.shape[0], dtype="int64")
            is_active_in_sales[(np.asarray(active_keys, dtype="int64") - 1)] = 1
        else:
            # assume all active
            is_active_in_sales = np.ones(customer_keys.shape[0], dtype="int64")
    else:
        is_active_in_sales = np.asarray(is_active_in_sales, dtype="int64")

    start_month = _get_state_attr("customer_start_month")
    if start_month is None:
        # If not yet wired, fall back: everyone starts at 0 (old behavior)
        start_month = np.zeros(customer_keys.shape[0], dtype="int64")
    else:
        start_month = np.asarray(start_month, dtype="int64")

    end_month = _get_state_attr("customer_end_month")
    end_month_norm = _normalize_end_month(end_month, customer_keys.shape[0])

    base_weight = _get_state_attr("customer_base_weight")
    if base_weight is not None:
        base_weight = np.asarray(base_weight, dtype="float64")

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
    # MONTH SLICES (date_pool indices per month)
    # ------------------------------------------------------------
    month_slices = _build_month_slices(date_pool)
    T = len(month_slices)

    # ------------------------------------------------------------
    # ROW BUDGET PER MONTH (BASE DEMAND NORMALIZATION)
    # ------------------------------------------------------------
    macro_cfg = State.models_cfg.get("macro_demand", {})

    eligible_counts = np.empty(T, dtype="float64")
    eligible_masks = []

    for m in range(T):
        mask = _eligible_customer_mask_for_month(
            m_offset=m,
            is_active_in_sales=is_active_in_sales,
            start_month=start_month,
            end_month_norm=end_month_norm,
        )
        eligible_masks.append(mask)
        eligible_counts[m] = float(mask.sum())

    if eligible_counts.sum() == 0:
        return pa.Table.from_arrays([], schema=schema)

    rows_per_month = build_rows_per_month(
        rng=rng,
        total_rows=n,
        eligible_counts=eligible_counts,
        macro_cfg=macro_cfg,
    )

    # ------------------------------------------------------------
    # DISCOVERY STATE (persist across chunks if State provides it)
    # ------------------------------------------------------------
    seen_customers = getattr(State, "seen_customers", None)
    if seen_customers is None:
        seen_customers = set()
    else:
        # ensure it's a set-like
        if not isinstance(seen_customers, set):
            seen_customers = set(seen_customers)
            State.seen_customers = seen_customers

    disc_cfg = State.models_cfg.get("customer_discovery", {})
    use_discovery = bool(disc_cfg.get("enabled", True))

    participation_cfg = State.models_cfg.get("customer_participation", {})
    use_participation = bool(participation_cfg)

    # ------------------------------------------------------------
    # BUILD MONTHLY TABLES
    # ------------------------------------------------------------
    tables = []

    for m_offset, m_rows in enumerate(rows_per_month):
        eligible_mask = eligible_masks[m_offset]
        if not eligible_mask.any():
            continue

        date_idx = month_slices.get(m_offset)
        if date_idx is None or len(date_idx) == 0:
            continue

        # month-specific date pool / probabilities
        month_date_pool = date_pool[date_idx]

        if date_prob is not None:
            month_date_prob = date_prob[date_idx].astype("float64", copy=False)
            s = month_date_prob.sum()
            if s > 0:
                month_date_prob = month_date_prob / s
            else:
                month_date_prob = None
        else:
            month_date_prob = None

        # Discovery config (copied per-month so we can inject per-month targets)
        disc_cfg_local = dict(disc_cfg)

        opc = float(disc_cfg_local.get("orders_per_new_customer", 20.0))
        min_month = int(disc_cfg_local.get("min_new_customers_per_month", 0) or 0)

        # demand-driven discovery
        target_new = int(max(1, round(int(m_rows) / max(opc, 1e-9))))

        # --------------------------------------------------------
        # DISCOVERY RATE / SEASONALITY (rate knobs)
        # Multiplies the demand-driven target_new; floors run after this.
        # --------------------------------------------------------
        base_rate = float(disc_cfg_local.get("base_discovery_rate", 0.06))
        seasonal_amp = float(disc_cfg_local.get("seasonal_amplitude", 0.0))
        seasonal_period = int(disc_cfg_local.get("seasonal_period_months", 24))
        min_p = float(disc_cfg_local.get("min_discovery_rate", base_rate))
        max_p = float(disc_cfg_local.get("max_discovery_rate", base_rate))

        if base_rate > 0.0 and (seasonal_amp != 0.0 or min_p != base_rate or max_p != base_rate):
            cyc = math.sin(2.0 * math.pi * float(m_offset) / max(seasonal_period, 1))
            p = base_rate * (1.0 + seasonal_amp * cyc)
            p = float(np.clip(p, min_p, max_p))
            target_new = int(max(0, round(target_new * (p / base_rate))))

        # HARD FLOOR (prevents year-2 collapse)
        if min_month > 0:
            ramp_months = int(disc_cfg_local.get("floor_ramp_months", 12) or 0)
            if ramp_months > 0:
                # ramp floor up over first N months
                t = min(1.0, max(0.0, m_offset / float(ramp_months)))
                floor0 = float(min_month)
                floor1 = float(disc_cfg_local.get("min_new_customers_steady", min_month))
                floor = int(round(floor0 + t * (floor1 - floor0)))
            else:
                floor = int(min_month)
            target_new = max(target_new, floor)

        # --------------------------------------------------------
        # BOOTSTRAP SUPPRESSION (removes Year-1 privilege)
        # --------------------------------------------------------
        bootstrap_months = int(disc_cfg_local.get("bootstrap_suppression_months", 12) or 0)
        if bootstrap_months > 0 and m_offset < bootstrap_months:
            scale = (m_offset + 1) / float(bootstrap_months)
            target_new = int(round(target_new * scale))

        disc_cfg_local["_target_new_customers"] = target_new

        target_distinct = None
        if use_participation:
            target_distinct = _participation_distinct_target(
                rng=rng,
                m_offset=int(m_offset),
                eligible_count=int(eligible_mask.sum()),
                n_orders=int(m_rows),
                cfg=participation_cfg,
            )

        customer_keys_for_orders = _sample_customers(
            rng=rng,
            customer_keys=customer_keys,
            eligible_mask=eligible_mask,
            seen_set=seen_customers,
            n=int(m_rows),
            use_discovery=use_discovery,
            discovery_cfg=disc_cfg_local,
            base_weight=base_weight,
            target_distinct=target_distinct,
        )

        if customer_keys_for_orders.size == 0:
            continue

        n_orders = int(customer_keys_for_orders.size)

        # --------------------------------------------------------
        # PRODUCTS (PER ORDER)
        # --------------------------------------------------------
        prod_idx = rng.integers(0, len(product_np), size=n_orders)
        prods = product_np[prod_idx]

        product_keys = prods[:, 0]
        unit_price = prods[:, 1].astype(np.float64, copy=False)
        unit_cost = prods[:, 2].astype(np.float64, copy=False)

        # --------------------------------------------------------
        # STORE → GEO → CURRENCY
        # --------------------------------------------------------
        store_key_arr = store_keys[rng.integers(0, len(store_keys), size=n_orders)]
        geo_arr = st2g_arr[store_key_arr]
        currency_arr = g2c_arr[geo_arr]

        # --------------------------------------------------------
        # ORDERS (use month-specific date pool so month loop is real)
        # --------------------------------------------------------
        if not skip_cols:
            orders = build_orders(
                rng=rng,
                n=n_orders,
                skip_cols=False,
                date_pool=month_date_pool,
                date_prob=month_date_prob,
                customers=customer_keys_for_orders,
                product_keys=product_keys,
                _len_date_pool=len(month_date_pool),
                _len_customers=n_orders,
            )

            customer_keys_out = orders["customer_keys"]
            order_dates = orders["order_dates"]
            order_ids_int = orders["order_ids_int"]
            line_num = orders["line_num"]
        else:
            customer_keys_out = customer_keys_for_orders
            order_dates = month_date_pool[rng.integers(0, len(month_date_pool), size=n_orders)]
            order_ids_int = None
            line_num = None

        # --------------------------------------------------------
        # DATE LOGIC
        # --------------------------------------------------------
        dates = compute_dates(
            rng=rng,
            n=n_orders,
            product_keys=product_keys,
            order_ids_int=order_ids_int,
            order_dates=order_dates,
        )

        # --------------------------------------------------------
        # PROMOTIONS
        # --------------------------------------------------------
        promo_keys, promo_pct = apply_promotions(
            rng=rng,
            n=n_orders,
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
        keep_mask = apply_activity_thinning(rng=rng, order_dates=order_dates)
        if not keep_mask.any():
            continue

        def _f(x):
            return x[keep_mask]

        customer_keys_out = _f(customer_keys_out)
        product_keys = _f(product_keys)
        unit_price = _f(unit_price)
        unit_cost = _f(unit_cost)
        store_key_arr = _f(store_key_arr)
        geo_arr = _f(geo_arr)
        currency_arr = _f(currency_arr)
        order_dates = _f(order_dates)

        promo_keys = _f(promo_keys)
        promo_pct = _f(promo_pct)

        if not skip_cols:
            order_ids_int = _f(order_ids_int)
            line_num = _f(line_num)

        # --------------------------------------------------------
        # UPDATE DISCOVERY STATE (persist)
        # --------------------------------------------------------
        # IMPORTANT: use post-thinning keys so "seen" means "actually appeared"
        seen_customers.update(customer_keys_out.tolist())
        State.seen_customers = seen_customers

        # --------------------------------------------------------
        # QUANTITY + PRICING
        # --------------------------------------------------------
        qty = build_quantity(rng, order_dates)

        price = compute_prices(
            rng=rng,
            n=len(customer_keys_out),
            unit_price=unit_price,
            unit_cost=unit_cost,
            promo_pct=promo_pct,
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

        add("CustomerKey", customer_keys_out)
        add("ProductKey", product_keys)
        add("StoreKey", store_key_arr)
        add("PromotionKey", promo_keys)
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
