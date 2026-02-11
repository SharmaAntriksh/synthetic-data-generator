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

def _normalize_prob(p: np.ndarray) -> np.ndarray | None:
    p = np.asarray(p, dtype="float64")
    p = np.where(np.isfinite(p) & (p > 0), p, 0.0)
    s = float(p.sum())
    if s <= 1e-12:
        return None
    return p / s


def _sample_product_row_indices(
    rng: np.random.Generator,
    n: int,
    product_np: np.ndarray,
    *,
    m_offset: int,
    enabled: bool,
) -> np.ndarray:
    """
    Return row indices into `product_np` for n orders.

    Brand-first sampling is used iff:
      - enabled == True (controlled by models_cfg.brand_popularity presence), AND
      - State provides brand buckets and optionally per-month brand probabilities.

    Required State attributes (to actually take effect):
      - State.brand_to_row_idx (list/tuple where entry b is np.ndarray of row indices into product_np)
        OR State.active_brand_to_row_idx if product_np is State.active_product_np
    Optional:
      - State.brand_prob_by_month: shape (T, B) or (B,)
        If absent or invalid, falls back to equal probability over non-empty brands.

    Safe fallback:
      - If anything is missing/invalid -> uniform sampling over product_np (old behavior).
    """
    if not enabled:
        return rng.integers(0, len(product_np), size=int(n)).astype("int64", copy=False)

    # Prefer "active" buckets if we're using active_product_np upstream.
    # Choose buckets that align with the passed `product_np`.
    if getattr(State, "active_product_np", None) is not None and product_np is State.active_product_np:
        brand_to_rows = getattr(State, "active_brand_to_row_idx", None)
    else:
        brand_to_rows = getattr(State, "brand_to_row_idx", None)
        
    if brand_to_rows is not None:
        mx = -1
        for b in brand_to_rows:
            if b is not None and len(b) > 0:
                mx = max(mx, int(np.max(b)))
        if mx >= len(product_np):
            brand_to_rows = None  # force uniform fallback

    brand_probs_by_month = _get_state_attr("brand_prob_by_month", default=None)

    if brand_to_rows is None or len(brand_to_rows) == 0:
        return rng.integers(0, len(product_np), size=int(n)).astype("int64", copy=False)

    B = int(len(brand_to_rows))

    # Select probability vector for this month
    p = None
    if brand_probs_by_month is not None:
        probs = np.asarray(brand_probs_by_month, dtype="float64")
        if probs.ndim == 1:
            cand = probs
        else:
            cand = probs[int(m_offset) % int(probs.shape[0])]
        cand = _normalize_prob(cand)
        if cand is not None and int(cand.size) == B:
            p = cand

    # Fallback: equal probability across brands with at least 1 SKU
    if p is None:
        avail = np.asarray([len(x) > 0 for x in brand_to_rows], dtype="float64")
        if float(avail.sum()) <= 0:
            return rng.integers(0, len(product_np), size=int(n)).astype("int64", copy=False)
        p = avail / float(avail.sum())

    brand_ids = rng.choice(B, size=int(n), p=p)

    # Fill output by grouping same-brand rows (minimizes Python overhead)
    out = np.empty(int(n), dtype="int64")

    order = np.argsort(brand_ids)
    b_sorted = brand_ids[order]

    starts = np.flatnonzero(np.r_[True, b_sorted[1:] != b_sorted[:-1]])
    ends = np.r_[starts[1:], b_sorted.size]

    for s, e in zip(starts, ends):
        b = int(b_sorted[int(s)])
        bucket = brand_to_rows[b]
        k = int(e - s)

        if bucket is None or len(bucket) == 0:
            out[order[s:e]] = rng.integers(0, len(product_np), size=k).astype("int64", copy=False)
        else:
            sel = rng.integers(0, len(bucket), size=k).astype("int64", copy=False)
            out[order[s:e]] = np.asarray(bucket, dtype="int64")[sel]

    return out


def _get_state_attr(*names, default=None):
    """Return the first non-None attribute from State among names."""
    for n in names:
        if hasattr(State, n):
            v = getattr(State, n)
            if v is not None:
                return v
    return default


def _build_month_slices(date_pool: np.ndarray):
    """
    Build a list mapping month_offset -> slice/indices in date_pool belonging to that month.
    month_offset is 0..T-1 where 0 is min month in date_pool (datetime64[M] int).

    Optimized fast-path for sorted date_pool:
      - returns `slice(start, end)` per month (view slicing; low alloc)
    Fallback for unsorted pools:
      - returns index arrays per month
    """
    if not isinstance(date_pool, np.ndarray):
        date_pool = np.asarray(date_pool)

    if date_pool.size == 0:
        return []

    months_int = date_pool.astype("datetime64[M]").astype("int64")
    min_m = int(months_int.min())
    max_m = int(months_int.max())
    T = int(max_m - min_m + 1)

    # Fast-path: non-decreasing months (typical for date_range-derived pools)
    if months_int.size <= 1 or np.all(months_int[:-1] <= months_int[1:]):
        cuts = np.flatnonzero(months_int[1:] != months_int[:-1]) + 1
        starts = np.concatenate(([0], cuts))
        ends = np.concatenate((cuts, [months_int.size]))

        out = [slice(0, 0)] * T
        for s, e in zip(starts, ends):
            mo = int(months_int[int(s)] - min_m)
            out[mo] = slice(int(s), int(e))
        return out

    # Fallback: non-sorted pools
    out = [slice(0, 0)] * T
    for m_int in range(min_m, max_m + 1):
        idx = np.nonzero(months_int == m_int)[0]
        out[int(m_int - min_m)] = idx
    return out


def _eligible_counts_fast(
    T: int,
    is_active_in_sales: np.ndarray,
    start_month: np.ndarray,
    end_month_norm: np.ndarray,
) -> np.ndarray:
    """
    Compute eligible customer counts per month in O(N + T) using a delta/cumsum approach.

    Eligibility rule matches _eligible_customer_mask_for_month:
      active == 1 AND start_month <= m AND (end_month == -1 OR end_month >= m)
    """
    if T <= 0:
        return np.zeros(0, dtype="float64")

    is_active_in_sales = np.asarray(is_active_in_sales, dtype="int64", order="C")
    start_month = np.asarray(start_month, dtype="int64", order="C")
    end_month_norm = np.asarray(end_month_norm, dtype="int64", order="C")

    if start_month.size == 0:
        return np.zeros(T, dtype="float64")

    active = is_active_in_sales == 1
    if not active.any():
        return np.zeros(T, dtype="float64")

    s = start_month[active]
    e = end_month_norm[active]

    # keep only sane starts
    valid_start = (s >= 0) & (s < T)
    s = s[valid_start]
    e = e[valid_start]
    if s.size == 0:
        return np.zeros(T, dtype="float64")

    # discard invalid ranges where end < start (and end != -1)
    valid_range = (e < 0) | (e >= s)
    s = s[valid_range]
    e = e[valid_range]
    if s.size == 0:
        return np.zeros(T, dtype="float64")

    delta = np.zeros(T + 1, dtype="int64")
    np.add.at(delta, s, 1)

    finite = e >= 0
    if finite.any():
        endp1 = e[finite] + 1
        # endp1==T doesn't affect delta[:-1] cumsum; skip
        endp1 = endp1[(endp1 > 0) & (endp1 < T)]
        if endp1.size:
            np.add.at(delta, endp1, -1)

    counts = np.cumsum(delta[:-1])
    counts = np.maximum(counts, 0)
    return counts.astype("float64", copy=False)


def _empty_table(schema: pa.Schema) -> pa.Table:
    """Return a zero-row table with the exact schema types."""
    arrays = [pa.array([], type=f.type, safe=False) for f in schema]
    return pa.Table.from_arrays(arrays, schema=schema)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def build_chunk_table(
    n: int,
    seed: int,
    no_discount_key: int = 1,
    *,
    chunk_idx: int,
    chunk_capacity_orders: int,
) -> pa.Table:

    """
    Build a chunk of synthetic sales data.

    Guarantees:
    - Customer lifecycle controls eligibility (IsActiveInSales + Start/End month)
    - Month loop generates orders within that month (date_pool sliced)
    - Optional discovery forcing (persistable across chunks via State.seen_customers)
      - IMPORTANT: discovery is OFF if customer_discovery block is absent
    - All per-order arrays remain aligned
    """
    if not PA_AVAILABLE:
        raise RuntimeError("pyarrow is required")

    rng = np.random.default_rng(int(seed))
    skip_cols = bool(State.skip_order_cols)
    chunk_idx = int(chunk_idx)
    cap = int(chunk_capacity_orders)

    MOD = np.int64(1_000_000_000)

    # Each chunk owns a disjoint suffix range: [base, base + cap)
    base = np.int64(chunk_idx) * np.int64(cap)

    # Advances as we allocate orders month-by-month inside this chunk
    order_cursor = np.int64(0)

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
    customer_keys = np.asarray(customer_keys, dtype="int64")

    # is_active_in_sales (new contract)
    is_active_in_sales = _get_state_attr("customer_is_active_in_sales", "is_active_in_sales")
    if is_active_in_sales is None:
        # backward compat: if State.active_customer_keys exists, treat those as active
        active_keys = getattr(State, "active_customer_keys", None)
        if active_keys is not None:
            is_active_in_sales = np.zeros(customer_keys.shape[0], dtype="int64")
            idx = (np.asarray(active_keys, dtype="int64") - 1)
            idx = idx[(idx >= 0) & (idx < customer_keys.shape[0])]
            is_active_in_sales[idx] = 1
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
    if st2g_arr is None or g2c_arr is None:
        raise RuntimeError("State must provide store_to_geo_arr and geo_to_currency_arr")

    file_format = State.file_format

    # IMPORTANT:
    # - State.sales_schema reflects the *Sales output* schema (driven by skip_order_cols_requested)
    # - chunk_builder must use the *generation* schema (driven by skip_order_cols effective)
    if file_format == "deltaparquet":
        schema = State.schema_no_order_delta if skip_cols else State.schema_with_order_delta
    else:
        schema = State.schema_no_order if skip_cols else State.schema_with_order

    schema_types = {f.name: f.type for f in schema}

    # ------------------------------------------------------------
    # MONTH SLICES (date_pool indices per month)
    # ------------------------------------------------------------
    month_slices = _build_month_slices(date_pool)
    T = len(month_slices)
    if T == 0:
        return _empty_table(schema)

    # ------------------------------------------------------------
    # ROW BUDGET PER MONTH (BASE DEMAND NORMALIZATION)
    # ------------------------------------------------------------
    macro_cfg = State.models_cfg.get("macro_demand", {}) or {}

    eligible_counts = _eligible_counts_fast(
        T=T,
        is_active_in_sales=is_active_in_sales,
        start_month=start_month,
        end_month_norm=end_month_norm,
    )
    if eligible_counts.sum() <= 0:
        return _empty_table(schema)

    rows_per_month = build_rows_per_month(
        rng=rng,
        total_rows=int(n),
        eligible_counts=eligible_counts,
        macro_cfg=macro_cfg,
    )

    # ------------------------------------------------------------
    # DISCOVERY / PARTICIPATION CONFIG (EXPLICIT PRESENCE SEMANTICS)
    # ------------------------------------------------------------
    # Discovery is OFF if customer_discovery is absent.
    disc_cfg = State.models_cfg.get("customer_discovery", None)
    use_discovery = bool(disc_cfg) and bool(disc_cfg.get("enabled", True))
    disc_cfg = disc_cfg or {}

    # Participation is OFF if block absent. If present without enabled, default-on.
    participation_cfg = State.models_cfg.get("customer_participation", None)
    use_participation = bool(participation_cfg) and bool(participation_cfg.get("enabled", True))
    participation_cfg = participation_cfg or {}

    # Brand popularity is OFF if block absent. If present without enabled, default-on.
    brand_cfg = State.models_cfg.get("brand_popularity", None)
    use_brand_popularity = bool(brand_cfg) and bool(brand_cfg.get("enabled", True))

    # Discovery state only matters if discovery is enabled
    if use_discovery:
        seen_customers = getattr(State, "seen_customers", None)
        if seen_customers is None:
            seen_customers = set()
        elif not isinstance(seen_customers, set):
            seen_customers = set(seen_customers)
    else:
        seen_customers = None

    # ------------------------------------------------------------
    # Generate month-by-month
    # ------------------------------------------------------------
    tables = []

    for m_offset in range(T):
        m_rows = int(rows_per_month[m_offset])
        if m_rows <= 0:
            continue

        date_idx = month_slices[m_offset]
        month_date_pool = date_pool[date_idx]
        if month_date_pool.size == 0:
            continue

        if date_prob is not None:
            # copy before normalization (slice can be a view)
            month_date_prob = np.asarray(date_prob[date_idx], dtype="float64").copy()
            s = float(month_date_prob.sum())
            if s > 1e-12:
                month_date_prob /= s
            else:
                month_date_prob = None
        else:
            month_date_prob = None

        # --------------------------------------------------------
        # Eligibility mask (compute only for months that generate rows)
        # --------------------------------------------------------
        eligible_mask = _eligible_customer_mask_for_month(
            m_offset=int(m_offset),
            is_active_in_sales=is_active_in_sales,
            start_month=start_month,
            end_month_norm=end_month_norm,
        )
        if not eligible_mask.any():
            continue

        # --------------------------------------------------------
        # DISCOVERY TARGET (ONLY IF DISCOVERY ENABLED) - preserved semantics
        # --------------------------------------------------------
        disc_cfg_local = {}
        if use_discovery:
            disc_cfg_local = dict(disc_cfg)

            opc = float(disc_cfg_local.get("orders_per_new_customer", 20.0))
            min_month = int(disc_cfg_local.get("min_new_customers_per_month", 0) or 0)

            # demand-driven discovery
            target_new = int(max(1, round(m_rows / max(opc, 1e-9))))

            # Discovery rate / seasonality (rate knobs)
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

            # Hard floor (helps prevent weak early years if desired)
            if min_month > 0:
                ramp_months = int(disc_cfg_local.get("floor_ramp_months", 12) or 0)
                if ramp_months > 0:
                    t = min(1.0, max(0.0, m_offset / float(ramp_months)))
                    floor0 = float(min_month)
                    floor1 = float(disc_cfg_local.get("min_new_customers_steady", min_month))
                    floor = int(round(floor0 + t * (floor1 - floor0)))
                else:
                    floor = int(min_month)
                target_new = max(target_new, floor)

            # Bootstrap suppression (optional)
            bootstrap_months = int(disc_cfg_local.get("bootstrap_suppression_months", 12) or 0)
            if bootstrap_months > 0 and m_offset < bootstrap_months:
                scale = (m_offset + 1) / float(bootstrap_months)
                target_new = int(round(target_new * scale))

            disc_cfg_local["_target_new_customers"] = int(max(0, target_new))

        # --------------------------------------------------------
        # PARTICIPATION DISTINCT TARGET (OPTIONAL)
        # --------------------------------------------------------
        target_distinct = None
        if use_participation:
            target_distinct = _participation_distinct_target(
                rng=rng,
                m_offset=int(m_offset),
                eligible_count=int(eligible_mask.sum()),
                n_orders=int(m_rows),
                cfg=participation_cfg,
            )

        # --------------------------------------------------------
        # CUSTOMER SAMPLING (discovery/participation aware)
        # --------------------------------------------------------
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
        # PRODUCTS (PER ORDER) - avoid temporary prods array
        # --------------------------------------------------------
        prod_idx = _sample_product_row_indices(
            rng=rng,
            n=n_orders,
            product_np=product_np,
            m_offset=int(m_offset),
            enabled=use_brand_popularity,
        )

        product_keys = product_np[prod_idx, 0].astype(np.int64, copy=False)
        unit_price  = product_np[prod_idx, 1].astype(np.float64, copy=False)
        unit_cost   = product_np[prod_idx, 2].astype(np.float64, copy=False)

        # --------------------------------------------------------
        # STORE → GEO → CURRENCY (guard missing mappings)
        # --------------------------------------------------------
        store_key_arr = store_keys[rng.integers(0, len(store_keys), size=n_orders)]
        geo_arr = st2g_arr[store_key_arr]
        if np.any(geo_arr < 0):
            raise RuntimeError("store_to_geo_arr missing mapping for sampled StoreKey(s)")
        currency_arr = g2c_arr[geo_arr]
        if np.any(currency_arr < 0):
            raise RuntimeError("geo_to_currency_arr missing mapping for sampled GeographyKey(s)")

        # --------------------------------------------------------
        # ORDERS (use month-specific date pool so month loop is real)
        # --------------------------------------------------------
        if not skip_cols:
            order_id_start = base + order_cursor
            if order_id_start + np.int64(n_orders) >= MOD:
                raise RuntimeError("SalesOrderNumber suffix overflow; increase suffix width/capacity.")

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
                order_id_start=int(order_id_start),
            )

            order_cursor += np.int64(n_orders)

            customer_keys_out = orders["customer_keys"]
            order_dates = orders["order_dates"]
            order_ids_int = orders["order_ids_int"]
            line_num = orders["line_num"]

        else:
            customer_keys_out = customer_keys_for_orders
            order_dates = month_date_pool[rng.integers(0, len(month_date_pool), size=n_orders)]
            order_ids_int = None
            line_num = None

        customer_keys_out = np.asarray(customer_keys_out, dtype=np.int64)
        order_dates = np.asarray(order_dates, dtype="datetime64[D]")

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
        promo_keys, _promo_pct = apply_promotions(
            rng=rng,
            n=n_orders,
            order_dates=order_dates,
            promo_keys_all=promo_keys_all,
            promo_pct_all=promo_pct_all,
            promo_start_all=promo_start_all,
            promo_end_all=promo_end_all,
            no_discount_key=no_discount_key,
        )
        promo_keys = np.asarray(promo_keys, dtype=np.int64)

        # --------------------------------------------------------
        # ACTIVITY THINNING
        # --------------------------------------------------------
        keep_mask = apply_activity_thinning(rng=rng, order_dates=order_dates)
        if not keep_mask.any():
            continue

        def _take(x):
            if not isinstance(x, np.ndarray):
                x = np.asarray(x)
            return x[keep_mask]

        customer_keys_out = _take(customer_keys_out)
        product_keys = _take(product_keys)
        unit_price = _take(unit_price)
        unit_cost = _take(unit_cost)
        store_key_arr = _take(store_key_arr)
        currency_arr = _take(currency_arr)
        order_dates = _take(order_dates)
        promo_keys = _take(promo_keys)

        if not skip_cols:
            order_ids_int = _take(order_ids_int)
            line_num = _take(line_num)

        # --------------------------------------------------------
        # UPDATE DISCOVERY STATE (persist)
        # --------------------------------------------------------
        if use_discovery:
            # use post-thinning keys so "seen" means "actually appeared"
            seen_customers.update(map(int, np.unique(customer_keys_out)))
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
            # promo_pct intentionally NOT passed: promo discount must be applied in analysis via PromotionKey join
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
        return _empty_table(schema)

    return pa.concat_tables(tables)
