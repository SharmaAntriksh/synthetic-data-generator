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


def _macro_month_weights(rng: np.random.Generator, T: int, cfg: dict) -> np.ndarray:
    """
    Create base demand weights per month, independent of eligible customer count.
    Produces a smooth trend + seasonality + optional shocks + noise.

    cfg example (models.yaml -> models.macro_demand):
      base_level: 1.0
      yearly_growth: 0.03               # 3% per year
      seasonality_amplitude: 0.12       # +/-12%
      seasonality_phase: 0.0            # radians
      noise_std: 0.05                   # month-to-month
      shock_probability: 0.06           # per month
      shock_impact: [-0.35, -0.10]      # multiplicative range (declines)
    """
    base_level = float(cfg.get("base_level", 1.0))
    yearly_growth = float(cfg.get("yearly_growth", 0.0))
    amp = float(cfg.get("seasonality_amplitude", 0.0))
    phase = float(cfg.get("seasonality_phase", 0.0))
    noise_std = float(cfg.get("noise_std", 0.0))

    shock_p = float(cfg.get("shock_probability", 0.0))
    shock_lo, shock_hi = cfg.get("shock_impact", [-0.25, -0.08])
    shock_lo = float(shock_lo); shock_hi = float(shock_hi)

    m = np.arange(T, dtype="float64")

    # gentle growth: per-month multiplier derived from yearly_growth
    if yearly_growth != 0.0:
        g = (1.0 + yearly_growth) ** (m / 12.0)
    else:
        g = 1.0

    # seasonality: sin wave (12-month cycle)
    if amp != 0.0:
        s = 1.0 + amp * np.sin((2.0 * np.pi * m / 12.0) + phase)
    else:
        s = 1.0

    # month-to-month noise (kept small)
    if noise_std > 0:
        n = rng.normal(loc=1.0, scale=noise_std, size=T)
        n = np.clip(n, 0.5, 1.5)
    else:
        n = 1.0

    # shocks: occasional negative multiplicative hits
    if shock_p > 0:
        shock = np.ones(T, dtype="float64")
        hit = rng.random(T) < shock_p
        if hit.any():
            shock[hit] = 1.0 + rng.uniform(shock_lo, shock_hi, size=int(hit.sum()))
            shock = np.clip(shock, 0.1, 1.0)
    else:
        shock = 1.0

    w = base_level * g * s * n * shock
    w = np.clip(w, 1e-9, None)
    return w / w.sum()


def _normalize_end_month(end_month_arr, n_customers: int) -> np.ndarray:
    """
    Convert nullable end-month representations into an int64 array with -1 meaning "no end inside window".
    Accepts:
      - None -> all -1
      - numpy array of ints -> returned as int64 (negative treated as -1)
      - pandas Int64 series/object array with pd.NA -> converted to -1
    """
    if end_month_arr is None:
        return np.full(n_customers, -1, dtype="int64")

    a = np.asarray(end_month_arr)

    # object arrays may contain pd.NA
    if a.dtype == object:
        out = np.empty(n_customers, dtype="int64")
        for i in range(n_customers):
            v = a[i]
            if v is None or v is np.nan:
                out[i] = -1
            else:
                try:
                    out[i] = int(v)
                except Exception:
                    out[i] = -1
        out[out < 0] = -1
        return out

    # pandas nullable ints often come through as float with nans depending on upstream
    if np.issubdtype(a.dtype, np.floating):
        out = np.where(np.isnan(a), -1, a).astype("int64")
        out[out < 0] = -1
        return out

    out = a.astype("int64", copy=False)
    out[out < 0] = -1
    return out


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


def _eligible_customer_mask_for_month(
    m_offset: int,
    is_active_in_sales: np.ndarray,
    start_month: np.ndarray,
    end_month_norm: np.ndarray,
) -> np.ndarray:
    """
    Returns boolean mask over customers dimension rows, true if eligible in this month.
    """
    # global gate
    mask = (is_active_in_sales == 1)

    # lifecycle start
    mask &= (start_month <= m_offset)

    # lifecycle end: -1 means no end
    has_end = (end_month_norm >= 0)
    mask &= (~has_end) | (m_offset <= end_month_norm)

    return mask


def _participation_distinct_target(
    rng: np.random.Generator,
    m_offset: int,
    eligible_count: int,
    n_orders: int,
    cfg: dict,
) -> int:
    """
    Compute the target number of distinct customers to appear in a given month.

    models.yaml -> models.customer_participation
      base_distinct_ratio: 0.26
      min_distinct_customers: 250
      max_distinct_ratio: 0.55
      cycles:
        enabled: true
        period_months: 24
        amplitude: 0.35
        phase: 0.0
        noise_std: 0.08

    Notes:
      - Returns 0 if eligible_count == 0 or n_orders == 0.
      - Always capped by eligible_count and n_orders.
      - Intended to shape *distinct-customer participation* independently from macro_demand row allocation.
    """
    if eligible_count <= 0 or n_orders <= 0:
        return 0

    base_ratio = float(cfg.get("base_distinct_ratio", 0.0))
    min_k = int(cfg.get("min_distinct_customers", 0))
    max_ratio = float(cfg.get("max_distinct_ratio", 1.0))

    k = eligible_count * base_ratio

    cycles_cfg = cfg.get("cycles", {}) or {}
    if bool(cycles_cfg.get("enabled", False)):
        period = int(cycles_cfg.get("period_months", 24))
        amp = float(cycles_cfg.get("amplitude", 0.0))
        phase = float(cycles_cfg.get("phase", 0.0))
        noise_std = float(cycles_cfg.get("noise_std", 0.0))

        cyc = math.sin((2.0 * math.pi * float(m_offset) / max(period, 1)) + phase)
        mult = 1.0 + (amp * cyc)

        if noise_std > 0:
            mult += float(rng.normal(loc=0.0, scale=noise_std))

        # Keep sane bounds so we don't get negative/huge distinct targets
        mult = float(np.clip(mult, 0.05, 3.0))
        k *= mult

    # hard floor / cap (ratio cap applies to eligible population)
    k = max(k, float(min_k))
    k = min(k, eligible_count * max_ratio)

    # final caps
    k = min(k, float(eligible_count), float(n_orders))

    return int(max(1, round(k)))

def _sample_customers(
    rng: np.random.Generator,
    customer_keys: np.ndarray,
    eligible_mask: np.ndarray,
    seen_set: set,
    n: int,
    use_discovery: bool,
    discovery_cfg: dict,
    base_weight: np.ndarray | None = None,
    target_distinct: int | None = None,
) -> np.ndarray:
    """
    Returns an array of CustomerKeys of length n, sampling from eligible customers.

    Features:
      - Optional discovery forcing (bring in newly-eligible-but-unseen customers).
      - Optional weighted repeat sampling (customer_base_weight).
      - Optional participation control: target_distinct enforces a target number of distinct customers
        to appear in the month, then fills remaining orders with repeats from that distinct pool.

    If use_discovery is True:
      - forces a slice of newly-eligible-but-unseen customers to appear
      - (then) fills the remainder with repeat customers from seen_set (or from eligible if empty)
        unless target_distinct is provided, in which case repeats are drawn from the month distinct pool.
    """
    eligible_keys = customer_keys[eligible_mask]
    if eligible_keys.size == 0 or n <= 0:
        return np.empty(0, dtype=customer_keys.dtype)

    # Normalize target distinct
    if target_distinct is not None:
        try:
            k = int(target_distinct)
        except Exception:
            k = None
        else:
            k = max(1, min(k, int(eligible_keys.size), int(n)))
    else:
        k = None

    # Helper: weighted choice without replacement (numpy supports p + replace=False)
    def _choice_unique(keys: np.ndarray, size: int) -> np.ndarray:
        if size <= 0:
            return np.empty(0, dtype=keys.dtype)
        if base_weight is None:
            return rng.choice(keys, size=size, replace=False)
        w = base_weight[eligible_mask].astype("float64", copy=False)
        # Map w onto the provided keys; fall back to uniform if mapping fails
        try:
            # assumes CustomerKey 1..N
            idx = (keys.astype("int64") - 1)
            ww = base_weight[idx].astype("float64", copy=False)
            ww = np.clip(ww, 1e-12, None)
            p = ww / ww.sum()
            return rng.choice(keys, size=size, replace=False, p=p)
        except Exception:
            return rng.choice(keys, size=size, replace=False)

    # Helper: sample repeats (with replacement) from a pool, optionally weighted by base_weight
    def _choice_repeat(keys: np.ndarray, size: int) -> np.ndarray:
        if size <= 0:
            return np.empty(0, dtype=keys.dtype)
        if base_weight is None:
            return rng.choice(keys, size=size, replace=True)
        try:
            idx = (keys.astype("int64") - 1)
            ww = base_weight[idx].astype("float64", copy=False)
            ww = np.clip(ww, 1e-12, None)
            p = ww / ww.sum()
            return rng.choice(keys, size=size, replace=True, p=p)
        except Exception:
            return rng.choice(keys, size=size, replace=True)

    # -----------------------------
    # No discovery: simple sampling
    # -----------------------------
    if not use_discovery:
        if k is None:
            # legacy behavior
            if base_weight is not None:
                w = base_weight[eligible_mask].astype("float64", copy=False)
                w = np.clip(w, 1e-12, None)
                p = w / w.sum()
                return rng.choice(eligible_keys, size=n, replace=True, p=p)
            return rng.choice(eligible_keys, size=n, replace=True)

        # participation-controlled: build a distinct pool then repeat from it
        distinct_pool = _choice_unique(eligible_keys, size=k)
        remaining = int(n - distinct_pool.size)
        if remaining <= 0:
            out = distinct_pool
            rng.shuffle(out)
            return out

        repeats = _choice_repeat(distinct_pool, size=remaining)
        out = np.concatenate([distinct_pool, repeats])
        rng.shuffle(out)
        return out

    # -----------------------------
    # Discovery mode
    # -----------------------------
    # Determine undiscovered among eligible
    if seen_set:
        seen_arr = np.fromiter(seen_set, dtype=eligible_keys.dtype)
        undiscovered_mask = ~np.isin(eligible_keys, seen_arr, assume_unique=False)
        undiscovered = eligible_keys[undiscovered_mask]
        seen_eligible = eligible_keys[~undiscovered_mask]
    else:
        undiscovered = eligible_keys
        seen_eligible = np.empty(0, dtype=eligible_keys.dtype)

    base_rate = float(discovery_cfg.get("base_discovery_rate", 0.12))
    seasonal_amp = float(discovery_cfg.get("seasonal_amplitude", 0.35))
    seasonal_period = int(discovery_cfg.get("seasonal_period_months", 24))
    min_p = float(discovery_cfg.get("min_discovery_rate", 0.02))
    max_p = float(discovery_cfg.get("max_discovery_rate", 0.60))

    forced = np.empty(0, dtype=customer_keys.dtype)

    if undiscovered.size > 0:
        cycle = np.sin(2 * np.pi * (discovery_cfg.get("_m_offset", 0)) / max(seasonal_period, 1))
        p = base_rate * (1.0 + seasonal_amp * cycle)
        p = float(np.clip(p, min_p, max_p))

        row_scale = np.sqrt(max(n, 1) / 10_000)
        discover_n = int(discovery_cfg.get("_target_new_customers", 1))
        
        # --- HARD CAP: prevent early discovery spike ---
        max_frac = discovery_cfg.get("max_fraction_per_month")
        if max_frac is not None:
            max_new = int(max_frac * customer_keys.size)
            discover_n = min(discover_n, max_new)

        forced = rng.choice(
            undiscovered,
            size=min(discover_n, undiscovered.size),
            replace=False,
        )

    # If no participation target: keep legacy discovery behavior
    if k is None:
        remaining = max(0, n - forced.size)
        if remaining <= 0:
            out = forced
            rng.shuffle(out)
            return out

        # Repeat sampling: prefer seen customers if any, else eligible
        if seen_set:
            repeat_pool = np.fromiter(seen_set, dtype=customer_keys.dtype)
        else:
            repeat_pool = eligible_keys

        if repeat_pool.size == 0:
            out = forced
            rng.shuffle(out)
            return out

        # weighted repeats if possible
        if base_weight is not None and seen_set:
            try:
                idx = (repeat_pool.astype("int64") - 1)
                ww = base_weight[idx].astype("float64", copy=False)
                ww = np.clip(ww, 1e-12, None)
                pp = ww / ww.sum()
                repeat = rng.choice(repeat_pool, size=remaining, replace=True, p=pp)
            except Exception:
                repeat = rng.choice(repeat_pool, size=remaining, replace=True)
        else:
            repeat = rng.choice(repeat_pool, size=remaining, replace=True)

        out = np.concatenate([forced, repeat])
        rng.shuffle(out)
        return out

    # Participation-controlled discovery:
    # Build the month distinct pool of size k, seeded with forced undiscovered customers.
    if forced.size > k:
        forced = rng.choice(forced, size=k, replace=False)

    distinct_pool = forced

    need = int(k - distinct_pool.size)
    if need > 0:
        # fill remaining distinct slots: prefer seen eligible first, then undiscovered
        fill_candidates = []
        if seen_eligible.size > 0:
            fill_candidates.append(seen_eligible)
        if undiscovered.size > 0:
            # exclude those already forced
            if distinct_pool.size > 0:
                u = undiscovered[~np.isin(undiscovered, distinct_pool, assume_unique=False)]
            else:
                u = undiscovered
            if u.size > 0:
                fill_candidates.append(u)

        if fill_candidates:
            pool = np.unique(np.concatenate(fill_candidates))
            if pool.size > 0:
                add_n = min(need, int(pool.size))
                extra = rng.choice(pool, size=add_n, replace=False)
                distinct_pool = np.concatenate([distinct_pool, extra])

    # If we still don't have enough distinct customers (tiny eligible), just use what we have.
    if distinct_pool.size == 0:
        return rng.choice(eligible_keys, size=n, replace=True)

    # Guarantee at least one order per distinct customer, then repeat from distinct pool
    remaining = int(n - distinct_pool.size)
    if remaining <= 0:
        out = distinct_pool
        rng.shuffle(out)
        return out

    repeats = _choice_repeat(distinct_pool, size=remaining)
    out = np.concatenate([distinct_pool, repeats])
    rng.shuffle(out)
    return out

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
    use_macro = bool(macro_cfg)  # if missing, keep old behavior

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

    eligible_nonzero = (eligible_counts > 0)

    if use_macro:
        # base demand weights independent of customer count
        macro_w = _macro_month_weights(rng, T, macro_cfg)

        # months with no eligible customers cannot receive demand
        macro_w = macro_w * eligible_nonzero.astype("float64")
        if macro_w.sum() <= 0:
            return pa.Table.from_arrays([], schema=schema)
        macro_w = macro_w / macro_w.sum()

        # initial allocation
        rows_per_month = np.floor(macro_w * n).astype("int64")

        # ensure we allocate all rows (fix rounding)
        remainder = int(n - rows_per_month.sum())
        if remainder > 0:
            # add remainder to highest-weight months
            add_idx = np.argsort(-macro_w)[:remainder]
            rows_per_month[add_idx] += 1

        # cap early months if eligible base is too small (prevents 2021 "few customers buying 100x/day")
        cap_cfg = macro_cfg.get("early_month_cap", {}) or {}
        cap_enabled = bool(cap_cfg.get("enabled", True))
        per_customer_cap = int(cap_cfg.get("max_rows_per_customer", 12))
        redistribute = bool(cap_cfg.get("redistribute_excess", True))

        if cap_enabled and per_customer_cap > 0:
            excess = 0
            for m in range(T):
                if not eligible_nonzero[m]:
                    continue
                max_rows = int(eligible_counts[m]) * per_customer_cap
                if rows_per_month[m] > max_rows:
                    excess += int(rows_per_month[m] - max_rows)
                    rows_per_month[m] = max_rows

            if redistribute and excess > 0:
                # redistribute excess to later months (or generally to months with capacity)
                capacity = np.maximum(0, (eligible_counts * per_customer_cap).astype("int64") - rows_per_month)
                cap_months = np.nonzero(capacity > 0)[0]
                if cap_months.size > 0:
                    # weight redistribution by macro_w among months with capacity
                    w = macro_w[cap_months]
                    w = w / w.sum()
                    add = rng.multinomial(excess, w)
                    rows_per_month[cap_months] += add
    else:
        # old behavior (backward compatibility)
        month_weights = eligible_counts / eligible_counts.sum()
        rows_per_month = np.maximum(1, (month_weights * n).astype("int64"))


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

        # Tag month offset into discovery config for seasonal cycle
        disc_cfg_local = dict(disc_cfg)
        disc_cfg_local["_m_offset"] = int(m_offset)

        opc = float(disc_cfg_local.get("orders_per_new_customer", 20.0))
        min_month = int(disc_cfg_local.get("min_new_customers_per_month", 0) or 0)

        # demand-driven discovery
        target_new = int(max(1, round(int(m_rows) / max(opc, 1e-9))))

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
