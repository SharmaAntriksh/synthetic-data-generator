import numpy as np
import pyarrow as pa

from src.utils.static_schemas import get_sales_schema

PA_AVAILABLE = pa is not None


def _logical_to_arrow_schema(logical_schema):
    """
    Convert logical (name, sql_type) schema from static_schemas
    into a PyArrow schema.

    Must stay aligned with chunk_builder output dtypes.
    """
    fields = []

    for name, sql_type in logical_schema:
        t = str(sql_type).upper()

        if "BIGINT" in t:
            pa_type = pa.int64()
        elif "SMALLINT" in t or "TINYINT" in t:
            # NOTE: your pipeline may use int8 for some flags; keep schema source-of-truth in static_schemas
            pa_type = pa.int16()
        elif "INT" in t:
            pa_type = pa.int32()
        elif "DECIMAL" in t or "FLOAT" in t:
            pa_type = pa.float64()
        elif "DATE" in t:
            pa_type = pa.date32()
        else:
            pa_type = pa.string()

        fields.append(pa.field(name, pa_type))

    return pa.schema(fields)


class State:
    """
    Shared global state for Sales runtime only.

    Holds cached dimension data, promotion context,
    and output configuration.

    NOTE:
    - Process-local (safe with multiprocessing)
    - Sealed after initialization
    """

    # --------------------------------------------------------------
    # Internal control
    # --------------------------------------------------------------
    _sealed = False

    # --------------------------------------------------------------
    # Core runtime flags / data
    # --------------------------------------------------------------
    skip_order_cols = None

    product_np = None
    active_product_np = None

    # Backward-compat customer key pool
    customers = None

    # New lifecycle-aware customer dimension arrays (aligned by row index)
    customer_keys = None
    customer_is_active_in_sales = None
    customer_start_month = None
    customer_end_month = None  # int64 with -1 meaning "no end"
    customer_base_weight = None  # optional float64

    # Discovery persistence (optional)
    seen_customers = None

    date_pool = None
    date_prob = None
    store_keys = None

    models_cfg = None

    # --------------------------------------------------------------
    # Promotions
    # --------------------------------------------------------------
    promo_keys_all = None
    promo_pct_all = None
    promo_start_all = None
    promo_end_all = None

    # --------------------------------------------------------------
    # Mappings
    # --------------------------------------------------------------
    store_to_geo_arr = None
    geo_to_currency_arr = None

    # (kept for compatibility; may be passed as dicts too)
    store_to_geo = None
    geo_to_currency = None

    # --------------------------------------------------------------
    # Output configuration
    # --------------------------------------------------------------
    file_format = None
    out_folder = None
    row_group_size = None
    compression = None

    # parquet tuning
    parquet_dict_exclude = None

    # --------------------------------------------------------------
    # Delta options
    # --------------------------------------------------------------
    no_discount_key = None
    delta_output_folder = None
    write_delta = None

    # --------------------------------------------------------------
    # Partitioning
    # --------------------------------------------------------------
    partition_enabled = None
    partition_cols = None

    # --------------------------------------------------------------
    # Schema (bound once per run)
    # --------------------------------------------------------------
    sales_schema = None

    # --------------------------------------------------------------
    # Lifecycle helpers
    # --------------------------------------------------------------
    @staticmethod
    def reset():
        """
        Reset all State fields.
        Intended for tests / development only.
        """
        for key in list(vars(State).keys()):
            if key.startswith("__"):
                continue
            # keep methods / functions / descriptors intact
            attr = getattr(State, key)
            if callable(attr):
                continue
            setattr(State, key, None)
        State._sealed = False

    @staticmethod
    def validate(required):
        """
        Validate required state fields exist (not None).
        """
        missing = [r for r in required if getattr(State, r, None) is None]
        if missing:
            raise RuntimeError(f"Missing State fields: {missing}")

    @staticmethod
    def seal():
        """
        Prevent further mutation of State.
        Called once during worker initialization.
        """
        if PA_AVAILABLE and State.sales_schema is None:
            raise RuntimeError("State.sales_schema was not bound before sealing")
        State._sealed = True


def bind_globals(gdict: dict):
    """
    Bind values into State and finalize the Sales Arrow schema.

    This must be called exactly once per process
    before workers start.
    """
    if State._sealed:
        raise RuntimeError("State is sealed and cannot be modified")

    # Bind raw values
    for k, v in gdict.items():
        setattr(State, k, v)

    # Ensure seen_customers is initialized if discovery is enabled
    if getattr(State, "seen_customers", None) is None:
        State.seen_customers = set()

    # --------------------------------------------------------------
    # Bind Sales schema ONCE, respecting skip_order_cols
    # --------------------------------------------------------------
    if PA_AVAILABLE and State.sales_schema is None:
        if State.skip_order_cols is None:
            raise RuntimeError(
                "skip_order_cols must be bound before Sales schema initialization"
            )

        logical_schema = get_sales_schema(State.skip_order_cols)
        State.sales_schema = _logical_to_arrow_schema(logical_schema)


def fmt(dt):
    """
    Format datetime64[D] as YYYYMMDD string array.
    """
    return np.char.replace(np.datetime_as_string(dt, unit="D"), "-", "")


__all__ = [
    "State",
    "bind_globals",
    "fmt",
    "PA_AVAILABLE",
]
