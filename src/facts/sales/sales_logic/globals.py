"""Sales runtime state + schema binding.

This module is imported by worker processes; keep it lightweight and deterministic.

The ``State`` class remains the canonical process-local singleton for
multiprocessing workers.  ``SalesContext`` is the new dependency-injection
friendly dataclass that makes dependencies explicit.  Use
``SalesContext.from_state()`` to snapshot the current ``State`` into a
context object, which can then be passed through function parameters
instead of relying on the global.

Migration path:
    1. New/refactored functions accept ``ctx: SalesContext`` as their
       first parameter.
    2. Legacy code continues to read from ``State`` directly.
    3. Over time, functions are converted to use ``ctx`` and ``State``
       access is phased out.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import pyarrow as pa  # type: ignore
except Exception:  # pragma: no cover
    pa = None

from src.utils.static_schemas import get_sales_schema

PA_AVAILABLE = pa is not None


# ===============================================================
# Schema helpers
# ===============================================================

def _sql_to_pa_type(sql_type: str):
    """
    Map SQL-ish type strings (from static_schemas) to PyArrow types.

    Must remain aligned with chunk_builder output dtypes.
    """
    t = str(sql_type).upper()

    # Order matters: BIGINT must be checked before INT, etc.
    if "BIGINT" in t:
        return pa.int64()

    # Prefer tighter mapping than the old "SMALLINT or TINYINT => int16"
    # because some flags/month columns are intentionally int8 downstream.
    if "TINYINT" in t:
        return pa.int8()
    if "SMALLINT" in t:
        return pa.int16()

    if "INT" in t:
        return pa.int32()

    # Keep numeric types as float64 for stability (DECIMAL varies in real systems)
    if "DECIMAL" in t or "NUMERIC" in t or "FLOAT" in t or "REAL" in t or "DOUBLE" in t:
        return pa.float64()

    if "DATE" in t:
        return pa.date32()

    # Default: string
    return pa.string()


def _logical_to_arrow_schema(logical_schema):
    """
    Convert logical (name, sql_type) schema from static_schemas into a PyArrow schema.
    """
    if not PA_AVAILABLE:
        raise RuntimeError("pyarrow is required to build Arrow schema")

    fields = []
    for name, sql_type in logical_schema:
        fields.append(pa.field(str(name), _sql_to_pa_type(sql_type)))
    return pa.schema(fields)


# ===============================================================
# Dependency-injection context (explicit alternative to State)
# ===============================================================

@dataclass
class SalesContext:
    """Explicit, testable container for all sales worker dependencies.

    Every field that ``State`` exposes as a class variable is represented
    here as a typed dataclass field.  Use ``SalesContext.from_state()``
    to snapshot the current global ``State`` into a portable context.
    """

    # -- Dimension data --
    product_np: Any = None
    active_product_np: Any = None
    customer_keys: Any = None
    customer_is_active_in_sales: Any = None
    customer_start_month: Any = None
    customer_end_month: Any = None
    customer_base_weight: Any = None
    seen_customers: Any = field(default_factory=set)
    date_pool: Any = None
    date_prob: Any = None
    store_keys: Any = None

    # -- Promotions --
    promo_keys_all: Any = None
    promo_start_all: Any = None
    promo_end_all: Any = None
    new_customer_promo_keys: Any = None
    new_customer_window_months: int = 3

    # -- Mappings --
    store_to_product_rows: Any = None
    store_to_geo_arr: Any = None
    geo_to_currency_arr: Any = None
    models_cfg: Optional[Dict[str, Any]] = None

    # -- Output config --
    file_format: Optional[str] = None
    out_folder: Optional[str] = None
    chunk_size: Optional[int] = None
    row_group_size: Optional[int] = None
    compression: Optional[str] = None
    order_id_stride_orders: Optional[int] = None
    skip_order_cols: Optional[bool] = None
    skip_order_cols_requested: Optional[bool] = None
    max_lines_per_order: int = 6

    # -- Delta / partitioning --
    no_discount_key: Any = None
    delta_output_folder: Optional[str] = None
    write_delta: Optional[bool] = None
    partition_enabled: Optional[bool] = None
    partition_cols: Optional[List[str]] = None

    # -- Budget --
    budget_enabled: Optional[bool] = None
    budget_store_to_country: Any = None
    budget_product_to_cat: Any = None

    # -- Schema --
    sales_schema: Any = None

    @classmethod
    def from_state(cls) -> "SalesContext":
        """Snapshot the current ``State`` singleton into a ``SalesContext``."""
        fields = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {}
        for name in fields:
            val = getattr(State, name, None)
            if val is not None:
                kwargs[name] = val
        return cls(**kwargs)


# ===============================================================
# Global Sales runtime state (process-local)
# ===============================================================

class State:
    """
    Shared global state for Sales runtime only.

    Holds cached dimension data, promotion context, and output configuration.

    Notes:
    - Process-local (safe with multiprocessing)
    - Sealed after initialization (bind_globals refuses mutation once sealed)
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
    promo_start_all = None
    promo_end_all = None
    new_customer_promo_keys = None
    new_customer_window_months = 3

    # --------------------------------------------------------------
    # Mappings
    # --------------------------------------------------------------
    store_to_product_rows = None  # assortment: list[StoreKey] -> np.ndarray of product row indices

    # --------------------------------------------------------------
    # Budget streaming aggregation (worker-side lookups)
    # --------------------------------------------------------------
    budget_enabled = None
    budget_store_to_country = None   # dense int32 array: StoreKey -> country_id
    budget_product_to_cat = None     # dense int32 array: ProductKey -> category_id
    
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

    # CRITICAL: constant per-run stride for chunk order-id ranges.
    # Also controls output chunking (row count per chunk file).
    # (task.py validates this; chunk_builder uses it to avoid overlaps)
    chunk_size = None

    row_group_size = None
    compression = None

    # Forward-compat aliases for SalesOrderNumber generation
    order_id_stride_orders = None      # usually == chunk_size

    # used by task.py when deciding to drop order cols in Sales output
    skip_order_cols_requested = None
    
    max_lines_per_order = 6

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

    # These may be injected by worker init for debugging/inspection.
    schema_no_order = None
    schema_with_order = None
    schema_no_order_delta = None
    schema_with_order_delta = None

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
        Prevent further mutation of State via bind_globals().
        Called once during worker initialization.
        """
        if PA_AVAILABLE and State.sales_schema is None:
            raise RuntimeError("State.sales_schema was not bound before sealing")
        State._sealed = True


# ===============================================================
# Binding
# ===============================================================

def bind_globals(gdict: dict):
    """
    Bind values into State and finalize the Sales Arrow schema.

    Must be called before workers start (per-process).
    """
    if State._sealed:
        raise RuntimeError("State is sealed and cannot be modified")

    if not isinstance(gdict, dict):
        raise TypeError("bind_globals expects a dict")

    # Bind raw values (allow injecting additional attrs for debugging)
    for k, v in gdict.items():
        setattr(State, k, v)

    # Ensure seen_customers exists (chunk_builder will only use it if discovery enabled)
    sc = getattr(State, "seen_customers", None)
    if sc is None:
        State.seen_customers = set()
    elif not isinstance(sc, set):
        # tolerate list/tuple/np arrays being passed by caller
        try:
            State.seen_customers = set(sc)
        except Exception:
            State.seen_customers = set()

    # --------------------------------------------------------------
    # Bind Sales schema ONCE, respecting skip_order_cols
    # (worker may pass an explicit sales_schema; if so, don't override)
    # --------------------------------------------------------------
    if PA_AVAILABLE and State.sales_schema is None:
        if State.skip_order_cols is None:
            raise RuntimeError("skip_order_cols must be bound before Sales schema initialization")

        logical_schema = get_sales_schema(bool(State.skip_order_cols))
        State.sales_schema = _logical_to_arrow_schema(logical_schema)


# ===============================================================
# Date formatting
# ===============================================================

def fmt(dt):
    """
    Format datetime64[D] as YYYYMMDD string array (fast path).

    Accepts scalar or array-like.
    """
    d = np.asarray(dt).astype("datetime64[D]", copy=False)

    # Extract Y/M/D in a vectorized way
    y = d.astype("datetime64[Y]").astype("int64") + 1970
    m = (
        d.astype("datetime64[M]").astype("int64")
        - d.astype("datetime64[Y]").astype("datetime64[M]").astype("int64")
        + 1
    )
    day = (d - d.astype("datetime64[M]")).astype("timedelta64[D]").astype("int64") + 1

    yyyymmdd = (y * 10000 + m * 100 + day).astype("int64")
    return yyyymmdd.astype(str)


__all__ = [
    "SalesContext",
    "State",
    "bind_globals",
    "fmt",
    "PA_AVAILABLE",
]
