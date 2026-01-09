import numpy as np
import pyarrow as pa

PA_AVAILABLE = pa is not None


class State:
    """
    Shared global state for Sales runtime only.

    Holds cached dimension data, promotion context,
    and output configuration.

    NOTE:
    - This is process-local (safe with multiprocessing)
    - It is sealed after initialization to prevent mutation
    """

    # ------------------------------------------------------------------
    # Internal control
    # ------------------------------------------------------------------
    _sealed = False

    # ------------------------------------------------------------------
    # Core runtime data
    # ------------------------------------------------------------------
    skip_order_cols = None
    product_np = None
    customers = None
    date_pool = None
    date_prob = None
    store_keys = None

    # ------------------------------------------------------------------
    # Promotions
    # ------------------------------------------------------------------
    promo_keys_all = None
    promo_pct_all = None
    promo_start_all = None
    promo_end_all = None

    # ------------------------------------------------------------------
    # Mappings
    # Dense arrays are authoritative at runtime
    # Dict versions are kept for fallback / debug
    # ------------------------------------------------------------------
    store_to_geo_arr = None
    geo_to_currency_arr = None
    store_to_geo = None
    geo_to_currency = None

    # ------------------------------------------------------------------
    # Output configuration
    # ------------------------------------------------------------------
    file_format = None
    out_folder = None
    row_group_size = None
    compression = None

    # ------------------------------------------------------------------
    # Delta options
    # ------------------------------------------------------------------
    no_discount_key = None
    delta_output_folder = None
    write_delta = None

    # ------------------------------------------------------------------
    # Partitioning
    # ------------------------------------------------------------------
    partition_enabled = None
    partition_cols = None

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------
    @staticmethod
    def reset():
        """
        Reset all state fields.

        Intended for tests or development only.
        Also unseals the State.
        """
        for key, val in list(vars(State).items()):
            if not key.startswith("__") and not callable(val):
                setattr(State, key, None)
        State._sealed = False

    @staticmethod
    def validate(required):
        """
        Validate that required state fields are populated.
        """
        missing = [r for r in required if getattr(State, r, None) is None]
        if missing:
            raise RuntimeError(f"Missing State fields: {missing}")

    @staticmethod
    def seal():
        """
        Prevent further mutation of State.
        Should be called once during worker initialization.
        """
        State._sealed = True


def bind_globals(gdict: dict):
    """
    Bind values into State.

    Raises if State has already been sealed.
    """
    if State._sealed:
        raise RuntimeError("State is sealed and cannot be modified")

    for k, v in gdict.items():
        setattr(State, k, v)


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
