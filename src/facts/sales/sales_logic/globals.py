import numpy as np
import pandas as pd
import pyarrow as pa

PA_AVAILABLE = pa is not None


class State:
    """
    Shared global state container for sales logic.
    All modules reference the same class attributes.
    """

    # core
    skip_order_cols = None
    product_np = None
    customers = None
    date_pool = None
    date_prob = None
    store_keys = None

    # promotions
    promo_keys_all = None
    promo_pct_all = None
    promo_start_all = None
    promo_end_all = None

    # mappings
    store_to_geo_arr = None
    geo_to_currency_arr = None
    store_to_geo = None
    geo_to_currency = None

    # output config
    file_format = None
    out_folder = None
    row_group_size = None
    compression = None

    # delta options
    no_discount_key = None
    delta_output_folder = None
    write_delta = None

    # partitioning
    partition_enabled = None
    partition_cols = None

    @staticmethod
    def reset():
        """
        Reset all state fields (useful for tests or dev).
        """
        for key, val in list(vars(State).items()):
            if not key.startswith("__") and not callable(val):
                setattr(State, key, None)

    @staticmethod
    def validate(required):
        """
        Validate that required state fields are populated.
        """
        for r in required:
            if getattr(State, r) is None:
                raise RuntimeError(f"State.{r} is not set")


def bind_globals(gdict: dict):
    for k, v in gdict.items():
        setattr(State, k, v)


def fmt(dt):
    return np.char.replace(np.datetime_as_string(dt, unit='D'), "-", "")


__all__ = ["State", "bind_globals", "fmt", "PA_AVAILABLE"]
