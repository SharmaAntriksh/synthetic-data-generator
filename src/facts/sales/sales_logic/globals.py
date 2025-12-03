import numpy as np
import pandas as pd
import pyarrow as pa

PA_AVAILABLE = pa is not None

# ============================================================
# GLOBAL STATE USED BY WORKERS + CHUNK BUILDER
# ============================================================

_G_skip_order_cols = None
_G_product_np = None
_G_customers = None
_G_date_pool = None
_G_date_prob = None
_G_store_keys = None

_G_promo_keys_all = None
_G_promo_pct_all = None
_G_promo_start_all = None
_G_promo_end_all = None

# Dense mapping arrays (fast path)
_G_store_to_geo_arr = None
_G_geo_to_currency_arr = None

# Fallback dict mapping
_G_store_to_geo = None
_G_geo_to_currency = None

# Output + format settings
_G_file_format = None
_G_out_folder = None
_G_row_group_size = None
_G_compression = None

# Extra worker settings
_G_no_discount_key = None
_G_delta_output_folder = None
_G_write_delta = None

_G_partition_enabled = None
_G_partition_cols = None

# Utility
_fmt = lambda dt: np.char.replace(np.datetime_as_string(dt, unit='D'), "-", "")

# Worker initializer injects all globals here
def bind_globals(gdict):
    globals().update(gdict)


__all__ = [
    "PA_AVAILABLE",
    "_G_skip_order_cols",
    "_G_product_np", "_G_customers",
    "_G_date_pool", "_G_date_prob", "_G_store_keys",
    "_G_promo_keys_all", "_G_promo_pct_all",
    "_G_promo_start_all", "_G_promo_end_all",
    "_G_store_to_geo_arr", "_G_geo_to_currency_arr",
    "_G_store_to_geo", "_G_geo_to_currency",

    "_G_file_format", "_G_out_folder",
    "_G_row_group_size", "_G_compression",

    "_G_no_discount_key",
    "_G_delta_output_folder",
    "_G_write_delta",

    "_G_partition_enabled",
    "_G_partition_cols",

    "_fmt", "bind_globals"
]
