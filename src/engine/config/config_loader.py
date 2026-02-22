from __future__ import annotations

"""
Compatibility shim.

Historically this module duplicated most of the load/normalize logic that also
existed in config.py. It now re-exports the single source of truth from config.py,
so adding a new section/table normalizer only requires changing one file.
"""

from .config import (  # noqa: F401
    apply_acquisition_tuning,
    get_global_dates,
    load_config,
    load_config_file,
    load_pipeline_config,
    normalize_customer_segments_config,
    normalize_defaults,
    normalize_sales_config,
    prepare_paths,
)

__all__ = [
    "load_pipeline_config",
    "load_config",
    "load_config_file",
    "apply_acquisition_tuning",
    "normalize_defaults",
    "get_global_dates",
    "normalize_sales_config",
    "normalize_customer_segments_config",
    "prepare_paths",
]
