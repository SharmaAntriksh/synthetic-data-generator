# ---------------------------------------------------------
#  DIMENSIONS ORCHESTRATOR (CLEAN + DATE-AWARE)
# ---------------------------------------------------------
from pathlib import Path
import copy
from typing import Dict, Any

from src.dimensions.geography import run_geography
from src.dimensions.customers import run_customers
from src.dimensions.stores import run_stores
from src.dimensions.promotions import run_promotions
from src.dimensions.dates import run_dates
from src.dimensions.currency import run_currency
from src.dimensions.exchange_rates import run_exchange_rates


def _get_defaults_dates(cfg: Dict[str, Any]):
    """
    Return the defaults.dates mapping from cfg.
    Support both 'defaults' and '_defaults' keys (backwards compatibility).
    If none found, return None.
    """
    defaults_section = cfg.get("defaults") or cfg.get("_defaults")
    if not defaults_section:
        return None
    return defaults_section.get("dates")


def _cfg_with_global_dates(cfg: Dict[str, Any], dim_key: str):
    """
    Produce a deep copy of cfg where cfg[dim_key] is augmented with
    a stable 'global_dates' entry that contains defaults.dates.

    This copy is used for version checks inside dimension code so that
    changes to defaults.dates will be detected for date-dependent dims.
    """
    cfg_for = copy.deepcopy(cfg)
    gd = _get_defaults_dates(cfg)
    if gd is None:
        # nothing to inject
        return cfg_for

    # Ensure the dimension section exists
    dim_section = cfg_for.get(dim_key, {})
    # Put the global dates under a named key so comparisons are stable
    dim_section["global_dates"] = gd
    cfg_for[dim_key] = dim_section
    return cfg_for


def generate_dimensions(cfg: dict, parquet_dims_folder: Path):
    """
    Orchestrates dimension generation in correct dependency order.

    This orchestrator ensures that:
      - dimensions that depend on the global default date window (like
        dates, exchange_rates, stores, promotions, currency) receive
        a cfg that includes those defaults under 'global_dates', so
        version checks will detect changes to defaults.dates.
      - other dimensions (customers, geography) continue to receive the
        normal cfg and only regenerate when their own section changes.
    """

    # date-dependent dimension keys (adjust if you add more date-sensitive dims)
    date_dependent = {"dates", "exchange_rates", "stores", "promotions", "currency"}

    # 1️⃣ Geography (root) — not date-dependent
    run_geography(cfg, parquet_dims_folder)

    # 2️⃣ Customers (depends on geography) — not date-dependent
    run_customers(cfg, parquet_dims_folder)

    # 3️⃣ Stores (depends on geography) — date-dependent: include defaults.dates
    cfg_for = _cfg_with_global_dates(cfg, "stores")
    run_stores(cfg_for, parquet_dims_folder)

    # 4️⃣ Promotions (may be date-sensitive) — include defaults.dates
    cfg_for = _cfg_with_global_dates(cfg, "promotions")
    run_promotions(cfg_for, parquet_dims_folder)

    # 5️⃣ Dates (obviously date-dependent) — include defaults.dates
    cfg_for = _cfg_with_global_dates(cfg, "dates")
    run_dates(cfg_for, parquet_dims_folder)

    # 6️⃣ Currency (FX master / slicing may depend on dates) — include defaults.dates
    cfg_for = _cfg_with_global_dates(cfg, "currency")
    run_currency(cfg_for, parquet_dims_folder)

    # 7️⃣ Exchange Rates (date-dependent)
    cfg_for = _cfg_with_global_dates(cfg, "exchange_rates")
    run_exchange_rates(cfg_for, parquet_dims_folder)
