# ---------------------------------------------------------
#  DIMENSIONS ORCHESTRATOR (CLEAN + DATE-AWARE)
# ---------------------------------------------------------
from pathlib import Path
from typing import Dict, Any, Set, Optional

from src.dimensions.geography import run_geography
from src.dimensions.customers import run_customers
from src.dimensions.stores import run_stores
from src.dimensions.promotions import run_promotions
from src.dimensions.dates import run_dates
from src.dimensions.currency import run_currency
from src.dimensions.exchange_rates import run_exchange_rates
from src.dimensions.products.products import (
    generate_product_dimension as run_products
)

from src.utils.logging_utils import done, skip


# =========================================================
# Helpers
# =========================================================

def _get_defaults_dates(cfg: Dict[str, Any]):
    """
    Return defaults.dates from cfg.
    Supports both 'defaults' and '_defaults' (backward compatibility).
    """
    defaults_section = cfg.get("defaults") or cfg.get("_defaults")
    if not defaults_section:
        return None
    return defaults_section.get("dates")


def _cfg_with_global_dates(
    cfg: Dict[str, Any],
    dim_key: str,
    global_dates,
) -> Dict[str, Any]:
    """
    Return a shallow-copied cfg where cfg[dim_key] is augmented
    with global_dates. Root cfg is never mutated.
    """
    if global_dates is None:
        return cfg

    cfg_for = cfg.copy()
    dim_section = dict(cfg.get(dim_key, {}))
    dim_section["global_dates"] = global_dates
    cfg_for[dim_key] = dim_section
    return cfg_for


def _cfg_for_dimension(
    cfg: Dict[str, Any],
    dim_key: str,
    force: bool,
) -> Dict[str, Any]:
    """
    Return a cfg where ONLY cfg[dim_key] is copied and optionally
    annotated with _force_regenerate.
    """
    cfg_for = cfg.copy()
    dim_section = dict(cfg.get(dim_key, {}))

    if force:
        dim_section["_force_regenerate"] = True

    cfg_for[dim_key] = dim_section
    return cfg_for


def _should_force(dim: str, force_regenerate: Set[str]) -> bool:
    return dim in force_regenerate or "all" in force_regenerate


# =========================================================
# Main Orchestrator
# =========================================================

def generate_dimensions(
    cfg: Dict[str, Any],
    parquet_dims_folder: Path,
    force_regenerate: Optional[Set[str]] = None,
):
    """
    Orchestrates dimension generation in correct dependency order.

    Guarantees:
    - Date-dependent dimensions regenerate when defaults.dates change
    - Non-date-dependent dimensions are isolated from date changes
    - Dependency order is strictly preserved
    - Forced regeneration is runtime-only (no config mutation)
    """
    force_regenerate = force_regenerate or set()

    parquet_dims_folder = Path(parquet_dims_folder).resolve()
    parquet_dims_folder.mkdir(parents=True, exist_ok=True)

    global_dates = _get_defaults_dates(cfg)

    # -----------------------------------------------------
    # 1. Geography
    # -----------------------------------------------------
    run_geography(
        _cfg_for_dimension(
            cfg,
            "geography",
            _should_force("geography", force_regenerate),
        ),
        parquet_dims_folder,
    )

    # -----------------------------------------------------
    # 2. Customers  ✅ FIXED
    # -----------------------------------------------------
    run_customers(
        _cfg_for_dimension(
            cfg,
            "customers",
            _should_force("customers", force_regenerate),
        ),
        parquet_dims_folder,
    )

    # -----------------------------------------------------
    # 3. Stores (date-dependent)
    # -----------------------------------------------------
    cfg_stores = _cfg_with_global_dates(cfg, "stores", global_dates)
    run_stores(
        _cfg_for_dimension(
            cfg_stores,
            "stores",
            _should_force("stores", force_regenerate),
        ),
        parquet_dims_folder,
    )

    # -----------------------------------------------------
    # 4. Promotions (date-dependent)
    # -----------------------------------------------------
    cfg_promotions = _cfg_with_global_dates(cfg, "promotions", global_dates)
    run_promotions(
        _cfg_for_dimension(
            cfg_promotions,
            "promotions",
            _should_force("promotions", force_regenerate),
        ),
        parquet_dims_folder,
    )

    # -----------------------------------------------------
    # 5. Products (static)
    # -----------------------------------------------------
    products = run_products(
        _cfg_for_dimension(
            cfg,
            "products",
            _should_force("products", force_regenerate),
        ),
        parquet_dims_folder,
    )

    if products.get("_regenerated"):
        done("Generating Product Dimension completed")
    else:
        skip("Product Dimension up-to-date; skipping.")

    # -----------------------------------------------------
    # 6. Dates (date-dependent)
    # -----------------------------------------------------
    cfg_dates = _cfg_with_global_dates(cfg, "dates", global_dates)
    run_dates(
        _cfg_for_dimension(
            cfg_dates,
            "dates",
            _should_force("dates", force_regenerate),
        ),
        parquet_dims_folder,
    )

    # -----------------------------------------------------
    # 7. Currency (date-dependent)
    # -----------------------------------------------------
    cfg_currency = _cfg_with_global_dates(cfg, "currency", global_dates)
    run_currency(
        _cfg_for_dimension(
            cfg_currency,
            "currency",
            _should_force("currency", force_regenerate),
        ),
        parquet_dims_folder,
    )

    # -----------------------------------------------------
    # 8. Exchange Rates (date-dependent)
    # -----------------------------------------------------
    cfg_fx = _cfg_with_global_dates(cfg, "exchange_rates", global_dates)
    run_exchange_rates(
        _cfg_for_dimension(
            cfg_fx,
            "exchange_rates",
            _should_force("exchange_rates", force_regenerate),
        ),
        parquet_dims_folder,
    )
