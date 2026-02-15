# ---------------------------------------------------------
#  DIMENSIONS ORCHESTRATOR (CLEAN + DATE-AWARE)
# ---------------------------------------------------------
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Set, Optional, Tuple

from src.dimensions.geography import run_geography
from src.dimensions.customers import run_customers
from src.dimensions.stores import run_stores
from src.dimensions.promotions import run_promotions
from src.dimensions.dates import run_dates
from src.dimensions.currency import run_currency
from src.dimensions.exchange_rates import run_exchange_rates
from src.dimensions.products.products import (
    generate_product_dimension as run_products,
)

from src.utils.logging_utils import done, skip, info
from src.dimensions.return_reasons import run_return_reasons
from src.dimensions.customer_segments import run_customer_segments

# =========================================================
# Helpers
# =========================================================

DATE_DEPENDENT_DIMS = {
    "customers",        # NEW: lifecycle uses defaults.dates (start/end months)
    "stores",
    "promotions",
    "dates",
    "currency",
    "exchange_rates",
    "customer_segments",
}

STATIC_DIMS = {
    "geography",
    "products",
    "return_reason",
}


def _get_defaults_dates(cfg: Dict[str, Any]):
    """
    Return defaults.dates from cfg.
    Supports both 'defaults' and '_defaults' (backward compatibility).
    """
    defaults_section = cfg.get("defaults") or cfg.get("_defaults")
    if not defaults_section:
        return None
    return defaults_section.get("dates")


def _cfg_with_global_dates(cfg: Dict[str, Any], dim_key: str, global_dates) -> Dict[str, Any]:
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


def _cfg_for_dimension(cfg: Dict[str, Any], dim_key: str, force: bool) -> Dict[str, Any]:
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


def _expand_force_for_date_dependencies(force_regenerate: Set[str]) -> Set[str]:
    """
    If the user forces any date-dependent dimension (or explicitly forces 'dates'),
    it is usually correct to regenerate all date-dependent dimensions to keep a
    consistent timeline.

    You can still force a subset manually by specifying exactly those keys,
    but forcing 'dates' implies the rest should be refreshed unless the user
    explicitly disables that behavior (not supported here; we keep it safe).
    """
    if "all" in force_regenerate:
        return force_regenerate

    # If dates is forced, expand to all date-dependent dims
    if "dates" in force_regenerate:
        return set(force_regenerate) | set(DATE_DEPENDENT_DIMS)

    return force_regenerate


def _returns_enabled(cfg: Dict[str, Any]) -> bool:
    returns_cfg = cfg.get("returns") if isinstance(cfg.get("returns"), dict) else {}
    enabled = bool(returns_cfg.get("enabled", False))

    facts = cfg.get("facts") if isinstance(cfg.get("facts"), dict) else {}
    facts_enabled = facts.get("enabled", [])
    if isinstance(facts_enabled, list) and facts_enabled:
        enabled = enabled and ("returns" in {str(x).strip().lower() for x in facts_enabled})

    return bool(enabled)

def _customer_segments_enabled(cfg: Dict[str, Any]) -> bool:
    seg_cfg = cfg.get("customer_segments") if isinstance(cfg.get("customer_segments"), dict) else {}
    return bool(seg_cfg.get("enabled", False))


# =========================================================
# Main Orchestrator
# =========================================================

def generate_dimensions(
    cfg: Dict[str, Any],
    parquet_dims_folder: Path,
    force_regenerate: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Orchestrates dimension generation in correct dependency order.

    Guarantees:
    - Date-dependent dimensions regenerate consistently when timeline is forced
    - Dependency order is strictly preserved
    - Forced regeneration is runtime-only (no config mutation)

    Returns:
      Summary dict (useful for UI/CLI):
        {
          "global_dates": {...} | None,
          "folder": "<abs path>",
          "regenerated": {dim_name: bool, ...}
        }
    """
    force_regenerate = set(force_regenerate or set())
    force_regenerate = _expand_force_for_date_dependencies(force_regenerate)
    
    if "customers" in force_regenerate:
        force_regenerate.add("customer_segments")

    parquet_dims_folder = Path(parquet_dims_folder).resolve()
    parquet_dims_folder.mkdir(parents=True, exist_ok=True)

    global_dates = _get_defaults_dates(cfg)

    if global_dates is not None:
        info(f"Using global dates: start={global_dates.get('start')} end={global_dates.get('end')}")

    regenerated: Dict[str, bool] = {}

    # -----------------------------------------------------
    # 1. Geography (static; upstream dependency for stores)
    # -----------------------------------------------------
    run_geography(
        _cfg_for_dimension(
            cfg,
            "geography",
            _should_force("geography", force_regenerate),
        ),
        parquet_dims_folder,
    )
    # run_geography doesn't consistently return status, so mark only when forced
    regenerated["geography"] = _should_force("geography", force_regenerate)

    # -----------------------------------------------------
    # 2. Customers (NOW date-dependent due to lifecycle months)
    # -----------------------------------------------------
    cfg_customers = _cfg_with_global_dates(cfg, "customers", global_dates)
    run_customers(
        _cfg_for_dimension(
            cfg_customers,
            "customers",
            _should_force("customers", force_regenerate),
        ),
        parquet_dims_folder,
    )
    regenerated["customers"] = _should_force("customers", force_regenerate)
    # -----------------------------------------------------
    # 2.5 Customer Segments (depends on Customers; date-dependent if validity enabled)
    # -----------------------------------------------------
    force_segs = _should_force("customer_segments", force_regenerate)

    if _customer_segments_enabled(cfg) or force_segs:
        cfg_segs = _cfg_with_global_dates(cfg, "customer_segments", global_dates)
        run_customer_segments(
            _cfg_for_dimension(cfg_segs, "customer_segments", force_segs),
            parquet_dims_folder,
        )
        regenerated["customer_segments"] = force_segs
    else:
        regenerated["customer_segments"] = False

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
    regenerated["stores"] = _should_force("stores", force_regenerate)

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
    regenerated["promotions"] = _should_force("promotions", force_regenerate)

    # -----------------------------------------------------
    # 4.5 Return Reasons (static-ish; only if returns enabled)
    # -----------------------------------------------------
    force_rr = _should_force("return_reason", force_regenerate)
    if _returns_enabled(cfg) or force_rr:
        run_return_reasons(
            _cfg_for_dimension(
                cfg,
                "return_reason",   # config section name; used only for _force_regenerate
                force_rr,
            ),
            parquet_dims_folder,
        )
        regenerated["return_reason"] = force_rr
    else:
        regenerated["return_reason"] = False

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

    if isinstance(products, dict) and products.get("_regenerated"):
        done("Generating Product Dimension completed")
        regenerated["products"] = True
    else:
        if _should_force("products", force_regenerate):
            # If forced but generator didn't report, still mark as forced for visibility
            regenerated["products"] = True
            done("Generating Product Dimension completed (forced)")
        else:
            regenerated["products"] = False
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
    regenerated["dates"] = _should_force("dates", force_regenerate)

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
    regenerated["currency"] = _should_force("currency", force_regenerate)

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
    regenerated["exchange_rates"] = _should_force("exchange_rates", force_regenerate)

    return {
        "global_dates": global_dates,
        "folder": str(parquet_dims_folder),
        "regenerated": regenerated,
    }
