"""
Sales models package.

Provides the three main data-shaping functions consumed by chunk_builder:
  - apply_activity_thinning  (month-to-month row density)
  - build_quantity           (items per order)
  - build_prices             (inflation drift + snapping + markdown)

And customer lifecycle helpers:
  - build_eligibility_by_month
  - apply_activity_overlay_by_month
  - build_active_customer_pool   (backward-compat shim)
  - ActivityOverlayCfg
"""
from .activity_model import apply_activity_thinning
from .customer_lifecycle import (
    ActivityOverlayCfg,
    apply_activity_overlay_by_month,
    build_active_customer_pool,
    build_eligibility_by_month,
)
from .pricing_pipeline import build_prices
from .quantity_model import build_quantity

__all__ = [
    "apply_activity_thinning",
    "ActivityOverlayCfg",
    "build_eligibility_by_month",
    "apply_activity_overlay_by_month",
    "build_active_customer_pool",
    "build_prices",
    "build_quantity",
]
