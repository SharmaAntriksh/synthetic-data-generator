"""
Sales models package.

Provides the two main data-shaping functions consumed by chunk_builder:
  - build_quantity           (items per order)
  - build_prices             (inflation drift + snapping + markdown)
"""
from .pricing_pipeline import build_prices
from .quantity_model import build_quantity

__all__ = [
    "build_prices",
    "build_quantity",
]
