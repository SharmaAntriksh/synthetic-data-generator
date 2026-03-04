# ui/constants.py
"""
Single source of truth for shared constants across the UI layer.
"""

# Dimensions that can be force-regenerated from the UI.
# Used by both regenerate.py and generate.py.
REGENERATABLE_DIMENSIONS: list[str] = [
    "customers",
    "products",
    "stores",
    "geography",
    "promotions",
    "dates",
    "currency",
    "exchange_rates",
]

# Dimension config keys that map to a "size" number_input in the UI.
DIMENSION_SIZE_FIELDS: dict[str, str] = {
    "customers": "total_customers",
    "products": "num_products",
    "stores": "num_stores",
}
