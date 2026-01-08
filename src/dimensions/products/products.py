from pathlib import Path
from src.utils import info

from src.dimensions.static_loader import load_static_dimension
from .product_loader import load_product_dimension


def generate_product_dimension(config, output_folder: Path):
    """
    Orchestrates the full Product dimension:
    1. Product Category (static)
    2. Product Subcategory (static)
    3. Product (scalable)
    """

    started = False

    # ---------------- Product Category ----------------
    df_cat, regen_cat = load_static_dimension(
        name="product_category",
        src_path=Path("data/contoso_products/product_category.parquet"),
        output_path=output_folder / "product_category.parquet",
    )
    if regen_cat and not started:
        info("Starting Product Dimension")
        started = True

    # ---------------- Product Subcategory ----------------
    df_sub, regen_sub = load_static_dimension(
        name="product_subcategory",
        src_path=Path("data/contoso_products/product_subcategory.parquet"),
        output_path=output_folder / "product_subcategory.parquet",
    )
    if regen_sub and not started:
        info("Starting Product Dimension")
        started = True

    # ---------------- Products ----------------
    df_prod, regen_prod = load_product_dimension(config, output_folder)
    if regen_prod and not started:
        info("Starting Product Dimension")
        started = True

    return {
        "category": df_cat,
        "subcategory": df_sub,
        "product": df_prod,
    }
