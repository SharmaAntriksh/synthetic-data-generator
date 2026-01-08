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

    Ensures ordering and proper regeneration.
    """

    info("Starting Product Dimension")

    df_cat = load_static_dimension(
        name="product_category",
        src_path=Path("data/contoso_products/product_category.parquet"),
        output_path=output_folder / "product_category.parquet",
    )

    df_sub = load_static_dimension(
        name="product_subcategory",
        src_path=Path("data/contoso_products/product_subcategory.parquet"),
        output_path=output_folder / "product_subcategory.parquet",
    )

    df_prod = load_product_dimension(config, output_folder)

    return {
        "category": df_cat,
        "subcategory": df_sub,
        "product": df_prod,
    }

