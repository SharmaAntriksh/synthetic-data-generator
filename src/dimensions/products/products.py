from pathlib import Path
from src.utils import info
from src.utils.logging_utils import stage

from .static_loader import load_static_dimension
from .generator import load_product_dimension


def generate_product_dimension(config, output_folder: Path):
    """
    Orchestrates the full Product dimension:
    1. Product Category (static)
    2. Product Subcategory (static)
    3. Product (scalable) + ProductProfile
    """

    with stage("Generating Products", lazy=True):
        df_cat, regen_cat = load_static_dimension(
            name="product_category",
            src_path=Path("data/contoso_products/product_category.parquet"),
            output_path=output_folder / "product_category.parquet",
        )
        if regen_cat:
            info(f"Product Category: {len(df_cat):,} rows")

        df_sub, regen_sub = load_static_dimension(
            name="product_subcategory",
            src_path=Path("data/contoso_products/product_subcategory.parquet"),
            output_path=output_folder / "product_subcategory.parquet",
        )
        if regen_sub:
            info(f"Product Subcategory: {len(df_sub):,} rows")

        df_prod, df_prod_profile, regen_prod = load_product_dimension(
            config, output_folder, log_skip=False,
        )
        if regen_prod:
            info(f"Products: {len(df_prod):,} rows")
            info(f"Product Profile: {len(df_prod_profile):,} rows")

    regenerated = regen_cat or regen_sub or regen_prod

    return {
        "category": df_cat,
        "subcategory": df_sub,
        "product": df_prod,
        "product_profile": df_prod_profile,
        "_regenerated": regenerated,
    }
