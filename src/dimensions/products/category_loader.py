import pandas as pd
from pathlib import Path
from src.utils import info, skip
from src.versioning import should_regenerate, save_version

from src.dimensions.products.seeds.categories import CATEGORIES


def load_category_dimension(config, output_folder: Path):
    p = config["products"]
    version_key = _version_key(p)
    parquet_path = output_folder / "product_category.parquet"

    # Skip if already up-to-date
    if not should_regenerate("product_category", version_key, parquet_path):
        skip("Product Category up-to-date; skipping regeneration")
        return pd.read_parquet(parquet_path)

    info("Loading Product Category")

    df = _load_contoso_category(parquet_path)

    save_version("product_category", version_key, parquet_path)
    return df


# ---------------------------------------------------------
# CONTOSO MODE — PASSTHROUGH
# ---------------------------------------------------------
def _load_contoso_category(parquet_path: Path):
    src = Path("data/contoso_products/product_category.parquet")

    df = pd.read_parquet(src)

    # Write as-is
    df.to_parquet(parquet_path, index=False)
    return df


# ---------------------------------------------------------
# FAKE MODE — Uses taxonomy seed list
# ---------------------------------------------------------
def _generate_fake_category(p, parquet_path: Path):
    selected_cats = list(CATEGORIES.keys())[: p["num_categories"]]

    df = pd.DataFrame({
        "CategoryKey": range(1, len(selected_cats) + 1),
        "Category": selected_cats,
    })

    df.to_parquet(parquet_path, index=False)
    return df


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def _version_key(p):
    return {
        "use_contoso_products": p["use_contoso_products"],
    }
