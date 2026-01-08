import pandas as pd
from pathlib import Path
from src.utils import info, skip
from src.versioning import should_regenerate, save_version

from src.dimensions.products.seeds.categories import CATEGORIES


def load_subcategory_dimension(config, output_folder: Path):
    p = config["products"]
    version_key = _version_key(p)
    parquet_path = output_folder / "product_subcategory.parquet"

    if not should_regenerate("product_subcategory", version_key, parquet_path):
        skip("Product Subcategory up-to-date; skipping regeneration")
        return pd.read_parquet(parquet_path)

    info("Loading Product Subcategory")

    df = _load_contoso_subcategory(output_folder, parquet_path)

    save_version("product_subcategory", version_key, parquet_path)
    return df


# ---------------------------------------------------------
# CONTOSO MODE — PASSTHROUGH
# ---------------------------------------------------------
def _load_contoso_subcategory(output_folder: Path, parquet_path: Path):
    src = Path("data/contoso_products/product_subcategory.parquet")

    df = pd.read_parquet(src)

    # Write as-is
    df.to_parquet(parquet_path, index=False)
    return df


# ---------------------------------------------------------
# FAKE MODE
# ---------------------------------------------------------
def _generate_fake_subcategory(config, output_folder: Path, parquet_path: Path):
    df_cat = pd.read_parquet(output_folder / "product_category.parquet")
    p = config["products"]

    rows = []
    sub_key = 1

    for _, row in df_cat.iterrows():
        category_key = row["CategoryKey"]
        category_name = row["Category"]

        subs = CATEGORIES[category_name]

        for sub in subs:
            rows.append({
                "SubcategoryKey": sub_key,
                "CategoryKey": category_key,
                "Subcategory": sub,
                "Category": category_name   # ← IMPORTANT
            })
            sub_key += 1

    df = pd.DataFrame(rows)
    df.to_parquet(parquet_path, index=False)
    return df


# ---------------------------------------------------------
# Version key
# ---------------------------------------------------------
def _version_key(p):
    return {
        "use_contoso_products": p["use_contoso_products"],
    }
