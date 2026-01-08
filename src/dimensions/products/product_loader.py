from pathlib import Path
import pandas as pd

from src.utils import info, skip
from src.versioning import should_regenerate, save_version

from .contoso_loader import load_contoso_products
from .contoso_expander import expand_contoso_products
from .pricing import apply_product_pricing


def load_product_dimension(config, output_folder: Path):
    """
    Product dimension loader.

    Behavior:
    - use_contoso_products = True
        → Load Contoso products
        → Apply product pricing rules

    - use_contoso_products = False
        → Expand Contoso products to num_products
        → Apply product pricing rules

    Notes:
    - BaseProductKey and VariantIndex are always present
    - Pricing is authoritative at the product level
    """
    p = config["products"]
    version_key = _version_key(p)
    parquet_path = output_folder / "products.parquet"

    # Skip if unchanged
    if not should_regenerate("products", version_key, parquet_path):
        skip("Products up-to-date; skipping regeneration")
        return pd.read_parquet(parquet_path)

    # Always load Contoso base products
    base_df = load_contoso_products(output_folder)
    
    # -------------------------------------------------
    # ENSURE VARIANT COLUMNS ALWAYS EXIST
    # -------------------------------------------------
    if "BaseProductKey" not in base_df.columns:
        base_df["BaseProductKey"] = base_df["ProductKey"]

    if "VariantIndex" not in base_df.columns:
        base_df["VariantIndex"] = 0

    if p["use_contoso_products"]:
        info("📦 USING CONTOSO PRODUCTS (AS-IS)")
        df = base_df
    else:
        info("📦 EXPANDING CONTOSO PRODUCTS")
        df = expand_contoso_products(
            base_products=base_df,
            num_products=int(p["num_products"]),
            seed=int(p.get("seed", 42)),
            price_jitter_pct=float(p.get("price_jitter_pct", 0.0)),
        )
    # -------------------------------------------------
    # APPLY PRODUCT PRICING (AUTHORITATIVE)
    # -------------------------------------------------
    df = apply_product_pricing(
        df=df,
        pricing_cfg=p.get("pricing"),
        seed=p.get("seed"),
    )

    # Required minimal fields for Sales
    required = [
        "ProductKey",
        "BaseProductKey",
        "VariantIndex",
        "SubcategoryKey",
        "UnitPrice",
        "UnitCost",
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Missing required field in Products: {col}"
            )

    # Write final parquet
    df.to_parquet(parquet_path, index=False)

    # Save version metadata
    save_version("products", version_key, parquet_path)

    return df


# ---------------------------------------------------------
# Version key
# ---------------------------------------------------------
def _version_key(p):
    return {
        "use_contoso_products": p["use_contoso_products"],
        "num_products": p.get("num_products"),
        "seed": p.get("seed"),
        "price_jitter_pct": p.get("price_jitter_pct", 0.0),
        "pricing": p.get("pricing"),
    }

