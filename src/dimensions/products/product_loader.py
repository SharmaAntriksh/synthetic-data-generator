from pathlib import Path
import pandas as pd
import numpy as np

from src.utils import info, skip
from src.versioning import should_regenerate, save_version

from .contoso_loader import load_contoso_products
from .contoso_expander import expand_contoso_products
from .pricing import apply_product_pricing


def load_product_dimension(config, output_folder: Path):
    """
    Product dimension loader.

    Returns:
        (DataFrame, regenerated: bool)
    """

    p = config["products"]
    active_ratio = p.get("active_ratio", 1.0)

    if not isinstance(active_ratio, (int, float)) or not (0 < active_ratio <= 1):
        raise ValueError("products.active_ratio must be a number in the range (0, 1]")

    version_key = _version_key(p)
    parquet_path = output_folder / "products.parquet"

    # ---------------- SKIP ----------------
    force = p.get("_force_regenerate", False)

    if not force and not should_regenerate("products", version_key, parquet_path):
        skip("Products up-to-date; skipping regeneration")
        return pd.read_parquet(parquet_path), False

    # ---------------- WORK ----------------
    base_df = load_contoso_products(output_folder)

    # Ensure variant columns exist
    if "BaseProductKey" not in base_df.columns:
        base_df["BaseProductKey"] = base_df["ProductKey"]

    if "VariantIndex" not in base_df.columns:
        base_df["VariantIndex"] = 0

    if p["use_contoso_products"]:
        info("USING CONTOSO PRODUCTS (AS-IS)")
        df = base_df
    else:
        info("EXPANDING CONTOSO PRODUCTS")
        df = expand_contoso_products(
            base_products=base_df,
            num_products=int(p["num_products"]),
            seed=int(p.get("seed", 42)),
            price_jitter_pct=float(p.get("price_jitter_pct", 0.0)),
        )

    # Apply pricing (authoritative)
    df = apply_product_pricing(
        df=df,
        pricing_cfg=p.get("pricing"),
        seed=p.get("seed"),
    )

    # ------------------------------------------------------------
    # Active products (eligibility for Sales)
    # ------------------------------------------------------------
    N = len(df)
    active_count = int(N * active_ratio)
    
    if active_count == 0:
        raise ValueError(
            "products.active_ratio results in zero active products; "
            "increase active_ratio or product count"
        )

    product_keys = df["ProductKey"].to_numpy()

    if active_count < N:
        rng = np.random.default_rng(int(p.get("seed", 42)))
        active_product_keys = rng.choice(
            product_keys,
            size=active_count,
            replace=False,
        )
        active_product_set = set(active_product_keys)
    else:
        active_product_set = set(product_keys)

    df["IsActiveInSales"] = df["ProductKey"].isin(active_product_set).astype("int64")

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
            raise ValueError(f"Missing required field in Products: {col}")

    # Write parquet
    df.to_parquet(parquet_path, index=False)

    # Save version metadata
    save_version("products", version_key, parquet_path)

    return df, True


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
