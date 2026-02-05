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

    Behavior (intended):
      - Load Contoso catalog as a base
      - If use_contoso_products == false, expand to num_products (identity-only expansion)
      - Apply authoritative pricing from products.pricing (UnitPrice/UnitCost finalized here)
      - Mark IsActiveInSales based on active_ratio
      - Write products.parquet (single writer)

    Returns:
        (DataFrame, regenerated: bool)
    """
    p = config["products"]
    active_ratio = p.get("active_ratio", 1.0)

    if not isinstance(active_ratio, (int, float)) or not (0 < float(active_ratio) <= 1.0):
        raise ValueError("products.active_ratio must be a number in the range (0, 1]")

    version_key = _version_key(p)
    parquet_path = output_folder / "products.parquet"

    # ---------------- SKIP ----------------
    force = bool(p.get("_force_regenerate", False))

    if not force and not should_regenerate("products", version_key, parquet_path):
        skip("Products up-to-date; skipping regeneration")
        return pd.read_parquet(parquet_path), False

    # ---------------- WORK ----------------
    # Base catalog (Contoso). This function MUST NOT write products.parquet anymore.
    base_df = load_contoso_products(output_folder)

    # Ensure variant lineage columns exist
    if "BaseProductKey" not in base_df.columns:
        base_df["BaseProductKey"] = base_df["ProductKey"]
    if "VariantIndex" not in base_df.columns:
        base_df["VariantIndex"] = 0

    use_contoso = bool(p.get("use_contoso_products", True))
    if use_contoso:
        info("USING CONTOSO BASE CATALOG")
        df = base_df
    else:
        info("EXPANDING CONTOSO BASE CATALOG")
        df = expand_contoso_products(
            base_products=base_df,
            num_products=int(p["num_products"]),
            seed=int(p.get("seed", 42)),
            # IMPORTANT: do NOT pass price_jitter_pct; prices/costs are centralized in products.pricing.*
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
    active_count = int(N * float(active_ratio))

    if active_count <= 0:
        raise ValueError(
            "products.active_ratio results in zero active products; "
            "increase active_ratio or product count"
        )

    product_keys = df["ProductKey"].to_numpy(dtype="int64", copy=False)

    if active_count < N:
        rng = np.random.default_rng(int(p.get("seed", 42)))
        active_product_keys = rng.choice(product_keys, size=active_count, replace=False)
        active_product_set = set(active_product_keys.tolist())
    else:
        active_product_set = set(product_keys.tolist())

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
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required field(s) in Products: {missing}")

    # Write parquet (single writer)
    df.to_parquet(parquet_path, index=False)

    # Save version metadata
    save_version("products", version_key, parquet_path)

    return df, True


# ---------------------------------------------------------
# Version key
# ---------------------------------------------------------
def _version_key(p):
    """
    IMPORTANT:
      - price_jitter_pct removed: expander no longer mutates UnitPrice/UnitCost.
      - products.pricing controls economics and SHOULD be the only pricing-related version input.
    """
    return {
        "use_contoso_products": bool(p.get("use_contoso_products", True)),
        "num_products": p.get("num_products"),
        "seed": p.get("seed"),
        "pricing": p.get("pricing"),
        "active_ratio": p.get("active_ratio", 1.0),
    }
