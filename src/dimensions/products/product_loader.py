from pathlib import Path
import pandas as pd
import numpy as np

from src.utils import info, skip
from src.versioning import should_regenerate, save_version

from .contoso_loader import load_contoso_products
from .contoso_expander import expand_contoso_products
from .pricing import apply_product_pricing


def _load_supplier_keys(output_folder: Path) -> np.ndarray:
    """
    Loads SupplierKey values from suppliers.parquet in the same dims folder.
    Returns sorted unique int64 keys.
    """
    sup_path = output_folder / "suppliers.parquet"
    if not sup_path.exists():
        raise FileNotFoundError(
            f"Missing suppliers dimension parquet: {sup_path}. "
            "Generate dimensions first (Suppliers)."
        )

    sup = pd.read_parquet(sup_path)
    # tolerate minor schema naming drift
    key_col = None
    for c in ["SupplierKey", "Key"]:
        if c in sup.columns:
            key_col = c
            break
    if key_col is None:
        raise KeyError(f"suppliers.parquet missing SupplierKey/Key. Available: {list(sup.columns)}")

    keys = pd.to_numeric(sup[key_col], errors="coerce").dropna().astype("int64").to_numpy()
    keys = np.unique(keys)
    if keys.size == 0:
        raise ValueError("suppliers.parquet has zero valid SupplierKey values")
    return np.sort(keys)


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
    # ------------------------------------------------------------
    # Supplier assignment
    # ------------------------------------------------------------
    sup_cfg = p.get("supplier_assignment") or {}
    sup_enabled = bool(sup_cfg.get("enabled", True))
    sup_seed = int(sup_cfg.get("seed", p.get("seed", 42)))
    sup_strategy = str(sup_cfg.get("strategy", "by_base_product")).lower()

    supplier_keys = None
    supplier_sig = None
    if sup_enabled:
        supplier_keys = _load_supplier_keys(output_folder)
        supplier_sig = {
            "n": int(supplier_keys.size),
            "min": int(supplier_keys.min()),
            "max": int(supplier_keys.max()),
        }

    if not isinstance(active_ratio, (int, float)) or not (0 < float(active_ratio) <= 1.0):
        raise ValueError("products.active_ratio must be a number in the range (0, 1]")

    version_key = _version_key(p)

    # Make supplier assignment part of the skip/version signature
    if sup_enabled:
        version_key = dict(version_key)
        version_key["supplier_assignment"] = {
            "enabled": True,
            "strategy": sup_strategy,
            "seed": sup_seed,
        }
        version_key["supplier_sig"] = supplier_sig

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
    # Assign SupplierKey (deterministic)
    # ------------------------------------------------------------
    if sup_enabled:
        n_sup = int(supplier_keys.size)

        if "BaseProductKey" in df.columns:
            base = pd.to_numeric(df["BaseProductKey"], errors="coerce").fillna(0).astype("int64").to_numpy()
        else:
            base = pd.to_numeric(df["ProductKey"], errors="coerce").fillna(0).astype("int64").to_numpy()

        if sup_strategy == "by_subcategory" and "SubcategoryKey" in df.columns:
            sub = pd.to_numeric(df["SubcategoryKey"], errors="coerce").fillna(0).astype("int64").to_numpy()
            idx = np.mod(sub, n_sup)
        elif sup_strategy == "uniform":
            rng_sup = np.random.default_rng(sup_seed)
            idx = rng_sup.integers(0, n_sup, size=len(df), dtype=np.int64)
        else:
            # default: keep variants on the same supplier
            idx = np.mod(base, n_sup)

        df["SupplierKey"] = supplier_keys[idx].astype("int64")

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
    if sup_enabled:
        required.append("SupplierKey")

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
