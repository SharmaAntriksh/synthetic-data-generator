from __future__ import annotations

from pathlib import Path
import datetime as _dt

import numpy as np
import pandas as pd

from src.utils import info, skip
from src.versioning import should_regenerate, save_version

from .contoso_loader import load_contoso_products
from .contoso_expander import expand_contoso_products
from .pricing import apply_product_pricing
from .product_profile import _enrich_products_attributes


# ---------------------------------------------------------------------
# Supplier assignment
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------
def load_product_dimension(config, output_folder: Path, *, log_skip: bool = True):
    """
    Product dimension loader (single method):

      - Load Contoso catalog as base (2517 rows typically)
      - Scale to target row count via expand_contoso_products():
          * stratified trim by SubcategoryKey when target < base_count
          * repeat/variants when target > base_count
      - Apply lifecycle (Launch/Discontinued) from products.lifecycle (date-typed)
      - Apply pricing from products.pricing (UnitPrice/UnitCost finalized here)
      - Enrich attributes (merch/channel/logistics/quality)
      - Assign SupplierKey (optional)
      - Mark IsActiveInSales based on active_ratio
      - Write products.parquet

    Returns:
        (DataFrame, regenerated: bool)
    """
    p = config["products"]
    seed = int(p.get("seed", 42))

    active_ratio = p.get("active_ratio", 1.0)
    if not isinstance(active_ratio, (int, float)) or not (0 < float(active_ratio) <= 1.0):
        raise ValueError("products.active_ratio must be a number in the range (0, 1]")

    # Supplier assignment
    sup_cfg = p.get("supplier_assignment") or {}
    sup_enabled = bool(sup_cfg.get("enabled", True))
    sup_seed = int(sup_cfg.get("seed", seed))
    sup_strategy = str(sup_cfg.get("strategy", "by_base_product")).lower()

    supplier_keys = None
    supplier_sig = None
    if sup_enabled:
        supplier_keys = _load_supplier_keys(output_folder)
        supplier_sig = {"n": int(supplier_keys.size), "min": int(supplier_keys.min()), "max": int(supplier_keys.max())}

    # Versioning / skip
    parquet_path = output_folder / "products.parquet"
    version_key = _version_key(p)

    if sup_enabled:
        version_key = dict(version_key)
        version_key["supplier_assignment"] = {"enabled": True, "strategy": sup_strategy, "seed": sup_seed}
        version_key["supplier_sig"] = supplier_sig

    if not should_regenerate("products", version_key, parquet_path):
        if log_skip:
            skip("Products up-to-date")
        profile_path = output_folder / "product_profile.parquet"
        profile_df = pd.read_parquet(profile_path) if profile_path.exists() else pd.DataFrame()
        return pd.read_parquet(parquet_path), profile_df, False

    # Base catalog
    base_df = load_contoso_products(output_folder)
    base_count = int(len(base_df))

    # Target row count
    target_n = p.get("num_products", None)
    if target_n is None:
        target_n = base_count
    target_n = int(target_n)

    if target_n <= 0:
        raise ValueError("products.num_products must be a positive integer")

    if "num_products" in p and "use_contoso_products" in p:
        info("products.use_contoso_products is deprecated; ignoring (num_products is set)")

    if target_n < base_count:
        info(f"Trimming Contoso: {base_count:,} -> {target_n:,} (stratified by SubcategoryKey)")
    elif target_n == base_count:
        info(f"Using Contoso catalog (standardized): {target_n:,}")
    else:
        info(f"Expanding Contoso: {base_count:,} -> {target_n:,} (variants)")

    df = expand_contoso_products(
        base_products=base_df,
        num_products=target_n,
        seed=seed,
    )

    # Defensive: ensure ProductCode exists
    if "ProductCode" not in df.columns:
        df["ProductCode"] = df["ProductKey"].astype(str).str.zfill(7)


    # Pricing (authoritative)
    df = apply_product_pricing(
        df=df,
        pricing_cfg=p.get("pricing"),
        seed=seed,
    )

    # Enrichment columns
    df = _enrich_products_attributes(df, config, seed=seed, output_folder=output_folder)

    # SupplierKey (deterministic)
    if sup_enabled:
        n_sup = int(supplier_keys.size)
        base = pd.to_numeric(df.get("BaseProductKey", df["ProductKey"]), errors="coerce").fillna(0).astype("int64").to_numpy()

        if sup_strategy == "by_subcategory" and "SubcategoryKey" in df.columns:
            sub = pd.to_numeric(df["SubcategoryKey"], errors="coerce").fillna(0).astype("int64").to_numpy()
            idx = np.mod(sub, n_sup)
        elif sup_strategy == "uniform":
            rng_sup = np.random.default_rng(sup_seed)
            idx = rng_sup.integers(0, n_sup, size=len(df), dtype=np.int64)
        else:
            idx = np.mod(base, n_sup)

        df["SupplierKey"] = supplier_keys[idx].astype("int64")

    # Active products for Sales
    N = len(df)
    active_count = int(N * float(active_ratio))
    if active_count <= 0:
        raise ValueError("products.active_ratio results in zero active products; increase active_ratio or product count")

    product_keys = df["ProductKey"].to_numpy(dtype="int64", copy=False)
    if active_count < N:
        rng = np.random.default_rng(seed)
        active_product_keys = rng.choice(product_keys, size=active_count, replace=False)
        active_product_set = set(active_product_keys.tolist())
    else:
        active_product_set = set(product_keys.tolist())

    df["IsActiveInSales"] = df["ProductKey"].isin(active_product_set).astype("int64")

    # Minimal required fields for Sales
    required = [
        "ProductKey",
        "BaseProductKey",
        "VariantIndex",
        "SubcategoryKey",
        "UnitPrice",
        "UnitCost",
        "IsActiveInSales",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required field(s) in Products: {missing}")

    # -----------------------------------------------------------------
    # Split into Products (core) and ProductProfile (analytical)
    # -----------------------------------------------------------------
    _PRODUCTS_CORE_COLS = [
        "ProductKey", "ProductCode", "ProductName", "ProductDescription",
        "SubcategoryKey", "Brand", "Class", "Color",
        "StockTypeCode", "StockType",
        "UnitCost", "UnitPrice",
        "BaseProductKey", "VariantIndex",
        "IsActiveInSales",
    ]

    core_cols = [c for c in _PRODUCTS_CORE_COLS if c in df.columns]
    profile_cols = ["ProductKey"] + [c for c in df.columns if c not in core_cols]

    products_df = df[core_cols].copy()
    profile_df = df[profile_cols].copy()

    profile_path = output_folder / "product_profile.parquet"
    products_df.to_parquet(parquet_path, index=False)
    profile_df.to_parquet(profile_path, index=False)

    save_version("products", version_key, parquet_path)
    return products_df, profile_df, True


def _version_key(p: dict) -> dict:
    """
    Version key for Products. Pricing is the economic source of truth.
    """
    return {
        "num_products": p.get("num_products"),
        "seed": p.get("seed"),
        "pricing": p.get("pricing"),
        "active_ratio": p.get("active_ratio", 1.0),
        # bump whenever you add/remove enrichment columns (forces one regen)
        "enrichment_v": 4,
    }
