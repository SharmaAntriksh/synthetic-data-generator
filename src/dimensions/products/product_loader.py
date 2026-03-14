from __future__ import annotations

from pathlib import Path
import datetime as _dt

import numpy as np
import pandas as pd

from src.utils import info, skip
from src.versioning import should_regenerate, save_version

from src.utils.config_precedence import resolve_dates

from .contoso_loader import load_contoso_products
from .contoso_expander import expand_contoso_products
from .pricing import apply_product_pricing
from .product_profile import _enrich_products_attributes


# ---------------------------------------------------------------------
# SCD Type 2 — price revision versions
# ---------------------------------------------------------------------

def _generate_scd2_versions(
    rng: np.random.Generator,
    base_df: pd.DataFrame,
    prod_cfg,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """
    Expand products into SCD2 version rows with price revisions.

    Each version represents a price revision period. ListPrice and UnitCost
    change per version (drift ± scd2_price_drift), while all other product
    attributes remain static.

    Version 1 = original product (EffectiveStartDate = config start_date).
    Subsequent versions have EffectiveStartDate spaced by scd2_revision_frequency
    months from the product's first revision date.

    Returns a new DataFrame sorted by ProductID + VersionNumber, with
    ProductKey reassigned sequentially (1..N_total_rows).
    """
    revision_freq = int(getattr(prod_cfg, "revision_frequency", 12))
    price_drift = float(getattr(prod_cfg, "price_drift", 0.05))
    max_versions = int(getattr(prod_cfg, "max_versions", 4))

    N = len(base_df)
    if max_versions <= 1 or revision_freq <= 0:
        # No revisions — just add SCD2 metadata with defaults
        return base_df

    # How many revision slots fit in the date range?
    total_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
    max_possible_versions = min(max_versions, max(1, total_months // revision_freq + 1))

    if max_possible_versions <= 1:
        return base_df

    # Determine how many versions each product gets (1 to max_possible_versions)
    # Use a geometric-ish distribution: most products get few revisions
    n_extra = rng.integers(0, max_possible_versions, size=N, dtype=np.int64)
    # Products with 0 extra versions stay at version 1 only

    version_rows = []
    base_records = base_df.to_dict("records")
    for i in range(N):
        base_rec = base_records[i]
        n_versions = int(n_extra[i]) + 1  # at least 1

        list_price = float(base_rec["ListPrice"])
        unit_cost = float(base_rec["UnitCost"])

        # First revision starts at a random offset within the first revision period
        first_revision_offset = rng.integers(1, max(2, revision_freq))
        revision_start = start_date + pd.DateOffset(months=int(first_revision_offset))

        for v in range(n_versions):
            version_data = base_rec.copy()
            version_data["VersionNumber"] = v + 1

            if v == 0:
                version_data["EffectiveStartDate"] = start_date
            else:
                eff_start = revision_start + pd.DateOffset(months=int((v - 1) * revision_freq))
                if eff_start > end_date:
                    break  # no more versions fit
                version_data["EffectiveStartDate"] = eff_start

                # Apply price drift
                drift = 1.0 + rng.uniform(-price_drift, price_drift * 2)
                list_price = round(list_price * drift, 2)
                unit_cost = round(min(unit_cost * drift, list_price), 2)
                version_data["ListPrice"] = list_price
                version_data["UnitCost"] = unit_cost

            version_rows.append(version_data)

    result = pd.DataFrame(version_rows)

    # Set EffectiveEndDate: next version's start - 1 day, or 9999-12-31 for current
    result = result.sort_values(["ProductID", "VersionNumber"]).reset_index(drop=True)

    # Vectorised EffectiveEndDate: shift within ProductID groups
    eff_start_arr = result["EffectiveStartDate"].to_numpy()
    pid_arr = result["ProductID"].to_numpy()
    # Data is already sorted by [ProductID, VersionNumber]
    # Next row's start date within same ProductID → current row's end date - 1 day
    same_pid_as_next = np.empty(len(result), dtype=bool)
    same_pid_as_next[:-1] = pid_arr[:-1] == pid_arr[1:]
    same_pid_as_next[-1] = False

    eff_end_arr = np.full(len(result), pd.Timestamp("9999-12-31"), dtype="datetime64[ns]")
    is_current_arr = np.ones(len(result), dtype=np.int64)
    _shift_mask = np.flatnonzero(same_pid_as_next)
    eff_end_arr[_shift_mask] = eff_start_arr[_shift_mask + 1] - np.timedelta64(1, "D")
    is_current_arr[_shift_mask] = 0

    result["EffectiveEndDate"] = eff_end_arr
    result["IsCurrent"] = is_current_arr

    # Reassign ProductKey sequentially (PK, unique per version row)
    result["ProductKey"] = np.arange(1, len(result) + 1, dtype="int64")

    n_with_history = int((n_extra > 0).sum())
    total_rows = len(result)
    info(f"Products SCD2: {n_with_history:,}/{N:,} products have price history "
         f"({total_rows:,} total rows, max {max_possible_versions} versions)")

    return result


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
      - Apply pricing from products.pricing (ListPrice/UnitCost finalized here)
      - Enrich attributes (merch/channel/logistics/quality)
      - Assign SupplierKey (optional)
      - Write products.parquet

    Returns:
        (DataFrame, regenerated: bool)
    """
    p = config["products"]
    seed = int(p.get("seed", 42))

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

    # Minimal required fields for Sales
    required = [
        "ProductKey",
        "BaseProductKey",
        "VariantIndex",
        "SubcategoryKey",
        "ListPrice",
        "UnitCost",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required field(s) in Products: {missing}")

    # -----------------------------------------------------------------
    # SCD Type 2 metadata (always present for consistent schema)
    # -----------------------------------------------------------------
    N = len(df)
    df["ProductID"] = df["ProductKey"].copy()
    df["VersionNumber"] = np.ones(N, dtype="int64")

    # Resolve date range for SCD2 effective dates
    try:
        start_date, end_date = resolve_dates(config, p, section_name="products")
    except (KeyError, ValueError):
        start_date = pd.Timestamp("2020-01-01")
        end_date = pd.Timestamp("2025-12-31")

    df["EffectiveStartDate"] = start_date
    df["EffectiveEndDate"] = pd.Timestamp("9999-12-31")
    df["IsCurrent"] = np.ones(N, dtype="int64")

    # SCD2 expansion (if enabled)
    scd2_cfg = getattr(p, "scd2", None)
    scd2_enabled = bool(getattr(scd2_cfg, "enabled", False)) if scd2_cfg else False
    if scd2_enabled:
        rng_scd2 = np.random.default_rng(seed + 7777)
        df = _generate_scd2_versions(rng_scd2, df, scd2_cfg, start_date, end_date)

    # -----------------------------------------------------------------
    # Split into Products (core) and ProductProfile (analytical)
    # -----------------------------------------------------------------
    _PRODUCTS_CORE_COLS = [
        "ProductKey", "ProductID",
        "ProductCode", "ProductName", "ProductDescription",
        "SubcategoryKey", "Brand", "Class", "Color",
        "StockTypeCode", "StockType",
        "UnitCost", "ListPrice",
        "BaseProductKey", "VariantIndex",
        "VersionNumber", "EffectiveStartDate", "EffectiveEndDate", "IsCurrent",
    ]

    core_cols = [c for c in _PRODUCTS_CORE_COLS if c in df.columns]
    # Profile links to IsCurrent=1 version's ProductKey
    current_mask = df["IsCurrent"] == 1
    profile_source = df.loc[current_mask]
    profile_cols = ["ProductKey"] + [c for c in df.columns if c not in core_cols]

    products_df = df[core_cols].copy()
    profile_df = profile_source[profile_cols].copy()

    profile_path = output_folder / "product_profile.parquet"
    products_df.to_parquet(parquet_path, index=False)
    profile_df.to_parquet(profile_path, index=False)

    save_version("products", version_key, parquet_path)
    return products_df, profile_df, True


def _version_key(p: dict) -> dict:
    """
    Version key for Products. Pricing is the economic source of truth.
    """
    key = {
        "num_products": p.get("num_products"),
        "seed": p.get("seed"),
        "pricing": p.get("pricing"),
        # bump whenever you add/remove enrichment columns (forces one regen)
        "enrichment_v": 6,
    }
    # SCD2 settings affect output shape
    scd2 = p.get("scd2")
    if scd2 and bool(scd2.get("enabled", False)):
        key["scd2"] = {
            "enabled": True,
            "revision_frequency": scd2.get("revision_frequency", 12),
            "price_drift": scd2.get("price_drift", 0.05),
            "max_versions": scd2.get("max_versions", 4),
        }
    return key
