from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils import info, skip, warn
from src.utils.output_utils import write_parquet_with_date32
from src.versioning import should_regenerate, save_version

from src.utils.config_precedence import resolve_dates
from src.defaults import SCD2_END_OF_TIME

from .contoso_loader import load_contoso_products
from .contoso_expander import expand_contoso_products
from .pricing import apply_product_pricing, snap_drifted_prices
from .product_profile import enrich_products_attributes, apply_post_merge_enrichment


# ---------------------------------------------------------------------
# SCD Type 2 — price revision versions
# ---------------------------------------------------------------------

def _generate_scd2_versions(
    rng: np.random.Generator,
    base_df: pd.DataFrame,
    prod_cfg,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    *,
    pricing_cfg=None,
) -> pd.DataFrame:
    """Expand products into SCD2 version rows with price revisions.

    Fully vectorized — no per-product Python loops.  Each version represents
    a price revision period.  Version 1 keeps the original price; subsequent
    versions apply cumulative drift.

    Returns a new DataFrame sorted by ProductID + VersionNumber, with
    ProductKey reassigned sequentially (1..N_total_rows).
    """
    revision_freq = int(getattr(prod_cfg, "revision_frequency", 12))
    price_drift = float(getattr(prod_cfg, "price_drift", 0.05))
    max_versions = int(getattr(prod_cfg, "max_versions", 4))

    N = len(base_df)
    if max_versions <= 1 or revision_freq <= 0:
        return base_df

    total_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
    max_possible_versions = min(max_versions, max(1, total_months // revision_freq + 1))

    if max_possible_versions <= 1:
        return base_df

    # How many versions each product gets (1-based)
    n_versions = rng.integers(1, max_possible_versions + 1, size=N, dtype=np.int64)

    # Random offset for first revision (months from start_date)
    first_offsets = rng.integers(1, max(2, revision_freq), size=N, dtype=np.int64)

    # Pre-generate all drift values for max_possible_versions-1 extra versions per product
    # Shape: (N, max_possible_versions-1)
    n_extra_max = max_possible_versions - 1
    if n_extra_max > 0:
        all_drifts = 1.0 + rng.uniform(-price_drift, price_drift, size=(N, n_extra_max))
    else:
        all_drifts = np.empty((N, 0), dtype=np.float64)

    # Build row indices: which base row does each output row come from?
    # Product i contributes n_versions[i] rows
    total_rows = int(n_versions.sum())
    row_idx = np.repeat(np.arange(N, dtype=np.int64), n_versions)
    # Version number within each product (0-based): position minus group start
    offsets = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(n_versions, out=offsets[1:])
    ver_within = np.arange(total_rows, dtype=np.int64) - offsets[row_idx]

    # Compute EffectiveStartDate as month offset from start_date
    # Version 0: start_date (offset=0)
    # Version v>=1: first_offsets[product] + (v-1) * revision_freq
    month_offsets = np.where(
        ver_within == 0,
        0,
        first_offsets[row_idx] + (ver_within - 1) * revision_freq,
    )

    # Convert month offsets to dates via pure numpy arithmetic (no Python loop)
    start_day = min(start_date.day, 28)  # safe day for all months
    start_year = start_date.year
    start_month = start_date.month
    total_month = start_month - 1 + month_offsets  # 0-based months from Jan of start_year
    eff_years = start_year + total_month // 12
    eff_months = total_month % 12 + 1  # back to 1-based

    eff_dates = (
        (eff_years - 1970).astype("datetime64[Y]")
        + (eff_months - 1).astype("timedelta64[M]")
        + np.timedelta64(start_day - 1, "D")
    ).astype("datetime64[D]")

    # Clip versions that exceed end_date
    end_np = np.datetime64(end_date.date(), "D")
    valid = (ver_within == 0) | (eff_dates <= end_np)

    # Filter to valid rows only
    row_idx = row_idx[valid]
    ver_within = ver_within[valid]
    eff_dates = eff_dates[valid]
    total_rows = int(valid.sum())

    # Build result DataFrame by repeating base rows
    result = base_df.iloc[row_idx].reset_index(drop=True)
    result["VersionNumber"] = ver_within + 1
    result["EffectiveStartDate"] = pd.to_datetime(eff_dates)

    # Apply cumulative price drift for versions > 1
    # For each row, compute cumulative drift product up to its version
    list_prices = result["ListPrice"].to_numpy(dtype=np.float64, copy=True)
    unit_costs = result["UnitCost"].to_numpy(dtype=np.float64, copy=True)

    if n_extra_max > 0:
        # Compute cumulative drift per product per version
        cum_drifts = np.cumprod(all_drifts, axis=1)  # shape (N, n_extra_max)

        drifted_mask = np.zeros(total_rows, dtype=bool)
        for v in range(1, max_possible_versions):
            mask = ver_within == v
            if not mask.any():
                continue
            prods = row_idx[mask]
            drift_v = cum_drifts[prods, v - 1]
            list_prices[mask] = list_prices[mask] * drift_v
            unit_costs[mask] = np.minimum(unit_costs[mask] * drift_v, list_prices[mask])
            drifted_mask |= mask

        # Re-snap drifted prices to the same appearance grid used by
        # apply_product_pricing so SCD2 versions look equally realistic.
        if drifted_mask.any():
            lp_snapped, uc_snapped = snap_drifted_prices(
                list_prices[drifted_mask], unit_costs[drifted_mask], pricing_cfg)
            list_prices[drifted_mask] = lp_snapped
            unit_costs[drifted_mask] = uc_snapped

    # Snapping can push cost above list price in edge cases
    unit_costs = np.minimum(unit_costs, list_prices)
    result["ListPrice"] = np.round(list_prices, 2)
    result["UnitCost"] = np.round(unit_costs, 2)

    # Sort by ProductID + VersionNumber
    result = result.sort_values(["ProductID", "VersionNumber"]).reset_index(drop=True)

    # Vectorised EffectiveEndDate
    eff_start_arr = result["EffectiveStartDate"].to_numpy()
    pid_arr = result["ProductID"].to_numpy()
    same_pid_as_next = np.empty(total_rows, dtype=bool)
    same_pid_as_next[:-1] = pid_arr[:-1] == pid_arr[1:]
    same_pid_as_next[-1] = False

    eff_end_arr = np.full(total_rows, SCD2_END_OF_TIME, dtype="datetime64[ns]")
    is_current_arr = np.ones(total_rows, dtype=np.int64)
    _shift_mask = np.flatnonzero(same_pid_as_next)
    eff_end_arr[_shift_mask] = eff_start_arr[_shift_mask + 1] - np.timedelta64(1, "D")
    is_current_arr[_shift_mask] = 0

    result["EffectiveEndDate"] = eff_end_arr
    result["IsCurrent"] = is_current_arr

    # Reassign ProductKey sequentially
    result["ProductKey"] = np.arange(1, total_rows + 1, dtype="int64")

    n_with_history = int((n_versions > 1).sum())
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
# Parallel enrichment orchestrator
# ---------------------------------------------------------------------
def _generate_parallel_enrichment(
    df: pd.DataFrame,
    config,
    seed: int,
    output_folder: Path,
    n_workers: int,
) -> pd.DataFrame:
    """Enrich products in parallel: chunk -> enrich -> merge -> rank columns."""
    import shutil
    from src.facts.sales.sales_worker.pool import PoolRunSpec, iter_imap_unordered
    from .worker import product_enrich_chunk_worker

    N = len(df)

    # Chunk partitioning (same formula as customers)
    n_chunks = min(n_workers * 2, max(2, N // 10_000))
    n_chunks = max(2, n_chunks)
    n_actual_workers = min(n_chunks, n_workers)

    # Serialize config for workers (must be picklable plain dict)
    from src.utils.config_helpers import as_dict
    cfg_dump = as_dict(config)

    # Scratch directory
    scratch_dir = output_folder / "_product_chunks"
    scratch_dir.mkdir(parents=True, exist_ok=True)

    # Split df into chunks and write to scratch
    chunk_boundaries = np.array_split(np.arange(N), n_chunks)

    tasks = []
    for i, indices in enumerate(chunk_boundaries):
        input_path = str(scratch_dir / f"chunk_{i:05d}_input.parquet")
        output_path = str(scratch_dir / f"chunk_{i:05d}_enriched.parquet")
        df.iloc[indices].to_parquet(input_path, index=False)
        tasks.append((
            i, seed,
            input_path, output_path,
            cfg_dump, str(output_folder),
        ))

    info(f"Product enrichment: {n_chunks} chunks across {n_actual_workers} workers")

    pool_spec = PoolRunSpec(
        processes=n_actual_workers,
        chunksize=1,
        label="product_enrichment",
    )

    try:
        chunk_results = []
        for result in iter_imap_unordered(
            tasks=tasks,
            task_fn=product_enrich_chunk_worker,
            spec=pool_spec,
        ):
            chunk_results.append(result)

        chunk_results.sort(key=lambda r: r["chunk_idx"])

        # Merge enriched chunks (read in order for determinism)
        enriched_dfs = []
        for i in range(n_chunks):
            path = scratch_dir / f"chunk_{i:05d}_enriched.parquet"
            enriched_dfs.append(pd.read_parquet(path))

        merged = pd.concat(enriched_dfs, ignore_index=True)
        del enriched_dfs

        # Apply rank-dependent columns on full dataset
        merged = apply_post_merge_enrichment(merged, seed)

        return merged
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


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

    # Enrichment columns (parallel when above threshold)
    from multiprocessing import cpu_count
    from src.defaults import PRODUCT_PARALLEL_THRESHOLD

    sales_cfg = getattr(config, "sales", None)
    configured_workers = getattr(sales_cfg, "workers", None) if sales_cfg else None
    from src.utils.config_helpers import int_or
    n_workers = max(1, int_or(configured_workers, cpu_count() - 1))

    if len(df) >= PRODUCT_PARALLEL_THRESHOLD and n_workers >= 2:
        df = _generate_parallel_enrichment(df, config, seed=seed,
                                            output_folder=output_folder,
                                            n_workers=n_workers)
    else:
        df = enrich_products_attributes(df, config, seed=seed, output_folder=output_folder)

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
        warn("Could not resolve dates for products; using fallback 2020-01-01 to 2025-12-31")
        start_date = pd.Timestamp("2020-01-01")
        end_date = pd.Timestamp("2025-12-31")

    df["EffectiveStartDate"] = start_date
    df["EffectiveEndDate"] = SCD2_END_OF_TIME
    df["IsCurrent"] = np.ones(N, dtype="int64")

    # SCD2 expansion (if enabled)
    scd2_cfg = getattr(p, "scd2", None)
    scd2_enabled = bool(getattr(scd2_cfg, "enabled", False)) if scd2_cfg else False
    if scd2_enabled:
        rng_scd2 = np.random.default_rng(seed + 7777)
        df = _generate_scd2_versions(
            rng_scd2, df, scd2_cfg, start_date, end_date,
            pricing_cfg=p.get("pricing"),
        )

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

    # Reorder profile columns to match static_schemas.py (SQL CREATE TABLE order).
    # BULK INSERT is positional — CSV column order must match the schema exactly.
    from src.utils.static_schemas import STATIC_SCHEMAS
    schema_cols = [name for name, _ in STATIC_SCHEMAS.get("ProductProfile", ())]
    if schema_cols:
        # Keep only columns present in both schema and DataFrame, in schema order
        ordered = [c for c in schema_cols if c in profile_df.columns]
        # Append any extra DataFrame columns not in schema (future-proof)
        extra = [c for c in profile_df.columns if c not in schema_cols]
        profile_df = profile_df[ordered + extra]

    profile_path = output_folder / "product_profile.parquet"
    write_parquet_with_date32(products_df, parquet_path, cast_all_datetime=True)
    write_parquet_with_date32(profile_df, profile_path, cast_all_datetime=True)

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
