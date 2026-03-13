"""Wishlist pipeline runner — generates customer_wishlists.parquet using
accumulated (CustomerKey, ProductKey) purchase pairs from the sales pipeline.

Runs AFTER sales generation, using the WishlistAccumulator to create realistic
wishlist-to-purchase conversion rates.  A configurable fraction of each
customer's wishlist items are drawn from products they actually bought;
the remainder are selected via popularity-weighted subcategory affinity.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.facts.wishlists.accumulator import WishlistAccumulator
from src.utils.logging_utils import info, skip


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NS_PER_DAY: int = 86_400_000_000_000

_PRIORITY_VALUES = np.array(["High", "Medium", "Low"], dtype=object)
_PRIORITY_WEIGHTS = np.array([0.20, 0.50, 0.30])


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _WishlistsCfg:
    enabled: bool = False
    participation_rate: float = 0.35
    avg_items: float = 3.5
    max_items: int = 20
    pre_browse_days: int = 90
    affinity_strength: float = 0.6
    conversion_rate: float = 0.30
    seed: int = 500
    write_chunk_rows: int = 250_000


def _read_cfg(cfg: Any) -> _WishlistsCfg:
    wl = getattr(cfg, "wishlists", None)
    if wl is None:
        return _WishlistsCfg()
    return _WishlistsCfg(
        enabled=bool(getattr(wl, "enabled", False)),
        participation_rate=float(getattr(wl, "participation_rate", 0.35)),
        avg_items=float(getattr(wl, "avg_items", 3.5)),
        max_items=int(getattr(wl, "max_items", 20)),
        pre_browse_days=int(getattr(wl, "pre_browse_days", 90)),
        affinity_strength=float(getattr(wl, "affinity_strength", 0.6)),
        conversion_rate=float(getattr(wl, "conversion_rate", 0.30)),
        seed=int(getattr(wl, "seed", None) or 500),
        write_chunk_rows=int(getattr(wl, "write_chunk_rows", 250_000)),
    )


def _parse_global_dates(cfg: Any) -> Tuple[pd.Timestamp, pd.Timestamp]:
    defaults = getattr(cfg, "defaults", None)
    if defaults is None:
        defaults = getattr(cfg, "_defaults", None)
    gd = getattr(defaults, "dates", None) if defaults else None
    if gd is None:
        raise ValueError("Cannot resolve global dates for wishlists.")
    start_raw = gd.get("start", None) if isinstance(gd, dict) else getattr(gd, "start", None)
    end_raw = gd.get("end", None) if isinstance(gd, dict) else getattr(gd, "end", None)
    if start_raw is None or end_raw is None:
        raise ValueError("Global dates must have both 'start' and 'end'.")
    return pd.Timestamp(start_raw), pd.Timestamp(end_raw)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def _bridge_schema() -> pa.Schema:
    return pa.schema([
        pa.field("WishlistKey", pa.int64()),
        pa.field("CustomerKey", pa.int64()),
        pa.field("ProductKey", pa.int64()),
        pa.field("AddedDate", pa.date32()),
        pa.field("Priority", pa.string()),
        pa.field("Quantity", pa.int32()),
        pa.field("NetPrice", pa.float64()),
    ])


# ---------------------------------------------------------------------------
# Customer windows
# ---------------------------------------------------------------------------

def _compute_customer_windows(
    customers: pd.DataFrame,
    g_start: pd.Timestamp,
    g_end: pd.Timestamp,
    pre_browse_days: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    cust_keys = customers["CustomerKey"].astype(np.int64).to_numpy()
    order = np.argsort(cust_keys)
    cust_keys = cust_keys[order]

    g_start_ns = np.int64(g_start.value)
    g_end_ns = np.int64(g_end.value)
    browse_offset_ns = np.int64(pre_browse_days) * _NS_PER_DAY

    if "CustomerStartDate" in customers.columns:
        start_vals = customers["CustomerStartDate"].to_numpy().astype("datetime64[ns]")
        start_ns = start_vals.view(np.int64).copy()
        start_ns[np.isnat(start_vals)] = g_start_ns
    else:
        start_ns = np.full(len(cust_keys), g_start_ns, dtype=np.int64)

    if "CustomerEndDate" in customers.columns:
        end_vals = customers["CustomerEndDate"].to_numpy().astype("datetime64[ns]")
        end_ns = end_vals.view(np.int64).copy()
        end_ns[np.isnat(end_vals)] = g_end_ns
    else:
        end_ns = np.full(len(cust_keys), g_end_ns, dtype=np.int64)

    start_ns = start_ns[order]
    end_ns = end_ns[order]

    earliest_ns = start_ns - browse_offset_ns
    floor_ns = g_start_ns - browse_offset_ns
    earliest_ns = np.maximum(earliest_ns, floor_ns)
    latest_ns = np.clip(end_ns, g_start_ns, g_end_ns)
    latest_ns = np.maximum(latest_ns, earliest_ns + _NS_PER_DAY)

    return cust_keys, earliest_ns, latest_ns


# ---------------------------------------------------------------------------
# Product popularity weights
# ---------------------------------------------------------------------------

def _build_product_weights(
    products: pd.DataFrame,
    parquet_dims: Path,
) -> np.ndarray:
    n = len(products)
    prod_keys = products["ProductKey"].to_numpy().astype(np.int64)

    brand_weight_per_product = np.ones(n, dtype=np.float64)
    if "Brand" in products.columns:
        brands = products["Brand"].fillna("Unknown").astype(str).to_numpy()
        unique_brands, brand_codes = np.unique(brands, return_inverse=True)
        brand_counts = np.bincount(brand_codes).astype(np.float64)
        brand_base = np.sqrt(np.maximum(brand_counts, 1.0))
        brand_base /= brand_base.sum()
        brand_weight_per_product = brand_base[brand_codes]

    pop_scores = np.full(n, 50.0, dtype=np.float64)
    for name in ("product_profile.parquet", "ProductProfile.parquet"):
        profile_path = parquet_dims / name
        if profile_path.exists():
            try:
                pp_df = pd.read_parquet(profile_path, columns=["ProductKey", "PopularityScore"])
                pp_df = pp_df.drop_duplicates("ProductKey", keep="first")
                pop_map = pd.Series(
                    pp_df["PopularityScore"].to_numpy(dtype=np.float64),
                    index=pp_df["ProductKey"].to_numpy(dtype=np.int64),
                )
                pop_scores = pop_map.reindex(prod_keys).fillna(50.0).to_numpy(dtype=np.float64)
            except (KeyError, OSError):
                pass
            break

    weights = brand_weight_per_product * pop_scores
    total = weights.sum()
    if total > 0:
        weights /= total
    else:
        weights = np.ones(n, dtype=np.float64) / n
    return weights


# ---------------------------------------------------------------------------
# Product selection with sales-driven conversion + affinity
# ---------------------------------------------------------------------------

def _pick_products_for_customer(
    rng: np.random.Generator,
    n_items: int,
    purchased_indices: np.ndarray,
    n_products: int,
    prod_subcat: np.ndarray,
    subcat_to_indices: Dict[int, np.ndarray],
    affinity: float,
    conversion_rate: float,
    product_weights: np.ndarray,
    subcat_weights: Dict[int, np.ndarray],
) -> np.ndarray:
    """Pick n_items product indices for a customer's wishlist.

    - With probability *conversion_rate* per slot, pick from products the
      customer actually purchased (if any available).
    - Otherwise, pick via popularity-weighted subcategory affinity.
    - No duplicates.
    """
    chosen = np.empty(n_items, dtype=np.int64)
    chosen_set: set = set()

    has_purchases = len(purchased_indices) > 0

    for j in range(n_items):
        # Conversion slot: pick from actual purchases
        if has_purchases and rng.random() < conversion_rate:
            available = [p for p in purchased_indices if p not in chosen_set]
            if available:
                pick = int(rng.choice(available))
                chosen[j] = pick
                chosen_set.add(pick)
                continue

        # Affinity slot: pick from same subcategory as a prior item
        if j > 0 and rng.random() < affinity:
            anchor = chosen[rng.integers(0, j)]
            sc = prod_subcat[anchor]
            pool = subcat_to_indices[sc]
            sc_w = subcat_weights[sc]
            mask = ~np.isin(pool, list(chosen_set))
            available = pool[mask]
            if len(available) > 0:
                w = sc_w[mask]
                w_sum = w.sum()
                pick = int(rng.choice(available, p=w / w_sum if w_sum > 0 else None))
                chosen[j] = pick
                chosen_set.add(pick)
                continue

        # Global weighted random (rejection-sample for no-dups)
        for _ in range(200):
            pick = int(rng.choice(n_products, p=product_weights))
            if pick not in chosen_set:
                chosen[j] = pick
                chosen_set.add(pick)
                break
        else:
            remaining = np.setdiff1d(np.arange(n_products), np.array(list(chosen_set)))
            w = product_weights[remaining]
            w_sum = w.sum()
            pick = int(rng.choice(remaining, p=w / w_sum if w_sum > 0 else None))
            chosen[j] = pick
            chosen_set.add(pick)

    return chosen


# ---------------------------------------------------------------------------
# Bridge writer
# ---------------------------------------------------------------------------

def _write_bridge(
    customers: pd.DataFrame,
    products: pd.DataFrame,
    product_weights: np.ndarray,
    purchased_pairs: pd.DataFrame,
    c: _WishlistsCfg,
    g_start: pd.Timestamp,
    g_end: pd.Timestamp,
    out_path: Path,
) -> int:
    rng = np.random.default_rng(c.seed)
    schema = _bridge_schema()

    cust_keys, earliest_ns, latest_ns = _compute_customer_windows(
        customers, g_start, g_end, c.pre_browse_days,
    )

    n_customers = len(cust_keys)
    n_participants = max(1, int(round(n_customers * c.participation_rate)))

    participant_idx = rng.choice(n_customers, size=n_participants, replace=False)
    participant_idx.sort()

    prod_keys = products["ProductKey"].to_numpy().astype(np.int64)
    prod_prices = products["UnitPrice"].to_numpy().astype(np.float64)
    prod_subcat = products["SubcategoryKey"].to_numpy().astype(np.int64)
    n_products = len(prod_keys)

    # Product key → index lookup
    prod_key_to_idx = pd.Series(np.arange(n_products), index=prod_keys)

    # Build per-customer purchased product indices from accumulated sales data
    cust_purchased: Dict[int, np.ndarray] = {}
    if len(purchased_pairs) > 0:
        for ck, grp in purchased_pairs.groupby("CustomerKey"):
            pks = grp["ProductKey"].to_numpy().astype(np.int64)
            idxs = prod_key_to_idx.reindex(pks).dropna().to_numpy(dtype=np.int64)
            if len(idxs) > 0:
                cust_purchased[int(ck)] = idxs

    unique_subcats = np.unique(prod_subcat)
    subcat_to_indices: Dict[int, np.ndarray] = {
        sc: np.where(prod_subcat == sc)[0] for sc in unique_subcats
    }
    subcat_weights: Dict[int, np.ndarray] = {
        sc: product_weights[idx] for sc, idx in subcat_to_indices.items()
    }

    items_per = rng.poisson(lam=c.avg_items, size=n_participants)
    items_per = np.clip(items_per, 1, min(c.max_items, n_products))
    total_rows = int(items_per.sum())

    out_wkey = np.empty(total_rows, dtype=np.int64)
    out_ckey = np.empty(total_rows, dtype=np.int64)
    out_pkey = np.empty(total_rows, dtype=np.int64)
    out_date_ns = np.empty(total_rows, dtype=np.int64)
    out_priority = np.empty(total_rows, dtype=object)
    out_quantity = np.empty(total_rows, dtype=np.int32)
    out_price = np.empty(total_rows, dtype=np.float64)

    row = 0
    for i in range(n_participants):
        cidx = participant_idx[i]
        n_items = int(items_per[i])
        ck = cust_keys[cidx]
        e_ns = earliest_ns[cidx]
        l_ns = latest_ns[cidx]

        purchased_indices = cust_purchased.get(int(ck), np.array([], dtype=np.int64))

        chosen_prod_idx = _pick_products_for_customer(
            rng, n_items, purchased_indices,
            n_products, prod_subcat, subcat_to_indices,
            c.affinity_strength, c.conversion_rate,
            product_weights, subcat_weights,
        )

        span = l_ns - e_ns
        if span <= 0:
            span = _NS_PER_DAY
        offsets = rng.integers(0, max(1, span), size=n_items)
        dates = e_ns + offsets

        priorities = rng.choice(_PRIORITY_VALUES, size=n_items, p=_PRIORITY_WEIGHTS)
        qtys = rng.choice([1, 1, 1, 1, 2, 2, 3], size=n_items).astype(np.int32)

        sl = slice(row, row + n_items)
        out_wkey[sl] = np.arange(row + 1, row + n_items + 1)
        out_ckey[sl] = ck
        out_pkey[sl] = prod_keys[chosen_prod_idx]
        out_date_ns[sl] = dates
        out_priority[sl] = priorities
        out_quantity[sl] = qtys
        out_price[sl] = prod_prices[chosen_prod_idx]

        row += n_items

    out_dates_dt = out_date_ns.view("datetime64[ns]").astype("datetime64[ms]")

    writer = pq.ParquetWriter(str(out_path), schema)
    chunk = c.write_chunk_rows
    for start in range(0, total_rows, chunk):
        end = min(start + chunk, total_rows)
        batch = pa.record_batch(
            [
                pa.array(out_wkey[start:end], type=pa.int64()),
                pa.array(out_ckey[start:end], type=pa.int64()),
                pa.array(out_pkey[start:end], type=pa.int64()),
                pa.array(out_dates_dt[start:end], type=pa.date32()),
                pa.array(out_priority[start:end], type=pa.string()),
                pa.array(out_quantity[start:end], type=pa.int32()),
                pa.array(out_price[start:end], type=pa.float64()),
            ],
            schema=schema,
        )
        writer.write_batch(batch)
    writer.close()

    return total_rows


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_wishlist_pipeline(
    *,
    accumulator: WishlistAccumulator,
    parquet_dims: Path,
    cfg: Any,
    file_format: str = "parquet",
) -> Optional[Dict[str, Any]]:
    """Generate customer_wishlists.parquet using accumulated sales data."""
    c = _read_cfg(cfg)
    if not c.enabled:
        return None

    if not accumulator.has_data:
        skip("Wishlists: no sales data accumulated; skipping.")
        return None

    g_start, g_end = _parse_global_dates(cfg)

    # Load dimension tables
    customers_fp = _find_parquet(parquet_dims, "customers")
    products_fp = _find_parquet(parquet_dims, "products")

    customers = pd.read_parquet(
        customers_fp,
        columns=["CustomerKey", "CustomerStartDate", "CustomerEndDate"],
    )
    products = pd.read_parquet(
        products_fp,
        columns=["ProductKey", "UnitPrice", "SubcategoryKey", "Brand"],
    )

    product_weights = _build_product_weights(products, parquet_dims)
    purchased_pairs = accumulator.finalize()

    out_path = parquet_dims / "customer_wishlists.parquet"

    n_rows = _write_bridge(
        customers=customers,
        products=products,
        product_weights=product_weights,
        purchased_pairs=purchased_pairs,
        c=c,
        g_start=g_start,
        g_end=g_end,
        out_path=out_path,
    )
    info(f"Customer wishlists written: {out_path} ({n_rows:,} rows)")

    return {
        "bridge": str(out_path),
        "bridge_rows": n_rows,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_parquet(folder: Path, name: str) -> Path:
    for variant in (f"{name}.parquet", f"{name.title()}.parquet"):
        p = folder / variant
        if p.exists():
            return p
    raise FileNotFoundError(f"{name}.parquet not found in {folder}")
