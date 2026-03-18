"""Wishlist pipeline runner — generates customer_wishlists.parquet using
accumulated (CustomerKey, ProductKey) purchase pairs from the sales pipeline.

Runs AFTER sales generation, using the WishlistAccumulator to create realistic
wishlist-to-purchase conversion rates.  A configurable fraction of each
customer's wishlist items are drawn from products they actually bought;
the remainder are selected via popularity-weighted subcategory affinity.

For large datasets (total_rows >= WISHLIST_PARALLEL_THRESHOLD), the per-customer
loop is distributed across multiple worker processes.  Each worker receives a
slice of participants plus shared read-only product data, writes a chunk parquet,
and the main process merges them.  Small datasets use the serial path unchanged.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.facts.wishlists.accumulator import WishlistAccumulator
from src.utils.logging_utils import info, skip
from src.defaults import WISHLIST_PARALLEL_THRESHOLD


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

def _weighted_pick(cdf: np.ndarray, rng: np.random.Generator) -> int:
    """Single weighted draw via pre-computed CDF + searchsorted (O(log n))."""
    idx = int(np.searchsorted(cdf, rng.random()))
    return min(idx, len(cdf) - 1)


def _pick_products_for_customer(
    rng: np.random.Generator,
    n_items: int,
    purchased_indices: np.ndarray,
    n_products: int,
    prod_subcat: np.ndarray,
    subcat_to_indices: Dict[int, np.ndarray],
    affinity: float,
    conversion_rate: float,
    global_cdf: np.ndarray,
    subcat_cdfs: Dict[int, np.ndarray],
    subcat_to_pool_and_weights: Dict[int, Tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Pick n_items product indices for a customer's wishlist.

    Uses pre-computed CDFs for O(log n) weighted sampling instead of
    rebuilding CDFs per call.
    """
    has_purchases = len(purchased_indices) > 0
    chosen = np.empty(n_items, dtype=np.int64)
    chosen_set: set = set()

    for j in range(n_items):
        picked = False

        # Conversion slot: pick from actual purchases
        if has_purchases and rng.random() < conversion_rate:
            if chosen_set:
                mask = np.ones(len(purchased_indices), dtype=bool)
                for exc in chosen_set:
                    mask &= (purchased_indices != exc)
                available = purchased_indices[mask]
            else:
                available = purchased_indices
            if len(available) > 0:
                chosen[j] = int(available[rng.integers(0, len(available))])
                chosen_set.add(chosen[j])
                picked = True

        # Affinity slot: pick from same subcategory as a prior item
        if not picked and j > 0 and rng.random() < affinity:
            anchor = chosen[rng.integers(0, j)]
            sc = int(prod_subcat[anchor])
            pool, sc_w = subcat_to_pool_and_weights[sc]
            if chosen_set:
                mask = np.ones(len(pool), dtype=bool)
                for exc in chosen_set:
                    mask &= (pool != exc)
                available = pool[mask]
            else:
                available = pool
            if len(available) > 0:
                if chosen_set:
                    w = sc_w[mask]
                    w_sum = w.sum()
                    if w_sum > 0:
                        _sc_cdf = np.cumsum(w / w_sum)
                        _sc_cdf[-1] = 1.0
                        chosen[j] = int(available[np.searchsorted(_sc_cdf, rng.random())])
                    else:
                        chosen[j] = int(available[rng.integers(0, len(available))])
                else:
                    sc_cdf = subcat_cdfs.get(sc)
                    if sc_cdf is not None:
                        idx = min(int(np.searchsorted(sc_cdf, rng.random())), len(pool) - 1)
                        chosen[j] = int(pool[idx])
                    else:
                        chosen[j] = int(pool[rng.integers(0, len(pool))])
                chosen_set.add(chosen[j])
                picked = True

        # Global weighted random via pre-computed CDF (rejection for no-dups)
        if not picked:
            for _ in range(200):
                pick = _weighted_pick(global_cdf, rng)
                if pick not in chosen_set:
                    chosen[j] = pick
                    chosen_set.add(pick)
                    break
            else:
                remaining = np.setdiff1d(
                    np.arange(n_products),
                    np.fromiter(chosen_set, dtype=np.int64, count=len(chosen_set)),
                )
                chosen[j] = int(remaining[rng.integers(0, len(remaining))])
                chosen_set.add(chosen[j])

    return chosen


# ---------------------------------------------------------------------------
# Bridge writer (serial path)
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
    prod_prices = products["ListPrice"].to_numpy().astype(np.float64)
    prod_subcat = products["SubcategoryKey"].to_numpy().astype(np.int64)
    n_products = len(prod_keys)

    # Product key → index lookup
    prod_key_to_idx = pd.Series(np.arange(n_products), index=prod_keys)

    # Build per-customer purchased product indices from accumulated sales data
    cust_purchased: Dict[int, np.ndarray] = {}
    if len(purchased_pairs) > 0:
        _pp_ckeys = purchased_pairs["CustomerKey"].to_numpy().astype(np.int64)
        _pp_pkeys = purchased_pairs["ProductKey"].to_numpy().astype(np.int64)
        # Vectorized lookup: map product keys to indices
        _pp_idx_series = prod_key_to_idx.reindex(_pp_pkeys)
        _pp_valid = _pp_idx_series.notna()
        _pp_ckeys_valid = _pp_ckeys[_pp_valid.to_numpy()]
        _pp_idxs_valid = _pp_idx_series[_pp_valid].to_numpy(dtype=np.int64)
        # Group by customer key using np.unique for speed
        if len(_pp_ckeys_valid) > 0:
            _sort_order = np.argsort(_pp_ckeys_valid)
            _sorted_ckeys = _pp_ckeys_valid[_sort_order]
            _sorted_idxs = _pp_idxs_valid[_sort_order]
            _uniq_ckeys, _split_pos = np.unique(_sorted_ckeys, return_index=True)
            _split_groups = np.split(_sorted_idxs, _split_pos[1:])
            for _ck, _idxs in zip(_uniq_ckeys, _split_groups):
                cust_purchased[int(_ck)] = _idxs

    unique_subcats = np.unique(prod_subcat)
    subcat_to_indices: Dict[int, np.ndarray] = {
        sc: np.where(prod_subcat == sc)[0] for sc in unique_subcats
    }
    # Pre-compute per-subcategory weights and CDFs (avoids rebuilding per call)
    subcat_to_pool_and_weights: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    subcat_cdfs: Dict[int, np.ndarray] = {}
    for sc, idx in subcat_to_indices.items():
        w = product_weights[idx]
        subcat_to_pool_and_weights[sc] = (idx, w)
        ws = w.sum()
        if ws > 0:
            sc_cdf = np.cumsum(w / ws)
            sc_cdf[-1] = 1.0
            subcat_cdfs[sc] = sc_cdf
        else:
            subcat_cdfs[sc] = None

    # Pre-compute global CDF for fast weighted sampling via searchsorted
    global_cdf = np.cumsum(product_weights)
    global_cdf[-1] = 1.0

    items_per = rng.poisson(lam=c.avg_items, size=n_participants)
    items_per = np.clip(items_per, 1, min(c.max_items, n_products))
    total_rows = int(items_per.sum())

    # ------------------------------------------------------------------
    # BATCH ALL RANDOM NUMBER GENERATION UPFRONT
    # ------------------------------------------------------------------
    # 1. Global weighted product picks for every item
    _base_prod_idx = np.searchsorted(global_cdf, rng.random(total_rows))
    np.clip(_base_prod_idx, 0, n_products - 1, out=_base_prod_idx)

    # 2. Per-item conversion and affinity rolls
    _conv_rolls = rng.random(total_rows)
    _aff_rolls = rng.random(total_rows)

    # 3. Misc random pool (conversion picks, affinity anchors/picks, dedup)
    _MISC_SIZE = max(total_rows * 6, 1024)
    _misc = rng.random(_MISC_SIZE)
    _mi = 0

    # 4. Dates: batch uniform [0,1) scaled per-customer below
    _date_rands = rng.random(total_rows)

    # 5. Priorities: batch via CDF searchsorted (3 elements)
    _prio_cdf = np.cumsum(_PRIORITY_WEIGHTS)
    _prio_cdf[-1] = 1.0
    _prio_idx = np.searchsorted(_prio_cdf, rng.random(total_rows))
    np.clip(_prio_idx, 0, len(_PRIORITY_VALUES) - 1, out=_prio_idx)
    out_priority = _PRIORITY_VALUES[_prio_idx]

    # 6. Quantities: batch uniform choice from [1,1,1,1,2,2,3]
    _qty_options = np.array([1, 1, 1, 1, 2, 2, 3], dtype=np.int32)
    out_quantity = _qty_options[rng.integers(0, 7, size=total_rows)]

    # ------------------------------------------------------------------
    # OUTPUT ARRAYS
    # ------------------------------------------------------------------
    out_prod_idx = _base_prod_idx.copy()
    out_ckey = np.empty(total_rows, dtype=np.int64)
    out_date_ns = np.empty(total_rows, dtype=np.int64)

    # Pre-compute per-customer offsets
    _offsets = np.zeros(n_participants + 1, dtype=np.int64)
    np.cumsum(items_per, out=_offsets[1:])

    # Cache subcat pool arrays as Python lists for fast iteration
    _sc_pool_list: Dict[int, list] = {
        sc: pool.tolist() for sc, (pool, _) in subcat_to_pool_and_weights.items()
    }

    # Localize frequently accessed values for inner loop speed
    _prod_subcat = prod_subcat  # numpy array
    _conversion_rate = c.conversion_rate
    _affinity = c.affinity_strength
    _empty_list: list = []
    _n_products = n_products

    # ------------------------------------------------------------------
    # PER-CUSTOMER PRODUCT SELECTION (tight loop, minimal numpy calls)
    # ------------------------------------------------------------------
    for i in range(n_participants):
        start = int(_offsets[i])
        end = int(_offsets[i + 1])
        n_items = end - start
        cidx = participant_idx[i]
        ck = int(cust_keys[cidx])

        # Fill customer key (batch)
        out_ckey[start:end] = ck

        # Fill dates: scale pre-generated [0,1) by customer's window
        e_ns = int(earliest_ns[cidx])
        l_ns = int(latest_ns[cidx])
        span = l_ns - e_ns
        if span <= 0:
            span = _NS_PER_DAY
        out_date_ns[start:end] = e_ns + (_date_rands[start:end] * span).astype(np.int64)

        # Product selection adjustments
        purchased = cust_purchased.get(ck)
        has_purch = purchased is not None
        purch_list = purchased.tolist() if has_purch else _empty_list

        chosen_set: set = set()
        for j in range(n_items):
            pos = start + j
            picked = False

            # --- Conversion: pick from actual purchases ---
            if has_purch and _conv_rolls[pos] < _conversion_rate:
                avail = [p for p in purch_list if p not in chosen_set]
                if avail:
                    r = float(_misc[_mi]); _mi += 1
                    out_prod_idx[pos] = avail[min(int(r * len(avail)), len(avail) - 1)]
                    chosen_set.add(int(out_prod_idx[pos]))
                    picked = True

            # --- Affinity: pick from same subcategory (rejection sampling) ---
            if not picked and j > 0 and _aff_rolls[pos] < _affinity:
                r = float(_misc[_mi]); _mi += 1
                anchor_j = min(int(r * j), j - 1)
                anchor = int(out_prod_idx[start + anchor_j])
                sc = int(_prod_subcat[anchor])
                sc_cdf = subcat_cdfs.get(sc)
                pool = subcat_to_indices[sc]
                pool_len = len(pool)
                if sc_cdf is not None and pool_len > 0:
                    for _ in range(50):
                        r = float(_misc[_mi]); _mi += 1
                        local_idx = int(np.searchsorted(sc_cdf, r))
                        if local_idx >= pool_len:
                            local_idx = pool_len - 1
                        pick = int(pool[local_idx])
                        if pick not in chosen_set:
                            out_prod_idx[pos] = pick
                            chosen_set.add(pick)
                            picked = True
                            break
                elif pool_len > 0:
                    r = float(_misc[_mi]); _mi += 1
                    pick = int(pool[min(int(r * pool_len), pool_len - 1)])
                    if pick not in chosen_set:
                        out_prod_idx[pos] = pick
                        chosen_set.add(pick)
                        picked = True

            # --- Default: use pre-generated global pick, check dedup ---
            if not picked:
                pick = int(out_prod_idx[pos])
                if pick in chosen_set:
                    for _ in range(200):
                        r = float(_misc[_mi]); _mi += 1
                        pick = int(np.searchsorted(global_cdf, r))
                        if pick >= _n_products:
                            pick = _n_products - 1
                        if pick not in chosen_set:
                            out_prod_idx[pos] = pick
                            break
                    else:
                        # Absolute fallback (astronomically rare)
                        for p in range(_n_products):
                            if p not in chosen_set:
                                out_prod_idx[pos] = p
                                break
                chosen_set.add(int(out_prod_idx[pos]))

            # Refill misc pool if running low
            if _mi + 60 > len(_misc):
                _misc = rng.random(_MISC_SIZE)
                _mi = 0

    # ------------------------------------------------------------------
    # BUILD FINAL OUTPUT (batch lookups)
    # ------------------------------------------------------------------
    out_wkey = np.arange(1, total_rows + 1, dtype=np.int64)
    out_pkey = prod_keys[out_prod_idx]
    out_price = prod_prices[out_prod_idx]
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
# Parallel path helpers
# ---------------------------------------------------------------------------

def _build_product_data_dict(
    prod_keys: np.ndarray,
    prod_prices: np.ndarray,
    prod_subcat: np.ndarray,
    product_weights: np.ndarray,
) -> Dict[str, Any]:
    """Serialize product arrays + CDFs into a picklable dict for workers."""
    n_products = len(prod_keys)

    global_cdf = np.cumsum(product_weights)
    global_cdf[-1] = 1.0

    unique_subcats = np.unique(prod_subcat)

    # Build sorted flat arrays of per-subcat pool indices and weights
    # so workers can receive everything as plain lists (no dict of arrays)
    subcat_keys_list: List[int] = []
    subcat_starts_list: List[int] = []
    subcat_ends_list: List[int] = []
    pool_indices_list: List[int] = []
    pool_weights_list: List[float] = []
    subcat_cdf_starts_list: List[int] = []
    subcat_cdf_ends_list: List[int] = []
    cdf_data_list: List[float] = []

    pool_cursor = 0
    cdf_cursor = 0
    for sc in unique_subcats:
        idx = np.where(prod_subcat == sc)[0]
        w = product_weights[idx]
        ws = float(w.sum())

        subcat_keys_list.append(int(sc))
        subcat_starts_list.append(pool_cursor)
        subcat_ends_list.append(pool_cursor + len(idx))
        pool_indices_list.extend(idx.tolist())
        pool_weights_list.extend(w.tolist())

        subcat_cdf_starts_list.append(cdf_cursor)
        if ws > 0:
            sc_cdf = np.cumsum(w / ws)
            sc_cdf[-1] = 1.0
            cdf_data_list.extend(sc_cdf.tolist())
            subcat_cdf_ends_list.append(cdf_cursor + len(sc_cdf))
            cdf_cursor += len(sc_cdf)
        else:
            subcat_cdf_ends_list.append(cdf_cursor)

        pool_cursor += len(idx)

    return {
        "prod_keys": prod_keys.tolist(),
        "prod_prices": prod_prices.tolist(),
        "prod_subcat": prod_subcat.tolist(),
        "global_cdf": global_cdf.tolist(),
        "subcat_keys": subcat_keys_list,
        "subcat_starts": subcat_starts_list,
        "subcat_ends": subcat_ends_list,
        "pool_indices": pool_indices_list,
        "pool_weights": pool_weights_list,
        "subcat_cdf_starts": subcat_cdf_starts_list,
        "subcat_cdf_ends": subcat_cdf_ends_list,
        "cdf_data": cdf_data_list,
    }


def _partition_participants(
    participant_cust_keys: np.ndarray,
    participant_earliest_ns: np.ndarray,
    participant_latest_ns: np.ndarray,
    items_per: np.ndarray,
    n_chunks: int,
) -> List[Dict[str, Any]]:
    """Split participants evenly across n_chunks; each chunk gets its slice."""
    n = len(participant_cust_keys)
    # Compute cumulative row offsets for WishlistKey assignment across chunks
    cumulative_offsets = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(items_per, out=cumulative_offsets[1:])

    chunks = []
    splits = np.array_split(np.arange(n), n_chunks)
    for indices in splits:
        if len(indices) == 0:
            continue
        global_offset_start = int(cumulative_offsets[indices[0]])
        chunk_items = items_per[indices]
        chunks.append({
            "cust_keys": participant_cust_keys[indices].tolist(),
            "earliest_ns": participant_earliest_ns[indices].tolist(),
            "latest_ns": participant_latest_ns[indices].tolist(),
            "items_per": chunk_items.tolist(),
            # offsets[0] = global row start for this chunk (for WishlistKey)
            "offsets": [global_offset_start],
        })
    return chunks


def _partition_purchased_pairs(
    purchased_pairs: pd.DataFrame,
    prod_key_to_idx: "pd.Series",
    participant_chunks: List[Dict[str, Any]],
) -> List[Dict[int, List[int]]]:
    """For each chunk, build {customer_key: [product_indices]} filtered to that chunk's customers."""
    # Build a full mapping: customer_key -> list[product_indices]
    full_map: Dict[int, List[int]] = {}
    if len(purchased_pairs) > 0:
        _pp_ckeys = purchased_pairs["CustomerKey"].to_numpy().astype(np.int64)
        _pp_pkeys = purchased_pairs["ProductKey"].to_numpy().astype(np.int64)
        _pp_idx_series = prod_key_to_idx.reindex(_pp_pkeys)
        _pp_valid = _pp_idx_series.notna().to_numpy()
        _pp_ckeys_valid = _pp_ckeys[_pp_valid]
        _pp_idxs_valid = _pp_idx_series[_pp_valid].to_numpy(dtype=np.int64)
        if len(_pp_ckeys_valid) > 0:
            _sort_order = np.argsort(_pp_ckeys_valid)
            _sorted_ckeys = _pp_ckeys_valid[_sort_order]
            _sorted_idxs = _pp_idxs_valid[_sort_order]
            _uniq_ckeys, _split_pos = np.unique(_sorted_ckeys, return_index=True)
            _split_groups = np.split(_sorted_idxs, _split_pos[1:])
            for _ck, _idxs in zip(_uniq_ckeys, _split_groups):
                full_map[int(_ck)] = _idxs.tolist()

    chunk_maps: List[Dict[int, List[int]]] = []
    for chunk in participant_chunks:
        ck_set = set(chunk["cust_keys"])
        chunk_maps.append({ck: v for ck, v in full_map.items() if ck in ck_set})
    return chunk_maps


def _merge_chunk_parquets(
    chunk_paths: List[Path],
    out_path: Path,
    schema: pa.Schema,
    delete_chunks: bool = True,
) -> None:
    """Concatenate chunk parquets in order into a single output parquet."""
    writer = pq.ParquetWriter(str(out_path), schema)
    for cp in sorted(chunk_paths):
        tbl = pq.read_table(str(cp))
        writer.write_table(tbl)
    writer.close()
    if delete_chunks:
        for cp in chunk_paths:
            try:
                cp.unlink()
            except OSError:
                pass
        # Remove chunk directory if empty
        if chunk_paths:
            try:
                chunk_paths[0].parent.rmdir()
            except OSError:
                pass


def _run_parallel(
    customers: pd.DataFrame,
    products: pd.DataFrame,
    product_weights: np.ndarray,
    purchased_pairs: pd.DataFrame,
    c: _WishlistsCfg,
    g_start: pd.Timestamp,
    g_end: pd.Timestamp,
    out_path: Path,
    workers: Optional[int] = None,
) -> int:
    """Parallel wishlist generation: distribute customers across worker pool."""
    from src.facts.sales.sales_worker.pool import PoolRunSpec, iter_imap_unordered
    from src.facts.wishlists.worker import _wishlist_worker_task

    n_cpus = max(1, cpu_count() - 1)
    if workers is not None and workers >= 1:
        n_cpus = min(n_cpus, workers)

    # --- Shared setup (runs in main process) ---
    rng = np.random.default_rng(c.seed)

    cust_keys, earliest_ns, latest_ns = _compute_customer_windows(
        customers, g_start, g_end, c.pre_browse_days,
    )
    n_customers = len(cust_keys)
    n_participants = max(1, int(round(n_customers * c.participation_rate)))

    participant_idx = rng.choice(n_customers, size=n_participants, replace=False)
    participant_idx.sort()

    participant_cust_keys = cust_keys[participant_idx]
    participant_earliest_ns = earliest_ns[participant_idx]
    participant_latest_ns = latest_ns[participant_idx]

    prod_keys = products["ProductKey"].to_numpy().astype(np.int64)
    prod_prices = products["ListPrice"].to_numpy().astype(np.float64)
    prod_subcat = products["SubcategoryKey"].to_numpy().astype(np.int64)
    n_products = len(prod_keys)

    items_per = rng.poisson(lam=c.avg_items, size=n_participants)
    items_per = np.clip(items_per, 1, min(c.max_items, n_products))
    total_rows = int(items_per.sum())

    # Determine actual chunk count
    n_chunks = min(n_participants, n_cpus * 2)
    n_chunks = max(2, n_chunks)
    n_workers = min(n_chunks, n_cpus)

    info(f"Wishlists parallel: {n_chunks} chunks across {n_workers} workers "
         f"({n_participants:,} participants, ~{total_rows:,} rows)")

    # Build shared product data dict (sent to all workers via pickling)
    product_data = _build_product_data_dict(prod_keys, prod_prices, prod_subcat, product_weights)

    # Build prod_key_to_idx for purchased_pairs partitioning
    prod_key_to_idx = pd.Series(np.arange(n_products), index=prod_keys)

    # Partition participants across chunks
    participant_chunks = _partition_participants(
        participant_cust_keys,
        participant_earliest_ns,
        participant_latest_ns,
        items_per,
        n_chunks,
    )
    n_chunks = len(participant_chunks)  # may be fewer if participants < n_chunks

    # Partition purchased_pairs per chunk
    chunk_purchased_maps = _partition_purchased_pairs(
        purchased_pairs, prod_key_to_idx, participant_chunks,
    )

    config_scalars = {
        "conversion_rate": c.conversion_rate,
        "affinity_strength": c.affinity_strength,
        "avg_items": c.avg_items,
        "max_items": c.max_items,
        "write_chunk_rows": c.write_chunk_rows,
    }

    # Prepare chunk output directory
    chunk_dir = out_path.parent / "_wishlist_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for idx, (pdata, pmap) in enumerate(zip(participant_chunks, chunk_purchased_maps)):
        chunk_out = str(chunk_dir / f"wishlist_chunk_{idx:05d}.parquet")
        tasks.append((
            idx,
            c.seed,
            n_chunks,
            pdata,
            product_data,
            pmap,
            config_scalars,
            chunk_out,
        ))

    pool_spec = PoolRunSpec(
        processes=n_workers,
        chunksize=1,
        label="wishlists",
    )

    total_written = 0
    completed = 0
    for result in iter_imap_unordered(tasks=tasks, task_fn=_wishlist_worker_task, spec=pool_spec):
        completed += 1
        total_written += result["rows"]

    info(f"Wishlists: {completed}/{n_chunks} chunks done ({total_written:,} rows)")

    # Merge chunk parquets into final output
    chunk_files = sorted(chunk_dir.glob("wishlist_chunk_*.parquet"))
    if chunk_files:
        _merge_chunk_parquets(chunk_files, out_path, _bridge_schema(), delete_chunks=True)

    return total_written


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_wishlist_pipeline(
    *,
    accumulator: WishlistAccumulator,
    parquet_dims: Path,
    fact_out: Path,
    cfg: Any,
    file_format: str = "parquet",
) -> Optional[Dict[str, Any]]:
    """Generate customer_wishlists into the facts output folder."""
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

    _cust_cols = ["CustomerKey", "CustomerStartDate", "CustomerEndDate"]
    _cust_schema_names = set(pq.read_schema(str(customers_fp)).names)
    if "IsCurrent" in _cust_schema_names:
        _cust_cols.append("IsCurrent")
    customers = pd.read_parquet(customers_fp, columns=_cust_cols)
    if "IsCurrent" in customers.columns:
        customers = customers[customers["IsCurrent"] == 1].copy()
        customers = customers.drop(columns=["IsCurrent"])
    _prod_cols = ["ProductKey", "ListPrice", "SubcategoryKey", "Brand"]
    _prod_schema_names = set(pq.read_schema(str(products_fp)).names) if hasattr(pq, "read_schema") else set()
    if "IsCurrent" in _prod_schema_names:
        _prod_cols.append("IsCurrent")
    products = pd.read_parquet(products_fp, columns=_prod_cols)
    if "IsCurrent" in products.columns:
        products = products[products["IsCurrent"] == 1].copy()
        products = products.drop(columns=["IsCurrent"])

    product_weights = _build_product_weights(products, parquet_dims)
    purchased_pairs = accumulator.finalize()

    wishlists_dir = fact_out / "customer_wishlists"
    wishlists_dir.mkdir(parents=True, exist_ok=True)
    pq_path = wishlists_dir / "customer_wishlists.parquet"

    # Estimate total rows to decide serial vs parallel
    _rng_est = np.random.default_rng(c.seed)
    _n_custs = len(customers)
    _n_participants_est = max(1, int(round(_n_custs * c.participation_rate)))
    _items_est = _rng_est.poisson(lam=c.avg_items, size=_n_participants_est)
    _items_est = np.clip(_items_est, 1, min(c.max_items, len(products)))
    _total_rows_est = int(_items_est.sum())

    # Workers from config (optional)
    _workers: Optional[int] = None
    _sales_cfg = getattr(cfg, "sales", None)
    if _sales_cfg is not None:
        _workers = getattr(_sales_cfg, "workers", None)

    if _total_rows_est >= WISHLIST_PARALLEL_THRESHOLD:
        n_rows = _run_parallel(
            customers=customers,
            products=products,
            product_weights=product_weights,
            purchased_pairs=purchased_pairs,
            c=c,
            g_start=g_start,
            g_end=g_end,
            out_path=pq_path,
            workers=_workers,
        )
    else:
        n_rows = _write_bridge(
            customers=customers,
            products=products,
            product_weights=product_weights,
            purchased_pairs=purchased_pairs,
            c=c,
            g_start=g_start,
            g_end=g_end,
            out_path=pq_path,
        )

    # Write CSV (chunked for large datasets)
    if file_format == "csv" and n_rows > 0:
        _csv_chunk = int(getattr(_sales_cfg, "chunk_size", 0)) if _sales_cfg else 0
        df = pq.read_table(str(pq_path)).to_pandas()
        pq_path.unlink()  # remove intermediate parquet in CSV mode

        if _csv_chunk and _csv_chunk > 0 and n_rows > _csv_chunk:
            n_files = 0
            for start in range(0, n_rows, _csv_chunk):
                chunk_path = wishlists_dir / f"customer_wishlists_{n_files:05d}.csv"
                df.iloc[start:start + _csv_chunk].to_csv(str(chunk_path), index=False)
                n_files += 1
        else:
            df.to_csv(str(wishlists_dir / "customer_wishlists.csv"), index=False)

    info(f"Customer wishlists: {n_rows:,} rows")

    return {
        "wishlists": str(wishlists_dir),
        "wishlists_rows": n_rows,
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
