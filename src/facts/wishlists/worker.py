"""Wishlist multiprocessing worker.

Each worker receives a slice of participant customers plus shared read-only
product data, generates wishlist rows for those customers, and writes a
chunk parquet file.  Returns lightweight stats — no large arrays cross the
IPC boundary on return.

The worker is a top-level function so it is importable / pickleable on
Windows (spawn start method).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# Constants (duplicated from runner to avoid circular imports)
# ---------------------------------------------------------------------------

_NS_PER_DAY: int = 86_400_000_000_000

_PRIORITY_VALUES = np.array(["High", "Medium", "Low"], dtype=object)
_PRIORITY_WEIGHTS = np.array([0.20, 0.50, 0.30])


def _wishlist_worker_task(args: Tuple) -> Dict[str, Any]:
    """
    Worker entry point — must be a top-level function for Windows spawn.

    Args (tuple):
        chunk_idx       : int — chunk sequence number
        seed            : int — base seed from config
        n_chunks        : int — total number of chunks (for SeedSequence.spawn)
        participant_data: dict with keys:
            cust_keys    : list[int]
            earliest_ns  : list[int]
            latest_ns    : list[int]
            items_per    : list[int]
            offsets      : list[int]  — length n_participants+1, cumulative
        product_data    : dict with keys:
            prod_keys    : list[int]
            prod_prices  : list[float]
            prod_subcat  : list[int]
            global_cdf   : list[float]
            subcat_keys  : list[int]         — unique subcategory ids
            subcat_starts: list[int]         — start of each subcat in sorted pool
            subcat_ends  : list[int]         — exclusive end
            pool_indices : list[int]         — sorted per-subcat pool indices
            pool_weights : list[float]       — weights parallel to pool_indices
            subcat_cdf_starts: list[int]     — start of each subcat's CDF in cdf_data
            subcat_cdf_ends  : list[int]
            cdf_data     : list[float]       — concatenated per-subcat CDFs
        purchased_pairs : dict[int, list[int]]  customer_key -> list of product indices
        config_scalars  : dict with conversion_rate, affinity_strength, avg_items,
                          max_items, write_chunk_rows
        out_path        : str — path to write chunk parquet

    Returns:
        dict with chunk_idx, rows, wkey_start (first WishlistKey in this chunk)
    """
    (
        chunk_idx,
        seed,
        n_chunks,
        participant_data,
        product_data,
        purchased_pairs,
        config_scalars,
        out_path,
    ) = args

    # --- Derive per-chunk RNG via SeedSequence for reproducible independence ---
    ss = np.random.SeedSequence(seed)
    child_seeds = ss.spawn(n_chunks)
    rng = np.random.default_rng(child_seeds[chunk_idx])

    # --- Unpack participant data ---
    cust_keys: np.ndarray = np.asarray(participant_data["cust_keys"], dtype=np.int64)
    earliest_ns: np.ndarray = np.asarray(participant_data["earliest_ns"], dtype=np.int64)
    latest_ns: np.ndarray = np.asarray(participant_data["latest_ns"], dtype=np.int64)
    items_per: np.ndarray = np.asarray(participant_data["items_per"], dtype=np.int64)
    offsets: np.ndarray = np.asarray(participant_data["offsets"], dtype=np.int64)

    n_participants = len(cust_keys)
    total_rows = int(items_per.sum()) if n_participants > 0 else 0

    # --- Unpack product data ---
    prod_keys: np.ndarray = np.asarray(product_data["prod_keys"], dtype=np.int64)
    prod_prices: np.ndarray = np.asarray(product_data["prod_prices"], dtype=np.float64)
    prod_subcat: np.ndarray = np.asarray(product_data["prod_subcat"], dtype=np.int64)
    global_cdf: np.ndarray = np.asarray(product_data["global_cdf"], dtype=np.float64)
    n_products = len(prod_keys)

    subcat_keys: np.ndarray = np.asarray(product_data["subcat_keys"], dtype=np.int64)
    subcat_starts: np.ndarray = np.asarray(product_data["subcat_starts"], dtype=np.int64)
    subcat_ends: np.ndarray = np.asarray(product_data["subcat_ends"], dtype=np.int64)
    pool_indices: np.ndarray = np.asarray(product_data["pool_indices"], dtype=np.int64)
    pool_weights: np.ndarray = np.asarray(product_data["pool_weights"], dtype=np.float64)
    subcat_cdf_starts: np.ndarray = np.asarray(product_data["subcat_cdf_starts"], dtype=np.int64)
    subcat_cdf_ends: np.ndarray = np.asarray(product_data["subcat_cdf_ends"], dtype=np.int64)
    cdf_data: np.ndarray = np.asarray(product_data["cdf_data"], dtype=np.float64)

    # Build fast subcategory lookup: sc_id -> (pool slice start, pool slice end)
    sc_idx_map: Dict[int, int] = {int(sc): i for i, sc in enumerate(subcat_keys)}

    wkey_start = int(offsets[0]) + 1  # 1-based global WishlistKey offset

    if total_rows == 0 or n_participants == 0:
        return {"chunk_idx": chunk_idx, "rows": 0, "wkey_start": wkey_start}

    # --- Batch random generation ---
    _base_prod_idx = np.searchsorted(global_cdf, rng.random(total_rows))
    np.clip(_base_prod_idx, 0, n_products - 1, out=_base_prod_idx)

    _conv_rolls = rng.random(total_rows)
    _aff_rolls = rng.random(total_rows)

    _MISC_SIZE = max(total_rows * 6, 1024)
    _misc = rng.random(_MISC_SIZE)
    _mi = 0

    _date_rands = rng.random(total_rows)

    _prio_cdf = np.cumsum(_PRIORITY_WEIGHTS)
    _prio_cdf[-1] = 1.0
    _prio_idx = np.searchsorted(_prio_cdf, rng.random(total_rows))
    np.clip(_prio_idx, 0, len(_PRIORITY_VALUES) - 1, out=_prio_idx)
    out_priority = _PRIORITY_VALUES[_prio_idx]

    _qty_options = np.array([1, 1, 1, 1, 2, 2, 3], dtype=np.int32)
    out_quantity = _qty_options[rng.integers(0, 7, size=total_rows)]

    # --- Output arrays ---
    out_prod_idx = _base_prod_idx.copy()
    out_ckey = np.empty(total_rows, dtype=np.int64)
    out_date_ns = np.empty(total_rows, dtype=np.int64)

    _conversion_rate = float(config_scalars["conversion_rate"])
    _affinity = float(config_scalars["affinity_strength"])
    _n_products = n_products

    # Pre-compute local offsets (O(n) cumsum instead of O(n²) repeated sums)
    local_offsets = np.zeros(n_participants + 1, dtype=np.int64)
    np.cumsum(items_per, out=local_offsets[1:])

    # --- Per-customer product selection loop ---
    for i in range(n_participants):
        start = int(local_offsets[i])
        end = int(local_offsets[i + 1])
        n_items = end - start
        ck = int(cust_keys[i])

        out_ckey[start:end] = ck

        e_ns = int(earliest_ns[i])
        l_ns = int(latest_ns[i])
        span = l_ns - e_ns
        if span <= 0:
            span = _NS_PER_DAY
        out_date_ns[start:end] = e_ns + (_date_rands[start:end] * span).astype(np.int64)

        purch_list: List[int] = purchased_pairs.get(ck) or []
        has_purch = len(purch_list) > 0

        chosen_set: set = set()
        for j in range(n_items):
            pos = start + j
            picked = False

            # Conversion: pick from actual purchases
            if has_purch and _conv_rolls[pos] < _conversion_rate:
                avail = [p for p in purch_list if p not in chosen_set]
                if avail:
                    r = float(_misc[_mi]); _mi += 1
                    out_prod_idx[pos] = avail[min(int(r * len(avail)), len(avail) - 1)]
                    chosen_set.add(int(out_prod_idx[pos]))
                    picked = True

            # Affinity: pick from same subcategory
            if not picked and j > 0 and _aff_rolls[pos] < _affinity:
                r = float(_misc[_mi]); _mi += 1
                anchor_j = min(int(r * j), j - 1)
                anchor = int(out_prod_idx[start + anchor_j])
                sc = int(prod_subcat[anchor])
                sc_i = sc_idx_map.get(sc)
                if sc_i is not None:
                    ps = int(subcat_starts[sc_i])
                    pe = int(subcat_ends[sc_i])
                    pool_len = pe - ps
                    cs = int(subcat_cdf_starts[sc_i])
                    ce = int(subcat_cdf_ends[sc_i])
                    has_cdf = ce > cs
                    if has_cdf and pool_len > 0:
                        sc_cdf_slice = cdf_data[cs:ce]
                        for _ in range(50):
                            r = float(_misc[_mi]); _mi += 1
                            local_idx = int(np.searchsorted(sc_cdf_slice, r))
                            if local_idx >= pool_len:
                                local_idx = pool_len - 1
                            pick = int(pool_indices[ps + local_idx])
                            if pick not in chosen_set:
                                out_prod_idx[pos] = pick
                                chosen_set.add(pick)
                                picked = True
                                break
                    elif pool_len > 0:
                        r = float(_misc[_mi]); _mi += 1
                        pick = int(pool_indices[ps + min(int(r * pool_len), pool_len - 1)])
                        if pick not in chosen_set:
                            out_prod_idx[pos] = pick
                            chosen_set.add(pick)
                            picked = True

            # Default: use pre-generated global pick, dedup
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
                        for p in range(_n_products):
                            if p not in chosen_set:
                                out_prod_idx[pos] = p
                                break
                chosen_set.add(int(out_prod_idx[pos]))

            if _mi + 60 > len(_misc):
                _misc = rng.random(_MISC_SIZE)
                _mi = 0

    # --- Build output arrays ---
    # WishlistKey: use global offset so keys are unique across chunks
    global_row_start = int(offsets[0])  # how many rows precede this chunk
    out_wkey = np.arange(global_row_start + 1, global_row_start + total_rows + 1, dtype=np.int64)
    out_pkey = prod_keys[out_prod_idx]
    out_price = prod_prices[out_prod_idx]
    out_dates_dt = out_date_ns.view("datetime64[ns]").astype("datetime64[ms]")

    # --- Write chunk parquet ---
    schema = pa.schema([
        pa.field("WishlistKey", pa.int64()),
        pa.field("CustomerKey", pa.int64()),
        pa.field("ProductKey", pa.int64()),
        pa.field("AddedDate", pa.date32()),
        pa.field("Priority", pa.string()),
        pa.field("Quantity", pa.int32()),
        pa.field("NetPrice", pa.float64()),
    ])

    write_chunk_rows = int(config_scalars.get("write_chunk_rows", 250_000))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    writer = pq.ParquetWriter(out_path, schema)
    chunk = write_chunk_rows
    for s in range(0, total_rows, chunk):
        e = min(s + chunk, total_rows)
        batch = pa.record_batch(
            [
                pa.array(out_wkey[s:e], type=pa.int64()),
                pa.array(out_ckey[s:e], type=pa.int64()),
                pa.array(out_pkey[s:e], type=pa.int64()),
                pa.array(out_dates_dt[s:e], type=pa.date32()),
                pa.array(out_priority[s:e], type=pa.string()),
                pa.array(out_quantity[s:e], type=pa.int32()),
                pa.array(out_price[s:e], type=pa.float64()),
            ],
            schema=schema,
        )
        writer.write_batch(batch)
    writer.close()

    return {
        "chunk_idx": chunk_idx,
        "rows": total_rows,
        "wkey_start": wkey_start,
    }
