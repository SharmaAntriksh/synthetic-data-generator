from __future__ import annotations

"""
Customer Wishlists (bridge table: Customers ↔ Products)

Writes:
  - customer_wishlists.parquet

Each row = one wishlisted product for a customer, with a snapshot of the
product price at the time the item was added.

AddedDate semantics:
  - Can be BEFORE CustomerStartDate (pre-purchase browsing intent)
  - Bounded by [global_start - pre_browse_days, CustomerEndDate or global_end]

Config (optional):
wishlists:
  enabled: true
  participation_rate: 0.35      # fraction of customers who wishlist
  avg_items: 3.5                # mean items per participant (Poisson λ)
  max_items: 20                 # cap per customer
  pre_browse_days: 90           # how many days before CustomerStartDate wishlisting can begin
  affinity_strength: 0.6        # probability of picking from same subcategory as a prior item
  seed: 500

Requires:
  customers.parquet, products.parquet in parquet_folder
  defaults.dates.start/end (or _defaults.dates.start/end) in cfg.

Power BI:
  Customers (1) ──< customer_wishlists >── (1) Products
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.logging_utils import info, skip, stage
from src.versioning.version_store import should_regenerate, save_version


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NS_PER_DAY: int = 86_400_000_000_000

_PRIORITY_VALUES = np.array(["High", "Medium", "Low"], dtype=object)
_PRIORITY_WEIGHTS = np.array([0.20, 0.50, 0.30])  # most items are Medium


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WishlistsCfg:
    enabled: bool = False
    participation_rate: float = 0.35
    avg_items: float = 3.5
    max_items: int = 20
    pre_browse_days: int = 90
    affinity_strength: float = 0.6
    seed: int = 500
    write_chunk_rows: int = 250_000


def _read_cfg(cfg: Dict[str, Any]) -> WishlistsCfg:
    wl = cfg.wishlists
    return WishlistsCfg(
        enabled=bool(wl.enabled),
        participation_rate=float(wl.participation_rate),
        avg_items=float(wl.avg_items),
        max_items=int(wl.max_items),
        pre_browse_days=int(wl.pre_browse_days),
        affinity_strength=float(wl.affinity_strength),
        seed=int(wl.seed if wl.seed is not None else 500),
        write_chunk_rows=int(wl.write_chunk_rows),
    )


def _parse_global_dates(cfg: Dict[str, Any]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Extract global start/end dates from cfg, with 3-level fallback."""
    wl = getattr(cfg, "wishlists", None)
    gd = getattr(wl, "global_dates", None) if wl else None

    if gd is None:
        defaults = getattr(cfg, "defaults", None)
        if defaults is None:
            defaults = getattr(cfg, "_defaults", None)
        gd = getattr(defaults, "dates", None)

    if gd is None:
        raise ValueError("Cannot resolve global dates for wishlists.")

    start_raw = gd.get("start", None) if isinstance(gd, dict) else getattr(gd, "start", None)
    end_raw = gd.get("end", None) if isinstance(gd, dict) else getattr(gd, "end", None)

    if start_raw is None or end_raw is None:
        raise ValueError("Global dates must have both 'start' and 'end'.")

    return pd.Timestamp(start_raw), pd.Timestamp(end_raw)


# ---------------------------------------------------------------------------
# Bridge schema
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
# Customer window extraction
# ---------------------------------------------------------------------------

def _compute_customer_windows(
    customers: pd.DataFrame,
    g_start: pd.Timestamp,
    g_end: pd.Timestamp,
    pre_browse_days: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (cust_keys, earliest_ns, latest_ns) arrays.

    earliest = max(global_start - pre_browse_days, CustomerStartDate - pre_browse_days)
    latest   = CustomerEndDate or global_end
    """
    cust_keys = customers["CustomerKey"].astype(np.int64).to_numpy()
    order = np.argsort(cust_keys)
    cust_keys = cust_keys[order]

    g_start_ns = np.int64(g_start.value)
    g_end_ns = np.int64(g_end.value)
    browse_offset_ns = np.int64(pre_browse_days) * _NS_PER_DAY

    # Start dates
    if "CustomerStartDate" in customers.columns:
        start_vals = customers["CustomerStartDate"].to_numpy().astype("datetime64[ns]")
        start_ns = start_vals.view(np.int64).copy()
        start_ns[np.isnat(start_vals)] = g_start_ns
    else:
        start_ns = np.full(len(cust_keys), g_start_ns, dtype=np.int64)

    # End dates
    if "CustomerEndDate" in customers.columns:
        end_vals = customers["CustomerEndDate"].to_numpy().astype("datetime64[ns]")
        end_ns = end_vals.view(np.int64).copy()
        end_ns[np.isnat(end_vals)] = g_end_ns
    else:
        end_ns = np.full(len(cust_keys), g_end_ns, dtype=np.int64)

    start_ns = start_ns[order]
    end_ns = end_ns[order]

    # Earliest wishlist date = CustomerStartDate - pre_browse_days, floored at global_start - pre_browse_days
    earliest_ns = start_ns - browse_offset_ns
    floor_ns = g_start_ns - browse_offset_ns
    earliest_ns = np.maximum(earliest_ns, floor_ns)

    # Latest wishlist date = CustomerEndDate (or global_end)
    latest_ns = np.clip(end_ns, g_start_ns, g_end_ns)

    # Ensure earliest < latest
    latest_ns = np.maximum(latest_ns, earliest_ns + _NS_PER_DAY)

    return cust_keys, earliest_ns, latest_ns


# ---------------------------------------------------------------------------
# Subcategory affinity product selection
# ---------------------------------------------------------------------------

def _pick_products_with_affinity(
    rng: np.random.Generator,
    n_items: int,
    n_products: int,
    prod_subcat: np.ndarray,
    subcat_to_indices: Dict[int, np.ndarray],
    affinity: float,
) -> np.ndarray:
    """
    Pick *n_items* unique product indices with subcategory affinity.

    - First product is chosen uniformly at random.
    - Each subsequent product: with probability *affinity*, pick from the
      same subcategory as a previously chosen product; otherwise pick
      uniformly from all products.
    - No duplicates within a single customer's wishlist.
    """
    chosen = np.empty(n_items, dtype=np.int64)
    chosen_set: set = set()

    # First item: random
    first = int(rng.integers(0, n_products))
    chosen[0] = first
    chosen_set.add(first)

    for j in range(1, n_items):
        if rng.random() < affinity:
            # Pick a random previously chosen item's subcategory
            anchor = chosen[rng.integers(0, j)]
            sc = prod_subcat[anchor]
            pool = subcat_to_indices[sc]
            # Filter out already-chosen indices
            available = pool[~np.isin(pool, list(chosen_set))]
            if len(available) > 0:
                pick = int(rng.choice(available))
                chosen[j] = pick
                chosen_set.add(pick)
                continue

        # Fallback: random from all products (excluding already chosen)
        # For efficiency, rejection-sample (expected <2 tries at low fill)
        for _ in range(100):
            pick = int(rng.integers(0, n_products))
            if pick not in chosen_set:
                chosen[j] = pick
                chosen_set.add(pick)
                break
        else:
            # Extremely unlikely: brute-force fallback
            remaining = np.setdiff1d(np.arange(n_products), np.array(list(chosen_set)))
            pick = int(rng.choice(remaining))
            chosen[j] = pick
            chosen_set.add(pick)

    return chosen


# ---------------------------------------------------------------------------
# Bridge writer (streaming)
# ---------------------------------------------------------------------------

def _write_bridge_streaming(
    customers: pd.DataFrame,
    products: pd.DataFrame,
    c: WishlistsCfg,
    g_start: pd.Timestamp,
    g_end: pd.Timestamp,
    out_path: Path,
) -> int:
    """Generate and write the customer_wishlists bridge table."""
    rng = np.random.default_rng(c.seed)
    schema = _bridge_schema()

    cust_keys, earliest_ns, latest_ns = _compute_customer_windows(
        customers, g_start, g_end, c.pre_browse_days,
    )

    n_customers = len(cust_keys)
    n_participants = max(1, int(round(n_customers * c.participation_rate)))

    # Choose participating customers
    participant_idx = rng.choice(n_customers, size=n_participants, replace=False)
    participant_idx.sort()

    # Product keys, prices, and subcategory index
    prod_keys = products["ProductKey"].to_numpy().astype(np.int64)
    prod_prices = products["UnitPrice"].to_numpy().astype(np.float64)
    prod_subcat = products["SubcategoryKey"].to_numpy().astype(np.int64)
    n_products = len(prod_keys)

    # Build subcategory → product index lookup for affinity selection
    unique_subcats = np.unique(prod_subcat)
    subcat_to_indices: Dict[int, np.ndarray] = {
        sc: np.where(prod_subcat == sc)[0] for sc in unique_subcats
    }
    affinity = c.affinity_strength

    # Number of items per participant (Poisson, clamped to [1, max_items])
    items_per = rng.poisson(lam=c.avg_items, size=n_participants)
    items_per = np.clip(items_per, 1, min(c.max_items, n_products))

    total_rows = int(items_per.sum())

    # Pre-allocate output arrays
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

        # Choose products with subcategory affinity (no duplicates)
        chosen_prod_idx = _pick_products_with_affinity(
            rng, n_items, n_products, prod_subcat, subcat_to_indices, affinity,
        )

        # Random dates within window
        span = l_ns - e_ns
        if span <= 0:
            span = _NS_PER_DAY
        offsets = rng.integers(0, max(1, span), size=n_items)
        dates = e_ns + offsets

        # Priorities
        priorities = rng.choice(_PRIORITY_VALUES, size=n_items, p=_PRIORITY_WEIGHTS)

        # Quantities (most wishlist 1, some wishlist 2-3)
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

    # Convert ns timestamps to date32
    out_dates_dt = out_date_ns.view("datetime64[ns]").astype("datetime64[ms]")

    # Write in chunks via PyArrow
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
# Runner
# ---------------------------------------------------------------------------

def run_wishlists(cfg: Dict[str, Any], parquet_folder: Path) -> Dict[str, Any]:
    parquet_folder = Path(parquet_folder)

    c = _read_cfg(cfg)
    if not c.enabled:
        skip("Wishlists disabled; skipping.")
        return {"_regenerated": False, "reason": "disabled"}

    out_bridge = parquet_folder / "customer_wishlists.parquet"

    # Locate upstream parquet files
    customers_fp = parquet_folder / "customers.parquet"
    if not customers_fp.exists():
        alt = parquet_folder / "Customers.parquet"
        if alt.exists():
            customers_fp = alt
        else:
            raise FileNotFoundError(
                f"Customers parquet not found at {parquet_folder}. "
                "Expected customers.parquet (or Customers.parquet)."
            )

    products_fp = parquet_folder / "products.parquet"
    if not products_fp.exists():
        alt = parquet_folder / "Products.parquet"
        if alt.exists():
            products_fp = alt
        else:
            raise FileNotFoundError(
                f"Products parquet not found at {parquet_folder}. "
                "Expected products.parquet (or Products.parquet)."
            )

    # Version hash includes upstream file metadata
    cust_st = os.stat(customers_fp)
    prod_st = os.stat(products_fp)
    version_cfg = dict(cfg.wishlists)
    version_cfg["_schema_version"] = 1
    version_cfg["_upstream_customers_sig"] = {
        "path": str(customers_fp),
        "size": int(cust_st.st_size),
        "mtime_ns": int(cust_st.st_mtime_ns),
    }
    version_cfg["_upstream_products_sig"] = {
        "path": str(products_fp),
        "size": int(prod_st.st_size),
        "mtime_ns": int(prod_st.st_mtime_ns),
    }

    if out_bridge.exists() and not should_regenerate("wishlists", version_cfg, out_bridge):
        skip("Wishlists up-to-date")
        return {"_regenerated": False, "reason": "version"}

    with stage("Generating Customer Wishlists"):
        g_start, g_end = _parse_global_dates(cfg)

        customers = pd.read_parquet(
            customers_fp,
            columns=["CustomerKey", "CustomerStartDate", "CustomerEndDate"],
        )
        products = pd.read_parquet(
            products_fp,
            columns=["ProductKey", "UnitPrice", "SubcategoryKey"],
        )

        n_rows = _write_bridge_streaming(
            customers=customers,
            products=products,
            c=c,
            g_start=g_start,
            g_end=g_end,
            out_path=out_bridge,
        )
        save_version("wishlists", version_cfg, out_bridge)
        info(f"Customer wishlists written: {out_bridge} ({n_rows:,} rows)")

    return {
        "_regenerated": True,
        "bridge": str(out_bridge),
        "bridge_rows": n_rows,
    }
