from __future__ import annotations

"""
Superpowers (FAST + low-memory; dimensions-only)

Writes:
  - superpowers.parquet
  - customer_superpowers.parquet

Bridge is written STREAMING with pyarrow ParquetWriter (does not hold the
whole bridge in RAM).  Per-power acquisition dates are biased by rarity
using beta-distributed offsets.

Bridge validity semantics:
  ValidFromDate = acquired date of that power (per row)
  ValidToDate   = customer end date (or global end date)

Bridge analytical columns:
  PowerLevel       – rarity-biased integer level (2-5)
  IsPrimaryFlag    – 1 for the highest-level power per customer
  AcquiredDate     – when the customer gained this power
  PowerRank        – ordinal rank within customer portfolio (1 = strongest)
  AcquisitionOrder – chronological order of acquisition (1 = first gained)
  RarityWeight     – normalised selection probability used for this power
  DaysToAcquire    – days between CustomerStartDate and AcquiredDate
  IsLatestPower    – 1 for the most recently acquired power per customer

Config (optional)
superpowers:
  enabled: true
  generate_bridge: true
  powers_count: 40
  powers_per_customer_min: 1
  powers_per_customer_max: 4
  include_power_level: true
  include_primary_flag: true
  include_acquired_date: true
  seed: 123
  write_chunk_rows: 250000

Requires:
  defaults.dates.start/end (or _defaults.dates.start/end) in cfg.

Power BI:
  Customers (1) ──< customer_superpowers >── (1) Superpowers
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.logging_utils import info, skip, stage
from src.versioning.version_store import should_regenerate, save_version


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

# Each entry: (PowerName, PowerType, Universe, Rarity, IconicExamples)
POWERS_CATALOG: List[Tuple[str, str, str, str, str]] = [
    ("Super Strength", "Physical", "Marvel", "Common", "Hulk, Thor, Captain Marvel"),
    ("Flight", "Physical", "Marvel", "Common", "Iron Man, Captain Marvel, Falcon"),
    ("Enhanced Agility", "Physical", "Marvel", "Common", "Spider-Man, Black Panther"),
    ("Web-Slinging", "Physical", "Marvel", "Rare", "Spider-Man"),
    ("Spider-Sense", "Psychic", "Marvel", "Rare", "Spider-Man"),
    ("Vibranium Mastery", "Tech", "Marvel", "Rare", "Black Panther, Captain America"),
    ("Energy Blasts", "Cosmic", "Marvel", "Common", "Iron Man, Captain Marvel"),
    ("Lightning Manipulation", "Elemental", "Marvel", "Rare", "Thor"),
    ("Weather Control", "Elemental", "Marvel", "Legendary", "Storm"),
    ("Telepathy", "Psychic", "Marvel", "Legendary", "Professor X, Jean Grey"),
    ("Telekinesis", "Psychic", "Marvel", "Legendary", "Jean Grey"),
    ("Healing Factor", "Biological", "Marvel", "Rare", "Wolverine, Deadpool"),
    ("Shapeshifting", "Biological", "Marvel", "Rare", "Mystique"),
    ("Size Shifting", "Tech", "Marvel", "Rare", "Ant-Man, Wasp"),
    ("Time Manipulation", "Cosmic", "Marvel", "Legendary", "Doctor Strange"),
    ("Reality Warping", "Cosmic", "Marvel", "Legendary", "Scarlet Witch"),
    ("Mystic Arts", "Magic", "Marvel", "Legendary", "Doctor Strange"),
    ("Genius Inventor", "Tech", "Marvel", "Common", "Tony Stark, Shuri"),
    ("Arc Reactor Tech", "Tech", "Marvel", "Rare", "Iron Man"),
    ("Phasing", "Cosmic", "Marvel", "Rare", "Vision"),
    ("Invisibility", "Cosmic", "Marvel", "Rare", "Invisible Woman"),
    ("Elasticity", "Biological", "Marvel", "Rare", "Mr. Fantastic"),
    ("Super Speed", "Physical", "Marvel", "Rare", "Quicksilver"),
    ("Master Martial Artist", "Physical", "Marvel", "Common", "Shang-Chi"),
    ("Shield Mastery", "Training", "Marvel", "Common", "Captain America"),
    ("Force Sensitivity", "Cosmic", "Star Wars", "Rare", "Luke, Rey"),
    ("Force Telekinesis", "Psychic", "Star Wars", "Legendary", "Yoda"),
    ("Mind Trick", "Psychic", "Star Wars", "Rare", "Obi-Wan"),
    ("Precognition", "Psychic", "Star Wars", "Rare", "Jedi (vibe)"),
    ("Lightsaber Mastery", "Training", "Star Wars", "Common", "Jedi/Sith"),
    ("Dark Side Channeling", "Cosmic", "Star Wars", "Legendary", "Vader, Palpatine"),
    ("Cryokinesis", "Elemental", "Disney", "Rare", "Elsa (Frozen)"),
    ("Healing Light", "Magic", "Disney", "Rare", "Rapunzel (Tangled)"),
    ("Ocean Calling", "Magic", "Disney", "Rare", "Moana (vibe)"),
    ("Super Stretch", "Biological", "Pixar", "Rare", "Elastigirl (Incredibles)"),
    ("Force Fields", "Cosmic", "Pixar", "Rare", "Violet (Incredibles)"),
    ("Super Speed (Dash)", "Physical", "Pixar", "Rare", "Dash (Incredibles)"),
    ("Tech Hero Armor", "Tech", "Disney", "Rare", "Baymax (Big Hero 6)"),
    ("Microbot Control", "Tech", "Disney", "Rare", "Hiro (Big Hero 6)"),
    ("Animal Communication", "Psychic", "Disney", "Common", "Various"),
    ("Potion Brewing", "Magic", "Disney", "Common", "Various"),
    ("Illusions", "Magic", "Marvel", "Rare", "Loki (vibe)"),
    ("Portal Travel", "Cosmic", "Marvel", "Legendary", "Doctor Strange (vibe)"),
    ("Magnetism Control", "Elemental", "Marvel", "Rare", "Magneto (vibe)"),
]

_RARITY_WEIGHT = {"Common": 6.0, "Rare": 2.5, "Legendary": 1.0}

_NS_PER_DAY: int = 86_400_000_000_000

_RARITY_BETA: Dict[str, Tuple[float, float]] = {
    "Common": (2.0, 5.0),
    "Legendary": (5.0, 2.0),
}
_BETA_FALLBACK: Tuple[float, float] = (2.0, 2.0)

_RARITY_LEVEL_BOUNDS: Dict[str, Tuple[int, int]] = {
    "Common": (2, 5),
    "Rare": (3, 6),
    "Legendary": (4, 6),
}
_LEVEL_BOUNDS_FALLBACK: Tuple[int, int] = (3, 6)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SuperpowersCfg:
    enabled: bool = True
    generate_bridge: bool = True
    powers_count: int = 40
    per_customer_min: int = 1
    per_customer_max: int = 4
    include_power_level: bool = True
    include_primary_flag: bool = True
    include_acquired_date: bool = True
    seed: int = 123
    write_chunk_rows: int = 250_000


def _read_cfg(cfg: Dict[str, Any]) -> SuperpowersCfg:
    sp = cfg.superpowers
    return SuperpowersCfg(
        enabled=bool(sp.enabled),
        generate_bridge=bool(sp.generate_bridge),
        powers_count=int(sp.powers_count),
        per_customer_min=int(sp.powers_per_customer_min),
        per_customer_max=int(sp.powers_per_customer_max),
        include_power_level=bool(sp.include_power_level),
        include_primary_flag=bool(sp.include_primary_flag),
        include_acquired_date=bool(sp.include_acquired_date),
        seed=int(sp.seed if sp.seed is not None else 123),
        write_chunk_rows=int(sp.write_chunk_rows),
    )


def _parse_global_dates(cfg: Dict[str, Any]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Resolve timeline dates from (priority order):
      1) cfg.superpowers.global_dates  (runner injected)
      2) cfg.defaults.dates
      3) cfg._defaults.dates           (backward compatibility)
    """
    sp = cfg.superpowers
    gd = sp.global_dates if sp is not None else None
    if isinstance(gd, Mapping) and gd.get("start") and gd.get("end"):
        start = pd.to_datetime(gd["start"]).normalize()
        end = pd.to_datetime(gd["end"]).normalize()
        if end < start:
            raise ValueError("defaults.dates.end must be >= defaults.dates.start")
        return start, end

    defaults = cfg.defaults if hasattr(cfg, "defaults") else getattr(cfg, "_defaults", None)
    if defaults is None:
        raise ValueError("Missing defaults.dates.start/end (or _defaults.dates.start/end)")
    d = defaults.dates
    d_start = d.start if hasattr(d, "start") else None
    d_end = d.end if hasattr(d, "end") else None
    if not d_start or not d_end:
        raise ValueError("Missing defaults.dates.start/end (or _defaults.dates.start/end)")
    start = pd.to_datetime(d_start).normalize()
    end = pd.to_datetime(d_end).normalize()
    if end < start:
        raise ValueError("defaults.dates.end must be >= defaults.dates.start")
    return start, end


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_dim_superpowers(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Build the superpowers dimension table (columnar, no list-of-dict)."""
    c = _read_cfg(cfg)
    k = min(max(c.powers_count, 1), len(POWERS_CATALOG))
    catalog = POWERS_CATALOG[:k]

    return pd.DataFrame({
        "SuperpowerKey": np.arange(1, k + 1, dtype=np.int64),
        "SuperpowerName": [row[0] for row in catalog],
        "PowerType": [row[1] for row in catalog],
        "Universe": [row[2] for row in catalog],
        "Rarity": [row[3] for row in catalog],
        "IconicExamples": [row[4] for row in catalog],
        "IsActiveFlag": np.ones(k, dtype=np.int8),
    })


def _bridge_schema(
    include_power_level: bool,
    include_primary_flag: bool,
    include_acquired_date: bool,
) -> pa.Schema:
    fields: List[pa.Field] = [
        pa.field("CustomerKey", pa.int64()),
        pa.field("SuperpowerKey", pa.int32()),
        pa.field("ValidFromDate", pa.timestamp("ns")),
        pa.field("ValidToDate", pa.timestamp("ns")),
    ]
    if include_power_level:
        fields.append(pa.field("PowerLevel", pa.int8()))
    if include_primary_flag:
        fields.append(pa.field("IsPrimaryFlag", pa.int8()))
    if include_acquired_date:
        fields.append(pa.field("AcquiredDate", pa.timestamp("ns")))

    fields.append(pa.field("PowerRank", pa.int8()))
    fields.append(pa.field("AcquisitionOrder", pa.int8()))
    fields.append(pa.field("RarityWeight", pa.float32()))
    fields.append(pa.field("DaysToAcquire", pa.int32()))
    fields.append(pa.field("IsLatestPower", pa.int8()))
    return pa.schema(fields)


def _compute_rarity_weights(rarity_arr: np.ndarray) -> np.ndarray:
    """Map rarity strings to normalised probability weights (numpy-native)."""
    weights = np.ones(len(rarity_arr), dtype=np.float64)
    for rarity, w in _RARITY_WEIGHT.items():
        weights[rarity_arr == rarity] = w
    total = weights.sum()
    if total > 0:
        weights /= total
    return weights


def _compute_customer_windows(
    customers: pd.DataFrame,
    g_start: pd.Timestamp,
    g_end: pd.Timestamp,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract sorted (CustomerKey, start_ns, end_ns) arrays from the customers
    DataFrame.  All dates are clamped to [g_start, g_end] and returned as
    int64 nanosecond timestamps.
    """
    cust_keys = customers["CustomerKey"].astype(np.int64).to_numpy()
    order = np.argsort(cust_keys)
    cust_keys = cust_keys[order]

    g_start_ns = np.int64(g_start.value)
    g_end_ns = np.int64(g_end.value)

    has_start = "CustomerStartDate" in customers.columns
    has_end = "CustomerEndDate" in customers.columns

    if has_start:
        start_vals = customers["CustomerStartDate"].to_numpy().astype("datetime64[ns]")
        start_ns = start_vals.view(np.int64).copy()
        start_ns[np.isnat(start_vals)] = g_start_ns
        start_ns = np.clip(start_ns, g_start_ns, g_end_ns)
    else:
        start_ns = np.full(len(cust_keys), g_start_ns, dtype=np.int64)

    if has_end:
        end_vals = customers["CustomerEndDate"].to_numpy().astype("datetime64[ns]")
        end_ns = end_vals.view(np.int64).copy()
        end_ns[np.isnat(end_vals)] = g_end_ns
        end_ns = np.clip(end_ns, g_start_ns, g_end_ns)
    else:
        end_ns = np.full(len(cust_keys), g_end_ns, dtype=np.int64)

    start_ns = start_ns[order]
    end_ns = end_ns[order]
    end_ns = np.maximum(end_ns, start_ns)

    return cust_keys, start_ns, end_ns


def _vectorized_levels(
    rng: np.random.Generator,
    chosen_rarity: np.ndarray,
    count: int,
) -> np.ndarray:
    """
    Assign power levels for *count* chosen powers in one vectorised
    ``rng.integers`` call (array low/high bounds).
    """
    lo = np.empty(count, dtype=np.int64)
    hi = np.empty(count, dtype=np.int64)
    for rarity, (l, h) in _RARITY_LEVEL_BOUNDS.items():
        mask = chosen_rarity == rarity
        lo[mask] = l
        hi[mask] = h
    unknown = ~np.isin(chosen_rarity, list(_RARITY_LEVEL_BOUNDS))
    if unknown.any():
        fb_lo, fb_hi = _LEVEL_BOUNDS_FALLBACK
        lo[unknown] = fb_lo
        hi[unknown] = fb_hi

    return rng.integers(lo, hi).astype(np.int8)


def _acquired_dates_ns(
    rng: np.random.Generator,
    chosen_rarity: np.ndarray,
    count: int,
    start_ns: int,
    span_days: int,
) -> np.ndarray:
    """
    Compute acquired-date nanosecond timestamps for *count* powers belonging
    to one customer.  Uses beta-distributed offsets biased by rarity.
    Vectorised by rarity bucket rather than per-element draws.
    """
    if span_days <= 0:
        return np.full(count, start_ns, dtype=np.int64)

    u = np.empty(count, dtype=np.float64)
    matched = np.zeros(count, dtype=bool)
    for rarity_str, (a, b) in _RARITY_BETA.items():
        mask = chosen_rarity == rarity_str
        n = int(mask.sum())
        if n:
            u[mask] = rng.beta(a, b, size=n)
            matched |= mask

    n_fb = int((~matched).sum())
    if n_fb:
        a, b = _BETA_FALLBACK
        u[~matched] = rng.beta(a, b, size=n_fb)

    offsets = np.clip(np.round(u * span_days).astype(np.int64), 0, span_days)
    return start_ns + offsets * _NS_PER_DAY


# ---------------------------------------------------------------------------
# Bridge writer (streaming, vectorised inner loop)
# ---------------------------------------------------------------------------

def _write_bridge_streaming(
    customers: pd.DataFrame,
    dim_powers: pd.DataFrame,
    c: SuperpowersCfg,
    g_start: pd.Timestamp,
    g_end: pd.Timestamp,
    out_bridge: Path,
) -> int:
    """
    Stream-write customer_superpowers bridge to parquet.
    Returns number of rows written.

    Uses pre-allocated numpy chunk buffers with vectorised slice assignment
    instead of per-element Python list appends.
    """
    if "CustomerKey" not in customers.columns:
        raise KeyError("customers must include CustomerKey")
    if "SuperpowerKey" not in dim_powers.columns or "Rarity" not in dim_powers.columns:
        raise KeyError("dim_powers must include SuperpowerKey and Rarity")

    power_keys = dim_powers["SuperpowerKey"].astype(np.int32).to_numpy()
    rarity_arr = dim_powers["Rarity"].astype(str).to_numpy()
    n_powers = len(power_keys)
    w = _compute_rarity_weights(rarity_arr)

    per_power_weight = w.astype(np.float32)

    cust_keys, cust_start_ns, cust_end_ns = _compute_customer_windows(
        customers, g_start, g_end,
    )

    mn = max(c.per_customer_min, 1)
    mx = max(c.per_customer_max, mn)
    write_chunk_rows = max(c.write_chunk_rows, 10_000)
    max_per_cust = min(mx, n_powers)

    rng = np.random.default_rng(c.seed)
    schema = _bridge_schema(c.include_power_level, c.include_primary_flag, c.include_acquired_date)

    buf_cap = write_chunk_rows + max_per_cust
    buf_ck = np.empty(buf_cap, dtype=np.int64)
    buf_sk = np.empty(buf_cap, dtype=np.int32)
    buf_vf = np.empty(buf_cap, dtype=np.int64)
    buf_vt = np.empty(buf_cap, dtype=np.int64)

    buf_lvl: Optional[np.ndarray] = np.empty(buf_cap, dtype=np.int8) if c.include_power_level else None
    buf_pri: Optional[np.ndarray] = np.empty(buf_cap, dtype=np.int8) if c.include_primary_flag else None
    buf_acq: Optional[np.ndarray] = np.empty(buf_cap, dtype=np.int64) if c.include_acquired_date else None

    buf_rank = np.empty(buf_cap, dtype=np.int8)
    buf_acq_ord = np.empty(buf_cap, dtype=np.int8)
    buf_rw = np.empty(buf_cap, dtype=np.float32)
    buf_days = np.empty(buf_cap, dtype=np.int32)
    buf_latest = np.empty(buf_cap, dtype=np.int8)

    total_rows = 0
    pos = 0

    def flush(writer: pq.ParquetWriter, n: int) -> None:
        nonlocal total_rows, pos
        if n == 0:
            return

        arrays: List[pa.Array] = [
            pa.array(buf_ck[:n], type=pa.int64()),
            pa.array(buf_sk[:n], type=pa.int32()),
            pa.array(buf_vf[:n].copy(), type=pa.timestamp("ns")),
            pa.array(buf_vt[:n].copy(), type=pa.timestamp("ns")),
        ]
        names = ["CustomerKey", "SuperpowerKey", "ValidFromDate", "ValidToDate"]

        if buf_lvl is not None:
            arrays.append(pa.array(buf_lvl[:n], type=pa.int8()))
            names.append("PowerLevel")
        if buf_pri is not None:
            arrays.append(pa.array(buf_pri[:n], type=pa.int8()))
            names.append("IsPrimaryFlag")
        if buf_acq is not None:
            arrays.append(pa.array(buf_acq[:n].copy(), type=pa.timestamp("ns")))
            names.append("AcquiredDate")

        arrays.append(pa.array(buf_rank[:n], type=pa.int8()))
        names.append("PowerRank")
        arrays.append(pa.array(buf_acq_ord[:n], type=pa.int8()))
        names.append("AcquisitionOrder")
        arrays.append(pa.array(buf_rw[:n], type=pa.float32()))
        names.append("RarityWeight")
        arrays.append(pa.array(buf_days[:n], type=pa.int32()))
        names.append("DaysToAcquire")
        arrays.append(pa.array(buf_latest[:n], type=pa.int8()))
        names.append("IsLatestPower")

        table = pa.Table.from_arrays(arrays, names=names).cast(schema)
        writer.write_table(table)
        total_rows += n
        pos = 0

    out_bridge.parent.mkdir(parents=True, exist_ok=True)
    if out_bridge.exists():
        out_bridge.unlink()

    n_cust = len(cust_keys)

    with pq.ParquetWriter(out_bridge, schema=schema, compression="snappy") as writer:
        for i in range(n_cust):
            ck = int(cust_keys[i])
            count = int(rng.integers(mn, mx + 1)) if mx > mn else mn
            if count > n_powers:
                count = n_powers

            chosen_idx = rng.choice(n_powers, size=count, replace=False, p=w)
            chosen_keys = power_keys[chosen_idx]
            chosen_rarity = rarity_arr[chosen_idx]

            if c.include_power_level:
                levels = _vectorized_levels(rng, chosen_rarity, count)
            else:
                levels = None

            lo_ns = int(cust_start_ns[i])
            hi_ns = int(cust_end_ns[i])
            span_days = max(0, (hi_ns - lo_ns) // _NS_PER_DAY)
            vt_ns = hi_ns

            acq_ns = _acquired_dates_ns(rng, chosen_rarity, count, lo_ns, span_days)

            # PowerRank: 1 = strongest (by level desc, key asc for ties)
            if levels is not None:
                rank_order = np.lexsort((chosen_keys, -levels.astype(np.int64)))
            else:
                rank_order = np.argsort(chosen_keys)
            power_rank = np.empty(count, dtype=np.int8)
            power_rank[rank_order] = np.arange(1, count + 1, dtype=np.int8)

            # IsPrimaryFlag derived from PowerRank
            if c.include_primary_flag:
                primary_flags = (power_rank == 1).astype(np.int8)

            # AcquisitionOrder: 1 = earliest (by acq date asc, key asc for ties)
            acq_order_idx = np.lexsort((chosen_keys, acq_ns))
            acq_order = np.empty(count, dtype=np.int8)
            acq_order[acq_order_idx] = np.arange(1, count + 1, dtype=np.int8)

            latest_flags = (acq_order == count).astype(np.int8)
            rarity_weights = per_power_weight[chosen_idx]
            days_to_acquire = ((acq_ns - lo_ns) // _NS_PER_DAY).astype(np.int32)

            sl = slice(pos, pos + count)
            buf_ck[sl] = ck
            buf_sk[sl] = chosen_keys
            buf_vf[sl] = acq_ns
            buf_vt[sl] = vt_ns

            if buf_lvl is not None:
                buf_lvl[sl] = levels
            if buf_pri is not None:
                buf_pri[sl] = primary_flags
            if buf_acq is not None:
                buf_acq[sl] = acq_ns

            buf_rank[sl] = power_rank
            buf_acq_ord[sl] = acq_order
            buf_rw[sl] = rarity_weights
            buf_days[sl] = days_to_acquire
            buf_latest[sl] = latest_flags

            pos += count

            if pos >= write_chunk_rows:
                flush(writer, pos)

        flush(writer, pos)

    return total_rows


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_superpowers(cfg: Dict[str, Any], parquet_folder: Path) -> Dict[str, Any]:
    parquet_folder = Path(parquet_folder)

    c = _read_cfg(cfg)
    if not c.enabled:
        skip("Superpowers disabled; skipping.")
        return {"_regenerated": False, "reason": "disabled"}

    out_dim = parquet_folder / "superpowers.parquet"
    out_bridge = parquet_folder / "customer_superpowers.parquet"

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

    st = os.stat(customers_fp)
    version_cfg = dict(cfg.superpowers)
    version_cfg["_schema_version"] = 4
    version_cfg["_upstream_customers_sig"] = {
        "path": str(customers_fp),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }

    version_files_exist = out_dim.exists() and (out_bridge.exists() or not c.generate_bridge)
    if version_files_exist and (not should_regenerate("superpowers", version_cfg, out_dim)):
        if not c.generate_bridge and out_bridge.exists():
            out_bridge.unlink()
            info("Removed stale customer_superpowers bridge file.")
        skip("Superpowers up-to-date")
        return {"_regenerated": False, "reason": "version"}

    with stage("Generating Superpowers"):
        g_start, g_end = _parse_global_dates(cfg)
        dim = build_dim_superpowers(cfg)
        dim.to_parquet(out_dim, index=False)
        info(f"Superpowers written: {out_dim} ({len(dim):,} rows)")

        n_rows = 0
        if c.generate_bridge:
            customers = pd.read_parquet(customers_fp)
            n_rows = _write_bridge_streaming(
                customers=customers,
                dim_powers=dim,
                c=c,
                g_start=g_start,
                g_end=g_end,
                out_bridge=out_bridge,
            )
            save_version("superpowers", version_cfg, out_bridge)
            info(f"Customer superpowers written: {out_bridge} ({n_rows:,} rows)")
        else:
            skip("customer_superpowers bridge skipped (generate_bridge: false)")
            if out_bridge.exists():
                out_bridge.unlink()

    return {
        "_regenerated": True,
        "dim": str(out_dim),
        "bridge": str(out_bridge) if c.generate_bridge else None,
        "bridge_rows": n_rows,
    }
