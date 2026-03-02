from __future__ import annotations

"""
Superpowers (FAST + low-memory; dimensions-only)

Writes:
  - superpowers.parquet
  - customer_superpowers.parquet

Key changes vs earlier versions:
  - ValidFromDate / ValidToDate are ALWAYS present (stable schema).
  - Bridge is written STREAMING with pyarrow ParquetWriter (does not hold the whole bridge in RAM).
  - Avoids per-row Pandas .loc lookups and list-of-dict construction (major perf/memory win).
  - Per-power acquisition dates (not identical within a customer), biased by rarity.

Bridge validity semantics (always):
  ValidFromDate = acquired date of that power (per row)
  ValidToDate   = customer end date (or global end date)

Config (optional)
superpowers:
  enabled: true
  powers_count: 40
  powers_per_customer_min: 1
  powers_per_customer_max: 4
  include_power_level: true
  include_primary_flag: true
  include_acquired_date: true
  seed: 123
  _force_regenerate: false
  write_chunk_rows: 250000       # optional: approximate max rows per parquet write batch

Requires:
  defaults.dates.start/end (or _defaults.dates.start/end) in cfg.

Power BI:
  Customers (1) ──< customer_superpowers >── (1) Superpowers
"""

import os
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

# Nanoseconds in one calendar day — used for integer datetime math.
_NS_PER_DAY: int = 86_400_000_000_000

# Beta distribution parameters keyed by rarity.
# Common → early-biased, Legendary → late-biased, Rare/fallback → symmetric.
_RARITY_BETA: Dict[str, Tuple[float, float]] = {
    "Common": (2.0, 5.0),
    "Legendary": (5.0, 2.0),
}
_BETA_FALLBACK: Tuple[float, float] = (2.0, 2.0)

# Level ranges keyed by rarity: (low_inclusive, high_exclusive) matching rng.integers.
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
    powers_count: int = 40
    per_customer_min: int = 1
    per_customer_max: int = 4
    include_power_level: bool = True
    include_primary_flag: bool = True
    include_acquired_date: bool = True
    seed: int = 123
    _force_regenerate: bool = False
    write_chunk_rows: int = 250_000  # rows per parquet write batch (bridge)


def _read_cfg(cfg: Dict[str, Any]) -> SuperpowersCfg:
    sp = cfg.get("superpowers") or {}
    if not isinstance(sp, dict):
        sp = {}
    return SuperpowersCfg(
        enabled=bool(sp.get("enabled", True)),
        powers_count=int(sp.get("powers_count", 40)),
        per_customer_min=int(sp.get("powers_per_customer_min", 1)),
        per_customer_max=int(sp.get("powers_per_customer_max", 4)),
        include_power_level=bool(sp.get("include_power_level", True)),
        include_primary_flag=bool(sp.get("include_primary_flag", True)),
        include_acquired_date=bool(sp.get("include_acquired_date", True)),
        seed=int(sp.get("seed", 123)),
        _force_regenerate=bool(sp.get("_force_regenerate", False)),
        write_chunk_rows=int(sp.get("write_chunk_rows", 250_000)),
    )


def _parse_global_dates(cfg: Dict[str, Any]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    defaults = cfg.get("defaults") or cfg.get("_defaults") or {}
    d = defaults.get("dates") or {}
    if not isinstance(d, dict) or not d.get("start") or not d.get("end"):
        raise ValueError("Missing defaults.dates.start/end (or _defaults.dates.start/end)")
    start = pd.to_datetime(d["start"]).normalize()
    end = pd.to_datetime(d["end"]).normalize()
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
    int64 nanosecond timestamps for zero-overhead arithmetic in the hot loop.
    """
    cust_keys = customers["CustomerKey"].astype(np.int64).to_numpy()
    order = np.argsort(cust_keys)
    cust_keys = cust_keys[order]

    g_start_ns = np.int64(g_start.value)
    g_end_ns = np.int64(g_end.value)

    # Start dates
    raw_start = (
        pd.to_datetime(customers.get("CustomerStartDate", g_start), errors="coerce")
        .dt.normalize()
        .fillna(g_start)
        .clip(lower=g_start, upper=g_end)
    )
    start_ns = raw_start.to_numpy().astype("datetime64[ns]").view(np.int64)[order]

    # End dates
    raw_end = (
        pd.to_datetime(customers.get("CustomerEndDate", pd.NaT), errors="coerce")
        .dt.normalize()
        .fillna(g_end)
        .clip(lower=g_start, upper=g_end)
    )
    end_ns = raw_end.to_numpy().astype("datetime64[ns]").view(np.int64)[order]

    # Clamp end < start
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
    # Fallback for unknown rarities
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
    to one customer.  Uses beta-distributed offsets biased by rarity — purely
    integer arithmetic, no pandas objects.
    """
    if span_days <= 0:
        return np.full(count, start_ns, dtype=np.int64)

    # Draw beta samples per-power (preserves per-element RNG ordering)
    u = np.empty(count, dtype=np.float64)
    for j in range(count):
        a, b = _RARITY_BETA.get(str(chosen_rarity[j]), _BETA_FALLBACK)
        u[j] = rng.beta(a, b)

    offsets = np.clip(np.round(u * span_days).astype(np.int64), 0, span_days)
    return start_ns + offsets * _NS_PER_DAY


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

    Accepts a pre-parsed ``SuperpowersCfg`` and global date boundaries to
    avoid redundant config / date parsing.
    """
    if "CustomerKey" not in customers.columns:
        raise KeyError("customers must include CustomerKey")
    if "SuperpowerKey" not in dim_powers.columns or "Rarity" not in dim_powers.columns:
        raise KeyError("dim_powers must include SuperpowerKey and Rarity")

    # ---- Power arrays ----
    power_keys = dim_powers["SuperpowerKey"].astype(np.int32).to_numpy()
    rarity_arr = dim_powers["Rarity"].astype(str).to_numpy()
    n_powers = len(power_keys)
    w = _compute_rarity_weights(rarity_arr)

    # ---- Customer windows (int64 ns — zero pandas overhead in loop) ----
    cust_keys, cust_start_ns, cust_end_ns = _compute_customer_windows(
        customers, g_start, g_end,
    )

    mn = max(c.per_customer_min, 1)
    mx = max(c.per_customer_max, mn)
    write_chunk_rows = max(c.write_chunk_rows, 10_000)

    rng = np.random.default_rng(c.seed)
    schema = _bridge_schema(c.include_power_level, c.include_primary_flag, c.include_acquired_date)

    # ---- Column buffers (primitive lists — lighter than list-of-dict) ----
    buf_ck: List[int] = []
    buf_sk: List[int] = []
    buf_vf: List[int] = []          # int64 nanosecond timestamps
    buf_vt: List[int] = []          # int64 nanosecond timestamps
    buf_lvl: Optional[List[int]] = [] if c.include_power_level else None
    buf_pri: Optional[List[int]] = [] if c.include_primary_flag else None
    buf_acq: Optional[List[int]] = [] if c.include_acquired_date else None

    total_rows = 0

    def flush(writer: pq.ParquetWriter) -> None:
        nonlocal total_rows
        if not buf_ck:
            return

        # Build arrow arrays — timestamps from int64 ns buffers directly.
        arrays: List[pa.Array] = [
            pa.array(buf_ck, type=pa.int64()),
            pa.array(buf_sk, type=pa.int32()),
            pa.array(np.array(buf_vf, dtype=np.int64), type=pa.timestamp("ns")),
            pa.array(np.array(buf_vt, dtype=np.int64), type=pa.timestamp("ns")),
        ]
        names = ["CustomerKey", "SuperpowerKey", "ValidFromDate", "ValidToDate"]

        if buf_lvl is not None:
            arrays.append(pa.array(buf_lvl, type=pa.int8()))
            names.append("PowerLevel")
        if buf_pri is not None:
            arrays.append(pa.array(buf_pri, type=pa.int8()))
            names.append("IsPrimaryFlag")
        if buf_acq is not None:
            arrays.append(pa.array(np.array(buf_acq, dtype=np.int64), type=pa.timestamp("ns")))
            names.append("AcquiredDate")

        table = pa.Table.from_arrays(arrays, names=names).cast(schema)
        writer.write_table(table)

        total_rows += len(buf_ck)

        # Clear buffers
        buf_ck.clear(); buf_sk.clear(); buf_vf.clear(); buf_vt.clear()
        if buf_lvl is not None:
            buf_lvl.clear()
        if buf_pri is not None:
            buf_pri.clear()
        if buf_acq is not None:
            buf_acq.clear()

    out_bridge.parent.mkdir(parents=True, exist_ok=True)
    if out_bridge.exists():
        out_bridge.unlink()

    with pq.ParquetWriter(out_bridge, schema=schema, compression="snappy") as writer:
        for i in range(len(cust_keys)):
            ck = int(cust_keys[i])
            count = int(rng.integers(mn, mx + 1)) if mx > mn else mn
            if count > n_powers:
                count = n_powers

            chosen_idx = rng.choice(n_powers, size=count, replace=False, p=w)
            chosen_keys = power_keys[chosen_idx]
            chosen_rarity = rarity_arr[chosen_idx]

            # ---- Levels (vectorised) ----
            if c.include_power_level:
                levels = _vectorized_levels(rng, chosen_rarity, count)
                max_lvl = int(levels.max())
                tie = chosen_keys[levels == max_lvl]
                primary_sk = int(tie.min())
            else:
                levels = None
                primary_sk = int(chosen_keys.min())

            # ---- Acquired / validity dates (int64 ns — no pandas) ----
            lo_ns = int(cust_start_ns[i])
            hi_ns = int(cust_end_ns[i])
            span_days = max(0, (hi_ns - lo_ns) // _NS_PER_DAY)
            vt_ns = hi_ns

            acq_ns = _acquired_dates_ns(rng, chosen_rarity, count, lo_ns, span_days)

            # ---- Append to column buffers ----
            for j in range(count):
                buf_ck.append(ck)
                buf_sk.append(int(chosen_keys[j]))
                buf_vf.append(int(acq_ns[j]))
                buf_vt.append(vt_ns)

                if buf_lvl is not None:
                    buf_lvl.append(int(levels[j]))       # type: ignore[index]
                if buf_pri is not None:
                    buf_pri.append(1 if int(chosen_keys[j]) == primary_sk else 0)
                if buf_acq is not None:
                    buf_acq.append(int(acq_ns[j]))

            # Flush when buffers exceed threshold
            if len(buf_ck) >= write_chunk_rows:
                flush(writer)

        # Final flush for remaining rows
        flush(writer)

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
    version_cfg = dict(cfg.get("superpowers") or {})
    version_cfg["_schema_version"] = 3  # streaming + always validity
    version_cfg["_upstream_customers_sig"] = {
        "path": str(customers_fp),
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
    }

    if not c._force_regenerate:
        if out_dim.exists() and out_bridge.exists() and (not should_regenerate("superpowers", version_cfg, out_bridge)):
            skip("Superpowers up-to-date; skipping.")
            return {"_regenerated": False, "reason": "version"}

    with stage("Generating Superpowers"):
        g_start, g_end = _parse_global_dates(cfg)
        dim = build_dim_superpowers(cfg)
        customers = pd.read_parquet(customers_fp)

        dim.to_parquet(out_dim, index=False)
        n_rows = _write_bridge_streaming(
            customers=customers,
            dim_powers=dim,
            c=c,
            g_start=g_start,
            g_end=g_end,
            out_bridge=out_bridge,
        )

    save_version("superpowers", version_cfg, out_bridge)
    info(f"Superpowers written: {out_dim} ({len(dim):,} rows)")
    info(f"Customer superpowers written: {out_bridge} ({n_rows:,} rows)")
    return {"_regenerated": True, "dim": str(out_dim), "bridge": str(out_bridge), "bridge_rows": int(n_rows)}
