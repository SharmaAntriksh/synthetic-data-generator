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
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq

# Logging + versioning: support both in-repo import styles.
try:
    from src.utils import info, skip, stage
except Exception:  # pragma: no cover
    from contextlib import contextmanager
    from src.utils.logging_utils import info, skip  # type: ignore

    @contextmanager
    def stage(_name: str):
        yield

try:
    from src.versioning import should_regenerate, save_version
except Exception:  # pragma: no cover
    from src.versioning.version_store import should_regenerate, save_version  # type: ignore


# -----------------------------------------------------------------------------
# Catalog
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Builders
# -----------------------------------------------------------------------------

def _biased_acquired_date(
    rng: np.random.Generator,
    start: pd.Timestamp,
    end: pd.Timestamp,
    rarity: str,
) -> pd.Timestamp:
    """Acquire date within [start, end], biased by rarity."""
    start = pd.to_datetime(start).normalize()
    end = pd.to_datetime(end).normalize()
    if end < start:
        end = start
    span_days = int((end - start).days)
    if span_days <= 0:
        return start

    rarity = str(rarity)
    if rarity == "Common":
        a, b = 2.0, 5.0
    elif rarity == "Legendary":
        a, b = 5.0, 2.0
    else:  # Rare / fallback
        a, b = 2.0, 2.0

    u = float(rng.beta(a, b))  # in (0..1)
    off = int(u * span_days + 0.5)
    if off < 0:
        off = 0
    elif off > span_days:
        off = span_days
    return (start + pd.Timedelta(days=off)).normalize()


def build_dim_superpowers(cfg: Dict[str, Any]) -> pd.DataFrame:
    c = _read_cfg(cfg)
    rows = []
    for i, (name, ptype, uni, rarity, examples) in enumerate(POWERS_CATALOG, start=1):
        rows.append(
            {
                "SuperpowerKey": i,
                "SuperpowerName": name,
                "PowerType": ptype,
                "Universe": uni,
                "Rarity": rarity,
                "IconicExamples": examples,
                "IsActiveFlag": np.int8(1),
            }
        )
    df = pd.DataFrame(rows)
    k = min(max(int(c.powers_count), 1), len(df))
    return df.iloc[:k].reset_index(drop=True)


def _bridge_schema(include_power_level: bool, include_primary_flag: bool, include_acquired_date: bool) -> pa.Schema:
    fields = [
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


def _write_bridge_streaming(
    customers: pd.DataFrame,
    dim_powers: pd.DataFrame,
    cfg: Dict[str, Any],
    out_bridge: Path,
) -> int:
    """
    Stream-write customer_superpowers bridge to parquet.
    Returns number of rows written.
    """
    c = _read_cfg(cfg)
    g_start, g_end = _parse_global_dates(cfg)

    if "CustomerKey" not in customers.columns:
        raise KeyError("customers must include CustomerKey")
    if "SuperpowerKey" not in dim_powers.columns or "Rarity" not in dim_powers.columns:
        raise KeyError("dim_powers must include SuperpowerKey and Rarity")

    # Powers arrays
    powers = dim_powers
    power_keys = powers["SuperpowerKey"].astype("int32").to_numpy()
    rarity_arr = powers["Rarity"].astype(str).to_numpy()
    n_powers = int(len(power_keys))

    # Weight by rarity
    w = pd.Series(rarity_arr).map(_RARITY_WEIGHT).fillna(1.0).astype("float64").to_numpy()
    w = w / float(w.sum())

    # Customer windows
    cust_start = pd.to_datetime(customers.get("CustomerStartDate", g_start), errors="coerce").dt.normalize()
    cust_start = cust_start.fillna(g_start).clip(lower=g_start, upper=g_end)

    cust_end_raw = pd.to_datetime(customers.get("CustomerEndDate", pd.NaT), errors="coerce").dt.normalize()
    cust_end = cust_end_raw.fillna(g_end).clip(lower=g_start, upper=g_end)

    # Clamp end < start
    cust_end = pd.Series(
        np.where(cust_end.to_numpy() < cust_start.to_numpy(), cust_start.to_numpy(), cust_end.to_numpy()),
        index=customers.index,
    )
    cust_end = pd.to_datetime(cust_end).dt.normalize()

    # Deterministic order: sort by CustomerKey
    cust_keys = customers["CustomerKey"].astype("int64").to_numpy()
    order = np.argsort(cust_keys)
    cust_keys = cust_keys[order]
    cust_start_arr = pd.to_datetime(cust_start).to_numpy()[order]
    cust_end_arr = pd.to_datetime(cust_end).to_numpy()[order]

    mn = max(int(c.per_customer_min), 1)
    mx = max(int(c.per_customer_max), mn)
    write_chunk_rows = max(int(c.write_chunk_rows), 10_000)

    rng = np.random.default_rng(int(c.seed))
    schema = _bridge_schema(c.include_power_level, c.include_primary_flag, c.include_acquired_date)

    # Column buffers (lists of primitives) — much lighter than list-of-dict
    buf_ck: List[int] = []
    buf_sk: List[int] = []
    buf_vf: List[np.datetime64] = []
    buf_vt: List[np.datetime64] = []
    buf_lvl: Optional[List[int]] = [] if c.include_power_level else None
    buf_pri: Optional[List[int]] = [] if c.include_primary_flag else None
    buf_acq: Optional[List[np.datetime64]] = [] if c.include_acquired_date else None

    total_rows = 0

    def flush(writer: pq.ParquetWriter):
        nonlocal total_rows
        if not buf_ck:
            return

        arrays = [
            pa.array(buf_ck, type=pa.int64()),
            pa.array(buf_sk, type=pa.int32()),
            pa.array(buf_vf, type=pa.timestamp("ns")),
            pa.array(buf_vt, type=pa.timestamp("ns")),
        ]
        names = ["CustomerKey", "SuperpowerKey", "ValidFromDate", "ValidToDate"]

        if buf_lvl is not None:
            arrays.append(pa.array(buf_lvl, type=pa.int8()))
            names.append("PowerLevel")
        if buf_pri is not None:
            arrays.append(pa.array(buf_pri, type=pa.int8()))
            names.append("IsPrimaryFlag")
        if buf_acq is not None:
            arrays.append(pa.array(buf_acq, type=pa.timestamp("ns")))
            names.append("AcquiredDate")

        table = pa.Table.from_arrays(arrays, names=names).cast(schema)
        writer.write_table(table)

        n = len(buf_ck)
        total_rows += n

        # clear buffers
        buf_ck.clear(); buf_sk.clear(); buf_vf.clear(); buf_vt.clear()
        if buf_lvl is not None: buf_lvl.clear()
        if buf_pri is not None: buf_pri.clear()
        if buf_acq is not None: buf_acq.clear()

    out_bridge.parent.mkdir(parents=True, exist_ok=True)
    if out_bridge.exists():
        out_bridge.unlink()

    writer = pq.ParquetWriter(out_bridge, schema=schema, compression="snappy")

    try:
        for i, ck in enumerate(cust_keys):
            count = int(rng.integers(mn, mx + 1)) if mx > mn else mn
            if count > n_powers:
                count = n_powers

            chosen_idx = rng.choice(n_powers, size=count, replace=False, p=w)
            chosen_keys = power_keys[chosen_idx]
            chosen_rarity = rarity_arr[chosen_idx]

            # Levels array (same order as chosen_keys)
            if c.include_power_level:
                levels = np.empty(count, dtype=np.int8)
                for j, r in enumerate(chosen_rarity):
                    if r == "Common":
                        levels[j] = np.int8(rng.integers(2, 5))
                    elif r == "Rare":
                        levels[j] = np.int8(rng.integers(3, 6))
                    else:
                        levels[j] = np.int8(rng.integers(4, 6))
                # Primary: max level, tie -> smallest SuperpowerKey
                max_lvl = int(levels.max())
                tie = chosen_keys[levels == max_lvl]
                primary_sk = int(tie.min())
            else:
                levels = None
                primary_sk = int(chosen_keys.min())

            a_lo = pd.Timestamp(cust_start_arr[i]).normalize()
            a_hi = pd.Timestamp(cust_end_arr[i]).normalize()
            if a_hi < a_lo:
                a_hi = a_lo
            vt = np.datetime64(a_hi.to_datetime64(), "ns")

            # Row-per-power
            for j, (sk, r) in enumerate(zip(chosen_keys, chosen_rarity)):
                acquired = _biased_acquired_date(rng, a_lo, a_hi, str(r))
                vf = np.datetime64(acquired.to_datetime64(), "ns")

                buf_ck.append(int(ck))
                buf_sk.append(int(sk))
                buf_vf.append(vf)
                buf_vt.append(vt)

                if buf_lvl is not None:
                    buf_lvl.append(int(levels[j]))  # type: ignore[index]
                if buf_pri is not None:
                    buf_pri.append(1 if int(sk) == primary_sk else 0)
                if buf_acq is not None:
                    buf_acq.append(vf)

            # Flush when buffers exceed threshold
            if len(buf_ck) >= write_chunk_rows:
                flush(writer)

        flush(writer)
    finally:
        writer.close()

    return total_rows


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------

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
        dim = build_dim_superpowers(cfg)
        customers = pd.read_parquet(customers_fp)

        dim.to_parquet(out_dim, index=False)
        n_rows = _write_bridge_streaming(customers=customers, dim_powers=dim, cfg=cfg, out_bridge=out_bridge)

    save_version("superpowers", version_cfg, out_bridge)
    info(f"Superpowers written: {out_dim} ({len(dim):,} rows)")
    info(f"Customer superpowers written: {out_bridge} ({n_rows:,} rows)")
    return {"_regenerated": True, "dim": str(out_dim), "bridge": str(out_bridge), "bridge_rows": int(n_rows)}
