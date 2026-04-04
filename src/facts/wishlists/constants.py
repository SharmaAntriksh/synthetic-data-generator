"""Shared constants, schema, and config for the wishlists pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa

from src.defaults import NS_PER_DAY  # noqa: F401 (re-export)
from src.utils.config_helpers import parse_global_dates as _parse_global_dates_shared


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRIORITY_VALUES = np.array(["High", "Medium", "Low"], dtype=object)
PRIORITY_WEIGHTS = np.array([0.20, 0.50, 0.30])


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def bridge_schema() -> pa.Schema:
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
    conversion_rate: float = 0.30
    seed: int = 500
    write_chunk_rows: int = 250_000


def read_cfg(cfg: Any) -> WishlistsCfg:
    wl = getattr(cfg, "wishlists", None)
    if wl is None:
        return WishlistsCfg()
    return WishlistsCfg(
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


def parse_global_dates(cfg: Any) -> Tuple[pd.Timestamp, pd.Timestamp]:
    return _parse_global_dates_shared(cfg, {}, dimension_name="wishlists")
