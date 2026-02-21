from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq


# ------------------------------------------------------------
# Profile codes (internal only)
# ------------------------------------------------------------
P_RETAIL = np.int8(0)
P_DIGITAL = np.int8(1)
P_BUSINESS = np.int8(2)
P_ASSISTED = np.int8(3)

_PROFILE_FROM_STR = {
    "RETAIL": P_RETAIL,
    "DIGITAL": P_DIGITAL,
    "BUSINESS": P_BUSINESS,
    "ASSISTED": P_ASSISTED,
    "NA": P_DIGITAL,
    "UNKNOWN": P_DIGITAL,
}

_PROFILE_FROM_GROUP = {
    "physical": P_RETAIL,
    "digital": P_DIGITAL,
    "business": P_BUSINESS,
    "assisted": P_ASSISTED,
    "na": P_DIGITAL,
}


# ------------------------------------------------------------
# Hour-weight profiles (bin-agnostic: works for 1h/4h/6h/8h/etc.)
# ------------------------------------------------------------
RETAIL_HOUR_W = np.array(
    [0.002, 0.001, 0.001, 0.001, 0.002, 0.004,
     0.010, 0.020, 0.050, 0.080, 0.100, 0.110,
     0.110, 0.105, 0.095, 0.090, 0.085, 0.090,
     0.100, 0.095, 0.070, 0.040, 0.015, 0.006],
    dtype=np.float64,
)

DIGITAL_HOUR_W = np.array(
    [0.030, 0.025, 0.020, 0.020, 0.022, 0.026,
     0.030, 0.040, 0.050, 0.055, 0.060, 0.065,
     0.070, 0.070, 0.065, 0.060, 0.060, 0.065,
     0.080, 0.090, 0.090, 0.070, 0.050, 0.040],
    dtype=np.float64,
)

BUSINESS_HOUR_W = np.array(
    [0.001, 0.001, 0.001, 0.001, 0.001, 0.002,
     0.006, 0.020, 0.060, 0.090, 0.110, 0.120,
     0.120, 0.110, 0.100, 0.090, 0.070, 0.040,
     0.020, 0.010, 0.006, 0.003, 0.002, 0.001],
    dtype=np.float64,
)

ASSISTED_HOUR_W = np.array(
    [0.002, 0.001, 0.001, 0.001, 0.001, 0.003,
     0.010, 0.030, 0.060, 0.090, 0.110, 0.120,
     0.120, 0.110, 0.100, 0.090, 0.070, 0.040,
     0.020, 0.010, 0.006, 0.004, 0.003, 0.002],
    dtype=np.float64,
)


def _normalize_prob(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    s = float(w.sum())
    if s <= 1e-12:
        return np.full(w.shape[0], 1.0 / w.shape[0], dtype=np.float64)
    return w / s


def _dims_parquet_folder(State: Any) -> Optional[Path]:
    # Common names across your codebase
    for attr in ("parquet_folder_p", "parquet_folder", "dimensions_parquet_folder", "dims_parquet_folder"):
        v = getattr(State, attr, None)
        if v:
            return Path(v)
    return None


def _load_sales_channels(State: Any) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Returns (keys:int16[], p:float64[], profile_lut:int8[max_key+1]).
    Cached on State.
    """
    cached = getattr(State, "_sales_channels_cache", None)
    if cached is not None:
        return cached

    folder = _dims_parquet_folder(State)
    if folder is None:
        return None

    pth = folder / "sales_channels.parquet"
    if not pth.exists():
        return None

    # Read what exists (your current dim guarantees these 3 columns) :contentReference[oaicite:4]{index=4}
    # If you later add TimeProfile or SamplingWeight, they will be used automatically.
    cols = ["SalesChannelKey", "ChannelGroup"]
    tab = pq.read_table(pth, columns=[c for c in cols if True])
    keys = np.asarray(tab["SalesChannelKey"].to_numpy(), dtype=np.int16)
    grp = np.asarray(tab["ChannelGroup"].to_numpy(), dtype=object)

    # Exclude Unknown (0) from sampling if present
    m = keys != np.int16(0)
    keys = keys[m]
    grp = grp[m]

    if keys.size == 0:
        return None

    # Sampling prob: uniform across keys (you can later add weights to the dim and load them here)
    p = np.full(keys.shape[0], 1.0 / keys.shape[0], dtype=np.float64)

    # Profile LUT by key based on ChannelGroup
    maxk = int(keys.max())
    lut = np.full(maxk + 1, P_DIGITAL, dtype=np.int8)

    for k, g in zip(keys.astype(np.int64), grp):
        gg = str(g).strip().lower()
        lut[k] = _PROFILE_FROM_GROUP.get(gg, P_DIGITAL)

    State._sales_channels_cache = (keys, p, lut)
    return State._sales_channels_cache


def _sample_hour_weighted_minute(rng: np.random.Generator, size: int, hour_w: np.ndarray) -> np.ndarray:
    size = int(size)
    if size <= 0:
        return np.empty(0, dtype=np.int16)

    w = _normalize_prob(hour_w)
    hours = rng.choice(24, size=size, p=w).astype(np.int32)
    mins = rng.integers(0, 60, size=size, dtype=np.int32)
    return (hours * 60 + mins).astype(np.int16, copy=False)


def _sample_timekey_by_profile(rng: np.random.Generator, prof_codes: np.ndarray) -> np.ndarray:
    prof = np.asarray(prof_codes, dtype=np.int8)
    out = np.empty(prof.shape[0], dtype=np.int16)

    m = prof == P_RETAIL
    if m.any():
        out[m] = _sample_hour_weighted_minute(rng, int(m.sum()), RETAIL_HOUR_W)

    m = prof == P_DIGITAL
    if m.any():
        out[m] = _sample_hour_weighted_minute(rng, int(m.sum()), DIGITAL_HOUR_W)

    m = prof == P_BUSINESS
    if m.any():
        out[m] = _sample_hour_weighted_minute(rng, int(m.sum()), BUSINESS_HOUR_W)

    m = prof == P_ASSISTED
    if m.any():
        out[m] = _sample_hour_weighted_minute(rng, int(m.sum()), ASSISTED_HOUR_W)

    return out


def build_extra_columns(ctx: Dict[str, Any]) -> Dict[str, Any]:
    schema_types = ctx["schema_types"]
    out: Dict[str, Any] = {}

    State = ctx["State"]
    rng: np.random.Generator = ctx["rng"]
    n = int(ctx["n"])
    order_ids_int = ctx.get("order_ids_int", None)

    cache = _load_sales_channels(State)

    # ----------------------------
    # SalesChannelKey
    # ----------------------------
    sales_channel = None
    if "SalesChannelKey" in schema_types:
        if cache is None:
            # last-resort fallback: match your default lookup keys
            keys = np.array([1, 2, 3, 4, 5], dtype=np.int16)
            p = np.full(keys.shape[0], 1.0 / keys.shape[0], dtype=np.float64)
            profile_lut = None
        else:
            keys, p, profile_lut = cache

        if order_ids_int is not None:
            oid = np.asarray(order_ids_int, dtype=np.int64)
            unique_orders, inv = np.unique(oid, return_inverse=True)
            per_order_channel = rng.choice(keys, size=unique_orders.shape[0], p=p).astype(np.int16, copy=False)
            sales_channel = per_order_channel[inv]
        else:
            sales_channel = rng.choice(keys, size=n, p=p).astype(np.int16, copy=False)

        out["SalesChannelKey"] = sales_channel
    else:
        profile_lut = cache[2] if cache is not None else None

    # ----------------------------
    # TimeKey depends on SalesChannel profile
    # ----------------------------
    if "TimeKey" in schema_types:
        if sales_channel is None:
            sc_base = ctx["cols"].get("SalesChannelKey")
            sales_channel = None if sc_base is None else np.asarray(sc_base, dtype=np.int16)

        if sales_channel is None or profile_lut is None:
            # digital-like fallback so night bins aren't empty
            timekey = _sample_hour_weighted_minute(rng, n, DIGITAL_HOUR_W)
        else:
            prof = profile_lut[sales_channel.astype(np.int64)]
            if order_ids_int is not None:
                oid = np.asarray(order_ids_int, dtype=np.int64)
                unique_orders, inv = np.unique(oid, return_inverse=True)
                # per-order profile from first occurrence
                first_idx = np.full(unique_orders.shape[0], -1, dtype=np.int64)
                for i in range(inv.shape[0]):
                    j = inv[i]
                    if first_idx[j] == -1:
                        first_idx[j] = i
                per_order_prof = prof[first_idx]
                per_order_time = _sample_timekey_by_profile(rng, per_order_prof)
                timekey = per_order_time[inv]
            else:
                timekey = _sample_timekey_by_profile(rng, prof)

        out["TimeKey"] = timekey

        # Rollups (only if present in schema)
        if "TimeKey15" in schema_types:
            out["TimeKey15"] = (timekey.astype(np.int32) // 15).astype(np.int16)
        if "TimeKey30" in schema_types:
            out["TimeKey30"] = (timekey.astype(np.int32) // 30).astype(np.int16)
        if "TimeKey60" in schema_types:
            out["TimeKey60"] = (timekey.astype(np.int32) // 60).astype(np.int16)
        if "TimeKey360" in schema_types:
            out["TimeKey360"] = (timekey.astype(np.int32) // 360).astype(np.int16)
        if "TimeKey720" in schema_types:
            out["TimeKey720"] = (timekey.astype(np.int32) // 720).astype(np.int16)
        if "TimeBucketKey4" in schema_types:
            out["TimeBucketKey4"] = (timekey.astype(np.int32) // 360).astype(np.int16)

    return out


__all__ = ["build_extra_columns"]