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


_PROFILE_HOUR_WEIGHTS = [RETAIL_HOUR_W, DIGITAL_HOUR_W, BUSINESS_HOUR_W, ASSISTED_HOUR_W]


def _parse_time_str(t: object) -> int:
    """Parse 'HH:MM' string to hour (0-23). Returns -1 for None/invalid."""
    if t is None or (isinstance(t, float) and np.isnan(t)):
        return -1
    s = str(t).strip()
    if not s or s.lower() in ("none", "nan", "nat"):
        return -1
    parts = s.split(":")
    if len(parts) >= 2:
        try:
            return int(parts[0])
        except (ValueError, TypeError):
            return -1
    return -1


def _build_channel_hour_weights(
    profile_code: int, open_hour: int, close_hour: int,
) -> np.ndarray:
    """
    Build a 24-element normalized hour-weight array for a channel.

    Takes the base profile weights and zeros out hours outside the
    operating window [open_hour, close_hour). For 24/7 channels
    (open_hour < 0), returns the unmasked profile weights.
    """
    base = _PROFILE_HOUR_WEIGHTS[min(profile_code, len(_PROFILE_HOUR_WEIGHTS) - 1)].copy()
    if open_hour >= 0 and close_hour > open_hour:
        mask = np.ones(24, dtype=np.bool_)
        mask[:open_hour] = False
        mask[close_hour:] = False
        base *= mask
    return _normalize_prob(base)


def _load_sales_channels(State: Any) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Returns (keys:int16[], p:float64[], channel_hour_lut:ndarray[max_key+1, 24]).

    channel_hour_lut[key] is a 24-element normalized hour-weight array
    that respects the channel's OpenTime/CloseTime operating window.
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

    read_cols = ["SalesChannelKey", "ChannelGroup"]
    # Read OpenTime/CloseTime if available
    schema = pq.read_schema(pth)
    col_names = set(schema.names)
    has_times = "OpenTime" in col_names and "CloseTime" in col_names
    if has_times:
        read_cols += ["OpenTime", "CloseTime"]

    tab = pq.read_table(pth, columns=read_cols)
    keys = np.asarray(tab["SalesChannelKey"].to_numpy(), dtype=np.int16)
    grp = np.asarray(tab["ChannelGroup"].to_numpy(), dtype=object)
    open_times = np.asarray(tab["OpenTime"].to_numpy(), dtype=object) if has_times else None
    close_times = np.asarray(tab["CloseTime"].to_numpy(), dtype=object) if has_times else None

    # Exclude Unknown (0) from sampling if present
    m = keys != np.int16(0)
    keys = keys[m]
    grp = grp[m]
    if open_times is not None:
        open_times = open_times[m]
        close_times = close_times[m]

    if keys.size == 0:
        return None

    # Sampling prob: uniform across keys
    p = np.full(keys.shape[0], 1.0 / keys.shape[0], dtype=np.float64)

    # Per-channel hour weight LUT: channel_hour_lut[key] -> 24-element array
    maxk = int(keys.max())
    digital_w = _normalize_prob(DIGITAL_HOUR_W)
    channel_hour_lut = np.tile(digital_w, (maxk + 1, 1))  # default: digital

    for i, (k, g) in enumerate(zip(keys.astype(np.int32), grp)):
        profile_code = int(_PROFILE_FROM_GROUP.get(str(g).strip().lower(), P_DIGITAL))
        oh = _parse_time_str(open_times[i]) if open_times is not None else -1
        ch = _parse_time_str(close_times[i]) if close_times is not None else -1
        channel_hour_lut[k] = _build_channel_hour_weights(profile_code, oh, ch)

    State._sales_channels_cache = (keys, p, channel_hour_lut)
    return State._sales_channels_cache


def _sample_hour_weighted_minute(rng: np.random.Generator, size: int, hour_w: np.ndarray) -> np.ndarray:
    size = int(size)
    if size <= 0:
        return np.empty(0, dtype=np.int16)

    w = _normalize_prob(hour_w)
    hours = rng.choice(24, size=size, p=w).astype(np.int32)
    mins = rng.integers(0, 60, size=size, dtype=np.int32)
    return (hours * 60 + mins).astype(np.int16, copy=False)


def _sample_timekey_by_channel(
    rng: np.random.Generator,
    channel_keys: np.ndarray,
    channel_hour_lut: np.ndarray,
) -> np.ndarray:
    """Sample minute-of-day TimeKey values using per-channel hour weights."""
    keys = np.asarray(channel_keys, dtype=np.int16)
    out = np.empty(keys.shape[0], dtype=np.int16)

    # Group by unique channel key to batch-sample each channel's distribution
    unique_keys = np.unique(keys)
    for ck in unique_keys:
        ck_int = int(ck)
        mask = keys == ck
        n = int(mask.sum())
        if ck_int < channel_hour_lut.shape[0]:
            hour_w = channel_hour_lut[ck_int]
        else:
            hour_w = _normalize_prob(DIGITAL_HOUR_W)
        out[mask] = _sample_hour_weighted_minute(rng, n, hour_w)

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
    channel_hour_lut = None
    if "SalesChannelKey" in schema_types:
        if cache is None:
            # last-resort fallback: match your default lookup keys
            keys = np.array([1, 2, 3, 4, 5], dtype=np.int16)
            p = np.full(keys.shape[0], 1.0 / keys.shape[0], dtype=np.float64)
        else:
            keys, p, channel_hour_lut = cache

        if order_ids_int is not None:
            oid = np.asarray(order_ids_int, dtype=np.int32)
            # Order IDs are sequential ints — skip O(n log n) np.unique
            _min_oid = int(oid[0])
            _n_orders = int(oid[-1]) - _min_oid + 1
            inv = oid - np.int32(_min_oid)
            per_order_channel = rng.choice(keys, size=_n_orders, p=p).astype(np.int16, copy=False)
            sales_channel = per_order_channel[inv]
        else:
            sales_channel = rng.choice(keys, size=n, p=p).astype(np.int16, copy=False)

        out["SalesChannelKey"] = sales_channel
    else:
        channel_hour_lut = cache[2] if cache is not None else None

    # ----------------------------
    # TimeKey depends on SalesChannel operating hours
    # ----------------------------
    if "TimeKey" in schema_types:
        if sales_channel is None:
            sc_base = ctx["cols"].get("SalesChannelKey")
            sales_channel = None if sc_base is None else np.asarray(sc_base, dtype=np.int16)

        if sales_channel is None or channel_hour_lut is None:
            # digital-like fallback so night bins aren't empty
            timekey = _sample_hour_weighted_minute(rng, n, DIGITAL_HOUR_W)
        else:
            if order_ids_int is not None:
                oid = np.asarray(order_ids_int, dtype=np.int32)
                # Order IDs are sequential — derive first-row indices in O(n)
                _min_oid = int(oid[0])
                _n_orders = int(oid[-1]) - _min_oid + 1
                inv = oid - np.int32(_min_oid)
                # First occurrence: where oid changes value (sorted+grouped)
                _changes = np.empty(len(oid), dtype=np.bool_)
                _changes[0] = True
                _changes[1:] = oid[1:] != oid[:-1]
                first_idx = np.flatnonzero(_changes)
                per_order_sc = sales_channel[first_idx]
                per_order_time = _sample_timekey_by_channel(rng, per_order_sc, channel_hour_lut)
                timekey = per_order_time[inv]
            else:
                timekey = _sample_timekey_by_channel(rng, sales_channel, channel_hour_lut)

        out["TimeKey"] = timekey

        # Rollups — single int32 cast shared across all divisions
        _tk32 = timekey.astype(np.int32)
        if "TimeKey15" in schema_types:
            out["TimeKey15"] = (_tk32 // 15).astype(np.int16)
        if "TimeKey30" in schema_types:
            out["TimeKey30"] = (_tk32 // 30).astype(np.int16)
        if "TimeKey60" in schema_types:
            out["TimeKey60"] = (_tk32 // 60).astype(np.int16)
        if "TimeKey360" in schema_types:
            out["TimeKey360"] = (_tk32 // 360).astype(np.int16)
        if "TimeKey720" in schema_types:
            out["TimeKey720"] = (_tk32 // 720).astype(np.int16)
        if "TimeBucketKey4" in schema_types:
            out["TimeBucketKey4"] = (_tk32 // 240).astype(np.int16)

    return out


__all__ = ["build_extra_columns"]
