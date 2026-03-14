# src/dimensions/time_table.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

# Reuse the existing dim runner utility (versioning + force flag + parquet write)
from src.dimensions.lookups import _run_lookup_dim  # internal helper used by all lookups :contentReference[oaicite:2]{index=2}


def _fmt_hhmm(minute_of_day: int) -> str:
    """Format minute-of-day (0..1440) as HH:MM; allow 1440 => 24:00."""
    if minute_of_day == 24 * 60:
        return "24:00"
    h, m = divmod(int(minute_of_day), 60)
    return f"{h:02d}:{m:02d}"


def _label_range(start_min: int, width_min: int) -> str:
    """
    Half-open label like 13:30-13:45.
    End can be 24:00 for the last bucket.
    """
    end_min = start_min + width_min
    return f"{_fmt_hhmm(start_min)}-{_fmt_hhmm(end_min)}"


def _label_block_inclusive(start_min: int, block_min: int) -> str:
    """
    Inclusive-end label like 00:00-05:59 (good for 6h/12h blocks).
    """
    end_min = min(24 * 60 - 1, start_min + block_min - 1)
    return f"{_fmt_hhmm(start_min)}-{_fmt_hhmm(end_min)}"


def _df_time_table(dim_cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Minute-grain Time dimension (1440 rows) with precomputed rollup bins.

    Config:
      time:
        include_labels: true|false   # default true
        parquet_compression: snappy  # handled by _run_lookup_dim
    """
    include_labels = bool(dim_cfg.get("include_labels", True))

    # Vectorised construction of 1440-row time dimension
    t_arr = np.arange(24 * 60, dtype=np.int16)
    hour = t_arr // 60
    minute = t_arr % 60
    k15 = t_arr // 15
    k30 = t_arr // 30
    k60 = t_arr // 60
    k360 = t_arr // 360
    k720 = t_arr // 720

    _bucket_names = np.array(["Night", "Morning", "Afternoon", "Evening"])
    bucket_name4 = _bucket_names[k360]

    # TimeText: "HH:MM"
    time_text = np.char.add(
        np.char.zfill(hour.astype(str), 2),
        np.char.add(":", np.char.zfill(minute.astype(str), 2)),
    )

    data: dict = {
        "TimeKey": t_arr,
        "Hour": hour,
        "Minute": minute,
        "TimeText": time_text,
    }

    if include_labels:
        _v_label_range = np.vectorize(lambda k, w: _label_range(int(k) * w, w), otypes=[object])
        _v_label_block = np.vectorize(lambda k, w: _label_block_inclusive(int(k) * w, w), otypes=[object])
        data["TimeKey15"] = k15
        data["Bin15Label"] = _v_label_range(k15, 15)
        data["TimeKey30"] = k30
        data["Bin30Label"] = _v_label_range(k30, 30)
        data["TimeKey60"] = k60
        data["Bin60Label"] = _v_label_range(k60, 60)
        data["TimeKey360"] = k360
        data["Bin6hLabel"] = _v_label_block(k360, 360)
        data["TimeKey720"] = k720
        data["Bin12hLabel"] = _v_label_block(k720, 720)
        data["TimeBucketKey4"] = k360
        data["TimeBucket4"] = bucket_name4
    else:
        data["TimeKey15"] = k15
        data["TimeKey30"] = k30
        data["TimeKey60"] = k60
        data["TimeKey360"] = k360
        data["TimeKey720"] = k720
        data["TimeBucketKey4"] = k360
        data["TimeBucket4"] = bucket_name4

    df = pd.DataFrame(data)
    # Duration since midnight as a proper duration type (timedelta64[ns])

    # ---- Power Query friendly time columns ----
    # Seconds since midnight (0..86340)
    df["TimeSeconds"] = (df["Hour"].astype(int) * 3600 + df["Minute"].astype(int) * 60).astype(np.int32)

    # HH:MM:SS text (Power Query can easily cast Time.FromText)
    df["TimeOfDay"] = df["TimeText"] + ":00"

    # Compact integer types
    int_cols = [
        "TimeKey",
        "Hour",
        "Minute",
        "TimeKey15",
        "TimeKey30",
        "TimeKey60",
        "TimeKey360",
        "TimeKey720",
        "TimeBucketKey4",
    ]
    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].astype(np.int16)

    return df


def run_time_table(cfg: Dict[str, Any], parquet_folder: Path) -> None:
    """
    Public runner:
      - cfg key: "time"
      - output:  time.parquet
    """
    _run_lookup_dim(
        cfg=cfg,
        dim_key="time",
        out_name="time.parquet",
        build_df=_df_time_table,
        parquet_folder=parquet_folder,
    )


__all__ = ["run_time_table"]