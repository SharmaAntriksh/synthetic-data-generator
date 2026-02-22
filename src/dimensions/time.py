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

    rows = []
    for t in range(24 * 60):  # 0..1439
        hour = t // 60
        minute = t % 60

        # Bin keys (integer rollups)
        k15 = t // 15     # 0..95
        k30 = t // 30     # 0..47
        k60 = t // 60     # 0..23
        k360 = t // 360   # 0..3 (6h)
        k720 = t // 720   # 0..1 (12h)

        # 6h-block-derived "4 bucket" names
        # 00:00–05:59 Night, 06:00–11:59 Morning, 12:00–17:59 Afternoon, 18:00–23:59 Evening
        bucket_key4 = k360
        bucket_name4 = ("Night", "Morning", "Afternoon", "Evening")[bucket_key4]

        if include_labels:
            rows.append(
                (
                    t,
                    hour,
                    minute,
                    f"{hour:02d}:{minute:02d}",
                    k15,
                    _label_range(k15 * 15, 15),
                    k30,
                    _label_range(k30 * 30, 30),
                    k60,
                    _label_range(k60 * 60, 60),
                    k360,
                    _label_block_inclusive(k360 * 360, 360),
                    k720,
                    _label_block_inclusive(k720 * 720, 720),
                    bucket_key4,
                    bucket_name4,
                )
            )
        else:
            rows.append(
                (
                    t,
                    hour,
                    minute,
                    f"{hour:02d}:{minute:02d}",
                    k15,
                    k30,
                    k60,
                    k360,
                    k720,
                    bucket_key4,
                    bucket_name4,
                )
            )

    if include_labels:
        cols = [
            "TimeKey",
            "Hour",
            "Minute",
            "TimeText",
            "TimeKey15",
            "Bin15Label",
            "TimeKey30",
            "Bin30Label",
            "TimeKey60",
            "Bin60Label",
            "TimeKey360",
            "Bin6hLabel",
            "TimeKey720",
            "Bin12hLabel",
            "TimeBucketKey4",
            "TimeBucket4",
        ]
    else:
        cols = [
            "TimeKey",
            "Hour",
            "Minute",
            "TimeText",
            "TimeKey15",
            "TimeKey30",
            "TimeKey60",
            "TimeKey360",
            "TimeKey720",
            "TimeBucketKey4",
            "TimeBucket4",
        ]

    df = pd.DataFrame(rows, columns=cols)
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