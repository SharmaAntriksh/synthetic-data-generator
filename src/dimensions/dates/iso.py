"""ISO week columns and offsets."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .helpers import _ISO_WEEK_REF


def add_iso_columns(df: pd.DataFrame, *, as_of: pd.Timestamp) -> pd.DataFrame:
    """Add ISO-8601 week columns and as-of week offset to *df*.

    Expects ``Date`` and ``WeekStartDate`` is NOT yet present (we compute it here
    using ISO Monday-based weeks).
    """
    iso = df["Date"].dt.isocalendar()
    df["WeekOfYearISO"] = iso.week.astype(np.int32)
    df["ISOYear"] = iso.year.astype(np.int32)

    df["WeekStartDate"] = (df["Date"] - pd.to_timedelta(df["Date"].dt.weekday, unit="D")).dt.normalize()
    df["WeekEndDate"] = (df["WeekStartDate"] + pd.Timedelta(days=6)).dt.normalize()

    # ISO week index: contiguous week number relative to _ISO_WEEK_REF.
    df["ISOYearWeekIndex"] = (((df["WeekStartDate"] - _ISO_WEEK_REF).dt.days) // 7).astype(np.int32)
    as_of_week_start = (as_of - pd.Timedelta(days=int(as_of.weekday()))).normalize()
    as_of_iso_year_week_index = int(((as_of_week_start - _ISO_WEEK_REF).days) // 7)
    df["ISOWeekOffset"] = (df["ISOYearWeekIndex"].astype(int) - as_of_iso_year_week_index).astype(np.int32)

    return df
