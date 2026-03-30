"""Base calendar columns and as-of relative offsets."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .helpers import _EXCEL_EPOCH


def add_calendar_columns(df: pd.DataFrame, *, as_of: pd.Timestamp) -> pd.DataFrame:
    """Add base calendar columns and as-of relative offsets to *df*.

    Expects *df* to contain a ``Date`` column (datetime64).
    """
    # DateKey: YYYYMMDD integer (arithmetic, avoids slow strftime).
    year = df["Date"].dt.year
    month = df["Date"].dt.month
    day = df["Date"].dt.day

    df["DateKey"] = (year * 10000 + month * 100 + day).astype(np.int64)

    # SequentialDayIndex: Excel-like serial (2006-12-31 → 39082).
    df["SequentialDayIndex"] = (df["Date"] - _EXCEL_EPOCH).dt.days.astype(np.int32)

    # Basic parts
    df["Year"] = year.astype(np.int32)
    df["Month"] = month.astype(np.int32)
    df["Day"] = day.astype(np.int32)
    df["Quarter"] = df["Date"].dt.quarter.astype(np.int32)

    df["MonthName"] = df["Date"].dt.strftime("%B")
    df["MonthShort"] = df["Date"].dt.strftime("%b")
    df["DayName"] = df["Date"].dt.strftime("%A")
    df["DayShort"] = df["Date"].dt.strftime("%a")

    df["DayOfYear"] = df["Date"].dt.dayofyear.astype(np.int32)

    df["MonthYear"] = df["Date"].dt.strftime("%b %Y")
    df["MonthYearNumber"] = (df["Year"].astype(int) * 100 + df["Month"].astype(int)).astype(np.int32)

    # Common BI convenience keys/labels
    df["YearQuarterKey"] = (df["Year"].astype(int) * 10 + df["Quarter"].astype(int)).astype(np.int32)
    df["QuarterYear"] = "Q" + df["Quarter"].astype(str) + " " + df["Year"].astype(str)

    df["CalendarMonthIndex"] = (df["Year"].astype(int) * 12 + df["Month"].astype(int)).astype(np.int32)
    df["CalendarQuarterIndex"] = (df["Year"].astype(int) * 4 + df["Quarter"].astype(int)).astype(np.int32)

    # DayOfWeek conventions (compatible with earlier implementation):
    # 0 = Sunday, 1 = Monday, ... 6 = Saturday
    weekday = df["Date"].dt.weekday  # 0=Mon..6=Sun
    df["DayOfWeek"] = ((weekday + 1) % 7).astype(np.int32)  # 0=Sun..6=Sat

    df["IsWeekend"] = df["DayOfWeek"].isin([0, 6]).astype(np.int32)
    df["IsBusinessDay"] = (df["IsWeekend"] == 0).astype(np.int32)

    # Month start/end (pandas-native to avoid datetime64[D] casts)
    df["MonthStartDate"] = df["Date"].dt.to_period("M").dt.start_time.dt.normalize()
    df["MonthEndDate"] = df["Date"].dt.to_period("M").dt.end_time.dt.normalize()

    # Quarter start/end
    qperiod = df["Date"].dt.to_period("Q")
    df["QuarterStartDate"] = qperiod.dt.start_time.dt.normalize()
    df["QuarterEndDate"] = qperiod.dt.end_time.dt.normalize()

    df["IsMonthStart"] = (df["Day"] == 1).astype(np.int32)
    df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(np.int32)
    df["IsQuarterStart"] = df["Date"].dt.is_quarter_start.astype(np.int32)
    df["IsQuarterEnd"] = df["Date"].dt.is_quarter_end.astype(np.int32)
    df["IsYearStart"] = ((df["Month"] == 1) & (df["Day"] == 1)).astype(np.int32)
    df["IsYearEnd"] = ((df["Month"] == 12) & (df["Day"] == 31)).astype(np.int32)

    df["WeekOfMonth"] = ((df["Day"] - 1) // 7 + 1).astype(np.int32)

    # ------------------------------------------------------------------
    # Next/Previous Business Day
    #
    # Semantics (strict):
    #   NextBusinessDay     = first business day *after* the current date
    #   PreviousBusinessDay = last business day *before* the current date
    #
    # Edge behaviour: if no such day exists within the generated date
    # range, the date falls back to itself.
    # ------------------------------------------------------------------
    biz_dates = df.loc[df["IsBusinessDay"] == 1, "Date"].to_numpy(dtype="datetime64[D]")
    date_vals = df["Date"].to_numpy(dtype="datetime64[D]")

    if biz_dates.size > 0:
        # side="right" → first biz_date strictly after current date
        idx_next = np.searchsorted(biz_dates, date_vals, side="right")
        # side="left" - 1 → last biz_date strictly before current date
        idx_prev = np.searchsorted(biz_dates, date_vals, side="left") - 1

        next_bd = date_vals.copy()
        prev_bd = date_vals.copy()

        ok_next = idx_next < biz_dates.size
        ok_prev = idx_prev >= 0

        next_bd[ok_next] = biz_dates[idx_next[ok_next]]
        prev_bd[ok_prev] = biz_dates[idx_prev[ok_prev]]

        df["NextBusinessDay"] = pd.to_datetime(next_bd)
        df["PreviousBusinessDay"] = pd.to_datetime(prev_bd)
    else:
        # Degenerate range with no business days; fall back to self.
        df["NextBusinessDay"] = df["Date"]
        df["PreviousBusinessDay"] = df["Date"]

    # ------------------------------------------------------------------
    # As-of relative columns
    # ------------------------------------------------------------------
    df["IsToday"] = (df["Date"] == as_of).astype(np.int32)
    df["IsCurrentYear"] = (df["Year"] == as_of.year).astype(np.int32)
    df["IsCurrentMonth"] = ((df["Year"] == as_of.year) & (df["Month"] == as_of.month)).astype(np.int32)
    current_quarter = (as_of.month - 1) // 3 + 1
    df["IsCurrentQuarter"] = ((df["Year"] == as_of.year) & (df["Quarter"] == current_quarter)).astype(np.int32)
    df["CurrentDayOffset"] = (df["Date"] - as_of).dt.days.astype(np.int32)

    # Offsets (relative to as_of) for fast "last N" slicing in Power BI
    df["YearOffset"] = (df["Year"].astype(int) - int(as_of.year)).astype(np.int32)
    as_of_cal_month_index = int(as_of.year) * 12 + int(as_of.month)
    as_of_cal_quarter_index = int(as_of.year) * 4 + int((as_of.month - 1) // 3 + 1)
    df["CalendarMonthOffset"] = (df["CalendarMonthIndex"].astype(int) - as_of_cal_month_index).astype(np.int32)
    df["CalendarQuarterOffset"] = (df["CalendarQuarterIndex"].astype(int) - as_of_cal_quarter_index).astype(np.int32)

    return df
