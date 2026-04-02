"""Weekly fiscal (4-4-5) calendar columns and offsets."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from src.utils import warn

from .helpers import _clamp_month


@dataclass(frozen=True)
class WeeklyFiscalConfig:
    enabled: bool = False
    first_day_of_week: int = 0            # 0 = Sunday, 1 = Monday, ... 6 = Saturday
    weekly_type: str = "Last"             # "Last" or "Nearest"
    quarter_week_type: str = "445"        # "445", "454", "544"
    type_start_fiscal_year: int = 1       # 0 = start-year labeling, 1 = end-year labeling (matches DAX)


def _weekday_num(date: pd.Timestamp, first_day_of_week: int) -> int:
    """Return 1..7 weekday number where 1 == *first_day_of_week* (0=Sunday..6=Saturday)."""
    # pandas weekday: Monday=0..Sunday=6
    sun0 = (date.weekday() + 1) % 7  # Sunday=0..Saturday=6
    pos0 = (sun0 - first_day_of_week) % 7
    return int(pos0 + 1)


def _weekly_fiscal_year_bounds(
    fw_year_number: int,
    first_fiscal_month: int,
    first_day_of_week: int,
    weekly_type: str,
    type_start_fiscal_year: int,
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """Python translation of the DAX "Last/Nearest" weekly fiscal year boundary logic.

    Returns an inclusive interval ``[start_of_year, end_of_year]`` where
    *end_of_year* is the last day that belongs to *fw_year_number* (i.e. the
    day before the next fiscal year starts).
    """
    first_fiscal_month = _clamp_month(first_fiscal_month)
    first_day_of_week = int(first_day_of_week) % 7
    weekly_type = str(weekly_type or "Last").strip().title()
    if weekly_type not in {"Last", "Nearest"}:
        weekly_type = "Last"

    type_start_fiscal_year = 1 if int(type_start_fiscal_year) != 0 else 0
    offset_fiscal_year = 1 if first_fiscal_month > 1 else 0

    start_fy_calendar_year = int(fw_year_number) - (offset_fiscal_year * type_start_fiscal_year)

    first_day_current = pd.Timestamp(start_fy_calendar_year, first_fiscal_month, 1)
    first_day_next = pd.Timestamp(start_fy_calendar_year + 1, first_fiscal_month, 1)

    dow_cur = _weekday_num(first_day_current, first_day_of_week)
    dow_next = _weekday_num(first_day_next, first_day_of_week)

    if weekly_type == "Last":
        offset_start_current = 1 - dow_cur
        offset_start_next = 1 - dow_next
    else:
        # "Nearest"
        offset_start_current = (8 - dow_cur) if dow_cur >= 5 else (1 - dow_cur)
        offset_start_next = (8 - dow_next) if dow_next >= 5 else (1 - dow_next)

    start_of_year = (first_day_current + pd.Timedelta(days=int(offset_start_current))).normalize()
    # end_of_year is the day *before* the next fiscal year starts (inclusive boundary).
    next_year_start = (first_day_next + pd.Timedelta(days=int(offset_start_next))).normalize()
    end_of_year = next_year_start - pd.Timedelta(days=1)
    return start_of_year, end_of_year


def _weeks_in_periods(quarter_week_type: str) -> Tuple[int, int, int]:
    """Return ``(w1, w2, w3)`` weeks-per-period for the given 4-4-5 variant."""
    qwt = str(quarter_week_type or "445").strip()
    if qwt not in {"445", "454", "544"}:
        qwt = "445"
    if qwt == "445":
        return (4, 4, 5)
    if qwt == "454":
        return (4, 5, 4)
    return (5, 4, 4)


def _wf_is_enabled(wf_cfg) -> bool:
    """Return True if the weekly_fiscal config block is present and enabled.

    Accepts both the new dict form (``{enabled: true, ...}``) and the legacy
    bare bool (``weekly_fiscal: true``).
    """
    if isinstance(wf_cfg, bool):
        return wf_cfg
    if isinstance(wf_cfg, dict):
        return bool(wf_cfg.get("enabled", True))
    if isinstance(wf_cfg, Mapping):
        return bool(getattr(wf_cfg, "enabled", True))
    return False


def add_weekly_fiscal_columns(
    df: pd.DataFrame,
    *,
    first_fiscal_month: int,
    cfg: WeeklyFiscalConfig,
    as_of: pd.Timestamp,
) -> pd.DataFrame:
    """Add weekly-fiscal (4-4-5) columns and offsets based on the DAX logic.

    All new columns are assembled into a side DataFrame and concatenated once
    to avoid pandas "highly fragmented" warnings from inserting many columns.
    """
    if not cfg.enabled:
        return df

    first_fiscal_month = _clamp_month(first_fiscal_month)

    fdow = int(cfg.first_day_of_week) % 7
    weekly_type = str(cfg.weekly_type or "Last").strip().title()
    qwt = str(cfg.quarter_week_type or "445").strip()
    tsy = 1 if int(cfg.type_start_fiscal_year) != 0 else 0

    w1, w2, _w3 = _weeks_in_periods(qwt)

    start_year = int(df["Date"].dt.year.min())
    end_year = int(df["Date"].dt.year.max())
    # ±1 buffer is sufficient; the fiscal year boundaries for year Y are
    # derived from calendar months in years Y-1..Y+1 at most.
    year_span = range(start_year - 1, end_year + 2)

    bounds: Dict[int, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    for y in year_span:
        s, e = _weekly_fiscal_year_bounds(y, first_fiscal_month, fdow, weekly_type, tsy)
        bounds[int(y)] = (s, e)

    # --- Assign FWYearNumber (vectorized) ---
    fw_year = np.full(len(df), -1, dtype=np.int32)
    dates = df["Date"].to_numpy(dtype="datetime64[D]")

    years_sorted = np.array(sorted(bounds.keys()), dtype=np.int32)
    starts = np.array([bounds[int(y)][0].to_datetime64() for y in years_sorted], dtype="datetime64[D]")
    ends = np.array([bounds[int(y)][1].to_datetime64() for y in years_sorted], dtype="datetime64[D]")
    pos = np.searchsorted(starts, dates, side="right") - 1
    pos_clip = np.clip(pos, 0, len(ends) - 1)
    ok = (pos >= 0) & (dates <= ends[pos_clip])
    fw_year[ok] = years_sorted[pos[ok]]

    # Rare fallback: dates outside all computed weekly-fiscal year boundaries
    # are assigned the month-based fiscal end-year as a best-effort label.
    n_fallback = int((fw_year < 0).sum())
    if n_fallback:
        warn(f"Fiscal year: {n_fallback} date(s) outside weekly boundaries, using month-based fallback")
        fy_start_year = np.where(df["Month"] >= first_fiscal_month, df["Year"], df["Year"] - 1).astype(int)
        fy_end_add = 0 if first_fiscal_month == 1 else 1
        fy_end_year = (fy_start_year + fy_end_add).astype(int)
        fw_year = np.where(fw_year < 0, fy_end_year, fw_year).astype(np.int32)

    fw_year_s = pd.Series(fw_year.astype(np.int32), index=df.index, name="FWYearNumber")
    fw_year_label = "FY " + fw_year_s.astype(str)

    start_map = {y: se[0] for y, se in bounds.items()}
    end_map = {y: se[1] for y, se in bounds.items()}
    fw_start_year = fw_year_s.map(start_map).astype("datetime64[ns]")
    fw_end_year = fw_year_s.map(end_map).astype("datetime64[ns]")

    fw_day_of_year = (df["Date"] - fw_start_year).dt.days.add(1).astype(np.int32)
    fw_week = ((fw_day_of_year.astype(int) - 1) // 7 + 1).astype(np.int32)

    week = fw_week.astype(int).to_numpy()
    fw_period = np.where(week > 52, 13, (week + 3) // 4).astype(np.int32)
    fw_quarter = np.where(week > 52, 4, (week + 12) // 13).astype(np.int32)

    fw_quarter_s = pd.Series(fw_quarter, index=df.index, name="FWQuarterNumber")
    week_in_q = np.where(week > 52, 14, week - 13 * (fw_quarter_s.astype(int).to_numpy() - 1)).astype(int)
    fw_week_in_quarter = pd.Series(week_in_q.astype(np.int32), index=df.index, name="FWWeekInQuarterNumber")

    m_in_q = np.select(
        [week_in_q <= w1, week_in_q <= (w1 + w2)],
        [1, 2],
        default=3,
    ).astype(np.int32)
    fw_month = ((fw_quarter_s.astype(int) - 1) * 3 + m_in_q).astype(np.int32)
    fw_month_s = pd.Series(fw_month, index=df.index, name="FWMonthNumber")

    fw_year_quarter = (fw_year_s.astype(int) * 4 - 1 + fw_quarter_s.astype(int)).astype(np.int32)
    fw_year_month = (fw_year_s.astype(int) * 12 - 1 + fw_month_s.astype(int)).astype(np.int32)

    # Weekday number relative to first day of week (1..7)
    sun0 = (df["Date"].dt.weekday + 1) % 7
    pos0 = (sun0 - fdow) % 7
    week_day_num = (pos0 + 1).astype(np.int32)
    week_day_name_short = df["Date"].dt.strftime("%a")

    fw_start_week = (df["Date"] - pd.to_timedelta(week_day_num - 1, unit="D")).dt.normalize()
    fw_end_week = (fw_start_week + pd.to_timedelta(6, unit="D")).dt.normalize()

    # Working day (Mon-Fri)
    is_work = df["Date"].dt.weekday.isin([0, 1, 2, 3, 4])
    is_working_day = is_work.astype(bool)
    day_type = np.where(is_work, "Working Day", "Non-Working Day")

    # Boundaries within weekly fiscal month/quarter using temporary frame
    tmp = pd.DataFrame(
        {
            "Date": df["Date"],
            "FWMonthIndex": fw_year_month,
            "FWQuarterIndex": fw_year_quarter,
        },
        index=df.index,
    )
    fw_start_month = tmp.groupby("FWMonthIndex")["Date"].transform("min")
    fw_end_month = tmp.groupby("FWMonthIndex")["Date"].transform("max")
    fw_day_of_month = (df["Date"] - fw_start_month).dt.days.add(1).astype(np.int32)

    fw_start_quarter = tmp.groupby("FWQuarterIndex")["Date"].transform("min")
    fw_end_quarter = tmp.groupby("FWQuarterIndex")["Date"].transform("max")
    fw_day_of_quarter = (df["Date"] - fw_start_quarter).dt.days.add(1).astype(np.int32)

    # DAX-like FWWeekIndex (global increasing week index)
    first_week_reference = pd.Timestamp("1900-12-30") + pd.Timedelta(days=fdow)
    fw_year_week = (((df["Date"] - first_week_reference).dt.days) // 7 + 1).astype(np.int32)

    # Labels (year suffix for readability)
    y = fw_year_s.astype(str)
    fw_quarter_label = "FQ" + fw_quarter_s.astype(str) + " - " + y
    fw_week_label = "FW" + fw_week.astype(str).str.zfill(2) + " - " + y
    fw_period_label = "P" + pd.Series(fw_period, index=df.index).astype(str).str.zfill(2) + " - " + y
    fw_month_label = "FM " + (fw_start_month + pd.Timedelta(days=14)).dt.strftime("%b") + " - " + y

    # Convenience labels
    fw_year_month_label = "FM " + (fw_start_month + pd.Timedelta(days=14)).dt.strftime("%b %Y")

    new_cols = pd.DataFrame(
        {
            "FWYearNumber": fw_year_s,
            "FWYearLabel": fw_year_label,
            "FWStartOfYear": fw_start_year,
            "FWEndOfYear": fw_end_year,
            "FWDayOfYear": fw_day_of_year,
            "FWWeekNumber": fw_week,
            "FWPeriodNumber": pd.Series(fw_period, index=df.index).astype(np.int32),
            "FWQuarterNumber": fw_quarter_s.astype(np.int32),
            "FWWeekInQuarterNumber": fw_week_in_quarter,
            "FWMonthNumber": fw_month_s.astype(np.int32),
            "FWQuarterIndex": fw_year_quarter,
            "FWMonthIndex": fw_year_month,
            "FWWeekDayNumber": week_day_num,
            "FWWeekDayNameShort": week_day_name_short,
            "FWStartOfWeek": fw_start_week,
            "FWEndOfWeek": fw_end_week,
            "FWIsWorkingDay": is_working_day,
            "FWDayType": day_type,
            "FWStartOfMonth": fw_start_month,
            "FWEndOfMonth": fw_end_month,
            "FWDayOfMonth": fw_day_of_month,
            "FWStartOfQuarter": fw_start_quarter,
            "FWEndOfQuarter": fw_end_quarter,
            "FWDayOfQuarter": fw_day_of_quarter,
            "FWWeekIndex": fw_year_week,
            "FWQuarterLabel": fw_quarter_label,
            "FWWeekLabel": fw_week_label,
            "FWPeriodLabel": fw_period_label,
            "FWMonthLabel": fw_month_label,
            "FWYearMonthLabel": fw_year_month_label,
        },
        index=df.index,
    )

    df = pd.concat([df, new_cols], axis=1)

    # Weekly fiscal offsets (relative to as_of)
    asof_mask = df["Date"] == as_of
    asof_idx = df.index[asof_mask]

    if len(asof_idx) > 0:
        _asof = df.loc[asof_idx[0]]
        as_of_fw_year_week_index = int(_asof["FWWeekIndex"])
        as_of_fw_year_month_index = int(_asof["FWMonthIndex"])
        as_of_fw_year_quarter_index = int(_asof["FWQuarterIndex"])

        df = df.assign(
            FWWeekOffset=(df["FWWeekIndex"].astype(int) - as_of_fw_year_week_index).astype(np.int32),
            FWMonthOffset=(df["FWMonthIndex"].astype(int) - as_of_fw_year_month_index).astype(np.int32),
            FWQuarterOffset=(df["FWQuarterIndex"].astype(int) - as_of_fw_year_quarter_index).astype(np.int32),
        )
    else:
        df = df.assign(
            FWWeekOffset=np.int32(0),
            FWMonthOffset=np.int32(0),
            FWQuarterOffset=np.int32(0),
        )

    return df
