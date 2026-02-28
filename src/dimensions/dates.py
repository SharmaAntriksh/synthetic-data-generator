from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import re

from src.utils import info, skip, stage
from src.utils.output_utils import write_parquet_with_date32
from src.versioning import should_regenerate, save_version


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _dedupe_preserve_order(cols: Sequence[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for c in cols:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def _int_or(value: Any, default: int) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _bool_or(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "t", "1", "yes", "y", "on"}:
            return True
        if s in {"false", "f", "0", "no", "n", "off"}:
            return False
    return bool(default)


def _humanize_col_name(name: str) -> str:
    """
    Convert CamelCase / mixedCase names into space-separated "Title Case" names,
    while preserving acronyms like ISO/FY/FW.

    Examples:
      - FiscalYearEndDate -> Fiscal Year End Date
      - CalendarMonthIndex -> Calendar Month Index
      - FWYearWeekNumber -> FW Year Week Number
    """

    s = name.replace("_", " ").strip()
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)          # aB / 1B -> a B
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", s)        # ISOWeek -> ISO Week

    words = s.split()
    out_words: List[str] = []
    for w in words:
        if w.isupper():
            out_words.append(w)
        else:
            out_words.append(w[:1].upper() + w[1:])
    return " ".join(out_words)


def _normalize_override_dates(dates_cfg: Dict[str, Any]) -> Dict[str, Any]:
    override = dates_cfg.get("override") or {}
    override2 = dates_cfg.get("_override") or {}
    out: Dict[str, Any] = {}
    out.update(override if isinstance(override, dict) else {})
    out.update(override2 if isinstance(override2, dict) else {})
    return out




def warn(msg: str) -> None:
    # Treat as WARN-level line using existing logging.
    info(f"WARN  | {msg}")

def _require_start_end(raw_start: Any, raw_end: Any) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if raw_start is None or raw_end is None or raw_start == "" or raw_end == "":
        raise ValueError("Missing required start/end dates (defaults.dates or dates.override).")

    start_ts = pd.to_datetime(raw_start).normalize()
    end_ts = pd.to_datetime(raw_end).normalize()
    if end_ts < start_ts:
        warn(f"dates: start/end swapped (start={raw_start!r}, end={raw_end!r})")
        start_ts, end_ts = end_ts, start_ts
    return start_ts, end_ts


# ---------------------------------------------------------------------
# Weekly fiscal (4-4-5) configuration
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class WeeklyFiscalConfig:
    enabled: bool = True
    first_day_of_week: int = 0            # 0 = Sunday, 1 = Monday, ... 6 = Saturday
    weekly_type: str = "Last"             # "Last" or "Nearest"
    quarter_week_type: str = "445"        # "445", "454", "544"
    type_start_fiscal_year: int = 1       # 0 = start-year labeling, 1 = end-year labeling (matches DAX)


def _weekday_num(date: pd.Timestamp, first_day_of_week: int) -> int:
    """Return 1..7 weekday number where 1 == first_day_of_week (0=Sunday..6=Saturday)."""
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
    """
    Python translation of the user's DAX "Last/Nearest" weekly fiscal year boundary logic.

    Returns an inclusive interval: [start_of_year, end_of_year].
    """
    first_fiscal_month = int(first_fiscal_month)
    first_fiscal_month = 1 if first_fiscal_month < 1 else 12 if first_fiscal_month > 12 else first_fiscal_month

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
        offset_start_next = -dow_next
    else:
        # "Nearest"
        offset_start_current = (8 - dow_cur) if dow_cur >= 5 else (1 - dow_cur)
        offset_start_next = (7 - dow_next) if dow_next >= 5 else (-dow_next)

    start_of_year = (first_day_current + pd.Timedelta(days=int(offset_start_current))).normalize()
    end_of_year = (first_day_next + pd.Timedelta(days=int(offset_start_next))).normalize()
    return start_of_year, end_of_year


def _weeks_in_periods(quarter_week_type: str) -> Tuple[int, int, int]:
    qwt = str(quarter_week_type or "445").strip()
    if qwt not in {"445", "454", "544"}:
        qwt = "445"
    if qwt == "445":
        return (4, 4, 5)
    if qwt == "454":
        return (4, 5, 4)
    return (5, 4, 4)


def _compute_weekly_fiscal_columns(
    df: pd.DataFrame,
    *,
    first_fiscal_month: int,
    cfg: WeeklyFiscalConfig,
) -> pd.DataFrame:
    """
    Add weekly-fiscal (4-4-5) columns based on the user's DAX logic.

    Performance note:
      - We build all new columns into a side DataFrame and concat once to avoid
        pandas "highly fragmented" warnings from inserting many columns.
    """
    if not cfg.enabled:
        return df

    first_fiscal_month = int(first_fiscal_month)
    first_fiscal_month = 1 if first_fiscal_month < 1 else 12 if first_fiscal_month > 12 else first_fiscal_month

    fdow = int(cfg.first_day_of_week) % 7
    weekly_type = str(cfg.weekly_type or "Last").strip().title()
    qwt = str(cfg.quarter_week_type or "445").strip()
    tsy = 1 if int(cfg.type_start_fiscal_year) != 0 else 0

    w1, w2, w3 = _weeks_in_periods(qwt)

    start_year = int(df["Date"].dt.year.min())
    end_year = int(df["Date"].dt.year.max())
    year_span = range(start_year - 3, end_year + 4)

    bounds: Dict[int, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    for y in year_span:
        s, e = _weekly_fiscal_year_bounds(y, first_fiscal_month, fdow, weekly_type, tsy)
        bounds[int(y)] = (s, e)

    # Assign FWYearNumber
    fw_year = np.full(len(df), -1, dtype=np.int32)
    dates = df["Date"].dt.normalize().to_numpy(dtype="datetime64[D]")

    # Vectorized assignment: find the latest year-start <= date, then verify date <= year-end.
    years_sorted = np.array(sorted(bounds.keys()), dtype=np.int32)
    starts = np.array([bounds[int(y)][0].to_datetime64() for y in years_sorted], dtype="datetime64[D]")
    ends = np.array([bounds[int(y)][1].to_datetime64() for y in years_sorted], dtype="datetime64[D]")
    pos = np.searchsorted(starts, dates, side="right") - 1
    pos_clip = np.clip(pos, 0, len(ends) - 1)
    ok = (pos >= 0) & (dates <= ends[pos_clip])
    fw_year[ok] = years_sorted[pos[ok]]

    # Rare fallback: month-based fiscal end-year label
    if (fw_year < 0).any():
        fy_start_year = np.where(df["Month"] >= first_fiscal_month, df["Year"], df["Year"] - 1).astype(int)
        fy_end_add = 0 if first_fiscal_month == 1 else 1
        fy_end_year = (fy_start_year + fy_end_add).astype(int)
        fw_year = np.where(fw_year < 0, fy_end_year, fw_year).astype(np.int32)

    fw_year_s = pd.Series(fw_year.astype(np.int16), index=df.index, name="FWYearNumber")
    fw_year_label = "FY " + fw_year_s.astype(str)

    start_map = {y: se[0] for y, se in bounds.items()}
    end_map = {y: se[1] for y, se in bounds.items()}
    fw_start_year = fw_year_s.map(start_map).astype("datetime64[ns]")
    fw_end_year = fw_year_s.map(end_map).astype("datetime64[ns]")

    fw_day_of_year = (df["Date"] - fw_start_year).dt.days.add(1).astype(np.int16)
    fw_week = ((fw_day_of_year.astype(int) - 1) // 7 + 1).astype(np.int16)

    week = fw_week.astype(int).to_numpy()
    fw_period = np.where(week > 52, 13, (week + 3) // 4).astype(np.int16)
    fw_quarter = np.where(week > 52, 4, (week + 12) // 13).astype(np.int8)

    fw_quarter_s = pd.Series(fw_quarter, index=df.index, name="FWQuarterNumber")
    week_in_q = np.where(week > 52, 14, week - 13 * (fw_quarter_s.astype(int).to_numpy() - 1)).astype(int)
    fw_week_in_quarter = pd.Series(week_in_q.astype(np.int16), index=df.index, name="FWWeekInQuarterNumber")

    m_in_q = np.select(
        [week_in_q <= w1, week_in_q <= (w1 + w2)],
        [1, 2],
        default=3,
    ).astype(np.int8)
    fw_month = ((fw_quarter_s.astype(int) - 1) * 3 + m_in_q).astype(np.int8)
    fw_month_s = pd.Series(fw_month, index=df.index, name="FWMonthNumber")

    fw_year_quarter = (fw_year_s.astype(int) * 4 - 1 + fw_quarter_s.astype(int)).astype(np.int32)
    fw_year_month = (fw_year_s.astype(int) * 12 - 1 + fw_month_s.astype(int)).astype(np.int32)

    # Weekday number relative to first day of week (1..7)
    sun0 = (df["Date"].dt.weekday + 1) % 7
    pos0 = (sun0 - fdow) % 7
    week_day_num = (pos0 + 1).astype(np.int8)
    week_day_name_short = df["Date"].dt.strftime("%a")

    fw_start_week = (df["Date"] - pd.to_timedelta(week_day_num - 1, unit="D")).dt.normalize()
    fw_end_week = (fw_start_week + pd.to_timedelta(6, unit="D")).dt.normalize()

    # Working day (Mon-Fri)
    is_work = df["Date"].dt.weekday.isin([0, 1, 2, 3, 4])
    is_working_day = is_work.astype(np.int8)
    day_type = np.where(is_work, "Working Day", "Non-Working day")

    # Boundaries within weekly fiscal month/quarter using temporary frame
    tmp = pd.DataFrame(
        {
            "Date": df["Date"],
            "FWYearMonthNumber": fw_year_month,
            "FWYearQuarterNumber": fw_year_quarter,
        },
        index=df.index,
    )
    fw_start_month = tmp.groupby("FWYearMonthNumber")["Date"].transform("min")
    fw_end_month = tmp.groupby("FWYearMonthNumber")["Date"].transform("max")
    fw_day_of_month = (df["Date"] - fw_start_month).dt.days.add(1).astype(np.int16)

    fw_start_quarter = tmp.groupby("FWYearQuarterNumber")["Date"].transform("min")
    fw_end_quarter = tmp.groupby("FWYearQuarterNumber")["Date"].transform("max")
    fw_day_of_quarter = (df["Date"] - fw_start_quarter).dt.days.add(1).astype(np.int16)

    # DAX-like FWYearWeekNumber (global increasing week index)
    first_week_reference = pd.Timestamp("1900-12-30") + pd.Timedelta(days=fdow)
    fw_year_week = (((df["Date"] - first_week_reference).dt.days) // 7 + 1).astype(np.int32)

    # Labels include year for readability
    y = fw_year_s.astype(str)
    fw_quarter_label = "FQ" + fw_quarter_s.astype(str) + " - " + y
    fw_week_label = "FW" + fw_week.astype(str).str.zfill(2) + " - " + y
    fw_period_label = "P" + pd.Series(fw_period, index=df.index).astype(str).str.zfill(2) + " - " + y
    fw_month_label = "FM " + (fw_start_month + pd.Timedelta(days=14)).dt.strftime("%b") + " - " + y

    # Convenience labels
    fw_year_week_label = fw_week_label
    fw_year_month_label = "FM " + (fw_start_month + pd.Timedelta(days=14)).dt.strftime("%b %Y")
    fw_year_quarter_label = fw_quarter_label

    new_cols = pd.DataFrame(
        {
            "FWYearNumber": fw_year_s,
            "FWYearLabel": fw_year_label,
            "FWStartOfYear": fw_start_year,
            "FWEndOfYear": fw_end_year,
            "FWDayOfYearNumber": fw_day_of_year,
            "FWWeekNumber": fw_week,
            "FWPeriodNumber": pd.Series(fw_period, index=df.index).astype(np.int16),
            "FWQuarterNumber": fw_quarter_s.astype(np.int8),
            "FWWeekInQuarterNumber": fw_week_in_quarter,
            "FWMonthNumber": fw_month_s.astype(np.int8),
            "FWYearQuarterNumber": fw_year_quarter,
            "FWYearMonthNumber": fw_year_month,
            "WeekDayNumber": week_day_num,
            "WeekDayNameShort": week_day_name_short,
            "FWStartOfWeek": fw_start_week,
            "FWEndOfWeek": fw_end_week,
            "IsWorkingDay": is_working_day,
            "DayType": day_type,
            "FWStartOfMonth": fw_start_month,
            "FWEndOfMonth": fw_end_month,
            "FWDayOfMonthNumber": fw_day_of_month,
            "FWStartOfQuarter": fw_start_quarter,
            "FWEndOfQuarter": fw_end_quarter,
            "FWDayOfQuarterNumber": fw_day_of_quarter,
            "FWYearWeekNumber": fw_year_week,
            "FWQuarterLabel": fw_quarter_label,
            "FWWeekLabel": fw_week_label,
            "FWPeriodLabel": fw_period_label,
            "FWMonthLabel": fw_month_label,
            "FWYearWeekLabel": fw_year_week_label,
            "FWYearMonthLabel": fw_year_month_label,
            "FWYearQuarterLabel": fw_year_quarter_label,
        },
        index=df.index,
    )

    return pd.concat([df, new_cols], axis=1)

# ---------------------------------------------------------------------
# Column resolver
# ---------------------------------------------------------------------

def resolve_date_columns(dates_cfg: Dict[str, Any]) -> List[str]:
    include = (dates_cfg or {}).get("include", {}) or {}
    weekly_cfg = (dates_cfg or {}).get("weekly_calendar", {}) or {}

    base_cols = ["Date", "DateKey", "SequentialDayIndex"]

    calendar_cols = [
        "Year", "IsYearStart", "IsYearEnd",
        "Quarter", "QuarterStartDate", "QuarterEndDate",
        "IsQuarterStart", "IsQuarterEnd",
        "QuarterYear",
        "Month", "MonthName", "MonthShort", "MonthNameShort",
        "MonthStartDate", "MonthEndDate",
        "MonthYear", "MonthYearNumber", "YearMonthKey", "YearMonthLabel", "YearQuarterKey", "YearQuarterLabel", "CalendarMonthIndex", "CalendarQuarterIndex",
        "IsMonthStart", "IsMonthEnd",
        "WeekOfMonth",
        "Day", "DayName", "DayShort", "DayNameShort", "DayOfYear", "DayOfWeek",
        "IsWeekend", "IsBusinessDay",
        "NextBusinessDay", "PreviousBusinessDay",
        "IsToday", "IsCurrentYear", "IsCurrentMonth", "IsCurrentQuarter",
        "CurrentDayOffset", "YearOffset", "CalendarMonthOffset", "CalendarQuarterOffset",
    ]

    iso_cols = [
        "WeekOfYearISO",
        "ISOYear",
        "ISOYearWeekIndex",
        "ISOWeekOffset",
        "WeekStartDate",
        "WeekEndDate",
    ]

    fiscal_cols = [
        "FiscalYearStartYear", "FiscalMonthNumber", "FiscalQuarterNumber",
        "FiscalMonthIndex", "FiscalQuarterIndex", "FiscalMonthOffset", "FiscalQuarterOffset",
        "FiscalQuarterName", "FiscalYearBin",
        "FiscalYearMonthNumber", "FiscalYearQuarterNumber",
        "FiscalYearStartDate", "FiscalYearEndDate",
        "FiscalQuarterStartDate", "FiscalQuarterEndDate",
        "IsFiscalYearStart", "IsFiscalYearEnd",
        "IsFiscalQuarterStart", "IsFiscalQuarterEnd",
        "FiscalYear", "FiscalYearLabel", "FiscalSystem", "WeeklyFiscalSystem",
    ]

    weekly_cols = [
        "FWYearNumber",
        "FWYearLabel",
        "FWQuarterNumber",
        "FWQuarterLabel",
        "FWYearQuarterNumber",
        "FWYearQuarterOffset",
        "FWMonthNumber",
        "FWMonthLabel",
        "FWYearMonthNumber",
        "FWYearMonthOffset",
        "FWWeekNumber",
        "FWWeekLabel",
        "FWYearWeekNumber",
        "FWYearWeekOffset",
        "FWYearWeekLabel",
        "FWPeriodNumber",
        "FWPeriodLabel",
        "FWStartOfYear",
        "FWEndOfYear",
        "FWStartOfQuarter",
        "FWEndOfQuarter",
        "FWStartOfMonth",
        "FWEndOfMonth",
        "FWStartOfWeek",
        "FWEndOfWeek",
        "WeekDayNumber",
        "WeekDayNameShort",
        "FWDayOfYearNumber",
        "FWDayOfQuarterNumber",
        "FWDayOfMonthNumber",
        "IsWorkingDay",
        "DayType",
        "FWWeekInQuarterNumber",
        "FWYearMonthLabel",
        "FWYearQuarterLabel",
    ]

    cols: List[str] = list(base_cols)

    if include.get("calendar", True):
        cols += calendar_cols
    if include.get("iso", True):
        cols += iso_cols
    if include.get("fiscal", True):
        cols += fiscal_cols

    if include.get("weekly_fiscal", True) and _bool_or(weekly_cfg.get("enabled", True), True):
        cols += weekly_cols

    return _dedupe_preserve_order(cols)


# ---------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------

def generate_date_table(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    fiscal_start_month: int,
    *,
    as_of_date: Optional[str] = None,
    weekly_cfg: Optional[WeeklyFiscalConfig] = None,
) -> pd.DataFrame:
    start_date = pd.to_datetime(start_date).normalize()
    end_date = pd.to_datetime(end_date).normalize()
    if end_date < start_date:
        start_date, end_date = end_date, start_date

    dates = pd.date_range(start_date, end_date, freq="D")
    df = pd.DataFrame({"Date": dates})
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    df["DateKey"] = df["Date"].dt.strftime("%Y%m%d").astype(np.int64)

    # SequentialDayIndex: Excel-like serial (2006-12-31 -> 39082)
    excel_base = pd.Timestamp("1899-12-30")
    df["SequentialDayIndex"] = (df["Date"] - excel_base).dt.days.astype(np.int32)

    # Basic parts
    df["Year"] = df["Date"].dt.year.astype(np.int16)
    df["Month"] = df["Date"].dt.month.astype(np.int8)
    df["Day"] = df["Date"].dt.day.astype(np.int8)
    df["Quarter"] = df["Date"].dt.quarter.astype(np.int8)

    df["MonthName"] = df["Date"].dt.strftime("%B")
    df["MonthShort"] = df["Date"].dt.strftime("%b")
    df["MonthNameShort"] = df["MonthShort"]
    df["DayName"] = df["Date"].dt.strftime("%A")
    df["DayShort"] = df["Date"].dt.strftime("%a")
    df["DayNameShort"] = df["DayShort"]

    df["DayOfYear"] = df["Date"].dt.dayofyear.astype(np.int16)

    df["MonthYear"] = df["Date"].dt.strftime("%b %Y")
    df["MonthYearNumber"] = (df["Year"].astype(int) * 100 + df["Month"].astype(int)).astype(np.int32)

    # Common BI convenience keys/labels
    df["YearMonthKey"] = df["MonthYearNumber"].astype(np.int32)
    df["YearMonthLabel"] = df["MonthYear"]
    df["YearQuarterKey"] = (df["Year"].astype(int) * 10 + df["Quarter"].astype(int)).astype(np.int32)
    df["YearQuarterLabel"] = "Q" + df["Quarter"].astype(str) + " " + df["Year"].astype(str)

    df["CalendarMonthIndex"] = (df["Year"].astype(int) * 12 + df["Month"].astype(int)).astype(np.int32)
    df["CalendarQuarterIndex"] = (df["Year"].astype(int) * 4 + df["Quarter"].astype(int)).astype(np.int32)

    df["QuarterYear"] = "Q" + df["Quarter"].astype(str) + " " + df["Year"].astype(str)

    # DayOfWeek conventions (compatible with earlier implementation):
    # 0 = Sunday, 1 = Monday, ... 6 = Saturday
    weekday = df["Date"].dt.weekday  # 0=Mon..6=Sun
    df["DayOfWeek"] = ((weekday + 1) % 7).astype(np.int8)  # 0=Sun..6=Sat

    df["IsWeekend"] = df["DayOfWeek"].isin([0, 6]).astype(np.int8)
    df["IsBusinessDay"] = (df["IsWeekend"] == 0).astype(np.int8)

    # Month start/end (pandas-native to avoid datetime64[D] casts)
    df["MonthStartDate"] = df["Date"].dt.to_period("M").dt.start_time.dt.normalize()
    df["MonthEndDate"] = df["Date"].dt.to_period("M").dt.end_time.dt.normalize()

    # Quarter start/end
    qperiod = df["Date"].dt.to_period("Q")
    df["QuarterStartDate"] = qperiod.dt.start_time.dt.normalize()
    df["QuarterEndDate"] = qperiod.dt.end_time.dt.normalize()

    df["IsMonthStart"] = (df["Day"] == 1).astype(np.int8)
    df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(np.int8)
    df["IsQuarterStart"] = df["Date"].dt.is_quarter_start.astype(np.int8)
    df["IsQuarterEnd"] = df["Date"].dt.is_quarter_end.astype(np.int8)
    df["IsYearStart"] = ((df["Month"] == 1) & (df["Day"] == 1)).astype(np.int8)
    df["IsYearEnd"] = ((df["Month"] == 12) & (df["Day"] == 31)).astype(np.int8)

    df["WeekOfMonth"] = ((df["Day"] - 1) // 7 + 1).astype(np.int8)

    # ISO week fields + week boundaries (Monday..Sunday)
    iso = df["Date"].dt.isocalendar()
    df["WeekOfYearISO"] = iso.week.astype(np.int16)
    df["ISOYear"] = iso.year.astype(np.int16)

    df["WeekStartDate"] = (df["Date"] - pd.to_timedelta(df["Date"].dt.weekday, unit="D")).dt.normalize()
    df["WeekEndDate"] = (df["WeekStartDate"] + pd.Timedelta(days=6)).dt.normalize()

    # Next/Previous Business Day (edge-safe; if outside range -> self)
    biz_dates = df.loc[df["IsBusinessDay"] == 1, "Date"].to_numpy(dtype="datetime64[D]")
    date_vals = df["Date"].to_numpy(dtype="datetime64[D]")

    if biz_dates.size > 0:
        idx_next = np.searchsorted(biz_dates, date_vals, side="left")
        idx_prev = np.searchsorted(biz_dates, date_vals, side="right") - 1

        next_bd = date_vals.copy()
        prev_bd = date_vals.copy()

        ok_next = idx_next < biz_dates.size
        ok_prev = idx_prev >= 0

        next_bd[ok_next] = biz_dates[idx_next[ok_next]]
        prev_bd[ok_prev] = biz_dates[idx_prev[ok_prev]]

        # pd.to_datetime(ndarray[datetime64]) returns a DatetimeIndex; assign directly (values already day-normalized)
        df["NextBusinessDay"] = pd.to_datetime(next_bd)
        df["PreviousBusinessDay"] = pd.to_datetime(prev_bd)
    else:
        df["NextBusinessDay"] = df["Date"]
        df["PreviousBusinessDay"] = df["Date"]

    # Fiscal (month-based) calendar
    fy_start_month = int(fiscal_start_month)
    fy_start_month = 1 if fy_start_month < 1 else 12 if fy_start_month > 12 else fy_start_month

    df["FiscalYearStartYear"] = np.where(df["Month"] >= fy_start_month, df["Year"], df["Year"] - 1).astype(np.int16)
    df["FiscalMonthNumber"] = (((df["Month"].astype(int) - fy_start_month + 12) % 12) + 1).astype(np.int8)
    df["FiscalQuarterNumber"] = (((df["FiscalMonthNumber"] - 1) // 3) + 1).astype(np.int8)

    fy_end_add = 0 if fy_start_month == 1 else 1
    fiscal_year_end = (df["FiscalYearStartYear"].astype(int) + fy_end_add).astype(np.int16)

    if fy_start_month == 1:
        df["FiscalYearBin"] = df["FiscalYearStartYear"].astype(str)
    else:
        df["FiscalYearBin"] = df["FiscalYearStartYear"].astype(str) + "-" + fiscal_year_end.astype(str)
    df["FiscalQuarterName"] = "Q" + df["FiscalQuarterNumber"].astype(str) + " FY" + fiscal_year_end.astype(str)

    df["FiscalYearMonthNumber"] = (df["FiscalYearStartYear"].astype(int) * 12 + df["FiscalMonthNumber"].astype(int)).astype(np.int32)
    df["FiscalYearQuarterNumber"] = (df["FiscalYearStartYear"].astype(int) * 4 + df["FiscalQuarterNumber"].astype(int)).astype(np.int32)

    df["FiscalMonthIndex"] = df["FiscalYearMonthNumber"]
    df["FiscalQuarterIndex"] = df["FiscalYearQuarterNumber"]

    df["FiscalYearStartDate"] = pd.to_datetime(
        df["FiscalYearStartYear"].astype(str) + "-" + str(fy_start_month).zfill(2) + "-01"
    ).dt.normalize()
    df["FiscalYearEndDate"] = (df["FiscalYearStartDate"] + pd.DateOffset(years=1) - pd.Timedelta(days=1)).dt.normalize()

    fq_shift = (df["FiscalQuarterNumber"].astype(int) - 1) * 3
    fq_year = df["FiscalYearStartDate"].dt.year + ((df["FiscalYearStartDate"].dt.month + fq_shift - 1) // 12)
    fq_month = (df["FiscalYearStartDate"].dt.month + fq_shift - 1) % 12 + 1

    df["FiscalQuarterStartDate"] = pd.to_datetime(
        fq_year.astype(str) + "-" + fq_month.astype(str).str.zfill(2) + "-01"
    ).dt.normalize()
    df["FiscalQuarterEndDate"] = (df["FiscalQuarterStartDate"] + pd.DateOffset(months=3) - pd.Timedelta(days=1)).dt.normalize()

    df["IsFiscalYearStart"] = (df["Date"] == df["FiscalYearStartDate"]).astype(np.int8)
    df["IsFiscalYearEnd"] = (df["Date"] == df["FiscalYearEndDate"]).astype(np.int8)
    df["IsFiscalQuarterStart"] = (df["Date"] == df["FiscalQuarterStartDate"]).astype(np.int8)
    df["IsFiscalQuarterEnd"] = (df["Date"] == df["FiscalQuarterEndDate"]).astype(np.int8)

    df["FiscalYear"] = fiscal_year_end.astype(np.int16)
    df["FiscalYearLabel"] = "FY " + df["FiscalYear"].astype(str)

    # Deterministic "current" flags relative to as_of_date (default: end_date)
    as_of = pd.to_datetime(as_of_date).normalize() if as_of_date else end_date
    # If config provides an as_of_date outside the generated date window, clamp to end_date to keep offsets meaningful.
    if as_of < start_date or as_of > end_date:
        warn(f"dates: as_of_date {as_of.date()} outside generated window [{start_date.date()}..{end_date.date()}]; clamping to {end_date.date()}")
        as_of = end_date
    df["IsToday"] = (df["Date"] == as_of).astype(np.int8)
    df["IsCurrentYear"] = (df["Year"] == as_of.year).astype(np.int8)
    df["IsCurrentMonth"] = ((df["Year"] == as_of.year) & (df["Month"] == as_of.month)).astype(np.int8)
    current_quarter = (as_of.month - 1) // 3 + 1
    df["IsCurrentQuarter"] = ((df["Year"] == as_of.year) & (df["Quarter"] == current_quarter)).astype(np.int8)
    df["CurrentDayOffset"] = (df["Date"] - as_of).dt.days.astype(np.int32)

    # Offsets (relative to as_of) for fast "last N" slicing in Power BI
    df["YearOffset"] = (df["Year"].astype(int) - int(as_of.year)).astype(np.int16)
    as_of_cal_month_index = int(as_of.year) * 12 + int(as_of.month)
    as_of_cal_quarter_index = int(as_of.year) * 4 + int((as_of.month - 1) // 3 + 1)
    df["CalendarMonthOffset"] = (df["CalendarMonthIndex"].astype(int) - as_of_cal_month_index).astype(np.int32)
    df["CalendarQuarterOffset"] = (df["CalendarQuarterIndex"].astype(int) - as_of_cal_quarter_index).astype(np.int32)

    iso_asof = as_of.isocalendar()
    as_of_iso_year = int(iso_asof.year)
    as_of_iso_week = int(iso_asof.week)
        # ISO week index: contiguous week number based on ISO week start (Monday) relative to a fixed ISO reference.
    iso_ref = pd.Timestamp("2000-01-03")  # Monday of ISO week 1 in 2000
    df["ISOYearWeekIndex"] = (((df["WeekStartDate"] - iso_ref).dt.days) // 7).astype(np.int32)
    as_of_week_start = (as_of - pd.Timedelta(days=int(as_of.weekday()))).normalize()
    as_of_iso_year_week_index = int(((as_of_week_start - iso_ref).days) // 7)
    df["ISOWeekOffset"] = (df["ISOYearWeekIndex"].astype(int) - int(as_of_iso_year_week_index)).astype(np.int32)

    # Monthly fiscal offsets (computed after fiscal indices exist)
    asof_row = df.loc[df["Date"] == as_of]
    if not asof_row.empty:
        as_of_fiscal_month_index = int(asof_row["FiscalMonthIndex"].iloc[0])
        as_of_fiscal_quarter_index = int(asof_row["FiscalQuarterIndex"].iloc[0])
        df["FiscalMonthOffset"] = (df["FiscalMonthIndex"].astype(int) - as_of_fiscal_month_index).astype(np.int32)
        df["FiscalQuarterOffset"] = (df["FiscalQuarterIndex"].astype(int) - as_of_fiscal_quarter_index).astype(np.int32)
    else:
        df["FiscalMonthOffset"] = 0
        df["FiscalQuarterOffset"] = 0

    # Weekly fiscal columns from DAX logic
    weekly_cfg = weekly_cfg or WeeklyFiscalConfig()
    df = _compute_weekly_fiscal_columns(df, first_fiscal_month=fy_start_month, cfg=weekly_cfg)

    # Weekly fiscal offsets + system hints
    asof_row2 = df.loc[df["Date"] == as_of]
    if not asof_row2.empty:
        as_of_fw_year_week_index = int(asof_row2["FWYearWeekNumber"].iloc[0])
        as_of_fw_year_month_index = int(asof_row2["FWYearMonthNumber"].iloc[0])
        as_of_fw_year_quarter_index = int(asof_row2["FWYearQuarterNumber"].iloc[0])

        df = df.assign(
            FWYearWeekOffset=(df["FWYearWeekNumber"].astype(int) - as_of_fw_year_week_index).astype(np.int32),
            FWYearMonthOffset=(df["FWYearMonthNumber"].astype(int) - as_of_fw_year_month_index).astype(np.int32),
            FWYearQuarterOffset=(df["FWYearQuarterNumber"].astype(int) - as_of_fw_year_quarter_index).astype(np.int32),
        )
    else:
        df = df.assign(
            FWYearWeekOffset=np.int32(0),
            FWYearMonthOffset=np.int32(0),
            FWYearQuarterOffset=np.int32(0),
        )

    df = df.assign(
        FiscalSystem="Monthly",
        WeeklyFiscalSystem=f"Weekly ({weekly_cfg.quarter_week_type} {str(weekly_cfg.weekly_type).strip().title()})",
    )
    return df


# ---------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------

def run_dates(cfg: Dict, parquet_folder: Path) -> None:
    out_path = parquet_folder / "dates.parquet"
    if "dates" not in cfg:
        raise KeyError("Missing required config section: 'dates'")

    dates_cfg = cfg["dates"] or {}

    defaults_dates = (
        (cfg.get("defaults", {}) or {}).get("dates")
        or (cfg.get("_defaults", {}) or {}).get("dates")
        or {}
    )

    version_cfg = {**dates_cfg, "global_dates": defaults_dates}

    force = _bool_or(dates_cfg.get("_force_regenerate", False), False)
    if not force and not should_regenerate("dates", version_cfg, out_path):
        skip("Dates up-to-date; skipping.")
        return

    override_dates = _normalize_override_dates(dates_cfg)

    raw_start = override_dates.get("start") or defaults_dates.get("start")
    raw_end = override_dates.get("end") or defaults_dates.get("end")
    raw_start_ts, raw_end_ts = _require_start_end(raw_start, raw_end)

    buffer_years = max(0, _int_or(dates_cfg.get("buffer_years", 1), 1))
    start_date = pd.Timestamp(raw_start_ts.year - buffer_years, 1, 1)
    end_date = pd.Timestamp(raw_end_ts.year + buffer_years, 12, 31)
    info(f"Dates window: requested [{raw_start_ts.date()}..{raw_end_ts.date()}], generated [{start_date.date()}..{end_date.date()}] (buffer_years={buffer_years})")

    fiscal_start_month = _int_or(dates_cfg.get("fiscal_start_month") or dates_cfg.get("fiscal_month_offset", 5), 5)
    if dates_cfg.get("fiscal_start_month") is None and "fiscal_month_offset" in dates_cfg:
        warn("dates.fiscal_month_offset is treated as fiscal start month (1-12). Prefer dates.fiscal_start_month.")
    fiscal_start_month = 1 if fiscal_start_month < 1 else 12 if fiscal_start_month > 12 else fiscal_start_month

    as_of_date = dates_cfg.get("as_of_date") or str(raw_end_ts.date())

    compression = dates_cfg.get("parquet_compression", "snappy")
    compression_level = dates_cfg.get("parquet_compression_level", None)
    force_date32 = _bool_or(dates_cfg.get("force_date32", True), True)

    weekly_calendar = dates_cfg.get("weekly_calendar", {}) or {}
    wf_cfg = WeeklyFiscalConfig(
        enabled=_bool_or(weekly_calendar.get("enabled", True), True),
        first_day_of_week=_int_or(weekly_calendar.get("first_day_of_week", 0), 0),
        weekly_type=str(weekly_calendar.get("weekly_type", "Last")),
        quarter_week_type=str(weekly_calendar.get("quarter_week_type", "445")),
        type_start_fiscal_year=_int_or(weekly_calendar.get("type_start_fiscal_year", 1), 1),
    )

    with stage("Generating Dates"):
        df = generate_date_table(
            start_date,
            end_date,
            fiscal_start_month,
            as_of_date=as_of_date,
            weekly_cfg=wf_cfg,
        )

        cols = resolve_date_columns(dates_cfg)
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise RuntimeError(f"Dates: requested columns missing from generator: {missing}")

        df = df[cols]

        # Option A: keep both monthly-fiscal and weekly-fiscal in one table, but disambiguate weekly-fiscal columns.
        weekly_internal_cols = {
            "FWYearNumber",
            "FWYearLabel",
            "FWQuarterNumber",
            "FWQuarterLabel",
            "FWYearQuarterNumber",
            "FWMonthNumber",
            "FWMonthLabel",
            "FWYearMonthNumber",
            "FWWeekNumber",
            "FWWeekLabel",
            "FWYearWeekNumber",
            "FWYearWeekLabel",
            "FWYearQuarterOffset",
            "FWYearMonthOffset",
            "FWYearWeekOffset",
            "FWPeriodNumber",
            "FWPeriodLabel",
            "FWStartOfYear",
            "FWEndOfYear",
            "FWStartOfQuarter",
            "FWEndOfQuarter",
            "FWStartOfMonth",
            "FWEndOfMonth",
            "FWStartOfWeek",
            "FWEndOfWeek",
            "WeekDayNumber",
            "WeekDayNameShort",
            "FWDayOfYearNumber",
            "FWDayOfQuarterNumber",
            "FWDayOfMonthNumber",
            "IsWorkingDay",
            "DayType",
            "FWWeekInQuarterNumber",
            "FWYearMonthLabel",
            "FWYearQuarterLabel",
        }

        rename_map: Dict[str, str] = {}
        for c in df.columns:
            human = _humanize_col_name(c)

            if c in weekly_internal_cols:
                if human.startswith("FW "):
                    human = human[3:]
                human = "Weekly Fiscal " + human

                # Specific index renames for readability
                if c == "FWYearMonthNumber":
                    human = "Weekly Fiscal Year Month Index"
                elif c == "FWYearQuarterNumber":
                    human = "Weekly Fiscal Year Quarter Index"
                elif c == "FWYearWeekNumber":
                    human = "Weekly Fiscal Year Week Index"
                elif c == "FWYearMonthOffset":
                    human = "Weekly Fiscal Year Month Offset"
                elif c == "FWYearQuarterOffset":
                    human = "Weekly Fiscal Year Quarter Offset"
                elif c == "FWYearWeekOffset":
                    human = "Weekly Fiscal Year Week Offset"
            rename_map[c] = human

        df = df.rename(columns=rename_map)

        write_parquet_with_date32(
            df,
            out_path,
            compression=compression,
            compression_level=compression_level,
            force_date32=force_date32,
        )

    save_version("dates", version_cfg, out_path)
    info(f"Dates dimension written: {out_path}")
