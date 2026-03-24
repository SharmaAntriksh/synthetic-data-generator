"""Column resolver for the dates dimension."""
from __future__ import annotations

from typing import Any, Dict, List

from .helpers import _dedupe_preserve_order
from .weekly_fiscal import _wf_is_enabled


# ---------------------------------------------------------------------
# Column resolver
# ---------------------------------------------------------------------
# Duplicate aliases removed.  The following columns were consolidated:
#
#   Removed alias          →  Canonical column kept
#   ─────────────────────────────────────────────────
#   MonthNameShort         →  MonthShort
#   DayNameShort           →  DayShort
#   YearMonthKey           →  MonthYearNumber
#   YearMonthLabel         →  MonthYear
#   YearQuarterLabel       →  QuarterYear
#   FiscalYearMonthNumber  →  FiscalMonthIndex
#   FiscalYearQuarterNumber→  FiscalQuarterIndex
#   FWYearWeekLabel        →  FWWeekLabel
#   FWYearQuarterLabel     →  FWQuarterLabel
# ---------------------------------------------------------------------

def resolve_date_columns(dates_cfg: Dict[str, Any]) -> List[str]:
    """Build the ordered output column list based on config ``include`` flags.

    Base columns (primary keys + fundamental date-part attributes like Year,
    Month, MonthName, Day, DayName, etc.) are always included regardless of
    ``include`` flags.  The ``include.calendar`` flag controls only calendar-
    specific flags and offsets (IsYearStart, IsToday, CurrentDayOffset, etc.).
    """
    if dates_cfg and hasattr(dates_cfg, "include"):
        include = dates_cfg.include or {}
    else:
        include = ((dates_cfg or {}).get("include", {}) if isinstance(dates_cfg, dict) else {}) or {}
    if hasattr(include, "weekly_fiscal"):
        wf_cfg = include.weekly_fiscal or {}
    else:
        wf_cfg = (include.get("weekly_fiscal", {}) if isinstance(include, dict) else {}) or {}

    # Always present: primary keys + fundamental date-part attributes.
    base_cols = [
        "Date", "DateKey", "SequentialDayIndex",
        "Year",
        "Quarter", "QuarterStartDate", "QuarterEndDate",
        "QuarterYear",
        "Month", "MonthName", "MonthShort",
        "MonthStartDate", "MonthEndDate",
        "MonthYear", "MonthYearNumber",
        "YearQuarterKey",
        "CalendarMonthIndex", "CalendarQuarterIndex",
        "WeekOfMonth",
        "Day", "DayName", "DayShort", "DayOfYear", "DayOfWeek",
        "IsWeekend", "IsBusinessDay",
        "NextBusinessDay", "PreviousBusinessDay",
    ]

    # Calendar flags and relative offsets (gated by include.calendar).
    calendar_cols = [
        "IsYearStart", "IsYearEnd",
        "IsQuarterStart", "IsQuarterEnd",
        "IsMonthStart", "IsMonthEnd",
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
    ]

    cols: List[str] = list(base_cols)

    _get = (lambda k, d: getattr(include, k, d)) if not isinstance(include, dict) else (lambda k, d: include.get(k, d))
    if _get("calendar", True):
        cols += calendar_cols
    if _get("iso", True):
        cols += iso_cols
    if _get("fiscal", True):
        cols += fiscal_cols

    if _wf_is_enabled(wf_cfg):
        cols += weekly_cols

    return _dedupe_preserve_order(cols)
