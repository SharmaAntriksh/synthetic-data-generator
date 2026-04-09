"""Column resolver for the dates dimension."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, List

from src.utils.config_helpers import bool_or as _bool_or
from src.utils.static_schemas import _WF_INTERNAL_COLS

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
#   YearMonthKey           →  MonthYearKey
#   YearMonthLabel         →  MonthYear
#   YearQuarterLabel       →  QuarterYear
#   FiscalYearMonthNumber  →  FiscalMonthIndex
#   FiscalYearQuarterNumber→  FiscalQuarterIndex
#   FWYearWeekLabel        →  FWWeekLabel
#   FWYearQuarterLabel     →  FWQuarterLabel
# ---------------------------------------------------------------------

# PascalCase → spaced display names.  Applied when dates.spaced_column_names
# is true.  Keys not listed here keep their PascalCase name unchanged.
_SPACED_NAMES: Dict[str, str] = {
    # Base columns
    "Date": "Date",
    "DateKey": "Date Key",
    "DateSerialNumber": "Date Serial Number",
    "Year": "Year",
    "Quarter": "Quarter",
    "QuarterStartDate": "Quarter Start Date",
    "QuarterEndDate": "Quarter End Date",
    "QuarterYear": "Quarter Year",
    "Month": "Month",
    "MonthName": "Month Name",
    "MonthShort": "Month Short",
    "MonthStartDate": "Month Start Date",
    "MonthEndDate": "Month End Date",
    "MonthYear": "Month Year",
    "MonthYearKey": "Month Year Key",
    "YearQuarterKey": "Year Quarter Key",
    "CalendarMonthIndex": "Calendar Month Index",
    "CalendarQuarterIndex": "Calendar Quarter Index",
    "WeekOfMonth": "Week Of Month",
    "CalendarWeekNumber": "Calendar Week Number",
    "CalendarWeekStartDate": "Calendar Week Start Date",
    "CalendarWeekEndDate": "Calendar Week End Date",
    "CalendarWeekDateRange": "Calendar Week Date Range",
    "CalendarWeekIndex": "Calendar Week Index",
    "Day": "Day",
    "DayName": "Day Name",
    "DayShort": "Day Short",
    "DayOfYear": "Day Of Year",
    "DayOfWeek": "Day Of Week",
    "IsWeekend": "Is Weekend",
    "IsBusinessDay": "Is Business Day",
    "NextBusinessDay": "Next Business Day",
    "PreviousBusinessDay": "Previous Business Day",
    "YearStartDate": "Year Start Date",
    "YearEndDate": "Year End Date",
    "MonthDays": "Month Days",
    "QuarterDays": "Quarter Days",
    "YearDays": "Year Days",
    "DayOfQuarter": "Day Of Quarter",
    "DatePreviousWeek": "Date Previous Week",
    "DatePreviousMonth": "Date Previous Month",
    "DatePreviousQuarter": "Date Previous Quarter",
    "DatePreviousYear": "Date Previous Year",
    # Calendar flags & offsets
    "IsYearStart": "Is Year Start",
    "IsYearEnd": "Is Year End",
    "IsQuarterStart": "Is Quarter Start",
    "IsQuarterEnd": "Is Quarter End",
    "IsMonthStart": "Is Month Start",
    "IsMonthEnd": "Is Month End",
    "IsToday": "Is Today",
    "IsCurrentYear": "Is Current Year",
    "IsCurrentMonth": "Is Current Month",
    "IsCurrentQuarter": "Is Current Quarter",
    "CurrentDayOffset": "Current Day Offset",
    "YearOffset": "Year Offset",
    "CalendarMonthOffset": "Calendar Month Offset",
    "CalendarQuarterOffset": "Calendar Quarter Offset",
    "CalendarWeekOffset": "Calendar Week Offset",
    # ISO week columns
    "ISOWeekNumber": "ISO Week Number",
    "ISOYear": "ISO Year",
    "ISOYearWeekIndex": "ISO Year Week Index",
    "ISOWeekOffset": "ISO Week Offset",
    "ISOWeekStartDate": "ISO Week Start Date",
    "ISOWeekEndDate": "ISO Week End Date",
    "ISOWeekDateRange": "ISO Week Date Range",
    # Fiscal columns
    "FiscalYearStartYear": "Fiscal Year Start Year",
    "FiscalMonthNumber": "Fiscal Month Number",
    "FiscalQuarterNumber": "Fiscal Quarter Number",
    "FiscalMonthIndex": "Fiscal Month Index",
    "FiscalQuarterIndex": "Fiscal Quarter Index",
    "FiscalMonthOffset": "Fiscal Month Offset",
    "FiscalQuarterOffset": "Fiscal Quarter Offset",
    "FiscalQuarterLabel": "Fiscal Quarter Label",
    "FiscalMonthName": "Fiscal Month Name",
    "FiscalMonthShort": "Fiscal Month Short",
    "FiscalYearRange": "Fiscal Year Range",
    "FiscalYearStartDate": "Fiscal Year Start Date",
    "FiscalYearEndDate": "Fiscal Year End Date",
    "FiscalQuarterStartDate": "Fiscal Quarter Start Date",
    "FiscalQuarterEndDate": "Fiscal Quarter End Date",
    "IsFiscalYearStart": "Is Fiscal Year Start",
    "IsFiscalYearEnd": "Is Fiscal Year End",
    "IsFiscalQuarterStart": "Is Fiscal Quarter Start",
    "IsFiscalQuarterEnd": "Is Fiscal Quarter End",
    "FiscalYear": "Fiscal Year",
    "FiscalYearLabel": "Fiscal Year Label",
    "FiscalQuarterDays": "Fiscal Quarter Days",
    "FiscalYearDays": "Fiscal Year Days",
    "FiscalDayOfQuarter": "Fiscal Day Of Quarter",
    "FiscalDayOfYear": "Fiscal Day Of Year",
    "FiscalYearOffset": "Fiscal Year Offset",
    # Weekly fiscal (4-4-5) columns
    "FWYearNumber": "FW Year Number",
    "FWYearLabel": "FW Year Label",
    "FWQuarterNumber": "FW Quarter Number",
    "FWQuarterLabel": "FW Quarter Label",
    "FWQuarterIndex": "FW Quarter Index",
    "FWQuarterOffset": "FW Quarter Offset",
    "FWMonthNumber": "FW Month Number",
    "FWMonthLabel": "FW Month Label",
    "FWMonthIndex": "FW Month Index",
    "FWMonthOffset": "FW Month Offset",
    "FWWeekNumber": "FW Week Number",
    "FWWeekLabel": "FW Week Label",
    "FWWeekDateRange": "FW Week Date Range",
    "FWWeekIndex": "FW Week Index",
    "FWWeekOffset": "FW Week Offset",
    "FWPeriodNumber": "FW Period Number",
    "FWPeriodLabel": "FW Period Label",
    "FWStartOfYear": "FW Start Of Year",
    "FWEndOfYear": "FW End Of Year",
    "FWStartOfQuarter": "FW Start Of Quarter",
    "FWEndOfQuarter": "FW End Of Quarter",
    "FWStartOfMonth": "FW Start Of Month",
    "FWEndOfMonth": "FW End Of Month",
    "FWStartOfWeek": "FW Start Of Week",
    "FWEndOfWeek": "FW End Of Week",
    "FWWeekDayNumber": "FW Week Day Number",
    "FWWeekDayNameShort": "FW Week Day Name Short",
    "FWDayOfYear": "FW Day Of Year",
    "FWDayOfQuarter": "FW Day Of Quarter",
    "FWDayOfMonth": "FW Day Of Month",
    "FWMonthDays": "FW Month Days",
    "FWQuarterDays": "FW Quarter Days",
    "FWYearDays": "FW Year Days",
    "FWDatePreviousMonth": "FW Date Previous Month",
    "FWDatePreviousQuarter": "FW Date Previous Quarter",
    "FWIsWorkingDay": "FW Is Working Day",
    "FWDayType": "FW Day Type",
    "FWWeekInQuarterNumber": "FW Week In Quarter Number",
    "FWYearMonthLabel": "FW Year Month Label",
    "WeeklyFiscalSystem": "Weekly Fiscal System",
}

def _spaced_column_names_enabled(dates_cfg: Dict[str, Any]) -> bool:
    """Return True if ``dates.spaced_column_names`` is set."""
    if isinstance(dates_cfg, dict):
        return _bool_or(dates_cfg.get("spaced_column_names"), False)
    if isinstance(dates_cfg, Mapping):
        return _bool_or(getattr(dates_cfg, "spaced_column_names", None), False)
    return False


def _apply_spaced_names(cols: List[str]) -> List[str]:
    """Replace PascalCase column names with spaced display names."""
    return [_SPACED_NAMES.get(c, c) for c in cols]


def get_date_rename_map(dates_cfg: Dict[str, Any]) -> Dict[str, str]:
    """Return ``{PascalCase: Spaced Name}`` dict when spaced names are enabled.

    Returns an empty dict when the feature is off, so callers can always do
    ``df.rename(columns=get_date_rename_map(cfg))``.
    """
    if not _spaced_column_names_enabled(dates_cfg):
        return {}
    return dict(_SPACED_NAMES)


def resolve_date_columns(dates_cfg: Dict[str, Any]) -> List[str]:
    """Build the ordered output column list based on config ``include`` flags.

    Base columns (primary keys + fundamental date-part attributes like Year,
    Month, MonthName, Day, DayName, etc.) are always included regardless of
    ``include`` flags.  The ``include.calendar`` flag controls only calendar-
    specific flags and offsets (IsYearStart, IsToday, CurrentDayOffset, etc.).

    When ``dates.spaced_column_names`` is true, the returned column names
    have spaces (e.g. ``"Calendar Week Start Date"``).
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
        "Date", "DateKey", "DateSerialNumber",
        "Year",
        "Quarter", "QuarterStartDate", "QuarterEndDate",
        "QuarterYear",
        "Month", "MonthName", "MonthShort",
        "MonthStartDate", "MonthEndDate",
        "MonthYear", "MonthYearKey",
        "YearQuarterKey",
        "CalendarMonthIndex", "CalendarQuarterIndex",
        "WeekOfMonth",
        "CalendarWeekNumber",
        "CalendarWeekStartDate", "CalendarWeekEndDate",
        "CalendarWeekDateRange",
        "CalendarWeekIndex",
        "Day", "DayName", "DayShort", "DayOfYear", "DayOfQuarter", "DayOfWeek",
        "IsWeekend", "IsBusinessDay",
        "NextBusinessDay", "PreviousBusinessDay",
        "YearStartDate", "YearEndDate",
        "MonthDays", "QuarterDays", "YearDays",
        "DatePreviousWeek", "DatePreviousMonth", "DatePreviousQuarter", "DatePreviousYear",
    ]

    # Calendar flags and relative offsets (gated by include.calendar).
    calendar_cols = [
        "IsYearStart", "IsYearEnd",
        "IsQuarterStart", "IsQuarterEnd",
        "IsMonthStart", "IsMonthEnd",
        "IsToday", "IsCurrentYear", "IsCurrentMonth", "IsCurrentQuarter",
        "CurrentDayOffset", "YearOffset",
        "CalendarMonthOffset", "CalendarQuarterOffset", "CalendarWeekOffset",
    ]

    iso_cols = [
        "ISOWeekNumber",
        "ISOYear",
        "ISOYearWeekIndex",
        "ISOWeekOffset",
        "ISOWeekStartDate",
        "ISOWeekEndDate",
        "ISOWeekDateRange",
    ]

    fiscal_cols = [
        "FiscalYearStartYear", "FiscalMonthNumber", "FiscalQuarterNumber",
        "FiscalMonthIndex", "FiscalQuarterIndex", "FiscalMonthOffset", "FiscalQuarterOffset",
        "FiscalQuarterLabel", "FiscalMonthName", "FiscalMonthShort", "FiscalYearRange",
        "FiscalYearStartDate", "FiscalYearEndDate",
        "FiscalQuarterStartDate", "FiscalQuarterEndDate",
        "IsFiscalYearStart", "IsFiscalYearEnd",
        "IsFiscalQuarterStart", "IsFiscalQuarterEnd",
        "FiscalYear", "FiscalYearLabel",
        "FiscalQuarterDays", "FiscalYearDays",
        "FiscalDayOfQuarter", "FiscalDayOfYear",
        "FiscalYearOffset",
    ]

    cols: List[str] = list(base_cols)

    _get = (lambda k, d: getattr(include, k, d)) if not isinstance(include, dict) else (lambda k, d: include.get(k, d))
    if _get("calendar", True):
        cols += calendar_cols
    if _get("iso", False):
        cols += iso_cols
    if _get("fiscal", True):
        cols += fiscal_cols

    if _wf_is_enabled(wf_cfg):
        cols += _WF_INTERNAL_COLS

    cols = _dedupe_preserve_order(cols)
    if _spaced_column_names_enabled(dates_cfg):
        cols = _apply_spaced_names(cols)
    return cols
