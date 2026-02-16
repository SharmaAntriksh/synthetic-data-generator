from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from src.utils.output_utils import write_parquet_with_date32

import numpy as np
import pandas as pd

from src.utils import info, skip, stage
from src.versioning import should_regenerate, save_version


# ---------------------------------------------------------
# DATE COLUMN GROUPS
# ---------------------------------------------------------

BASE_COLUMNS = ["Date", "DateKey"]

CALENDAR_COLUMNS = [
    "Date", "DateKey",
    "Year", "IsYearStart", "IsYearEnd",
    "Quarter", "QuarterStartDate", "QuarterEndDate",
    "IsQuarterStart", "IsQuarterEnd",
    "QuarterYear",
    "Month", "MonthName", "MonthShort",
    "MonthStartDate", "MonthEndDate",
    "MonthYear", "MonthYearNumber", "CalendarMonthIndex", "CalendarQuarterIndex",
    "IsMonthStart", "IsMonthEnd",
    "WeekOfMonth",
    "Day", "DayName", "DayShort", "DayOfYear", "DayOfWeek",
    "IsWeekend", "IsBusinessDay",
    "NextBusinessDay", "PreviousBusinessDay",
    "IsToday", "IsCurrentYear", "IsCurrentMonth",
    "IsCurrentQuarter", "CurrentDayOffset",
]

ISO_COLUMNS = [
    "WeekOfYearISO",
    "ISOYear",
    "WeekStartDate",
    "WeekEndDate",
]

FISCAL_COLUMNS = [
    "FiscalYearStartYear", "FiscalMonthNumber", "FiscalQuarterNumber",
    "FiscalMonthIndex", "FiscalQuarterIndex",
    "FiscalQuarterName", "FiscalYearBin",
    "FiscalYearMonthNumber", "FiscalYearQuarterNumber",
    "FiscalYearStartDate", "FiscalYearEndDate",
    "FiscalQuarterStartDate", "FiscalQuarterEndDate",
    "IsFiscalYearStart", "IsFiscalYearEnd",
    "IsFiscalQuarterStart", "IsFiscalQuarterEnd",
    "FiscalYear", "FiscalYearLabel",
]


def _dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def resolve_date_columns(dates_cfg: Dict) -> List[str]:
    """
    Resolve output columns based on dates.include, but ALWAYS keep Date + DateKey.

    dates.include:
      calendar: true/false
      iso: true/false
      fiscal: true/false
    """
    include_cfg = (dates_cfg or {}).get("include", {}) or {}

    cols: List[str] = []
    cols.extend(BASE_COLUMNS)  # always present

    if include_cfg.get("calendar", True):
        cols.extend(CALENDAR_COLUMNS)
    if include_cfg.get("iso", False):
        cols.extend(ISO_COLUMNS)
    if include_cfg.get("fiscal", False):
        cols.extend(FISCAL_COLUMNS)

    return _dedupe_preserve_order(cols)


# ---------------------------------------------------------
# DATE GENERATOR
# ---------------------------------------------------------

def generate_date_table(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    fiscal_start_month: int,
    *,
    as_of_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Build a date dimension between start_date and end_date inclusive.

    - Uses datetime64[ns] internally (fast)
    - 'as_of_date' controls IsToday/IsCurrent* and CurrentDayOffset deterministically.
      If None, defaults to end_date (stable for synthetic datasets).
    """
    if fiscal_start_month < 1 or fiscal_start_month > 12:
        raise ValueError(f"fiscal_start_month must be 1..12, got {fiscal_start_month}")

    start_date = pd.to_datetime(start_date).normalize()
    end_date = pd.to_datetime(end_date).normalize()

    if as_of_date is None:
        as_of_date = end_date
    else:
        as_of_date = pd.to_datetime(as_of_date).normalize()

    df = pd.DataFrame({"Date": pd.date_range(start_date, end_date, freq="D")})
    df["DateKey"] = df["Date"].dt.strftime("%Y%m%d").astype(np.int64)

    # Basic parts
    df["Year"] = df["Date"].dt.year.astype(int)
    df["Month"] = df["Date"].dt.month.astype(int)
    df["Day"] = df["Date"].dt.day.astype(int)
    df["Quarter"] = df["Date"].dt.quarter.astype(int)

    df["MonthName"] = df["Date"].dt.strftime("%B")
    df["MonthShort"] = df["Date"].dt.strftime("%b")
    df["DayName"] = df["Date"].dt.strftime("%A")
    df["DayShort"] = df["Date"].dt.strftime("%a")

    df["DayOfYear"] = df["Date"].dt.dayofyear.astype(int)
    df["MonthYear"] = df["Date"].dt.strftime("%b %Y")
    df["MonthYearNumber"] = (df["Year"] * 100 + df["Month"]).astype(int)

    # Indexes (monotonic but not "dense from 0" across multiple years – consistent with original intent)
    df["CalendarMonthIndex"] = (df["Year"] * 12 + df["Month"]).astype(int)
    df["CalendarQuarterIndex"] = (df["Year"] * 4 + df["Quarter"]).astype(int)

    # Day-of-week conventions (kept compatible with your existing formula)
    weekday = df["Date"].dt.weekday  # 0=Mon..6=Sun
    df["DayOfWeek"] = ((weekday + 1) % 7).astype(int)  # 0=Sun, 1=Mon, ... 6=Sat
    df["IsWeekend"] = df["DayOfWeek"].isin([0, 6]).astype(int)
    df["IsBusinessDay"] = (df["IsWeekend"] == 0).astype(int)

    # Month start/end (fast)
    df["MonthStartDate"] = df["Date"].values.astype("datetime64[M]")
    df["MonthEndDate"] = (df["MonthStartDate"] + pd.offsets.MonthEnd(1)).dt.normalize()

    # Quarter start/end (clean + fast)
    qperiod = df["Date"].dt.to_period("Q")
    df["QuarterStartDate"] = qperiod.dt.start_time.dt.normalize()
    df["QuarterEndDate"] = qperiod.dt.end_time.dt.normalize()

    df["IsMonthStart"] = (df["Day"] == 1).astype(int)
    df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(int)
    df["IsQuarterStart"] = df["Date"].dt.is_quarter_start.astype(int)
    df["IsQuarterEnd"] = df["Date"].dt.is_quarter_end.astype(int)
    df["IsYearStart"] = ((df["Month"] == 1) & (df["Day"] == 1)).astype(int)
    df["IsYearEnd"] = ((df["Month"] == 12) & (df["Day"] == 31)).astype(int)

    df["QuarterYear"] = "Q" + df["Quarter"].astype(str) + " " + df["Year"].astype(str)
    df["WeekOfMonth"] = ((df["Day"] - 1) // 7 + 1).astype(int)

    # ISO week fields
    iso = df["Date"].dt.isocalendar()
    df["WeekOfYearISO"] = iso.week.astype(int)
    df["ISOYear"] = iso.year.astype(int)

    # Week start/end (Monday..Sunday)
    df["WeekStartDate"] = (df["Date"] - pd.to_timedelta(df["Date"].dt.weekday, unit="D")).dt.normalize()
    df["WeekEndDate"] = (df["WeekStartDate"] + pd.Timedelta(days=6)).dt.normalize()

    # Next/Previous Business Day (vectorized, replaces merge_asof)
    # Semantics preserved: if Date is a business day, Next/Prev = itself.
    biz_dates = df.loc[df["IsBusinessDay"] == 1, "Date"].to_numpy()
    date_vals = df["Date"].to_numpy()

    if biz_dates.size > 0:
        idx_next = np.searchsorted(biz_dates, date_vals, side="left")
        idx_next = np.clip(idx_next, 0, biz_dates.size - 1)
        df["NextBusinessDay"] = pd.to_datetime(biz_dates[idx_next]).normalize()

        idx_prev = np.searchsorted(biz_dates, date_vals, side="right") - 1
        idx_prev = np.clip(idx_prev, 0, biz_dates.size - 1)
        df["PreviousBusinessDay"] = pd.to_datetime(biz_dates[idx_prev]).normalize()
    else:
        df["NextBusinessDay"] = df["Date"]
        df["PreviousBusinessDay"] = df["Date"]

    # Fiscal calendar
    fy_start_month = fiscal_start_month

    df["FiscalYearStartYear"] = np.where(df["Month"] >= fy_start_month, df["Year"], df["Year"] - 1).astype(int)
    df["FiscalMonthNumber"] = (((df["Month"] - fy_start_month + 12) % 12) + 1).astype(int)
    df["FiscalQuarterNumber"] = (((df["FiscalMonthNumber"] - 1) // 3) + 1).astype(int)

    df["FiscalYearBin"] = df["FiscalYearStartYear"].astype(str) + "-" + (df["FiscalYearStartYear"] + 1).astype(str)
    df["FiscalQuarterName"] = (
        "Q" + df["FiscalQuarterNumber"].astype(str) + " FY" + (df["FiscalYearStartYear"] + 1).astype(str)
    )

    df["FiscalYearMonthNumber"] = (df["FiscalYearStartYear"] * 12 + df["FiscalMonthNumber"]).astype(int)
    df["FiscalYearQuarterNumber"] = (df["FiscalYearStartYear"] * 4 + df["FiscalQuarterNumber"]).astype(int)

    df["FiscalMonthIndex"] = (df["FiscalYearStartYear"] * 12 + df["FiscalMonthNumber"]).astype(int)
    df["FiscalQuarterIndex"] = (df["FiscalYearStartYear"] * 4 + df["FiscalQuarterNumber"]).astype(int)

    df["FiscalYearStartDate"] = pd.to_datetime(
        df["FiscalYearStartYear"].astype(str) + "-" + str(fy_start_month).zfill(2) + "-01"
    ).dt.normalize()
    df["FiscalYearEndDate"] = (df["FiscalYearStartDate"] + pd.DateOffset(years=1) - pd.Timedelta(days=1)).dt.normalize()

    fq_shift = (df["FiscalQuarterNumber"] - 1) * 3
    fq_year = df["FiscalYearStartDate"].dt.year + ((df["FiscalYearStartDate"].dt.month + fq_shift - 1) // 12)
    fq_month = (df["FiscalYearStartDate"].dt.month + fq_shift - 1) % 12 + 1

    df["FiscalQuarterStartDate"] = pd.to_datetime(
        fq_year.astype(str) + "-" + fq_month.astype(str).str.zfill(2) + "-01"
    ).dt.normalize()
    df["FiscalQuarterEndDate"] = (
        df["FiscalQuarterStartDate"] + pd.DateOffset(months=3) - pd.Timedelta(days=1)
    ).dt.normalize()

    df["IsFiscalYearStart"] = (df["Date"] == df["FiscalYearStartDate"]).astype(int)
    df["IsFiscalYearEnd"] = (df["Date"] == df["FiscalYearEndDate"]).astype(int)
    df["IsFiscalQuarterStart"] = (df["Date"] == df["FiscalQuarterStartDate"]).astype(int)
    df["IsFiscalQuarterEnd"] = (df["Date"] == df["FiscalQuarterEndDate"]).astype(int)

    df["FiscalYear"] = np.where(df["Month"] < fy_start_month, df["Year"], df["Year"] + 1).astype(int)
    df["FiscalYearLabel"] = "FY " + df["FiscalYear"].astype(str)

    # Deterministic "current" flags relative to as_of_date
    df["IsToday"] = (df["Date"] == as_of_date).astype(int)
    df["IsCurrentYear"] = (df["Year"] == as_of_date.year).astype(int)
    df["IsCurrentMonth"] = ((df["Year"] == as_of_date.year) & (df["Month"] == as_of_date.month)).astype(int)
    current_quarter = (as_of_date.month - 1) // 3 + 1
    df["IsCurrentQuarter"] = ((df["Year"] == as_of_date.year) & (df["Quarter"] == current_quarter)).astype(int)
    df["CurrentDayOffset"] = (df["Date"] - as_of_date).dt.days.astype(int)

    # Stable column ordering (full superset; caller can subset)
    df = df[CALENDAR_COLUMNS + ISO_COLUMNS + FISCAL_COLUMNS]
    return df


# ---------------------------------------------------------
# PIPELINE WRAPPER
# ---------------------------------------------------------

def _normalize_override_dates(dates_cfg: Dict) -> Dict:
    """
    Supports either:
      dates.override: { dates: {start,end} }
    or:
      dates.override: { start,end }
    """
    override = (dates_cfg or {}).get("override", {}) or {}
    if isinstance(override, dict) and isinstance(override.get("dates"), dict):
        return override["dates"] or {}
    return override if isinstance(override, dict) else {}


def _require_start_end(raw_start: Optional[str], raw_end: Optional[str]) -> Tuple[pd.Timestamp, pd.Timestamp]:
    if not raw_start or not raw_end:
        raise ValueError(
            "Dates config is missing start/end. Provide defaults.dates.start/end "
            "or set dates.override.dates.start/end."
        )
    return pd.to_datetime(raw_start), pd.to_datetime(raw_end)


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

    force = dates_cfg.get("_force_regenerate", False)
    if not force and not should_regenerate("dates", version_cfg, out_path):
        skip("Dates up-to-date; skipping.")
        return

    override_dates = _normalize_override_dates(dates_cfg)

    raw_start = override_dates.get("start") or defaults_dates.get("start")
    raw_end = override_dates.get("end") or defaults_dates.get("end")
    raw_start_ts, raw_end_ts = _require_start_end(raw_start, raw_end)

    # Expand to full years + buffer (default 1 year)
    buffer_years = int(dates_cfg.get("buffer_years", 1))
    start_date = pd.Timestamp(raw_start_ts.year - buffer_years, 1, 1)
    end_date = pd.Timestamp(raw_end_ts.year + buffer_years, 12, 31)

    fiscal_start_month = int(dates_cfg.get("fiscal_start_month") or dates_cfg.get("fiscal_month_offset", 5))

    # Deterministic "as of" (default: raw_end, can be overridden)
    as_of_date = dates_cfg.get("as_of_date") or str(raw_end_ts.date())

    # Parquet options (optional keys; safe defaults)
    compression = dates_cfg.get("parquet_compression", "snappy")
    compression_level = dates_cfg.get("parquet_compression_level", None)
    force_date32 = bool(dates_cfg.get("force_date32", True))

    with stage("Generating Dates"):
        df = generate_date_table(
            start_date,
            end_date,
            fiscal_start_month,
            as_of_date=as_of_date,
        )

        cols = resolve_date_columns(dates_cfg)
        df = df[cols]

        # Write so Power Query sees Date (Arrow date32) without PQ transforms
        write_parquet_with_date32(
            df,
            out_path,
            compression=compression,
            compression_level=compression_level,
            force_date32=force_date32,
        )

    save_version("dates", version_cfg, out_path)
    info(f"Dates dimension written: {out_path}")