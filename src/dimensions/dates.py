# ---------------------------------------------------------
#  DATES DIMENSION (PIPELINE READY)
# ---------------------------------------------------------

import pandas as pd
import numpy as np
import re
from pathlib import Path

from src.utils import info, skip, stage
from src.versioning import should_regenerate, save_version


# ---------------------------------------------------------
# DATE COLUMN GROUPS
# ---------------------------------------------------------

CALENDAR_COLUMNS = [
    "Date","DateKey",
    "Year","IsYearStart","IsYearEnd",
    "Quarter","QuarterStartDate","QuarterEndDate",
    "IsQuarterStart","IsQuarterEnd",
    "QuarterYear",
    "Month","MonthName","MonthShort",
    "MonthStartDate","MonthEndDate",
    "MonthYear","MonthYearNumber","CalendarMonthIndex","CalendarQuarterIndex",
    "IsMonthStart","IsMonthEnd",
    "WeekOfMonth",
    "Day","DayName","DayShort","DayOfYear","DayOfWeek",
    "IsWeekend","IsBusinessDay",
    "NextBusinessDay","PreviousBusinessDay",
    "IsToday","IsCurrentYear","IsCurrentMonth",
    "IsCurrentQuarter","CurrentDayOffset"
]

ISO_COLUMNS = [
    "WeekOfYearISO",
    "ISOYear",
    "WeekStartDate",
    "WeekEndDate",
]

FISCAL_COLUMNS = [
    "FiscalYearStartYear","FiscalMonthNumber","FiscalQuarterNumber",
    "FiscalMonthIndex","FiscalQuarterIndex",
    "FiscalQuarterName","FiscalYearBin",
    "FiscalYearMonthNumber","FiscalYearQuarterNumber",
    "FiscalYearStartDate","FiscalYearEndDate",
    "FiscalQuarterStartDate","FiscalQuarterEndDate",
    "IsFiscalYearStart","IsFiscalYearEnd",
    "IsFiscalQuarterStart","IsFiscalQuarterEnd",
    "FiscalYear","FiscalYearLabel",
]


def resolve_date_columns(dates_cfg):
    include_cfg = dates_cfg.get("include", {})

    cols = []
    if include_cfg.get("calendar", True):
        cols.extend(CALENDAR_COLUMNS)
    if include_cfg.get("iso", False):
        cols.extend(ISO_COLUMNS)
    if include_cfg.get("fiscal", False):
        cols.extend(FISCAL_COLUMNS)

    return cols


# ---------------------------------------------------------
# DATE GENERATOR (LOGIC UNCHANGED)
# ---------------------------------------------------------

def generate_date_table(start_date, end_date, first_fy_month):

    start_date = start_date or "2020-01-01"
    end_date = end_date or "2026-12-31"
    fy_start_month = first_fy_month or 5

    df = pd.DataFrame({"Date": pd.date_range(start_date, end_date, freq="D")})

    df["DateKey"] = df["Date"].dt.strftime("%Y%m%d").astype(int)

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Quarter"] = df["Date"].dt.quarter

    df["MonthName"] = df["Date"].dt.strftime("%B")
    df["MonthShort"] = df["Date"].dt.strftime("%b")

    df["MonthStartDate"] = df["Date"].values.astype("datetime64[M]")
    df["MonthEndDate"] = df["MonthStartDate"] + pd.offsets.MonthEnd(1)

    df["DayName"] = df["Date"].dt.strftime("%A")
    df["DayShort"] = df["Date"].dt.strftime("%a")
    df["DayOfYear"] = df["Date"].dt.dayofyear

    df["MonthYear"] = df["Date"].dt.strftime("%b %Y")
    df["MonthYearNumber"] = df["Year"] * 100 + df["Month"]
    df["CalendarMonthIndex"] = df["Year"] * 12 + df["Month"]
    df["CalendarQuarterIndex"] = df["Year"] * 4 + df["Quarter"]

    weekday = df["Date"].dt.weekday
    df["DayOfWeek"] = (weekday + 1) % 7
    df["IsWeekend"] = df["DayOfWeek"].isin([0, 6]).astype(int)
    df["IsBusinessDay"] = (df["IsWeekend"] == 0).astype(int)

    iso = df["Date"].dt.isocalendar()
    df["WeekOfYearISO"] = iso.week.astype(int)
    df["ISOYear"] = iso.year.astype(int)

    df["QuarterStartDate"] = pd.to_datetime(
        df["Year"].astype(str)
        + "-"
        + ((df["Quarter"] - 1) * 3 + 1).astype(str).str.zfill(2)
        + "-01"
    )
    df["QuarterEndDate"] = df["QuarterStartDate"] + pd.offsets.QuarterEnd(0)

    df["IsMonthStart"] = (df["Day"] == 1).astype(int)
    df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(int)
    df["IsQuarterStart"] = df["Date"].dt.is_quarter_start.astype(int)
    df["IsQuarterEnd"] = df["Date"].dt.is_quarter_end.astype(int)
    df["IsYearStart"] = ((df["Month"] == 1) & (df["Day"] == 1)).astype(int)
    df["IsYearEnd"] = ((df["Month"] == 12) & (df["Day"] == 31)).astype(int)

    df["QuarterYear"] = "Q" + df["Quarter"].astype(str) + " " + df["Year"].astype(str)
    df["WeekOfMonth"] = ((df["Day"] - 1) // 7 + 1).astype(int)

    df["WeekStartDate"] = df["Date"] - pd.to_timedelta(df["Date"].dt.weekday, unit="D")
    df["WeekEndDate"] = df["WeekStartDate"] + pd.Timedelta(days=6)

    biz = df.loc[df["IsBusinessDay"] == 1, ["Date"]].rename(columns={"Date": "BizDate"})

    df = pd.merge_asof(
        df.sort_values("Date"),
        biz.assign(NextBD=biz["BizDate"]),
        left_on="Date",
        right_on="BizDate",
        direction="forward"
    ).drop(columns=["BizDate"])
    df.rename(columns={"NextBD": "NextBusinessDay"}, inplace=True)

    df = pd.merge_asof(
        df,
        biz.sort_values("BizDate").assign(PrevBD=biz["BizDate"]),
        left_on="Date",
        right_on="BizDate",
        direction="backward"
    ).drop(columns=["BizDate"])
    df.rename(columns={"PrevBD": "PreviousBusinessDay"}, inplace=True)

    df["FiscalYearStartYear"] = np.where(
        df["Month"] >= fy_start_month, df["Year"], df["Year"] - 1
    )

    df["FiscalMonthNumber"] = ((df["Month"] - fy_start_month + 12) % 12) + 1
    df["FiscalQuarterNumber"] = ((df["FiscalMonthNumber"] - 1) // 3 + 1)

    df["FiscalYearBin"] = (
        df["FiscalYearStartYear"].astype(str)
        + "-"
        + (df["FiscalYearStartYear"] + 1).astype(str)
    )

    df["FiscalQuarterName"] = (
        "Q" + df["FiscalQuarterNumber"].astype(str)
        + " FY" + (df["FiscalYearStartYear"] + 1).astype(str)
    )

    df["FiscalYearMonthNumber"] = df["FiscalYearStartYear"] * 12 + df["FiscalMonthNumber"]
    df["FiscalYearQuarterNumber"] = df["FiscalYearStartYear"] * 4 + df["FiscalQuarterNumber"]

    df["FiscalMonthIndex"] = df["FiscalYearStartYear"] * 12 + df["FiscalMonthNumber"]
    df["FiscalQuarterIndex"] = df["FiscalYearStartYear"] * 4 + df["FiscalQuarterNumber"]

    df["FiscalYearStartDate"] = pd.to_datetime(
        df["FiscalYearStartYear"].astype(str)
        + "-"
        + str(fy_start_month).zfill(2)
        + "-01"
    )
    df["FiscalYearEndDate"] = df["FiscalYearStartDate"] + pd.DateOffset(years=1) - pd.Timedelta(days=1)

    fq_shift = (df["FiscalQuarterNumber"] - 1) * 3
    fq_year = df["FiscalYearStartDate"].dt.year + ((df["FiscalYearStartDate"].dt.month + fq_shift - 1) // 12)
    fq_month = (df["FiscalYearStartDate"].dt.month + fq_shift - 1) % 12 + 1

    df["FiscalQuarterStartDate"] = pd.to_datetime(
        fq_year.astype(str) + "-" + fq_month.astype(str).str.zfill(2) + "-01"
    )
    df["FiscalQuarterEndDate"] = df["FiscalQuarterStartDate"] + pd.DateOffset(months=3) - pd.Timedelta(days=1)

    df["IsFiscalYearStart"] = (df["Date"] == df["FiscalYearStartDate"]).astype(int)
    df["IsFiscalYearEnd"] = (df["Date"] == df["FiscalYearEndDate"]).astype(int)
    df["IsFiscalQuarterStart"] = (df["Date"] == df["FiscalQuarterStartDate"]).astype(int)
    df["IsFiscalQuarterEnd"] = (df["Date"] == df["FiscalQuarterEndDate"]).astype(int)

    df["FiscalYear"] = np.where(
        df["Month"] < fy_start_month, df["Year"], df["Year"] + 1
    ).astype(int)

    df["FiscalYearLabel"] = "FY " + df["FiscalYear"].astype(str)

    today = pd.Timestamp.today().normalize()
    df["IsToday"] = (df["Date"] == today).astype(int)
    df["IsCurrentYear"] = (df["Year"] == today.year).astype(int)
    df["IsCurrentMonth"] = ((df["Year"] == today.year) & (df["Month"] == today.month)).astype(int)

    current_quarter = (today.month - 1) // 3 + 1
    df["IsCurrentQuarter"] = ((df["Year"] == today.year) & (df["Quarter"] == current_quarter)).astype(int)
    df["CurrentDayOffset"] = (df["Date"] - today).dt.days
    
    # date_cols = df.select_dtypes(include="datetime").columns
    # df[date_cols] = df[date_cols].apply(lambda col: pd.to_datetime(col).dt.date)

    df = df[CALENDAR_COLUMNS + ISO_COLUMNS + FISCAL_COLUMNS]
    return df


# ---------------------------------------------------------
# PIPELINE WRAPPER
# ---------------------------------------------------------

def run_dates(cfg, parquet_folder: Path):

    out_path = parquet_folder / "dates.parquet"
    dates_cfg = cfg["dates"]
    defaults_dates = cfg.get("defaults", {}).get("dates") or cfg.get("_defaults", {}).get("dates")

    version_cfg = {**dates_cfg, "global_dates": defaults_dates}

    if not should_regenerate("dates", version_cfg, out_path):
        skip("Dates up-to-date; skipping.")
        return

    override = dates_cfg.get("override", {}).get("dates", {})
    start_date = override.get("start") or defaults_dates.get("start")
    end_date = override.get("end") or defaults_dates.get("end")

    fiscal_start_month = dates_cfg.get("fiscal_month_offset", 5)

    with stage("Generating Dates"):
        df = generate_date_table(start_date, end_date, fiscal_start_month)
        df = df[resolve_date_columns(dates_cfg)]
        df.to_parquet(out_path, index=False)

    save_version("dates", version_cfg, out_path)
    info(f"Dates dimension written → {out_path}")
