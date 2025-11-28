import pandas as pd
import numpy as np
import re
from datetime import date


def generate_date_table(start_date, end_date, first_fy_month):
    """
    Full calendar + fiscal + business-day + offset date table generator.
    Output columns remain identical to original.
    """

    # ================================
    # CONFIG
    # ================================
    start_date = start_date or "2020-01-01"
    end_date   = end_date or "2026-12-31"
    fiscal_year_start_month = first_fy_month or 5

    # ================================
    # BASE DATE RANGE
    # ================================
    df = pd.DataFrame({"Date": pd.date_range(start_date, end_date, freq="D")})

    # ================================
    # BASE FIELDS
    # ================================
    df["DateKey"] = df["Date"].dt.strftime("%Y%m%d").astype(int)
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Quarter"] = df["Date"].dt.quarter
    df["MonthName"] = df["Date"].dt.strftime("%B")
    df["MonthShort"] = df["Date"].dt.strftime("%b")

    # ================================
    # CALENDAR QUARTER START / END
    # ================================
    df["Quarter Start Date"] = pd.to_datetime(
        df["Year"].astype(str)
        + "-"
        + ((df["Quarter"] - 1) * 3 + 1).astype(str).str.zfill(2)
        + "-01"
    )
    df["Quarter End Date"] = df["Quarter Start Date"] + pd.offsets.QuarterEnd(0)

    # ================================
    # MONTH BOUNDARIES
    # ================================
    df["MonthStartDate"] = df["Date"].values.astype("datetime64[M]")
    df["MonthEndDate"] = df["MonthStartDate"] + pd.offsets.MonthEnd(1)

    df["DayName"] = df["Date"].dt.strftime("%A")
    df["DayShort"] = df["Date"].dt.strftime("%a")
    df["DayOfYear"] = df["Date"].dt.dayofyear

    # ================================
    # DAY OF WEEK + WEEKEND
    # ================================
    df["DayOfWeek"] = (df["Date"].dt.weekday + 1) % 7
    df["IsWeekend"] = df["DayOfWeek"].isin([0, 6]).astype(int)
    df["IsBusinessDay"] = (df["IsWeekend"] == 0).astype(int)

    # ================================
    # ISO WEEK/YEAR
    # ================================
    iso = df["Date"].dt.isocalendar()
    df["WeekOfYearISO"] = iso.week.astype(int)
    df["ISOYear"] = iso.year.astype(int)

    # ================================
    # CALENDAR SEQUENCES
    # ================================
    df["YearMonthNumber"] = df["Year"] * 12 + df["Month"]
    df["YearQuarterNumber"] = df["Year"] * 4 + df["Quarter"]

    df["IsMonthStart"] = (df["Day"] == 1).astype(int)
    df["IsMonthEnd"] = df["Date"].dt.is_month_end.astype(int)
    df["IsQuarterStart"] = df["Date"].dt.is_quarter_start.astype(int)
    df["IsQuarterEnd"] = df["Date"].dt.is_quarter_end.astype(int)
    df["IsYearStart"] = ((df["Month"] == 1) & (df["Day"] == 1)).astype(int)
    df["IsYearEnd"] = ((df["Month"] == 12) & (df["Day"] == 31)).astype(int)

    df["MonthYear"] = df["Date"].dt.strftime("%b %Y")
    df["MonthYearNumber"] = df["Year"] * 100 + df["Month"]
    df["QuarterYear"] = "Q" + df["Quarter"].astype(str) + " " + df["Year"].astype(str)

    # ================================
    # WEEK OF MONTH + WEEK START/END
    # ================================
    df["WeekOfMonth"] = ((df["Day"] - 1) // 7 + 1).astype(int)
    df["WeekStartDate"] = df["Date"] - pd.to_timedelta(df["Date"].dt.weekday, unit="D")
    df["WeekEndDate"] = df["WeekStartDate"] + pd.Timedelta(days=6)

    # ================================
    # BUSINESS DAY — NEXT / PREVIOUS
    # ================================
    biz = df.loc[df["IsBusinessDay"] == 1, ["Date"]].rename(columns={"Date": "BizDate"})

    # Next business day
    df = pd.merge_asof(
        df.sort_values("Date"),
        biz.assign(NextBD=biz["BizDate"]),
        left_on="Date",
        right_on="BizDate",
        direction="forward"
    ).drop(columns=["BizDate"])
    df.rename(columns={"NextBD": "NextBusinessDay"}, inplace=True)

    # Previous business day
    biz_sorted = biz.sort_values("BizDate")
    df = pd.merge_asof(
        df,
        biz_sorted.assign(PrevBD=biz_sorted["BizDate"]),
        left_on="Date",
        right_on="BizDate",
        direction="backward"
    ).drop(columns=["BizDate"])
    df.rename(columns={"PrevBD": "PreviousBusinessDay"}, inplace=True)

    # ================================
    # FISCAL YEAR
    # ================================
    df["FiscalYearStartYear"] = np.where(
        df["Month"] >= fiscal_year_start_month,
        df["Year"],
        df["Year"] - 1
    )

    df["FiscalMonthNumber"] = ((df["Month"] - fiscal_year_start_month + 12) % 12) + 1
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

    df["FiscalYearStartDate"] = pd.to_datetime(
        df["FiscalYearStartYear"].astype(str)
        + "-"
        + str(fiscal_year_start_month).zfill(2)
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
        df["Month"] < fiscal_year_start_month,
        df["Year"],
        df["Year"] + 1
    ).astype(int)

    df["FiscalYearLabel"] = "FY " + df["FiscalYear"].astype(str)

    # ================================
    # CURRENT PERIOD FLAGS
    # ================================
    today = pd.Timestamp.today().normalize()

    df["IsToday"] = (df["Date"] == today).astype(int)
    df["IsCurrentYear"] = (df["Year"] == today.year).astype(int)
    df["IsCurrentMonth"] = ((df["Year"] == today.year) & (df["Month"] == today.month)).astype(int)

    current_quarter = (today.month - 1) // 3 + 1
    df["IsCurrentQuarter"] = ((df["Year"] == today.year) &
                              (df["Quarter"] == current_quarter)).astype(int)

    # Current Day Offset
    df["CurrentDayOffset"] = (df["Date"] - today).dt.days

    # ================================
    # PRETTY COLUMN NAMES
    # ================================
    def camel_to_title(name):
        # Convert PascalCase / camelCase → Title Case
        name = re.sub(r"([A-Z]+)$", r" \1", name)
        name = re.sub(r"(?<!^)(?<![A-Z])(?=[A-Z])", " ", name)
        parts = [(p if p.isupper() else p.title()) for p in name.split()]
        return " ".join(parts)

    df.columns = [camel_to_title(col) for col in df.columns]
    df.rename(columns={"Isoyear": "ISO Year"}, inplace=True)

    # Convert datetime columns → date objects
    date_cols = df.select_dtypes(include="datetime").columns
    df[date_cols] = df[date_cols].apply(
        lambda col: pd.to_datetime(col, errors="coerce").dt.date
    )

    # ================================
    # COLUMN ORDER (unchanged)
    # ================================
    df = df[[ 
        "Date","Date Key",

        "Year","Is Year Start","Is Year End","Year Month Number","Year Quarter Number",

        "Quarter","Quarter Year","Quarter Start Date","Quarter End Date",
        "Is Quarter Start","Is Quarter End",

        "Month","Month Name","Month Short","Month Start Date","Month End Date",
        "Month Year","Month Year Number","Is Month Start","Is Month End",

        "Week Of Year ISO","ISO Year","Week Of Month","Week Start Date","Week End Date",

        "Day","Day Name","Day Short","Day Of Year","Day Of Week",
        "Is Weekend","Is Business Day","Next Business Day","Previous Business Day",

        "Fiscal Year Start Year","Fiscal Month Number","Fiscal Quarter Number",
        "Fiscal Quarter Name","Fiscal Year Bin","Fiscal Year Month Number",
        "Fiscal Year Quarter Number",

        "Fiscal Year Start Date","Fiscal Year End Date",
        "Fiscal Quarter Start Date","Fiscal Quarter End Date",

        "Is Fiscal Year Start","Is Fiscal Year End",
        "Is Fiscal Quarter Start","Is Fiscal Quarter End",

        "Fiscal Year","Fiscal Year Label",

        "Is Today","Is Current Year","Is Current Month","Is Current Quarter",
        "Current Day Offset"
    ]]

    return df
