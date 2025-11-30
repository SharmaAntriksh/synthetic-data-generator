import pandas as pd
import numpy as np
import re
from datetime import datetime


def generate_date_table(start_date, end_date, first_fy_month):
    """
    Full calendar + fiscal + business-day enriched date dimension.
    Output schema identical to original version.
    """

    # ============================================================
    # CONFIG
    # ============================================================
    start_date = start_date or "2020-01-01"
    end_date   = end_date or "2026-12-31"
    fy_start_month = first_fy_month or 5

    # ============================================================
    # BASE RANGE
    # ============================================================
    df = pd.DataFrame({"Date": pd.date_range(start_date, end_date, freq="D")})

    # ============================================================
    # BASIC FIELDS
    # ============================================================
    df["Date Key"] = df["Date"].dt.strftime("%Y%m%d").astype(int)

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["Quarter"] = df["Date"].dt.quarter

    df["Month Name"] = df["Date"].dt.strftime("%B")
    df["Month Short"] = df["Date"].dt.strftime("%b")

    df["Month Start Date"] = df["Date"].values.astype("datetime64[M]")
    df["Month End Date"] = df["Month Start Date"] + pd.offsets.MonthEnd(1)

    df["Day Name"] = df["Date"].dt.strftime("%A")
    df["Day Short"] = df["Date"].dt.strftime("%a")
    df["Day Of Year"] = df["Date"].dt.dayofyear

    # ============================================================
    # WEEKDAY / BUSINESS-DAY LOGIC
    # ============================================================
    weekday = df["Date"].dt.weekday
    df["Day Of Week"] = (weekday + 1) % 7
    df["Is Weekend"] = df["Day Of Week"].isin([0, 6]).astype(int)
    df["Is Business Day"] = (df["Is Weekend"] == 0).astype(int)

    # ISO week/year
    iso = df["Date"].dt.isocalendar()
    df["Week Of Year ISO"] = iso.week.astype(int)
    df["ISO Year"] = iso.year.astype(int)

    # Calendar boundaries
    df["Quarter Start Date"] = pd.to_datetime(
        df["Year"].astype(str)
        + "-"
        + ((df["Quarter"] - 1) * 3 + 1).astype(str).str.zfill(2)
        + "-01"
    )
    df["Quarter End Date"] = df["Quarter Start Date"] + pd.offsets.QuarterEnd(0)

    df["Is Month Start"] = (df["Day"] == 1).astype(int)
    df["Is Month End"] = df["Date"].dt.is_month_end.astype(int)
    df["Is Quarter Start"] = df["Date"].dt.is_quarter_start.astype(int)
    df["Is Quarter End"] = df["Date"].dt.is_quarter_end.astype(int)
    df["Is Year Start"] = ((df["Month"] == 1) & (df["Day"] == 1)).astype(int)
    df["Is Year End"] = ((df["Month"] == 12) & (df["Day"] == 31)).astype(int)

    df["Month Year"] = df["Date"].dt.strftime("%b %Y")
    df["Month Year Number"] = df["Year"] * 100 + df["Month"]
    df["Quarter Year"] = "Q" + df["Quarter"].astype(str) + " " + df["Year"].astype(str)

    # Week of month / week boundaries
    df["Week Of Month"] = ((df["Day"] - 1) // 7 + 1).astype(int)
    df["Week Start Date"] = df["Date"] - pd.to_timedelta(df["Date"].dt.weekday, unit="D")
    df["Week End Date"] = df["Week Start Date"] + pd.Timedelta(days=6)

    # ============================================================
    # BUSINESS-DAY NEXT / PREVIOUS
    # ============================================================
    biz = df.loc[df["Is Business Day"] == 1, ["Date"]].rename(columns={"Date": "BizDate"})

    # Next business day
    df = pd.merge_asof(
        df.sort_values("Date"),
        biz.assign(NextBD=biz["BizDate"]),
        left_on="Date",
        right_on="BizDate",
        direction="forward"
    ).drop(columns=["BizDate"])
    df.rename(columns={"NextBD": "Next Business Day"}, inplace=True)

    # Previous business day
    biz_sorted = biz.sort_values("BizDate")
    df = pd.merge_asof(
        df,
        biz_sorted.assign(PrevBD=biz_sorted["BizDate"]),
        left_on="Date",
        right_on="BizDate",
        direction="backward"
    ).drop(columns=["BizDate"])
    df.rename(columns={"PrevBD": "Previous Business Day"}, inplace=True)

    # ============================================================
    # FISCAL CALCULATIONS
    # ============================================================
    df["Fiscal Year Start Year"] = np.where(
        df["Month"] >= fy_start_month,
        df["Year"],
        df["Year"] - 1
    )

    df["Fiscal Month Number"] = ((df["Month"] - fy_start_month + 12) % 12) + 1
    df["Fiscal Quarter Number"] = ((df["Fiscal Month Number"] - 1) // 3 + 1)

    df["Fiscal Year Bin"] = (
        df["Fiscal Year Start Year"].astype(str)
        + "-"
        + (df["Fiscal Year Start Year"] + 1).astype(str)
    )

    df["Fiscal Quarter Name"] = (
        "Q" + df["Fiscal Quarter Number"].astype(str)
        + " FY" + (df["Fiscal Year Start Year"] + 1).astype(str)
    )

    df["Fiscal Year Month Number"] = df["Fiscal Year Start Year"] * 12 + df["Fiscal Month Number"]
    df["Fiscal Year Quarter Number"] = df["Fiscal Year Start Year"] * 4 + df["Fiscal Quarter Number"]

    df["Fiscal Year Start Date"] = pd.to_datetime(
        df["Fiscal Year Start Year"].astype(str)
        + "-"
        + str(fy_start_month).zfill(2)
        + "-01"
    )
    df["Fiscal Year End Date"] = df["Fiscal Year Start Date"] + pd.DateOffset(years=1) - pd.Timedelta(days=1)

    fq_shift = (df["Fiscal Quarter Number"] - 1) * 3
    fq_year = df["Fiscal Year Start Date"].dt.year + ((df["Fiscal Year Start Date"].dt.month + fq_shift - 1) // 12)
    fq_month = (df["Fiscal Year Start Date"].dt.month + fq_shift - 1) % 12 + 1

    df["Fiscal Quarter Start Date"] = pd.to_datetime(
        fq_year.astype(str) + "-" + fq_month.astype(str).str.zfill(2) + "-01"
    )
    df["Fiscal Quarter End Date"] = (
        df["Fiscal Quarter Start Date"] + pd.DateOffset(months=3) - pd.Timedelta(days=1)
    )

    df["Is Fiscal Year Start"] = (df["Date"] == df["Fiscal Year Start Date"]).astype(int)
    df["Is Fiscal Year End"] = (df["Date"] == df["Fiscal Year End Date"]).astype(int)
    df["Is Fiscal Quarter Start"] = (df["Date"] == df["Fiscal Quarter Start Date"]).astype(int)
    df["Is Fiscal Quarter End"] = (df["Date"] == df["Fiscal Quarter End Date"]).astype(int)

    df["Fiscal Year"] = np.where(
        df["Month"] < fy_start_month,
        df["Year"],
        df["Year"] + 1
    ).astype(int)

    df["Fiscal Year Label"] = "FY " + df["Fiscal Year"].astype(str)

    # ============================================================
    # CURRENT PERIOD FLAGS
    # ============================================================
    today = pd.Timestamp.today().normalize()

    df["Is Today"] = (df["Date"] == today).astype(int)
    df["Is Current Year"] = (df["Year"] == today.year).astype(int)
    df["Is Current Month"] = ((df["Year"] == today.year) & (df["Month"] == today.month)).astype(int)

    current_quarter = (today.month - 1) // 3 + 1
    df["Is Current Quarter"] = ((df["Year"] == today.year) &
                                (df["Quarter"] == current_quarter)).astype(int)

    df["Current Day Offset"] = (df["Date"] - today).dt.days

    # ============================================================
    # COLUMN NAMES → Title Case
    # ============================================================
    def camel_to_title(x):
        x = re.sub(r"([A-Z]+)$", r" \1", x)
        x = re.sub(r"(?<!^)(?<![A-Z])(?=[A-Z])", " ", x)
        parts = [p if p.isupper() else p.title() for p in x.split()]
        return " ".join(parts)

    df.columns = [camel_to_title(col) for col in df.columns]
    df.rename(columns={"Iso Year": "ISO Year"}, inplace=True)

    # convert datetimes → date
    date_cols = df.select_dtypes(include="datetime").columns
    df[date_cols] = df[date_cols].apply(lambda col: pd.to_datetime(col).dt.date)

    # ============================================================
    # COLUMN ORDER (unchanged from your version)
    # ============================================================
    df = df[
        [
            "Date","Date Key",

            "Year","Is Year Start","Is Year End",

            "Quarter","Quarter Start Date","Quarter End Date",
            "Is Quarter Start","Is Quarter End",
            "Quarter Year",

            "Month","Month Name","Month Short",
            "Month Start Date","Month End Date",
            "Month Year","Month Year Number",
            "Is Month Start","Is Month End",

            "Week Of Year ISO","ISO Year","Week Of Month",
            "Week Start Date","Week End Date",

            "Day","Day Name","Day Short","Day Of Year","Day Of Week",
            "Is Weekend","Is Business Day",
            "Next Business Day","Previous Business Day",

            "Fiscal Year Start Year","Fiscal Month Number","Fiscal Quarter Number",
            "Fiscal Quarter Name","Fiscal Year Bin",
            "Fiscal Year Month Number","Fiscal Year Quarter Number",

            "Fiscal Year Start Date","Fiscal Year End Date",
            "Fiscal Quarter Start Date","Fiscal Quarter End Date",

            "Is Fiscal Year Start","Is Fiscal Year End",
            "Is Fiscal Quarter Start","Is Fiscal Quarter End",

            "Fiscal Year","Fiscal Year Label",

            "Is Today","Is Current Year","Is Current Month",
            "Is Current Quarter","Current Day Offset"
        ]
    ]

    return df
