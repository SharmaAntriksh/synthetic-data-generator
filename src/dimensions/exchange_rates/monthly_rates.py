"""Monthly exchange rates aggregation.

Produces ``exchange_rates_monthly.parquet`` with columns:
  Date (first of month), FromCurrencyKey, ToCurrencyKey,
  FromCurrency, ToCurrency,
  AvgRate, MinRate, MaxRate, EndOfMonthRate
"""
from __future__ import annotations

import pandas as pd


def build_monthly_rates(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily exchange rates to monthly grain.

    Parameters
    ----------
    daily_df : DataFrame
        Daily exchange rates with columns including Date, FromCurrencyKey,
        ToCurrencyKey, FromCurrency, ToCurrency, Rate.

    Returns
    -------
    DataFrame
        Monthly aggregation with AvgRate, MinRate, MaxRate, EndOfMonthRate.
    """
    df = daily_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["MonthStart"] = df["Date"].dt.to_period("M").dt.to_timestamp()

    group_cols = [
        "MonthStart", "FromCurrencyKey", "ToCurrencyKey",
        "FromCurrency", "ToCurrency",
    ]

    # Single groupby: aggregate stats + end-of-month rate in one pass
    agg = (
        df.sort_values("Date")
        .groupby(group_cols)["Rate"]
        .agg(AvgRate="mean", MinRate="min", MaxRate="max", EndOfMonthRate="last")
        .reset_index()
    )

    agg = agg.rename(columns={"MonthStart": "Date"})
    agg = agg.sort_values(["Date", "FromCurrency", "ToCurrency"]).reset_index(drop=True)

    out_cols = [
        "Date", "FromCurrencyKey", "ToCurrencyKey",
        "FromCurrency", "ToCurrency",
        "AvgRate", "MinRate", "MaxRate", "EndOfMonthRate",
    ]
    return agg[out_cols]
