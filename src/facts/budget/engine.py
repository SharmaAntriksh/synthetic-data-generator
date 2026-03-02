"""Budget engine: pandas translation of the SQL budget views.

Stages (matching the SQL):
  1. Yearly budget (vw_Budget)          -> _compute_yearly_budget()
  2. Channel + month allocation          -> _allocate_channel_month()
  3. FX conversion                       -> _apply_fx()

Input:  actuals DataFrame from BudgetAccumulator.finalize_sales()
Output: final budget DataFrame(s) written as parquet
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ================================================================
# Config
# ================================================================

@dataclass(frozen=True)
class BudgetConfig:
    """Budget generation parameters (from config.yaml budget: section)."""
    enabled: bool = False

    # Scenario adjustments
    scenarios: Dict[str, float] = field(default_factory=lambda: {
        "Low": -0.03, "Medium": 0.00, "High": 0.05,
    })

    # Growth blending weights
    weight_local: float = 0.60
    weight_category: float = 0.30
    weight_global: float = 0.10

    # Growth caps
    growth_cap_high: float = 0.30
    growth_cap_low: float = -0.20

    # Backcast default growth
    default_backcast_growth: float = 0.05

    # Default fallback growth (when all tiers are NULL)
    fallback_growth: float = 0.05

    # Channel digital/physical shift factors
    digital_shift: float = 1.02
    physical_shift: float = 0.98

    # Month/channel mix weights (LY vs LY-1)
    mix_current_weight: float = 0.70
    mix_prior_weight: float = 0.30

    # Return rate cap
    return_rate_cap: float = 0.30

    # FX
    report_currency: str = "USD"

    # Rolling window for category/global growth
    rolling_window: int = 3   # 3-year rolling average


def load_budget_config(cfg: Dict[str, Any]) -> BudgetConfig:
    """Extract budget config from the main config dict."""
    raw = cfg.get("budget", {}) or {}
    if not isinstance(raw, dict):
        return BudgetConfig()

    # Nested sub-dicts in config.yaml
    growth_caps = raw.get("growth_caps", {}) or {}
    weights = raw.get("weights", {}) or {}

    return BudgetConfig(
        enabled=bool(raw.get("enabled", False)),
        scenarios=raw.get("scenarios", {"Low": -0.03, "Medium": 0.00, "High": 0.05}),
        report_currency=raw.get("report_currency", "USD"),
        growth_cap_high=float(growth_caps.get("high", 0.30)),
        growth_cap_low=float(growth_caps.get("low", -0.20)),
        weight_local=float(weights.get("local", 0.60)),
        weight_category=float(weights.get("category", 0.30)),
        weight_global=float(weights.get("global", 0.10)),
        default_backcast_growth=float(raw.get("default_backcast_growth", 0.05)),
        return_rate_cap=float(raw.get("return_rate_cap", 0.30)),
    )

# ================================================================
# Deterministic jitter (replaces SQL CHECKSUM)
# ================================================================

def _jitter_pct(country: str, category: str, year: int) -> float:
    """
    Deterministic pseudo-random jitter in [-0.02, +0.02].
    Replaces SQL: (ABS(CHECKSUM(country, category, year)) % 401 - 200) / 10000.

    NOTE: Won't match SQL Server CHECKSUM byte-for-byte, but produces
    equivalent statistical distribution. Use a stable hash so results
    are reproducible across Python versions.
    """
    key = f"{country}|{category}|{year}".encode("utf-8")
    h = int(hashlib.md5(key).hexdigest(), 16)
    return ((h % 401) - 200) / 10000.0


def _jitter_series(df: pd.DataFrame) -> pd.Series:
    """Vectorized jitter over a DataFrame with Country, Category, BudgetYear."""
    return df.apply(
        lambda r: _jitter_pct(r["Country"], r["Category"], int(r["BudgetYear"])),
        axis=1,
    )


# ================================================================
# Stage 1: Yearly budget (vw_Budget equivalent)
# ================================================================

def _compute_yearly_budget(
    actuals_annual: pd.DataFrame,
    bcfg: BudgetConfig,
) -> pd.DataFrame:
    """
    Input: actuals_annual with (Country, Category, Year, SalesAmount, SalesQuantity)
           aggregated to annual grain (no channel/month).

    Returns: budget at (Country, Category, BudgetYear, Scenario) grain with
             BudgetGrowthPct, BudgetSalesAmount, BudgetSalesQuantity, BudgetMethod.
    """
    a = actuals_annual.copy()
    min_year = int(a["Year"].min())
    max_year = int(a["Year"].max())

    # ---- Local YoY ----
    a = a.sort_values(["Country", "Category", "Year"])
    a["PrevAmount"] = a.groupby(["Country", "Category"])["SalesAmount"].shift(1)
    a["LocalYoY"] = _capped_growth(a["SalesAmount"], a["PrevAmount"], bcfg)

    # ---- Category-Year rolling ----
    cat_year = a.groupby(["Category", "Year"], as_index=False).agg(
        CatAmount=("SalesAmount", "sum")
    )
    cat_year = cat_year.sort_values(["Category", "Year"])
    cat_year["PrevCat"] = cat_year.groupby("Category")["CatAmount"].shift(1)
    cat_year["CatYoY"] = _capped_growth(cat_year["CatAmount"], cat_year["PrevCat"], bcfg)
    cat_year["CatYoY_3yr"] = (
        cat_year.groupby("Category")["CatYoY"]
        .rolling(bcfg.rolling_window, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )

    # ---- Global-Year rolling ----
    glob_year = a.groupby("Year", as_index=False).agg(
        GlobalAmount=("SalesAmount", "sum")
    )
    glob_year = glob_year.sort_values("Year")
    glob_year["PrevGlobal"] = glob_year["GlobalAmount"].shift(1)
    glob_year["GlobalYoY"] = _capped_growth(
        glob_year["GlobalAmount"], glob_year["PrevGlobal"], bcfg
    )
    glob_year["GlobalYoY_3yr"] = (
        glob_year["GlobalYoY"]
        .rolling(bcfg.rolling_window, min_periods=1).mean()
    )

    # ---- Join growth tiers back ----
    a = a.merge(cat_year[["Category", "Year", "CatYoY_3yr"]],
                on=["Category", "Year"], how="left")
    a = a.merge(glob_year[["Year", "GlobalYoY_3yr"]],
                on="Year", how="left")

    # ---- Blended growth ----
    a["BaseGrowth"] = _blend_growth(
        a["LocalYoY"], a["CatYoY_3yr"], a["GlobalYoY_3yr"], bcfg
    )

    # ---- Standard budgets: BudgetYear = ActualYear + 1 ----
    standard = a[a["Year"] + 1 <= max_year].copy()
    standard["BudgetYear"] = standard["Year"] + 1

    # ---- Backfill budgets for first 2 years ----
    first_year = a[a["Year"] == min_year].copy()

    backfill_same = first_year.copy()
    backfill_same["BudgetYear"] = min_year
    backfill_same["BaseGrowth"] = 0.0
    backfill_same["_method"] = "Backfill: same-year baseline"

    backfill_prior = first_year.copy()
    backfill_prior["BudgetYear"] = min_year - 1
    backfill_prior["BaseGrowth"] = -bcfg.default_backcast_growth
    backfill_prior["SalesAmount"] = (
        backfill_prior["SalesAmount"] / (1.0 + bcfg.default_backcast_growth)
    )
    backfill_prior["SalesQuantity"] = (
        backfill_prior["SalesQuantity"] / (1.0 + bcfg.default_backcast_growth)
    )
    backfill_prior["_method"] = "Backfill: back-cast from first-year"

    standard["_method"] = "Standard: LY actual + blended growth + scenario + jitter"

    combined = pd.concat([standard, backfill_same, backfill_prior], ignore_index=True)

    # ---- Expand across scenarios ----
    rows = []
    for scenario_name, scenario_adj in bcfg.scenarios.items():
        s = combined.copy()
        s["Scenario"] = scenario_name
        s["ScenarioAdj"] = scenario_adj
        s["Jitter"] = _jitter_series(s)
        s["BudgetGrowthPct"] = s["BaseGrowth"] + s["Jitter"] + s["ScenarioAdj"]
        s["BudgetSalesAmount"] = s["SalesAmount"] * (1.0 + s["BudgetGrowthPct"])
        s["BudgetSalesQuantity"] = s["SalesQuantity"] * (1.0 + s["BudgetGrowthPct"])
        rows.append(s)

    result = pd.concat(rows, ignore_index=True)

    return result[["Country", "Category", "BudgetYear", "Scenario",
                    "BudgetGrowthPct", "BudgetSalesAmount", "BudgetSalesQuantity",
                    "_method"]].rename(columns={"_method": "BudgetMethod"})


# ================================================================
# Stage 2: Channel + Month allocation (vw_Budget_ChannelMonth)
# ================================================================

def _allocate_channel_month(
    yearly_budget: pd.DataFrame,
    actuals_monthly: pd.DataFrame,
    returns_annual: Optional[pd.DataFrame],
    bcfg: BudgetConfig,
) -> pd.DataFrame:
    """
    Distribute yearly budget across channels and months.

    actuals_monthly: (Country, Category, Year, Month, SalesChannelKey,
                      SalesAmount, SalesQuantity) — the raw monthly from accumulator

    Returns: (Country, Category, BudgetYear, BudgetMonthStart, Scenario,
              SalesChannelKey, BudgetGrossAmount, BudgetNetAmount,
              BudgetGrossQuantity, BudgetNetQuantity, BudgetGrowthPct,
              audit columns...)
    """
    am = actuals_monthly.copy()

    # ---- Channel mix from actuals (70/30 weighted with prior year) ----
    # Annual by (Country, Category, Year, Channel)
    chan_year = am.groupby(
        ["Country", "Category", "Year", "SalesChannelKey"], as_index=False
    ).agg(SalesAmount_Y=("SalesAmount", "sum"))

    # Self-join for prior year
    chan_prior = chan_year.rename(columns={
        "Year": "Year", "SalesAmount_Y": "PriorAmount"
    }).copy()
    chan_prior["Year"] = chan_prior["Year"] + 1

    chan_mix = chan_year.merge(
        chan_prior[["Country", "Category", "Year", "SalesChannelKey", "PriorAmount"]],
        on=["Country", "Category", "Year", "SalesChannelKey"],
        how="left",
    )
    chan_mix["PriorAmount"] = chan_mix["PriorAmount"].fillna(0)
    chan_mix["MixAmount"] = (
        bcfg.mix_current_weight * chan_mix["SalesAmount_Y"]
        + bcfg.mix_prior_weight * chan_mix["PriorAmount"]
    )

    # TODO: apply digital/physical shift factor from budget_channel_is_digital
    # chan_mix["MixAmount"] *= shift_factor

    # Normalize to shares within (Country, Category, Year)
    chan_mix["ChannelTotal"] = chan_mix.groupby(
        ["Country", "Category", "Year"]
    )["MixAmount"].transform("sum")
    chan_mix["ChannelShare"] = np.where(
        chan_mix["ChannelTotal"] > 0,
        chan_mix["MixAmount"] / chan_mix["ChannelTotal"],
        0,
    )

    # ---- Month mix from actuals (same 70/30 pattern) ----
    month_prior = am.copy()
    month_prior["Year"] = month_prior["Year"] + 1
    month_prior = month_prior.rename(columns={"SalesAmount": "PriorMonthAmt"})

    month_mix = am.merge(
        month_prior[["Country", "Category", "Year", "Month",
                      "SalesChannelKey", "PriorMonthAmt"]],
        on=["Country", "Category", "Year", "Month", "SalesChannelKey"],
        how="left",
    )
    month_mix["PriorMonthAmt"] = month_mix["PriorMonthAmt"].fillna(0)
    month_mix["MonthAmount"] = (
        bcfg.mix_current_weight * month_mix["SalesAmount"]
        + bcfg.mix_prior_weight * month_mix["PriorMonthAmt"]
    )

    month_mix["MonthTotal"] = month_mix.groupby(
        ["Country", "Category", "Year", "SalesChannelKey"]
    )["MonthAmount"].transform("sum")
    month_mix["MonthShare"] = np.where(
        month_mix["MonthTotal"] > 0,
        month_mix["MonthAmount"] / month_mix["MonthTotal"],
        1.0 / 12.0,
    )

    # ---- Return rate (smoothed, capped) ----
    return_rate = _compute_return_rates(
        actuals_annual=am.groupby(
            ["Country", "Category", "Year", "SalesChannelKey"], as_index=False
        ).agg(SalesAmount=("SalesAmount", "sum")),
        returns_annual=returns_annual,
        bcfg=bcfg,
    )

    # ---- Expand yearly budget -> channel -> month ----
    # Join yearly budget with channel mix (BudgetYear uses LY = BudgetYear - 1)
    yb = yearly_budget.copy()
    yb["MixYear"] = yb["BudgetYear"] - 1  # look up LY channel mix

    budget_chan = yb.merge(
        chan_mix[["Country", "Category", "Year", "SalesChannelKey", "ChannelShare"]],
        left_on=["Country", "Category", "MixYear"],
        right_on=["Country", "Category", "Year"],
        how="left",
        suffixes=("", "_cm"),
    )
    # Where no LY channel mix exists, use default equal shares
    # TODO: apply ChannelDefaultsNorm fallback logic

    budget_chan["BudgetChannelAmount"] = (
        budget_chan["BudgetSalesAmount"] * budget_chan["ChannelShare"]
    )
    budget_chan["BudgetChannelQty"] = (
        budget_chan["BudgetSalesQuantity"] * budget_chan["ChannelShare"]
    )

    # Expand across 12 months using MonthShare from LY
    months = pd.DataFrame({"Month": range(1, 13)})
    budget_month = budget_chan.merge(months, how="cross")
    budget_month["BudgetMonthStart"] = pd.to_datetime(
        budget_month["BudgetYear"].astype(str) + "-"
        + budget_month["Month"].astype(str).str.zfill(2) + "-01"
    )

    # Join month shares (LY)
    budget_month = budget_month.merge(
        month_mix[["Country", "Category", "Year", "Month",
                    "SalesChannelKey", "MonthShare"]],
        left_on=["Country", "Category", "MixYear", "Month", "SalesChannelKey"],
        right_on=["Country", "Category", "Year", "Month", "SalesChannelKey"],
        how="left",
        suffixes=("", "_mm"),
    )
    budget_month["MonthShare"] = budget_month["MonthShare"].fillna(1.0 / 12.0)

    budget_month["BudgetGrossAmount"] = (
        budget_month["BudgetChannelAmount"] * budget_month["MonthShare"]
    )
    budget_month["BudgetGrossQty"] = (
        budget_month["BudgetChannelQty"] * budget_month["MonthShare"]
    )

    # ---- Apply return rate adjustment ----
    budget_month = budget_month.merge(
        return_rate,
        left_on=["Country", "Category", "MixYear", "SalesChannelKey"],
        right_on=["Country", "Category", "Year", "SalesChannelKey"],
        how="left",
        suffixes=("", "_rr"),
    )
    rr = budget_month["ReturnRateCapped"].fillna(0)
    budget_month["BudgetNetAmount"] = budget_month["BudgetGrossAmount"] * (1 - rr)
    budget_month["BudgetNetQuantity"] = budget_month["BudgetGrossQty"] * (1 - rr)

    # ---- Select output columns ----
    out_cols = [
        "Country", "Category", "SalesChannelKey",
        "BudgetYear", "BudgetMonthStart", "Scenario",
        "BudgetGrowthPct",
        "ChannelShare", "MonthShare", "ReturnRateCapped",
        "BudgetGrossAmount", "BudgetNetAmount",
        "BudgetGrossQuantity", "BudgetNetQuantity",
        "BudgetMethod",
    ]
    # Rename audit columns
    result = budget_month.rename(columns={
        "ChannelShare": "Audit_ChannelShare",
        "MonthShare": "Audit_MonthShare",
        "ReturnRateCapped": "Audit_ReturnRate",
    })

    return result  # TODO: select final columns


# ================================================================
# Stage 3: FX conversion (vw_Budget_ChannelMonth_FX)
# ================================================================

def _apply_fx(
    budget_local: pd.DataFrame,
    exchange_rates: pd.DataFrame,
    country_to_currency: Dict[int, str],
    country_labels: np.ndarray,
    bcfg: BudgetConfig,
) -> pd.DataFrame:
    """
    Convert local-currency budget amounts to report currency.

    exchange_rates: the generated ExchangeRates dimension parquet
        columns: Date, FromCurrency, ToCurrency, Rate

    Returns budget with additional columns:
        LocalCurrency, ReportCurrency, FxRate_ToReport,
        BudgetGrossAmount_Local, BudgetNetAmount_Local,
        BudgetGrossAmount_Report, BudgetNetAmount_Report
    """
    # Average FX rates to monthly grain
    er = exchange_rates.copy()
    er["MonthStart"] = er["Date"].values.astype("datetime64[M]")
    fx_month = er.groupby(
        ["MonthStart", "FromCurrency", "ToCurrency"], as_index=False
    ).agg(AvgRate=("Rate", "mean"))

    # Map Country -> LocalCurrency on the budget
    budget = budget_local.copy()
    # TODO: map budget["Country"] -> local currency using country_to_currency

    budget["ReportCurrency"] = bcfg.report_currency

    # Resolve FX: direct, then inverse, then identity
    # TODO: merge-asof or left-join with carry-forward logic
    # matching the SQL OUTER APPLY ... TOP 1 ... ORDER BY MonthStart DESC pattern

    # For each budget row:
    #   if LocalCurrency == ReportCurrency -> FxRate = 1.0
    #   elif direct rate exists for (LocalCurrency -> ReportCurrency, MonthStart <= BudgetMonthStart) -> use it
    #   elif inverse rate exists -> 1.0 / rate
    #   else -> NULL

    # budget["BudgetGrossAmount_Report"] = budget["BudgetGrossAmount"] * budget["FxRate_ToReport"]
    # budget["BudgetNetAmount_Report"] = budget["BudgetNetAmount"] * budget["FxRate_ToReport"]

    return budget  # placeholder


# ================================================================
# Helpers
# ================================================================

def _capped_growth(
    current: pd.Series,
    previous: pd.Series,
    bcfg: BudgetConfig,
) -> pd.Series:
    """YoY growth, capped to [growth_cap_low, growth_cap_high]."""
    raw = (current - previous) / previous.replace(0, np.nan)
    return raw.clip(lower=bcfg.growth_cap_low, upper=bcfg.growth_cap_high)


def _blend_growth(
    local: pd.Series,
    cat_3yr: pd.Series,
    global_3yr: pd.Series,
    bcfg: BudgetConfig,
) -> pd.Series:
    """
    Dynamic weighted blend of 3 growth tiers.
    Weights are zeroed for NULL tiers, then re-normalized.
    Falls back to bcfg.fallback_growth if all tiers are NULL.
    """
    w_l = np.where(local.notna(), bcfg.weight_local, 0)
    w_c = np.where(cat_3yr.notna(), bcfg.weight_category, 0)
    w_g = np.where(global_3yr.notna(), bcfg.weight_global, 0)
    w_total = w_l + w_c + w_g

    blended = (
        local.fillna(0) * w_l
        + cat_3yr.fillna(0) * w_c
        + global_3yr.fillna(0) * w_g
    )

    return pd.Series(
        np.where(w_total > 0, blended / w_total, bcfg.fallback_growth),
        index=local.index,
    )


def _compute_return_rates(
    actuals_annual: pd.DataFrame,
    returns_annual: Optional[pd.DataFrame],
    bcfg: BudgetConfig,
) -> pd.DataFrame:
    """
    Compute smoothed, capped return rates by (Country, Category, Year, Channel).

    Returns DataFrame with ReturnRateCapped column.
    """
    if returns_annual is None or returns_annual.empty:
        # No returns data -> 0% return rate everywhere
        result = actuals_annual[["Country", "Category", "Year", "SalesChannelKey"]].copy()
        result["ReturnRateCapped"] = 0.0
        return result

    # Join returns to actuals
    merged = actuals_annual.merge(
        returns_annual,
        on=["Country", "Category", "Year", "SalesChannelKey"],
        how="left",
    )
    merged["ReturnAmount"] = merged["ReturnAmount"].fillna(0)
    merged["ReturnRate"] = np.where(
        merged["SalesAmount"] > 0,
        merged["ReturnAmount"] / merged["SalesAmount"],
        0,
    )

    # Smooth: 70% current year + 30% prior year
    merged = merged.sort_values(
        ["Country", "Category", "SalesChannelKey", "Year"]
    )
    merged["PriorRate"] = merged.groupby(
        ["Country", "Category", "SalesChannelKey"]
    )["ReturnRate"].shift(1)
    merged["SmoothedRate"] = (
        bcfg.mix_current_weight * merged["ReturnRate"]
        + bcfg.mix_prior_weight * merged["PriorRate"].fillna(merged["ReturnRate"])
    )

    # Cap to [0, return_rate_cap]
    merged["ReturnRateCapped"] = merged["SmoothedRate"].clip(0, bcfg.return_rate_cap)

    return merged[["Country", "Category", "Year", "SalesChannelKey", "ReturnRateCapped"]]


# ================================================================
# Public API
# ================================================================

def compute_budget(
    actuals_monthly: pd.DataFrame,
    returns_annual: Optional[pd.DataFrame],
    exchange_rates_path: Path,
    country_to_currency: Dict[int, str],
    country_labels: np.ndarray,
    bcfg: BudgetConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Full budget computation.

    Returns:
        (yearly_budget, channel_month_budget, channel_month_fx_budget)

    channel_month_fx_budget is None if exchange_rates unavailable.
    """
    # Annual actuals (for yearly budget)
    actuals_annual = actuals_monthly.groupby(
        ["Country", "Category", "Year"], as_index=False
    ).agg(SalesAmount=("SalesAmount", "sum"), SalesQuantity=("SalesQuantity", "sum"))

    # Stage 1
    yearly = _compute_yearly_budget(actuals_annual, bcfg)

    # Stage 2
    channel_month = _allocate_channel_month(
        yearly, actuals_monthly, returns_annual, bcfg
    )

    # Stage 3 (optional)
    fx_budget = None
    if exchange_rates_path.exists():
        er = pd.read_parquet(exchange_rates_path)
        fx_budget = _apply_fx(
            channel_month, er, country_to_currency, country_labels, bcfg
        )

    return yearly, channel_month, fx_budget
