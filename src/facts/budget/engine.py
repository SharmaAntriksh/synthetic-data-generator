"""Budget engine: pandas translation of the SQL budget views.

Stages:
  1. Yearly budget (vw_Budget)          -> _compute_yearly_budget()
  2. Monthly budget (category × month)  -> _compute_monthly_budget()

Input:  actuals DataFrame from BudgetAccumulator.finalize_sales()
Output: (yearly, monthly) budget DataFrames
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

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

    out = result[["Country", "Category", "BudgetYear", "Scenario",
                    "BudgetGrowthPct", "BudgetSalesAmount", "BudgetSalesQuantity",
                    "_method"]].rename(columns={"_method": "BudgetMethod"})
    out["BudgetYear"] = out["BudgetYear"].astype(int)
    return out



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



# ================================================================
# Stage 2: Monthly budget (category × month × scenario)
# ================================================================

def _compute_monthly_budget(
    yearly: pd.DataFrame,
    actuals_monthly: pd.DataFrame,
    bcfg: BudgetConfig,
) -> pd.DataFrame:
    """
    Spread yearly budget to monthly grain using seasonal weights from actuals.

    Input:
      - yearly: BudgetYearly at (Country, Category, BudgetYear, Scenario) grain
      - actuals_monthly: raw monthly actuals with (Category, Year, Month, SalesAmount, SalesQuantity)

    Output:
      Country, Category, BudgetYear, BudgetMonthStart, Scenario,
      BudgetAmount, BudgetQuantity, BudgetMethod
    """
    # ---- 1. Budget at Country × Category level (keep countries separate) ----
    cat_yearly = yearly.groupby(
        ["Country", "Category", "BudgetYear", "Scenario"], as_index=False
    ).agg(
        BudgetSalesAmount=("BudgetSalesAmount", "sum"),
        BudgetSalesQuantity=("BudgetSalesQuantity", "sum"),
    )

    # ---- 2. Compute seasonal weights from actuals ----
    # Category × Month share, averaged across all available years
    cat_month = actuals_monthly.groupby(
        ["Category", "Year", "Month"], as_index=False
    ).agg(SalesAmount=("SalesAmount", "sum"))

    cat_year_total = cat_month.groupby(
        ["Category", "Year"], as_index=False
    ).agg(YearTotal=("SalesAmount", "sum"))

    cat_month = cat_month.merge(cat_year_total, on=["Category", "Year"], how="left")
    cat_month["MonthShare"] = np.where(
        cat_month["YearTotal"] > 0,
        cat_month["SalesAmount"] / cat_month["YearTotal"],
        1.0 / 12.0,
    )

    # Average month share across years → stable seasonal profile per category
    seasonal = cat_month.groupby(
        ["Category", "Month"], as_index=False
    ).agg(MonthShare=("MonthShare", "mean"))

    # Normalize so shares sum to 1.0 per category (safety)
    share_totals = seasonal.groupby("Category", as_index=False).agg(
        ShareSum=("MonthShare", "sum")
    )
    seasonal = seasonal.merge(share_totals, on="Category", how="left")
    seasonal["MonthShare"] = seasonal["MonthShare"] / seasonal["ShareSum"]
    seasonal.drop(columns=["ShareSum"], inplace=True)

    # ---- 3. Cross join budget × 12 months, then apply weights ----
    months = pd.DataFrame({"Month": range(1, 13)})
    budget_months = cat_yearly.merge(months, how="cross")

    budget_months = budget_months.merge(
        seasonal, on=["Category", "Month"], how="left"
    )
    # Fallback for categories with no seasonal data
    budget_months["MonthShare"] = budget_months["MonthShare"].fillna(1.0 / 12.0)

    budget_months["BudgetAmount"] = (
        budget_months["BudgetSalesAmount"] * budget_months["MonthShare"]
    )
    budget_months["BudgetQuantity"] = (
        budget_months["BudgetSalesQuantity"] * budget_months["MonthShare"]
    )

    # ---- 4. Build BudgetMonthStart date ----
    budget_months["BudgetMonthStart"] = pd.to_datetime(
        budget_months["BudgetYear"].astype(str) + "-"
        + budget_months["Month"].astype(str).str.zfill(2) + "-01"
    )

    budget_months["BudgetMethod"] = "Monthly: yearly total x seasonal share"

    out = budget_months[[
        "Country", "Category", "BudgetYear", "BudgetMonthStart", "Scenario",
        "BudgetAmount", "BudgetQuantity", "BudgetMethod",
    ]].copy()

    out["BudgetYear"] = out["BudgetYear"].astype(int)
    return out


# ================================================================
# Public API
# ================================================================

def compute_budget(
    actuals_monthly: pd.DataFrame,
    bcfg: BudgetConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute all budget tables.

    Returns:
        (yearly_budget, monthly_budget)
    """
    actuals_annual = actuals_monthly.groupby(
        ["Country", "Category", "Year"], as_index=False
    ).agg(SalesAmount=("SalesAmount", "sum"), SalesQuantity=("SalesQuantity", "sum"))

    yearly = _compute_yearly_budget(actuals_annual, bcfg)
    monthly = _compute_monthly_budget(yearly, actuals_monthly, bcfg)

    return yearly, monthly
