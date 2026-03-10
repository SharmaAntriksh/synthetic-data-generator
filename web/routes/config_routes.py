"""
web/routes/config_routes.py -- All /api/config/* endpoints.
"""

from __future__ import annotations

from typing import Any, Dict

import yaml
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from web.shared_state import (
    _config_path,
    _base_config,
    _g,
    _promo_total,
    _set_promotions_total,
    normalize_config_yaml,
)
import web.shared_state as _state

router = APIRouter(prefix="/api/config", tags=["config"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ConfigUpdate(BaseModel):
    values: Dict[str, Any]


class ConfigYamlUpdate(BaseModel):
    yaml_text: str


# ---------------------------------------------------------------------------
# GET /api/config -- flat shape for the frontend
# ---------------------------------------------------------------------------

@router.get("")
def get_config():
    cfg = _state._cfg
    defaults = _g(cfg, "defaults", "dates", default={})
    sales = _g(cfg, "sales", default={})
    cust = _g(cfg, "customers", default={})
    stores = _g(cfg, "stores", default={})
    prods = _g(cfg, "products", default={})
    promos = _g(cfg, "promotions", default={})
    pricing = _g(cfg, "products", "pricing", "base", default={})
    returns = _g(cfg, "returns", default={})
    geo = _g(cfg, "geography", default={})
    dates_cfg = _g(cfg, "dates", default={})
    include = _g(dates_cfg, "include", default={})
    wf = _g(include, "weekly_fiscal", default={})
    if isinstance(wf, bool):
        wf = {"enabled": wf}

    # Extra sections
    cs = _g(cfg, "customer_segments", default={})
    sp = _g(cfg, "superpowers", default={})
    emp = _g(cfg, "employees", default={})
    er = _g(cfg, "exchange_rates", default={})
    budget = _g(cfg, "budget", default={})
    inv = _g(cfg, "inventory", default={})
    cost = _g(cfg, "products", "pricing", "cost", default={})
    brand_norm = _g(cfg, "products", "pricing", "brand_normalization", default={})
    cs_validity = _g(cs, "validity", default={})
    store_assigns = _g(emp, "store_assignments", default={})

    return {
        # Global
        "seed": int(_g(cfg, "defaults", "seed", default=42) or 42),
        # Output
        "format": str(sales.get("file_format", "parquet")),
        "salesOutput": str(sales.get("sales_output", "sales")),
        "skipOrderCols": bool(sales.get("skip_order_cols", False)),
        "compression": str(_g(sales, "compression", default="snappy")),
        "rowGroupSize": int(_g(sales, "row_group_size", default=2000000)),
        "mergeParquet": bool(sales.get("merge_parquet", True)),
        "partitionEnabled": bool(_g(sales, "partitioning", "enabled", default=True)),
        "maxLinesPerOrder": int(sales.get("max_lines_per_order", 5)),
        "salesOptimize": bool(sales.get("optimize", True)),
        # Dates
        "startDate": str(defaults.get("start", "2023-01-01")),
        "endDate": str(defaults.get("end", "2026-12-31")),
        "fiscalMonthOffset": int(dates_cfg.get("fiscal_month_offset", dates_cfg.get("fiscal_start_month", 0)) or 0),
        "asOfDate": str(dates_cfg.get("as_of_date", "") or ""),
        "includeCalendar": True,
        "includeIso": bool(include.get("iso", False)),
        "includeFiscal": bool(include.get("fiscal", True)),
        "includeWeeklyFiscal": bool(wf.get("enabled", False)),
        "wfFirstDay": int(wf.get("first_day_of_week", 0)),
        "wfWeeklyType": str(wf.get("weekly_type", "Last")),
        "wfQuarterType": str(wf.get("quarter_week_type", "445")),
        "wfTypeStartFiscalYear": int(wf.get("type_start_fiscal_year", 1)),
        # Volume
        "salesRows": int(sales.get("total_rows", 100000)),
        "chunkSize": int(sales.get("chunk_size", 200000)),
        "workers": int(sales.get("workers", 0)),
        # Dimensions
        "customers": int(cust.get("total_customers", 10000)),
        "stores": int(stores.get("num_stores", 100)),
        "products": int(prods.get("num_products", 5000)),
        "promotions": _promo_total(promos),
        # Customers detail
        "pctIndia": float(cust.get("pct_india", 10)),
        "pctUs": float(cust.get("pct_us", 51)),
        "pctEu": float(cust.get("pct_eu", 39)),
        "pctAsia": float(cust.get("pct_asia", 0)),
        "pctOrg": float(cust.get("pct_org", 10)),
        "customerActiveRatio": float(cust.get("active_ratio", 0.90)),
        "profile": str(cust.get("profile", "steady")),
        "firstYearPct": float(cust.get("first_year_pct", 0.27)),
        # Products detail
        "valueScale": float(pricing.get("value_scale", 1.0)),
        "minPrice": float(pricing.get("min_unit_price", 10)),
        "maxPrice": float(pricing.get("max_unit_price", 3000)),
        "productActiveRatio": float(prods.get("active_ratio", 0.94)),
        "marginMin": float(cost.get("min_margin_pct", 0.20)),
        "marginMax": float(cost.get("max_margin_pct", 0.35)),
        "brandNormalize": bool(brand_norm.get("enabled", False)),
        "brandNormalizeAlpha": float(brand_norm.get("alpha", 0.35)),
        # Geography
        "geoWeights": dict(geo.get("country_weights", {})),
        # Returns
        "returnsEnabled": bool(returns.get("enabled", True)),
        "returnRate": float(returns.get("return_rate", 0.03)),
        "returnMinDays": int(returns.get("min_days_after_sale", 1)),
        "returnMaxDays": int(returns.get("max_days_after_sale", 60)),
        # Promotions
        "promoNewCustWindow": int(_g(promos, "new_customer_window_months", default=3)),
        "promoSeasonal": int(promos.get("num_seasonal", 20)),
        "promoClearance": int(promos.get("num_clearance", 8)),
        "promoLimited": int(promos.get("num_limited", 12)),
        "promoFlash": int(promos.get("num_flash", 6)),
        "promoVolume": int(promos.get("num_volume", 4)),
        "promoLoyalty": int(promos.get("num_loyalty", 3)),
        "promoBundle": int(promos.get("num_bundle", 3)),
        "promoNewCustomer": int(promos.get("num_new_customer", 3)),
        # Customer Segments
        "csEnabled": bool(cs.get("enabled", False)),
        "csGenerateBridge": bool(cs.get("generate_bridge", False)),
        "csSegmentCount": int(cs.get("segment_count", 10)),
        "csPerCustomerMin": int(cs.get("segments_per_customer_min", 1)),
        "csPerCustomerMax": int(cs.get("segments_per_customer_max", 2)),
        "csIncludeScore": bool(cs.get("include_score", True)),
        "csIncludePrimaryFlag": bool(cs.get("include_primary_flag", True)),
        "csIncludeValidity": bool(cs.get("include_validity", True)),
        "csValidityGrain": str(cs_validity.get("grain", "month")),
        "csChurnRateQtr": float(cs_validity.get("churn_rate_qtr", 0.08)),
        "csNewCustomerMonths": int(cs_validity.get("new_customer_months", 2)),
        "csSeed": int(cs.get("seed", 123) or 123),
        # Superpowers
        "spEnabled": bool(sp.get("enabled", False)),
        "spGenerateBridge": bool(sp.get("generate_bridge", False)),
        "spPowersCount": int(sp.get("powers_count", 20)),
        "spPerCustomerMin": int(sp.get("powers_per_customer_min", 1)),
        "spPerCustomerMax": int(sp.get("powers_per_customer_max", 3)),
        "spIncludePowerLevel": bool(sp.get("include_power_level", True)),
        "spIncludePrimaryFlag": bool(sp.get("include_primary_flag", True)),
        "spIncludeAcquiredDate": bool(sp.get("include_acquired_date", True)),
        "spIncludeValidity": bool(sp.get("include_validity", False)),
        "spSeed": int(sp.get("seed", 123) or 123),
        # Stores detail
        "storeEnsureIsoCoverage": bool(stores.get("ensure_iso_coverage", True)),
        "storeDistrictSize": int(stores.get("district_size", 10)),
        "storeDistrictsPerRegion": int(stores.get("districts_per_region", 8)),
        "storeOpeningStart": str(stores.get("opening", {}).get("start", "1995-01-01")),
        "storeOpeningEnd": str(stores.get("opening", {}).get("end", "2023-12-31")),
        "storeClosingEnd": str(stores.get("closing_end", "2028-12-31")),
        "storeAssortmentEnabled": bool(_g(stores, "assortment", "enabled", default=True)),
        # Employees
        "employeeMinStaff": int(emp.get("min_staff_per_store", 3)),
        "employeeMaxStaff": int(emp.get("max_staff_per_store", 5)),
        "employeeEmailDomain": str(_g(emp, "hr", "email_domain", default="contoso.com")),
        "employeeStoreAssignments": bool(store_assigns.get("enabled", True)),
        # Exchange Rates
        "erCurrencies": list(er.get("currencies", ["CAD", "GBP", "EUR", "INR", "AUD", "CNY", "JPY"])),
        "erBaseCurrency": str(er.get("base_currency", "USD")),
        "erVolatility": float(er.get("volatility", 0.02)),
        "erFutureDrift": float(er.get("future_annual_drift", 0.02)),
        "erUseGlobalDates": bool(er.get("use_global_dates", True)),
        # Budget detail
        "budgetEnabled": bool(budget.get("enabled", True)),
        "budgetReportCurrency": str(budget.get("report_currency", "USD")),
        "budgetDefaultGrowth": float(budget.get("default_backcast_growth", 0.05)),
        "budgetReturnRateCap": float(budget.get("return_rate_cap", 0.30)),
        # Inventory detail
        "inventoryEnabled": bool(inv.get("enabled", True)),
        "inventoryGrain": str(inv.get("grain", "monthly")),
        "inventoryShrinkageEnabled": bool(_g(inv, "shrinkage", "enabled", default=True)),
        "inventoryShrinkageRate": float(_g(inv, "shrinkage", "rate", default=0.02)),
    }


# ---------------------------------------------------------------------------
# POST /api/config -- apply partial updates from frontend
# ---------------------------------------------------------------------------

@router.post("")
def update_config(body: ConfigUpdate):
    cfg = _state._cfg
    v = body.values

    cfg.setdefault("defaults", {}).setdefault("dates", {})
    cfg["defaults"].setdefault("seed", 42)
    cfg.setdefault("sales", {}).setdefault("partitioning", {})
    cfg.setdefault("customers", {})
    cfg.setdefault("stores", {}).setdefault("opening", {})
    cfg["stores"].setdefault("assortment", {})
    cfg.setdefault("products", {}).setdefault("pricing", {}).setdefault("base", {})
    cfg["products"]["pricing"].setdefault("cost", {})
    cfg["products"]["pricing"].setdefault("brand_normalization", {})
    cfg.setdefault("promotions", {})
    cfg.setdefault("returns", {})
    cfg.setdefault("geography", {}).setdefault("country_weights", {})
    cfg.setdefault("dates", {}).setdefault("include", {}).setdefault("weekly_fiscal", {})
    cfg.setdefault("customer_segments", {}).setdefault("validity", {})
    cfg.setdefault("superpowers", {})
    cfg.setdefault("employees", {}).setdefault("hr", {})
    cfg["employees"].setdefault("store_assignments", {})
    cfg.setdefault("exchange_rates", {})
    cfg.setdefault("budget", {})
    cfg.setdefault("inventory", {}).setdefault("shrinkage", {})

    # Global
    if "seed" in v: cfg["defaults"]["seed"] = int(v["seed"])

    # Output
    if "format" in v: cfg["sales"]["file_format"] = v["format"]
    if "salesOutput" in v: cfg["sales"]["sales_output"] = v["salesOutput"]
    if "skipOrderCols" in v: cfg["sales"]["skip_order_cols"] = bool(v["skipOrderCols"])
    if "compression" in v: cfg["sales"]["compression"] = v["compression"]
    if "rowGroupSize" in v: cfg["sales"]["row_group_size"] = int(v["rowGroupSize"])
    if "mergeParquet" in v: cfg["sales"]["merge_parquet"] = bool(v["mergeParquet"])
    if "partitionEnabled" in v: cfg["sales"]["partitioning"]["enabled"] = bool(v["partitionEnabled"])
    if "maxLinesPerOrder" in v: cfg["sales"]["max_lines_per_order"] = int(v["maxLinesPerOrder"])
    if "salesOptimize" in v: cfg["sales"]["optimize"] = bool(v["salesOptimize"])

    # Dates
    if "startDate" in v: cfg["defaults"]["dates"]["start"] = v["startDate"]
    if "endDate" in v: cfg["defaults"]["dates"]["end"] = v["endDate"]
    if "fiscalMonthOffset" in v: cfg["dates"]["fiscal_month_offset"] = int(v["fiscalMonthOffset"])
    if "asOfDate" in v: cfg["dates"]["as_of_date"] = v["asOfDate"] or None
    if "includeIso" in v: cfg["dates"]["include"]["iso"] = bool(v["includeIso"])
    if "includeFiscal" in v: cfg["dates"]["include"]["fiscal"] = bool(v["includeFiscal"])
    if "includeWeeklyFiscal" in v: cfg["dates"]["include"]["weekly_fiscal"]["enabled"] = bool(v["includeWeeklyFiscal"])
    if "wfFirstDay" in v: cfg["dates"]["include"]["weekly_fiscal"]["first_day_of_week"] = int(v["wfFirstDay"])
    if "wfWeeklyType" in v: cfg["dates"]["include"]["weekly_fiscal"]["weekly_type"] = v["wfWeeklyType"]
    if "wfQuarterType" in v: cfg["dates"]["include"]["weekly_fiscal"]["quarter_week_type"] = v["wfQuarterType"]
    if "wfTypeStartFiscalYear" in v: cfg["dates"]["include"]["weekly_fiscal"]["type_start_fiscal_year"] = int(v["wfTypeStartFiscalYear"])

    # Volume
    if "salesRows" in v: cfg["sales"]["total_rows"] = int(v["salesRows"])
    if "chunkSize" in v: cfg["sales"]["chunk_size"] = int(v["chunkSize"])
    if "workers" in v: cfg["sales"]["workers"] = int(v["workers"])

    # Dimensions
    if "customers" in v: cfg["customers"]["total_customers"] = int(v["customers"])
    if "stores" in v: cfg["stores"]["num_stores"] = int(v["stores"])
    if "products" in v: cfg["products"]["num_products"] = int(v["products"])
    if "promotions" in v: _set_promotions_total(cfg["promotions"], int(v["promotions"]))

    # Customers detail
    if "pctIndia" in v: cfg["customers"]["pct_india"] = float(v["pctIndia"])
    if "pctUs" in v: cfg["customers"]["pct_us"] = float(v["pctUs"])
    if "pctEu" in v: cfg["customers"]["pct_eu"] = float(v["pctEu"])
    if "pctAsia" in v: cfg["customers"]["pct_asia"] = float(v["pctAsia"])
    if "pctOrg" in v: cfg["customers"]["pct_org"] = float(v["pctOrg"])
    if "customerActiveRatio" in v: cfg["customers"]["active_ratio"] = float(v["customerActiveRatio"])
    if "profile" in v: cfg["customers"]["profile"] = v["profile"]
    if "firstYearPct" in v: cfg["customers"]["first_year_pct"] = float(v["firstYearPct"])

    # Products detail
    if "valueScale" in v: cfg["products"]["pricing"]["base"]["value_scale"] = float(v["valueScale"])
    if "minPrice" in v: cfg["products"]["pricing"]["base"]["min_unit_price"] = float(v["minPrice"])
    if "maxPrice" in v: cfg["products"]["pricing"]["base"]["max_unit_price"] = float(v["maxPrice"])
    if "productActiveRatio" in v: cfg["products"]["active_ratio"] = float(v["productActiveRatio"])
    if "marginMin" in v: cfg["products"]["pricing"]["cost"]["min_margin_pct"] = float(v["marginMin"])
    if "marginMax" in v: cfg["products"]["pricing"]["cost"]["max_margin_pct"] = float(v["marginMax"])
    if "brandNormalize" in v: cfg["products"]["pricing"]["brand_normalization"]["enabled"] = bool(v["brandNormalize"])
    if "brandNormalizeAlpha" in v: cfg["products"]["pricing"]["brand_normalization"]["alpha"] = float(v["brandNormalizeAlpha"])

    # Geography
    if "geoWeights" in v and isinstance(v["geoWeights"], dict):
        cfg["geography"]["country_weights"] = v["geoWeights"]

    # Returns
    if "returnsEnabled" in v: cfg["returns"]["enabled"] = bool(v["returnsEnabled"])
    if "returnRate" in v: cfg["returns"]["return_rate"] = float(v["returnRate"])
    if "returnMinDays" in v: cfg["returns"]["min_days_after_sale"] = int(v["returnMinDays"])
    if "returnMaxDays" in v: cfg["returns"]["max_days_after_sale"] = int(v["returnMaxDays"])

    # Promotions
    if "promoNewCustWindow" in v: cfg["promotions"]["new_customer_window_months"] = int(v["promoNewCustWindow"])
    if "promoSeasonal" in v: cfg["promotions"]["num_seasonal"] = int(v["promoSeasonal"])
    if "promoClearance" in v: cfg["promotions"]["num_clearance"] = int(v["promoClearance"])
    if "promoLimited" in v: cfg["promotions"]["num_limited"] = int(v["promoLimited"])
    if "promoFlash" in v: cfg["promotions"]["num_flash"] = int(v["promoFlash"])
    if "promoVolume" in v: cfg["promotions"]["num_volume"] = int(v["promoVolume"])
    if "promoLoyalty" in v: cfg["promotions"]["num_loyalty"] = int(v["promoLoyalty"])
    if "promoBundle" in v: cfg["promotions"]["num_bundle"] = int(v["promoBundle"])
    if "promoNewCustomer" in v: cfg["promotions"]["num_new_customer"] = int(v["promoNewCustomer"])

    # Customer Segments
    if "csEnabled" in v: cfg["customer_segments"]["enabled"] = bool(v["csEnabled"])
    if "csGenerateBridge" in v: cfg["customer_segments"]["generate_bridge"] = bool(v["csGenerateBridge"])
    if "csSegmentCount" in v: cfg["customer_segments"]["segment_count"] = int(v["csSegmentCount"])
    if "csPerCustomerMin" in v: cfg["customer_segments"]["segments_per_customer_min"] = int(v["csPerCustomerMin"])
    if "csPerCustomerMax" in v: cfg["customer_segments"]["segments_per_customer_max"] = int(v["csPerCustomerMax"])
    if "csIncludeScore" in v: cfg["customer_segments"]["include_score"] = bool(v["csIncludeScore"])
    if "csIncludePrimaryFlag" in v: cfg["customer_segments"]["include_primary_flag"] = bool(v["csIncludePrimaryFlag"])
    if "csIncludeValidity" in v: cfg["customer_segments"]["include_validity"] = bool(v["csIncludeValidity"])
    if "csValidityGrain" in v: cfg["customer_segments"]["validity"]["grain"] = v["csValidityGrain"]
    if "csChurnRateQtr" in v: cfg["customer_segments"]["validity"]["churn_rate_qtr"] = float(v["csChurnRateQtr"])
    if "csNewCustomerMonths" in v: cfg["customer_segments"]["validity"]["new_customer_months"] = int(v["csNewCustomerMonths"])
    if "csSeed" in v: cfg["customer_segments"]["seed"] = int(v["csSeed"])

    # Superpowers
    if "spEnabled" in v: cfg["superpowers"]["enabled"] = bool(v["spEnabled"])
    if "spGenerateBridge" in v: cfg["superpowers"]["generate_bridge"] = bool(v["spGenerateBridge"])
    if "spPowersCount" in v: cfg["superpowers"]["powers_count"] = int(v["spPowersCount"])
    if "spPerCustomerMin" in v: cfg["superpowers"]["powers_per_customer_min"] = int(v["spPerCustomerMin"])
    if "spPerCustomerMax" in v: cfg["superpowers"]["powers_per_customer_max"] = int(v["spPerCustomerMax"])
    if "spIncludePowerLevel" in v: cfg["superpowers"]["include_power_level"] = bool(v["spIncludePowerLevel"])
    if "spIncludePrimaryFlag" in v: cfg["superpowers"]["include_primary_flag"] = bool(v["spIncludePrimaryFlag"])
    if "spIncludeAcquiredDate" in v: cfg["superpowers"]["include_acquired_date"] = bool(v["spIncludeAcquiredDate"])
    if "spIncludeValidity" in v: cfg["superpowers"]["include_validity"] = bool(v["spIncludeValidity"])
    if "spSeed" in v: cfg["superpowers"]["seed"] = int(v["spSeed"])

    # Stores detail
    if "storeEnsureIsoCoverage" in v: cfg["stores"]["ensure_iso_coverage"] = bool(v["storeEnsureIsoCoverage"])
    if "storeDistrictSize" in v: cfg["stores"]["district_size"] = int(v["storeDistrictSize"])
    if "storeDistrictsPerRegion" in v: cfg["stores"]["districts_per_region"] = int(v["storeDistrictsPerRegion"])
    if "storeOpeningStart" in v: cfg["stores"]["opening"]["start"] = v["storeOpeningStart"]
    if "storeOpeningEnd" in v: cfg["stores"]["opening"]["end"] = v["storeOpeningEnd"]
    if "storeClosingEnd" in v: cfg["stores"]["closing_end"] = v["storeClosingEnd"]
    if "storeAssortmentEnabled" in v: cfg["stores"]["assortment"]["enabled"] = bool(v["storeAssortmentEnabled"])

    # Employees
    if "employeeMinStaff" in v: cfg["employees"]["min_staff_per_store"] = int(v["employeeMinStaff"])
    if "employeeMaxStaff" in v: cfg["employees"]["max_staff_per_store"] = int(v["employeeMaxStaff"])
    if "employeeEmailDomain" in v: cfg["employees"]["hr"]["email_domain"] = v["employeeEmailDomain"]
    if "employeeStoreAssignments" in v: cfg["employees"]["store_assignments"]["enabled"] = bool(v["employeeStoreAssignments"])

    # Exchange Rates
    if "erCurrencies" in v and isinstance(v["erCurrencies"], list): cfg["exchange_rates"]["currencies"] = v["erCurrencies"]
    if "erBaseCurrency" in v: cfg["exchange_rates"]["base_currency"] = v["erBaseCurrency"]
    if "erVolatility" in v: cfg["exchange_rates"]["volatility"] = float(v["erVolatility"])
    if "erFutureDrift" in v: cfg["exchange_rates"]["future_annual_drift"] = float(v["erFutureDrift"])
    if "erUseGlobalDates" in v: cfg["exchange_rates"]["use_global_dates"] = bool(v["erUseGlobalDates"])

    # Budget detail
    if "budgetEnabled" in v: cfg["budget"]["enabled"] = bool(v["budgetEnabled"])
    if "budgetReportCurrency" in v: cfg["budget"]["report_currency"] = v["budgetReportCurrency"]
    if "budgetDefaultGrowth" in v: cfg["budget"]["default_backcast_growth"] = float(v["budgetDefaultGrowth"])
    if "budgetReturnRateCap" in v: cfg["budget"]["return_rate_cap"] = float(v["budgetReturnRateCap"])

    # Inventory detail
    if "inventoryEnabled" in v: cfg["inventory"]["enabled"] = bool(v["inventoryEnabled"])
    if "inventoryGrain" in v: cfg["inventory"]["grain"] = v["inventoryGrain"]
    if "inventoryShrinkageEnabled" in v: cfg["inventory"]["shrinkage"]["enabled"] = bool(v["inventoryShrinkageEnabled"])
    if "inventoryShrinkageRate" in v: cfg["inventory"]["shrinkage"]["rate"] = float(v["inventoryShrinkageRate"])

    return {"ok": True}


@router.get("/download")
def download_config():
    return _state._cfg


# ---------------------------------------------------------------------------
# Config YAML editor (in-memory only, never writes to disk)
# ---------------------------------------------------------------------------

@router.get("/yaml")
def get_config_yaml():
    """Return the current in-memory config serialized as YAML text."""
    text = yaml.safe_dump(_state._cfg, sort_keys=False, default_flow_style=False)
    return Response(content=text, media_type="text/plain")


@router.get("/yaml/disk")
def get_config_yaml_disk():
    """Return the original config.yaml from disk."""
    return Response(content=_state._cfg_disk_yaml, media_type="text/plain")


@router.post("/yaml")
def update_config_yaml(body: ConfigYamlUpdate):
    """Parse YAML, normalize, replace in-memory config. Original file untouched."""
    text = body.yaml_text
    try:
        parsed = yaml.safe_load(text)
        if not isinstance(parsed, dict):
            raise HTTPException(400, "Config YAML must be a mapping at the top level.")
    except yaml.YAMLError as e:
        raise HTTPException(400, f"Invalid YAML: {e}")

    _state._cfg = normalize_config_yaml(parsed)
    return {"ok": True}


@router.post("/yaml/reset")
def reset_config_yaml():
    """Reload config from disk, discarding in-memory edits."""
    _state._cfg = _base_config()
    _state._cfg_disk_yaml = _config_path.read_text(encoding="utf-8") if _config_path.exists() else ""
    return {"ok": True}
