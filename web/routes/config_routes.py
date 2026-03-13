"""
web/routes/config_routes.py -- All /api/config/* endpoints.
"""

from __future__ import annotations

from typing import Any, Dict

import copy

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
    cfg_to_dict,
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
    with _state._cfg_lock:
        cfg = copy.deepcopy(_state._cfg)
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
    sp = _g(cfg, "superpowers", default={})
    emp = _g(cfg, "employees", default={})
    er = _g(cfg, "exchange_rates", default={})
    budget = _g(cfg, "budget", default={})
    inv = _g(cfg, "inventory", default={})
    cost = _g(cfg, "products", "pricing", "cost", default={})
    brand_norm = _g(cfg, "products", "pricing", "brand_normalization", default={})
    store_assigns = _g(emp, "store_assignments", default={})

    return {
        # Global
        "seed": int(_g(cfg, "defaults", "seed", default=42) or 42),
        # Output
        "format": str(getattr(sales, "file_format", "parquet")),
        "salesOutput": str(getattr(sales, "sales_output", "sales")),
        "skipOrderCols": bool(getattr(sales, "skip_order_cols", False)),
        "compression": str(_g(sales, "compression", default="snappy")),
        "rowGroupSize": int(_g(sales, "row_group_size", default=2000000)),
        "mergeParquet": bool(getattr(sales, "merge_parquet", True)),
        "partitionEnabled": bool(_g(sales, "partitioning", "enabled", default=True)),
        "maxLinesPerOrder": int(getattr(sales, "max_lines_per_order", 5)),
        "salesOptimize": bool(getattr(sales, "optimize", True)),
        # Dates
        "startDate": str(_g(defaults, "start", default="2023-01-01")),
        "endDate": str(_g(defaults, "end", default="2026-12-31")),
        "fiscalMonthOffset": int(_g(dates_cfg, "fiscal_month_offset", default=_g(dates_cfg, "fiscal_start_month", default=0)) or 0),
        "asOfDate": str(_g(dates_cfg, "as_of_date", default="") or ""),
        "includeCalendar": True,
        "includeIso": bool(_g(include, "iso", default=False)),
        "includeFiscal": bool(_g(include, "fiscal", default=True)),
        "includeWeeklyFiscal": bool(_g(wf, "enabled", default=False)),
        "wfFirstDay": int(_g(wf, "first_day_of_week", default=0)),
        "wfWeeklyType": str(_g(wf, "weekly_type", default="Last")),
        "wfQuarterType": str(_g(wf, "quarter_week_type", default="445")),
        "wfTypeStartFiscalYear": int(_g(wf, "type_start_fiscal_year", default=1)),
        # Volume
        "salesRows": int(getattr(sales, "total_rows", 100000)),
        "chunkSize": int(getattr(sales, "chunk_size", 200000)),
        "workers": int(getattr(sales, "workers", 0) or 0),
        # Dimensions
        "customers": int(getattr(cust, "total_customers", 10000) or 10000),
        "stores": int(getattr(stores, "num_stores", 100) or 100),
        "products": int(getattr(prods, "num_products", 5000) or 5000),
        "promotions": _promo_total(promos),
        # Customers detail
        "pctIndia": float(getattr(cust, "pct_india", 10)),
        "pctUs": float(getattr(cust, "pct_us", 51)),
        "pctEu": float(getattr(cust, "pct_eu", 39)),
        "pctAsia": float(getattr(cust, "pct_asia", 0)),
        "pctOrg": float(getattr(cust, "pct_org", 10)),
        "customerActiveRatio": float(getattr(cust, "active_ratio", 0.90)),
        "profile": str(getattr(cust, "profile", "steady")),
        "firstYearPct": float(getattr(cust, "first_year_pct", 0.27)),
        # Products detail (pricing/cost/brand_norm are plain dicts from products.pricing)
        "valueScale": float(pricing.get("value_scale", 1.0) if isinstance(pricing, dict) else getattr(pricing, "value_scale", 1.0)),
        "minPrice": float(pricing.get("min_unit_price", 10) if isinstance(pricing, dict) else getattr(pricing, "min_unit_price", 10)),
        "maxPrice": float(pricing.get("max_unit_price", 3000) if isinstance(pricing, dict) else getattr(pricing, "max_unit_price", 3000)),
        "productActiveRatio": float(getattr(prods, "active_ratio", 0.94)),
        "marginMin": float(cost.get("min_margin_pct", 0.20) if isinstance(cost, dict) else getattr(cost, "min_margin_pct", 0.20)),
        "marginMax": float(cost.get("max_margin_pct", 0.35) if isinstance(cost, dict) else getattr(cost, "max_margin_pct", 0.35)),
        "brandNormalize": bool(brand_norm.get("enabled", False) if isinstance(brand_norm, dict) else getattr(brand_norm, "enabled", False)),
        "brandNormalizeAlpha": float(brand_norm.get("alpha", 0.35) if isinstance(brand_norm, dict) else getattr(brand_norm, "alpha", 0.35)),
        # Geography
        "geoWeights": dict((geo.get("country_weights", {}) if isinstance(geo, dict) else getattr(geo, "country_weights", {})) or {}),
        # Returns
        "returnsEnabled": bool(getattr(returns, "enabled", True)),
        "returnRate": float(getattr(returns, "return_rate", 0.03)),
        "returnMinDays": int(getattr(returns, "min_days_after_sale", 1)),
        "returnMaxDays": int(getattr(returns, "max_days_after_sale", 60)),
        # Promotions
        "promoNewCustWindow": int(_g(promos, "new_customer_window_months", default=3)),
        "promoSeasonal": int(getattr(promos, "num_seasonal", 20) or 20),
        "promoClearance": int(getattr(promos, "num_clearance", 8) or 8),
        "promoLimited": int(getattr(promos, "num_limited", 12) or 12),
        "promoFlash": int(getattr(promos, "num_flash", 6) or 6),
        "promoVolume": int(getattr(promos, "num_volume", 4) or 4),
        "promoLoyalty": int(getattr(promos, "num_loyalty", 3) or 3),
        "promoBundle": int(getattr(promos, "num_bundle", 3) or 3),
        "promoNewCustomer": int(getattr(promos, "num_new_customer", 3) or 3),
        # Superpowers
        "spEnabled": bool(getattr(sp, "enabled", False)),
        "spGenerateBridge": bool(getattr(sp, "generate_bridge", False)),
        "spPowersCount": int(getattr(sp, "powers_count", 20)),
        "spPerCustomerMin": int(getattr(sp, "powers_per_customer_min", 1)),
        "spPerCustomerMax": int(getattr(sp, "powers_per_customer_max", 3)),
        "spIncludePowerLevel": bool(getattr(sp, "include_power_level", True)),
        "spIncludePrimaryFlag": bool(getattr(sp, "include_primary_flag", True)),
        "spIncludeAcquiredDate": bool(getattr(sp, "include_acquired_date", True)),
        "spIncludeValidity": bool(getattr(sp, "include_validity", False)),
        "spSeed": int(getattr(sp, "seed", 123) or 123),
        # Stores detail
        "storeEnsureIsoCoverage": bool(getattr(stores, "ensure_iso_coverage", True)),
        "storeDistrictSize": int(getattr(stores, "district_size", 10)),
        "storeDistrictsPerRegion": int(getattr(stores, "districts_per_region", 8)),
        "storeOpeningStart": str(_g(stores, "opening", "start", default="1995-01-01")),
        "storeOpeningEnd": str(_g(stores, "opening", "end", default="2023-12-31")),
        "storeClosingEnd": str(getattr(stores, "closing_end", "2028-12-31")),
        "storeAssortmentEnabled": bool(_g(stores, "assortment", "enabled", default=True)),
        # Employees
        "employeeMinStaff": int(getattr(emp, "min_staff_per_store", 3)),
        "employeeMaxStaff": int(getattr(emp, "max_staff_per_store", 5)),
        "employeeEmailDomain": str(_g(emp, "hr", "email_domain", default="contoso.com")),
        "employeeStoreAssignments": bool(getattr(store_assigns, "enabled", True)),
        # Exchange Rates
        "erCurrencies": list(getattr(er, "currencies", ["CAD", "GBP", "EUR", "INR", "AUD", "CNY", "JPY"])),
        "erBaseCurrency": str(getattr(er, "base_currency", "USD")),
        "erVolatility": float(getattr(er, "volatility", 0.02)),
        "erFutureDrift": float(getattr(er, "future_annual_drift", 0.02)),
        "erUseGlobalDates": bool(getattr(er, "use_global_dates", True)),
        # Budget detail
        "budgetEnabled": bool(getattr(budget, "enabled", True)),
        "budgetReportCurrency": str(getattr(budget, "report_currency", "USD")),
        "budgetDefaultGrowth": float(getattr(budget, "default_backcast_growth", 0.05)),
        "budgetReturnRateCap": float(getattr(budget, "return_rate_cap", 0.30)),
        # Inventory detail
        "inventoryEnabled": bool(getattr(inv, "enabled", True)),
        "inventoryGrain": str(getattr(inv, "grain", "monthly")),
        "inventoryShrinkageEnabled": bool(_g(inv, "shrinkage", "enabled", default=True)),
        "inventoryShrinkageRate": float(_g(inv, "shrinkage", "rate", default=0.02)),
    }


# ---------------------------------------------------------------------------
# POST /api/config -- apply partial updates from frontend
# ---------------------------------------------------------------------------

@router.post("")
def update_config(body: ConfigUpdate):
    with _state._cfg_lock:
        cfg = copy.deepcopy(_state._cfg)
        v = body.values

        # Ensure nested sub-models exist (Pydantic models have defaults,
        # but products.pricing is Optional[Dict] and geography may be None).
        if cfg.products.pricing is None:
            cfg.products.pricing = {}
        cfg.products.pricing.setdefault("base", {})
        cfg.products.pricing.setdefault("cost", {})
        cfg.products.pricing.setdefault("brand_normalization", {})
        if cfg.geography is None:
            from src.engine.config.config_schema import GeographyConfig
            cfg.geography = GeographyConfig()
        if cfg.dates.include.weekly_fiscal is None:
            from src.engine.config.config_schema import WeeklyFiscalConfig
            cfg.dates.include.weekly_fiscal = WeeklyFiscalConfig()

        # Global
        if "seed" in v: cfg.defaults.seed = int(v["seed"])

        # Output
        if "format" in v: cfg.sales.file_format = v["format"]
        if "salesOutput" in v: cfg.sales.sales_output = v["salesOutput"]
        if "skipOrderCols" in v: cfg.sales.skip_order_cols = bool(v["skipOrderCols"])
        if "compression" in v: cfg.sales.compression = v["compression"]
        if "rowGroupSize" in v: cfg.sales.row_group_size = int(v["rowGroupSize"])
        if "mergeParquet" in v: cfg.sales.merge_parquet = bool(v["mergeParquet"])
        if "partitionEnabled" in v:
            if cfg.sales.partitioning is None:
                cfg.sales.partitioning = {}
            cfg.sales.partitioning["enabled"] = bool(v["partitionEnabled"])
        if "maxLinesPerOrder" in v: cfg.sales.max_lines_per_order = int(v["maxLinesPerOrder"])
        if "salesOptimize" in v: cfg.sales.optimize = bool(v["salesOptimize"])

        # Dates
        if "startDate" in v: cfg.defaults.dates.start = v["startDate"]
        if "endDate" in v: cfg.defaults.dates.end = v["endDate"]
        if "fiscalMonthOffset" in v: cfg.dates.fiscal_start_month = int(v["fiscalMonthOffset"])
        if "asOfDate" in v: cfg.dates.as_of_date = v["asOfDate"] or None
        if "includeIso" in v: cfg.dates.include.iso = bool(v["includeIso"])
        if "includeFiscal" in v: cfg.dates.include.fiscal = bool(v["includeFiscal"])
        if "includeWeeklyFiscal" in v: cfg.dates.include.weekly_fiscal.enabled = bool(v["includeWeeklyFiscal"])
        if "wfFirstDay" in v: cfg.dates.include.weekly_fiscal.first_day_of_week = int(v["wfFirstDay"])
        if "wfWeeklyType" in v: cfg.dates.include.weekly_fiscal.weekly_type = v["wfWeeklyType"]
        if "wfQuarterType" in v: cfg.dates.include.weekly_fiscal.quarter_week_type = v["wfQuarterType"]
        if "wfTypeStartFiscalYear" in v: cfg.dates.include.weekly_fiscal.type_start_fiscal_year = int(v["wfTypeStartFiscalYear"])

        # Volume
        if "salesRows" in v: cfg.sales.total_rows = int(v["salesRows"])
        if "chunkSize" in v: cfg.sales.chunk_size = int(v["chunkSize"])
        if "workers" in v: cfg.sales.workers = int(v["workers"])

        # Dimensions
        if "customers" in v: cfg.customers.total_customers = int(v["customers"])
        if "stores" in v: cfg.stores.num_stores = int(v["stores"])
        if "products" in v: cfg.products.num_products = int(v["products"])
        if "promotions" in v: _set_promotions_total(cfg.promotions, int(v["promotions"]))

        # Customers detail
        if "pctIndia" in v: cfg.customers.pct_india = float(v["pctIndia"])
        if "pctUs" in v: cfg.customers.pct_us = float(v["pctUs"])
        if "pctEu" in v: cfg.customers.pct_eu = float(v["pctEu"])
        if "pctAsia" in v: cfg.customers.pct_asia = float(v["pctAsia"])
        if "pctOrg" in v: cfg.customers.pct_org = float(v["pctOrg"])
        if "customerActiveRatio" in v: cfg.customers.active_ratio = float(v["customerActiveRatio"])
        if "profile" in v: cfg.customers.profile = v["profile"]
        if "firstYearPct" in v: cfg.customers.first_year_pct = float(v["firstYearPct"])

        # Products detail (pricing is Dict[str, Any], so dict access is correct)
        if "valueScale" in v: cfg.products.pricing["base"]["value_scale"] = float(v["valueScale"])
        if "minPrice" in v: cfg.products.pricing["base"]["min_unit_price"] = float(v["minPrice"])
        if "maxPrice" in v: cfg.products.pricing["base"]["max_unit_price"] = float(v["maxPrice"])
        if "productActiveRatio" in v: cfg.products.active_ratio = float(v["productActiveRatio"])
        if "marginMin" in v: cfg.products.pricing["cost"]["min_margin_pct"] = float(v["marginMin"])
        if "marginMax" in v: cfg.products.pricing["cost"]["max_margin_pct"] = float(v["marginMax"])
        if "brandNormalize" in v: cfg.products.pricing["brand_normalization"]["enabled"] = bool(v["brandNormalize"])
        if "brandNormalizeAlpha" in v: cfg.products.pricing["brand_normalization"]["alpha"] = float(v["brandNormalizeAlpha"])

        # Geography
        if "geoWeights" in v and isinstance(v["geoWeights"], dict):
            object.__setattr__(cfg.geography, "country_weights", v["geoWeights"])

        # Returns
        if "returnsEnabled" in v: cfg.returns.enabled = bool(v["returnsEnabled"])
        if "returnRate" in v: cfg.returns.return_rate = float(v["returnRate"])
        if "returnMinDays" in v: cfg.returns.min_days_after_sale = int(v["returnMinDays"])
        if "returnMaxDays" in v: cfg.returns.max_days_after_sale = int(v["returnMaxDays"])

        # Promotions
        if "promoNewCustWindow" in v: cfg.promotions.new_customer_window_months = int(v["promoNewCustWindow"])
        if "promoSeasonal" in v: cfg.promotions.num_seasonal = int(v["promoSeasonal"])
        if "promoClearance" in v: cfg.promotions.num_clearance = int(v["promoClearance"])
        if "promoLimited" in v: cfg.promotions.num_limited = int(v["promoLimited"])
        if "promoFlash" in v: cfg.promotions.num_flash = int(v["promoFlash"])
        if "promoVolume" in v: cfg.promotions.num_volume = int(v["promoVolume"])
        if "promoLoyalty" in v: cfg.promotions.num_loyalty = int(v["promoLoyalty"])
        if "promoBundle" in v: cfg.promotions.num_bundle = int(v["promoBundle"])
        if "promoNewCustomer" in v: cfg.promotions.num_new_customer = int(v["promoNewCustomer"])

        # Superpowers
        if "spEnabled" in v: cfg.superpowers.enabled = bool(v["spEnabled"])
        if "spGenerateBridge" in v: cfg.superpowers.generate_bridge = bool(v["spGenerateBridge"])
        if "spPowersCount" in v: cfg.superpowers.powers_count = int(v["spPowersCount"])
        if "spPerCustomerMin" in v: cfg.superpowers.powers_per_customer_min = int(v["spPerCustomerMin"])
        if "spPerCustomerMax" in v: cfg.superpowers.powers_per_customer_max = int(v["spPerCustomerMax"])
        if "spIncludePowerLevel" in v: cfg.superpowers.include_power_level = bool(v["spIncludePowerLevel"])
        if "spIncludePrimaryFlag" in v: cfg.superpowers.include_primary_flag = bool(v["spIncludePrimaryFlag"])
        if "spIncludeAcquiredDate" in v: cfg.superpowers.include_acquired_date = bool(v["spIncludeAcquiredDate"])
        if "spIncludeValidity" in v: cfg.superpowers.include_validity = bool(v["spIncludeValidity"])
        if "spSeed" in v: cfg.superpowers.seed = int(v["spSeed"])

        # Stores detail
        if "storeEnsureIsoCoverage" in v: cfg.stores.ensure_iso_coverage = bool(v["storeEnsureIsoCoverage"])
        if "storeDistrictSize" in v: cfg.stores.district_size = int(v["storeDistrictSize"])
        if "storeDistrictsPerRegion" in v: cfg.stores.districts_per_region = int(v["storeDistrictsPerRegion"])
        if "storeOpeningStart" in v: cfg.stores.opening.start = v["storeOpeningStart"]
        if "storeOpeningEnd" in v: cfg.stores.opening.end = v["storeOpeningEnd"]
        if "storeClosingEnd" in v: cfg.stores.closing_end = v["storeClosingEnd"]
        if "storeAssortmentEnabled" in v: cfg.stores.assortment.enabled = bool(v["storeAssortmentEnabled"])

        # Employees
        if "employeeMinStaff" in v: cfg.employees.min_staff_per_store = int(v["employeeMinStaff"])
        if "employeeMaxStaff" in v: cfg.employees.max_staff_per_store = int(v["employeeMaxStaff"])
        if "employeeEmailDomain" in v: cfg.employees.hr.email_domain = v["employeeEmailDomain"]
        if "employeeStoreAssignments" in v: cfg.employees.store_assignments.enabled = bool(v["employeeStoreAssignments"])

        # Exchange Rates
        if "erCurrencies" in v and isinstance(v["erCurrencies"], list): cfg.exchange_rates.currencies = v["erCurrencies"]
        if "erBaseCurrency" in v: cfg.exchange_rates.base_currency = v["erBaseCurrency"]
        if "erVolatility" in v: cfg.exchange_rates.volatility = float(v["erVolatility"])
        if "erFutureDrift" in v: cfg.exchange_rates.future_annual_drift = float(v["erFutureDrift"])
        if "erUseGlobalDates" in v: cfg.exchange_rates.use_global_dates = bool(v["erUseGlobalDates"])

        # Budget detail
        if "budgetEnabled" in v: cfg.budget.enabled = bool(v["budgetEnabled"])
        if "budgetReportCurrency" in v: cfg.budget.report_currency = v["budgetReportCurrency"]
        if "budgetDefaultGrowth" in v: cfg.budget.default_backcast_growth = float(v["budgetDefaultGrowth"])
        if "budgetReturnRateCap" in v: cfg.budget.return_rate_cap = float(v["budgetReturnRateCap"])

        # Inventory detail
        if "inventoryEnabled" in v: cfg.inventory.enabled = bool(v["inventoryEnabled"])
        if "inventoryGrain" in v: cfg.inventory.grain = v["inventoryGrain"]
        if "inventoryShrinkageEnabled" in v: cfg.inventory.shrinkage.enabled = bool(v["inventoryShrinkageEnabled"])
        if "inventoryShrinkageRate" in v: cfg.inventory.shrinkage.rate = float(v["inventoryShrinkageRate"])

        _state._cfg = cfg
        return {"ok": True}


@router.get("/download")
def download_config():
    with _state._cfg_lock:
        cfg = copy.deepcopy(_state._cfg)
    return cfg_to_dict(cfg)


# ---------------------------------------------------------------------------
# Config YAML editor (in-memory only, never writes to disk)
# ---------------------------------------------------------------------------

@router.get("/yaml")
def get_config_yaml():
    """Return the current in-memory config serialized as YAML text."""
    with _state._cfg_lock:
        text = yaml.safe_dump(cfg_to_dict(_state._cfg), sort_keys=False, default_flow_style=False)
    return Response(content=text, media_type="text/plain")


@router.get("/yaml/disk")
def get_config_yaml_disk():
    """Return the original config.yaml from disk."""
    with _state._cfg_lock:
        text = _state._cfg_disk_yaml
    return Response(content=text, media_type="text/plain")


@router.post("/yaml")
def update_config_yaml(body: ConfigYamlUpdate):
    """Parse YAML, normalize, replace in-memory config. Original file untouched."""
    text = body.yaml_text
    if len(text) > 1_048_576:
        raise HTTPException(413, "YAML text exceeds 1 MB limit")
    try:
        parsed = yaml.safe_load(text)
        if not isinstance(parsed, dict):
            raise HTTPException(400, "Config YAML must be a mapping at the top level.")
    except yaml.YAMLError as e:
        raise HTTPException(400, f"Invalid YAML: {e}")

    with _state._cfg_lock:
        _state._cfg = normalize_config_yaml(parsed)
    return {"ok": True}


@router.post("/yaml/reset")
def reset_config_yaml():
    """Reload config from disk, discarding in-memory edits."""
    with _state._cfg_lock:
        _state._cfg = _base_config()
        _state._cfg_disk_yaml = _config_path.read_text(encoding="utf-8") if _config_path.exists() else ""
    return {"ok": True}
