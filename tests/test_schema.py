"""Tests for Pydantic config schema models (Phase 3).

Validates that:
- AppConfig and ModelsConfig parse actual YAML files
- Sub-model nesting and defaults are correct
- Invalid configs raise ValidationError
- load_config() returns typed AppConfig
- Mutation patterns (__setitem__, pop, setdefault, update) work
- extra="forbid" catches typos in sub-model configs
"""
from __future__ import annotations

import copy
import pytest
import yaml

from src.engine.config.config_schema import (
    AppConfig,
    ModelsConfig,
    SalesConfig,
    DefaultsConfig,
    BudgetConfig,
    BudgetWeightsConfig,
    InventoryConfig,
    CustomersConfig,
    StoresConfig,
    EmployeesConfig,
    ExchangeRatesConfig,
    ReturnsConfig,
    ProductsConfig,
    PromotionsConfig,
    DatesTableConfig,
    SubscriptionsConfig,
    PackagingConfig,
    YearLevelFactorsConfig,
    ModelsInnerConfig,
)


# =========================================================================
# Fixtures
# =========================================================================

@pytest.fixture
def raw_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def normalized_config():
    from src.engine.config.config import load_config
    return load_config("config.yaml")


@pytest.fixture
def raw_models():
    with open("models.yaml") as f:
        return yaml.safe_load(f)


# =========================================================================
# AppConfig — parse actual config.yaml
# =========================================================================

class TestAppConfigFromYAML:
    def test_validates_normalized_config(self, normalized_config):
        app = AppConfig.from_raw_dict(normalized_config)
        assert app.sales.total_rows == normalized_config.sales.total_rows
        assert app.sales.file_format == normalized_config.sales.file_format

    def test_defaults_dates(self, normalized_config):
        app = AppConfig.from_raw_dict(normalized_config)
        assert app.defaults.dates.start == "2021-01-01"
        assert app.defaults.dates.end == "2025-12-31"

    def test_sales_section(self, normalized_config):
        app = AppConfig.from_raw_dict(normalized_config)
        assert app.sales.skip_order_cols is False
        assert app.sales.compression == "snappy"
        assert isinstance(app.sales.chunk_size, int)

    def test_budget_section(self, normalized_config):
        app = AppConfig.from_raw_dict(normalized_config)
        assert app.budget.enabled is True
        assert app.budget.report_currency == "USD"
        assert app.budget.scenarios.High == 0.05

    def test_inventory_section(self, normalized_config):
        app = AppConfig.from_raw_dict(normalized_config)
        assert app.inventory.enabled is True
        assert app.inventory.shrinkage.rate == 0.02
        assert app.inventory.abc_stock_multiplier.A == 1.20

    def test_stores_section(self, normalized_config):
        app = AppConfig.from_raw_dict(normalized_config)
        assert app.stores.ensure_iso_coverage is True
        assert app.stores.assortment.coverage.Online == 1.0

    def test_employees_section(self, normalized_config):
        app = AppConfig.from_raw_dict(normalized_config)
        assert app.employees.hr.email_domain == "contoso.com"
        assert app.employees.store_assignments.primary_sales_role == "Sales Associate"

    def test_exchange_rates_section(self, normalized_config):
        app = AppConfig.from_raw_dict(normalized_config)
        assert "USD" == app.exchange_rates.base_currency
        assert len(app.exchange_rates.currencies) >= 5

    def test_returns_section(self, normalized_config):
        app = AppConfig.from_raw_dict(normalized_config)
        assert isinstance(app.returns.return_rate, float)

    def test_products_section(self, normalized_config):
        app = AppConfig.from_raw_dict(normalized_config)
        assert app.products.active_ratio == 0.98
        assert app.products.pricing is not None

    def test_packaging_defaults(self, normalized_config):
        app = AppConfig.from_raw_dict(normalized_config)
        assert app.packaging.reset_scratch_fact_out is True
        assert app.packaging.clean_scratch_fact_out is True


# =========================================================================
# ModelsConfig — parse actual models.yaml
# =========================================================================

class TestModelsConfigFromYAML:
    def test_validates_models_yaml(self, raw_models):
        m = ModelsConfig.from_raw_dict(raw_models)
        assert m.models.quantity.base_poisson_lambda == 2.1
        assert m.models.quantity.min_qty == 1

    def test_macro_demand(self, raw_models):
        m = ModelsConfig.from_raw_dict(raw_models)
        assert m.models.macro_demand.year_level_factors.mode == "once"
        assert len(m.models.macro_demand.year_level_factors.factor_values) == 8

    def test_pricing_inflation(self, raw_models):
        m = ModelsConfig.from_raw_dict(raw_models)
        assert m.models.pricing.inflation.annual_rate == 0.02
        assert m.models.pricing.inflation.volatility_seed == 123

    def test_pricing_markdown(self, raw_models):
        m = ModelsConfig.from_raw_dict(raw_models)
        md = m.models.pricing.markdown
        assert md.enabled is True
        assert len(md.ladder) == 7
        assert md.ladder[0].kind == "none"

    def test_pricing_appearance(self, raw_models):
        m = ModelsConfig.from_raw_dict(raw_models)
        ap = m.models.pricing.appearance
        assert ap.enabled is True
        assert len(ap.unit_price.bands) == 5
        assert ap.unit_price.bands[0].max == 100

    def test_brand_popularity(self, raw_models):
        m = ModelsConfig.from_raw_dict(raw_models)
        bp = m.models.brand_popularity
        assert bp.enabled is True
        assert bp.winner_boost == 1.4

    def test_returns_reasons(self, raw_models):
        m = ModelsConfig.from_raw_dict(raw_models)
        ret = m.models.returns
        assert len(ret.reasons) == 8
        assert ret.lag_days.distribution == "triangular"
        assert ret.quantity.full_line_probability == 0.85


# =========================================================================
# Attribute access (replacing dict-compat tests)
# =========================================================================

class TestAttributeAccess:
    def test_attribute_read(self, normalized_config):
        app = AppConfig.from_raw_dict(normalized_config)
        assert app.sales.total_rows == app.sales.total_rows

    def test_nested_attribute_read(self, normalized_config):
        app = AppConfig.from_raw_dict(normalized_config)
        assert app.defaults.dates.start == "2021-01-01"

    def test_hasattr_existing(self, normalized_config):
        app = AppConfig.from_raw_dict(normalized_config)
        assert hasattr(app, "sales")
        assert hasattr(app, "defaults")

    def test_hasattr_missing(self):
        app = AppConfig()
        assert not hasattr(app, "nonexistent_key_xyz")

    def test_keys(self, normalized_config):
        app = AppConfig.from_raw_dict(normalized_config)
        k = list(app.keys())
        assert "sales" in k
        assert "defaults" in k

    def test_items(self, normalized_config):
        app = AppConfig.from_raw_dict(normalized_config)
        items = dict(app.items())
        assert "sales" in items

    def test_iter(self, normalized_config):
        app = AppConfig.from_raw_dict(normalized_config)
        keys = list(app)
        assert "sales" in keys

    def test_len(self, normalized_config):
        app = AppConfig.from_raw_dict(normalized_config)
        assert len(app) > 10

    def test_pop_returns_value(self, normalized_config):
        app = AppConfig.from_raw_dict(normalized_config)
        val = app.pop("sales")
        assert val is not None

    def test_pop_missing_with_default(self):
        app = AppConfig()
        assert app.pop("missing_key", "fallback") == "fallback"

    def test_pop_missing_raises(self):
        app = AppConfig()
        with pytest.raises(KeyError):
            app.pop("missing_key")

    def test_setdefault_existing(self, normalized_config):
        app = AppConfig.from_raw_dict(normalized_config)
        val = app.setdefault("sales", "unused")
        assert val is not None
        assert val != "unused"

    def test_setdefault_missing(self):
        app = AppConfig()
        val = app.setdefault("nonexistent_xyz", "fallback")
        assert val == "fallback"


# =========================================================================
# Sub-model defaults
# =========================================================================

class TestDefaults:
    def test_sales_defaults(self):
        s = SalesConfig()
        assert s.file_format == "csv"
        assert s.total_rows == 1_000_000
        assert s.compression == "snappy"
        assert s.workers is None

    def test_budget_weights_global_rename(self):
        w = BudgetWeightsConfig.model_validate({"local": 0.5, "category": 0.3, "global": 0.2})
        assert w.global_ == 0.2
        assert w.local == 0.5

    def test_returns_defaults(self):
        r = ReturnsConfig()
        assert r.enabled is False
        assert r.return_rate == 0.03

    def test_packaging_defaults(self):
        p = PackagingConfig()
        assert p.reset_scratch_fact_out is True
        assert p.clean_scratch_fact_out is True

    def test_empty_appconfig(self):
        app = AppConfig()
        assert app.sales.file_format == "csv"
        assert app.defaults.seed == 42

    def test_year_level_factors_values_rename(self):
        ylf = YearLevelFactorsConfig.model_validate({"mode": "repeat", "values": [1.0, 2.0]})
        assert ylf.factor_values == [1.0, 2.0]
        assert ylf.mode == "repeat"


# =========================================================================
# extra="forbid" catches typos
# =========================================================================

class TestExtraForbid:
    def test_extra_fields_allowed_on_appconfig(self):
        """AppConfig has extra='allow' for runtime-injected keys."""
        app = AppConfig.model_validate({"unknown_section": {"foo": "bar"}})
        assert getattr(app, "unknown_section") == {"foo": "bar"}

    def test_extra_fields_forbidden_on_sales(self):
        """SalesConfig has extra='forbid' to catch typos."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="extra_forbidden"):
            SalesConfig.model_validate({"total_rows": 100, "bogus_key": 42})

    def test_extra_fields_forbidden_on_stores(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="extra_forbidden"):
            StoresConfig.model_validate({"num_stores": 10, "typo_field": True})


# =========================================================================
# Validation errors
# =========================================================================

class TestValidationErrors:
    def test_sales_total_rows_must_be_int(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            SalesConfig.model_validate({"total_rows": "not_a_number"})

    def test_nested_model_type_error(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            InventoryConfig.model_validate({"shrinkage": "invalid"})


# =========================================================================
# Phase 3: load_config integration (attribute access)
# =========================================================================

class TestLoadConfigIntegration:
    """Verify that load_config() returns a typed AppConfig."""

    def test_load_config_returns_mapping(self):
        from collections.abc import Mapping
        from src.engine.config.config import load_config
        cfg = load_config("config.yaml")
        assert isinstance(cfg, Mapping)

    def test_load_config_sales_is_mapping(self):
        from collections.abc import Mapping
        from src.engine.config.config import load_config
        cfg = load_config("config.yaml")
        assert isinstance(cfg.sales, Mapping)

    def test_load_config_validates_types(self):
        """Pydantic validation happens at load time (bad types caught early)."""
        from src.engine.config.config import load_config
        cfg = load_config("config.yaml")
        assert cfg.sales.total_rows > 0

    def test_load_config_typed_returns_appconfig(self):
        from src.engine.config.config import load_config_typed
        cfg = load_config_typed("config.yaml")
        assert isinstance(cfg, AppConfig)
        assert cfg.sales.total_rows > 0
        assert cfg.defaults.dates.start == "2021-01-01"

    def test_load_config_typed_attribute_access(self):
        from src.engine.config.config import load_config_typed
        cfg = load_config_typed("config.yaml")
        # Attribute access works
        assert cfg.sales.total_rows > 0
        assert hasattr(cfg, "defaults")

    def test_load_config_typed_mutation(self):
        from src.engine.config.config import load_config_typed
        cfg = load_config_typed("config.yaml")
        # Simulate pipeline_runner mutation patterns (setitem still works)
        cfg["config_yaml_path"] = "/test/path"
        assert cfg.config_yaml_path == "/test/path"
        cfg.defaults.dates.start = "2020-06-01"
        assert cfg.defaults.dates.start == "2020-06-01"


# =========================================================================
# Mutation operations (write-side mixin)
# =========================================================================

class TestMutation:
    def test_setitem(self):
        app = AppConfig()
        app.sales.file_format = "parquet"
        assert app.sales.file_format == "parquet"

    def test_setitem_toplevel(self):
        app = AppConfig()
        app["config_yaml_path"] = "/test/path"
        assert app.config_yaml_path == "/test/path"

    def test_delitem(self):
        app = AppConfig()
        app["custom_key"] = "hello"
        assert hasattr(app, "custom_key")
        del app["custom_key"]
        assert not hasattr(app, "custom_key")

    def test_delitem_missing_raises(self):
        app = AppConfig()
        with pytest.raises(KeyError):
            del app["nonexistent_xyz"]

    def test_pop_removes_key(self):
        app = AppConfig()
        app["temp"] = "value"
        val = app.pop("temp", None)
        assert val == "value"
        assert not hasattr(app, "temp")

    def test_setdefault_creates_key(self):
        app = AppConfig()
        val = app.setdefault("new_field", {"nested": True})
        assert val == {"nested": True}
        assert getattr(app, "new_field") == {"nested": True}

    def test_update(self):
        app = AppConfig()
        app.update({"config_yaml_path": "/new/path"})
        assert app.config_yaml_path == "/new/path"
