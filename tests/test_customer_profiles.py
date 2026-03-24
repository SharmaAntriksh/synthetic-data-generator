"""Tests for customer behavior profiles."""
from __future__ import annotations

import pytest

from src.engine.config.config_schema import ModelsInnerConfig
from src.utils.config_merge import deep_merge
from src.utils.customer_profiles import (
    VALID_PROFILES,
    _PROFILES,
    get_profile_defaults,
    get_profile_names,
    resolve_customer_profile,
)


# ===================================================================
# Profile definitions integrity
# ===================================================================

class TestProfileDefinitions:
    def test_all_expected_profiles_exist(self):
        expected = {"gradual", "steady", "aggressive", "instant"}

        assert expected == VALID_PROFILES

    def test_each_profile_has_required_sections(self):
        for name, profile in _PROFILES.items():
            assert "lifecycle" in profile, f"{name} missing lifecycle"
            assert "demand" in profile, f"{name} missing demand"

    def test_lifecycle_has_expected_keys(self):
        required = {"enable_churn", "base_monthly_churn", "initial_active_customers"}

        for name, profile in _PROFILES.items():
            lc = profile["lifecycle"]
            missing = required - set(lc.keys())
            assert not missing, f"{name} lifecycle missing: {missing}"

    def test_demand_seasonal_spikes_are_valid(self):
        for name, profile in _PROFILES.items():
            spikes = profile["demand"].get("seasonal_spikes", [])
            for spike in spikes:
                assert 1 <= spike["month"] <= 12, f"{name}: invalid spike month"
                assert spike["boost"] > 0, f"{name}: boost must be positive"

    def test_profiles_do_not_contain_macro_demand(self):
        """Macro demand is now owned by trend presets, not customer profiles."""
        for name, profile in _PROFILES.items():
            assert "macro_demand" not in profile, (
                f"{name} still has macro_demand — should be in trend_presets.py"
            )


# ===================================================================
# get_profile_names / get_profile_defaults
# ===================================================================

class TestProfileAPI:
    def test_get_profile_names_sorted(self):
        names = get_profile_names()

        assert names == sorted(names)
        assert len(names) == len(VALID_PROFILES)

    def test_get_profile_defaults_valid(self):
        for name in VALID_PROFILES:
            result = get_profile_defaults(name)
            assert result is not None

    def test_get_profile_defaults_unknown(self):
        assert get_profile_defaults("nonexistent") is None


# ===================================================================
# deep_merge
# ===================================================================

class TestDeepMerge:
    def test_overrides_win(self):
        base = {"a": 1, "b": 2}

        result = deep_merge(base, {"b": 99})

        assert result == {"a": 1, "b": 99}

    def test_nested_dict_merged(self):
        base = {"a": {"x": 1, "y": 2}}

        result = deep_merge(base, {"a": {"y": 99, "z": 3}})

        assert result["a"] == {"x": 1, "y": 99, "z": 3}

    def test_non_dict_override_replaces(self):
        base = {"a": {"x": 1}}

        result = deep_merge(base, {"a": "replaced"})

        assert result["a"] == "replaced"

    def test_new_keys_added(self):
        result = deep_merge({"a": 1}, {"b": 2})

        assert result == {"a": 1, "b": 2}

    def test_base_not_mutated(self):
        base = {"a": 1}

        deep_merge(base, {"a": 2})

        assert base["a"] == 1


# ===================================================================
# resolve_customer_profile
# ===================================================================

class TestResolveCustomerProfile:
    def _cfg(self, d):
        from src.engine.config.config_schema import AppConfig
        return AppConfig.model_validate(d)

    def _models(self, d=None):
        return ModelsInnerConfig.model_validate(d or {})

    def test_unknown_profile_raises(self):
        cfg = self._cfg({"customers": {"profile": "unknown"}})

        with pytest.raises(ValueError, match="Unknown customer profile"):
            resolve_customer_profile(cfg, self._models())

    def test_no_profile_applies_default_steady(self):
        """With Pydantic, customers always has profile='steady' by default."""
        cfg = self._cfg({"customers": {"total_customers": 100}})
        models = self._models()

        result_cfg, result_models = resolve_customer_profile(cfg, models)

        # Steady profile is applied by default
        assert result_cfg.customers.lifecycle is not None

    def test_no_customers_section_returns_unchanged(self):
        """AppConfig always has a customers section, so this test uses a bare cfg."""
        cfg = self._cfg({})
        models = self._models()

        result_cfg, result_models = resolve_customer_profile(cfg, models)

        # Even with empty input, AppConfig has customers with profile="steady"
        assert result_cfg.customers.lifecycle is not None

    def test_gradual_injects_lifecycle(self):
        cfg = self._cfg({"customers": {"profile": "gradual"}})
        models = self._models()

        result_cfg, result_models = resolve_customer_profile(cfg, models)
        lc = result_cfg.customers.lifecycle

        assert lc["enable_churn"] is True
        assert result_models.customers is not None
        assert result_models.macro_demand is not None

    def test_instant_no_churn(self):
        cfg = self._cfg({"customers": {"profile": "instant"}})
        models = self._models()

        result_cfg, _ = resolve_customer_profile(cfg, models)

        assert result_cfg.customers.lifecycle["enable_churn"] is False
        assert result_cfg.customers.lifecycle["base_monthly_churn"] == 0.0

    def test_explicit_override_wins(self):
        cfg = self._cfg({"customers": {
            "profile": "steady",
            "lifecycle": {"base_monthly_churn": 0.99},
        }})
        models = self._models()

        result_cfg, _ = resolve_customer_profile(cfg, models)

        assert result_cfg.customers.lifecycle["base_monthly_churn"] == 0.99

    def test_first_year_pct_applied(self):
        cfg = self._cfg({"customers": {"profile": "gradual", "first_year_pct": 0.50}})
        models = self._models()

        result_cfg, result_models = resolve_customer_profile(cfg, models)

        assert result_cfg.customers.lifecycle["initial_active_customers"] == 0.50

    def test_first_year_pct_out_of_range_raises(self):
        cfg = self._cfg({"customers": {"profile": "gradual", "first_year_pct": 0.01}})

        with pytest.raises(ValueError, match="between 0.05 and 1.0"):
            resolve_customer_profile(cfg, self._models())

    def test_case_insensitive(self):
        cfg = self._cfg({"customers": {"profile": "STEADY"}})
        models = self._models()

        result_cfg, _ = resolve_customer_profile(cfg, models)

        assert result_cfg.customers.lifecycle is not None
