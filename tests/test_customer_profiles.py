"""Tests for customer behavior profiles."""
from __future__ import annotations

import pytest

from src.utils.customer_profiles import (
    VALID_PROFILES,
    _deep_merge,
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
            assert "macro_demand" in profile, f"{name} missing macro_demand"

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

    def test_macro_demand_year_level_factors(self):
        for name, profile in _PROFILES.items():
            ylf = profile["macro_demand"]["year_level_factors"]

            assert ylf["mode"] == "once"
            assert len(ylf["values"]) >= 1
            assert ylf["values"][0] == 1.0, f"{name}: first year factor should be 1.0"


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
# _deep_merge
# ===================================================================

class TestDeepMerge:
    def test_overrides_win(self):
        base = {"a": 1, "b": 2}

        result = _deep_merge(base, {"b": 99})

        assert result == {"a": 1, "b": 99}

    def test_nested_dict_merged(self):
        base = {"a": {"x": 1, "y": 2}}

        result = _deep_merge(base, {"a": {"y": 99, "z": 3}})

        assert result["a"] == {"x": 1, "y": 99, "z": 3}

    def test_non_dict_override_replaces(self):
        base = {"a": {"x": 1}}

        result = _deep_merge(base, {"a": "replaced"})

        assert result["a"] == "replaced"

    def test_new_keys_added(self):
        result = _deep_merge({"a": 1}, {"b": 2})

        assert result == {"a": 1, "b": 2}

    def test_base_not_mutated(self):
        base = {"a": 1}

        _deep_merge(base, {"a": 2})

        assert base["a"] == 1


# ===================================================================
# resolve_customer_profile
# ===================================================================

class TestResolveCustomerProfile:
    def test_unknown_profile_raises(self):
        cfg = {"customers": {"profile": "unknown"}}

        with pytest.raises(ValueError, match="Unknown customer profile"):
            resolve_customer_profile(cfg, {})

    def test_no_profile_returns_unchanged(self):
        cfg = {"customers": {"total_customers": 100}}
        models = {"quantity": {}}

        result_cfg, result_models = resolve_customer_profile(cfg, models)

        assert "lifecycle" not in result_cfg["customers"]

    def test_no_customers_section_returns_unchanged(self):
        cfg = {"sales": {}}
        models = {}

        result_cfg, result_models = resolve_customer_profile(cfg, models)

        assert result_cfg is cfg

    def test_gradual_injects_lifecycle(self):
        cfg = {"customers": {"profile": "gradual"}}
        models = {}

        result_cfg, result_models = resolve_customer_profile(cfg, models)
        lc = result_cfg["customers"]["lifecycle"]

        assert lc["enable_churn"] is True
        assert "customers" in result_models
        assert "macro_demand" in result_models

    def test_instant_no_churn(self):
        cfg = {"customers": {"profile": "instant"}}
        models = {}

        result_cfg, _ = resolve_customer_profile(cfg, models)

        assert result_cfg["customers"]["lifecycle"]["enable_churn"] is False
        assert result_cfg["customers"]["lifecycle"]["base_monthly_churn"] == 0.0

    def test_explicit_override_wins(self):
        cfg = {"customers": {
            "profile": "steady",
            "lifecycle": {"base_monthly_churn": 0.99},
        }}
        models = {}

        result_cfg, _ = resolve_customer_profile(cfg, models)

        assert result_cfg["customers"]["lifecycle"]["base_monthly_churn"] == 0.99

    def test_first_year_pct_applied(self):
        cfg = {"customers": {"profile": "gradual", "first_year_pct": 0.50}}
        models = {}

        result_cfg, result_models = resolve_customer_profile(cfg, models)

        assert result_cfg["customers"]["lifecycle"]["initial_active_customers"] == 0.50

    def test_first_year_pct_out_of_range_raises(self):
        cfg = {"customers": {"profile": "gradual", "first_year_pct": 0.01}}

        with pytest.raises(ValueError, match="between 0.05 and 1.0"):
            resolve_customer_profile(cfg, {})

    def test_case_insensitive(self):
        cfg = {"customers": {"profile": "STEADY"}}
        models = {}

        result_cfg, _ = resolve_customer_profile(cfg, models)

        assert "lifecycle" in result_cfg["customers"]
