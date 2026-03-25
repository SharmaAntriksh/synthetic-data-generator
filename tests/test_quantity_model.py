"""Tests for the quantity / basket-size model."""
from __future__ import annotations

import numpy as np
import pytest

from src.exceptions import SalesError
from src.facts.sales.sales_models.quantity_model import (
    _DEFAULTS,
    _load_cfg,
    _reset_cache,
    build_quantity,
)
from src.facts.sales.sales_logic.globals import State


# ===================================================================
# Default constants
# ===================================================================

class TestDefaults:
    def test_monthly_factors_length(self):
        assert len(_DEFAULTS["monthly_factors"]) == 12

    def test_min_qty_at_least_one(self):
        assert _DEFAULTS["min_qty"] >= 1

    def test_max_gte_min(self):
        assert _DEFAULTS["max_qty"] >= _DEFAULTS["min_qty"]

    def test_lambda_positive(self):
        assert _DEFAULTS["base_poisson_lambda"] > 0

    def test_noise_sigma_non_negative(self):
        assert _DEFAULTS["noise_sigma"] >= 0


# ===================================================================
# _load_cfg with mocked State
# ===================================================================

class TestLoadCfg:
    def setup_method(self):
        State.reset()
        _reset_cache()

    def teardown_method(self):
        State.reset()
        _reset_cache()

    def test_loads_defaults_when_no_models(self):
        State.models_cfg = {}

        cfg = _load_cfg()

        assert cfg["base_poisson_lambda"] == _DEFAULTS["base_poisson_lambda"]
        assert cfg["min_qty"] == _DEFAULTS["min_qty"]
        assert cfg["max_qty"] == _DEFAULTS["max_qty"]

    def test_custom_lambda(self):
        State.models_cfg = {"quantity": {"base_poisson_lambda": 3.0}}

        cfg = _load_cfg()

        assert cfg["base_poisson_lambda"] == 3.0

    def test_negative_lambda_raises(self):
        State.models_cfg = {"quantity": {"base_poisson_lambda": -1.0}}

        with pytest.raises(SalesError, match="must be >= 0"):
            _load_cfg()

    def test_wrong_monthly_factors_length_raises(self):
        State.models_cfg = {"quantity": {"monthly_factors": [1.0, 1.0]}}

        with pytest.raises(SalesError, match="12 floats"):
            _load_cfg()

    def test_min_max_swap_if_inverted(self):
        State.models_cfg = {"quantity": {"min_qty": 10, "max_qty": 2}}

        cfg = _load_cfg()

        assert cfg["min_qty"] == 2
        assert cfg["max_qty"] == 10

    def test_legacy_noise_sd_key(self):
        State.models_cfg = {"quantity": {"noise_sd": 0.25}}

        cfg = _load_cfg()

        assert cfg["noise_sigma"] == 0.25


# ===================================================================
# build_quantity
# ===================================================================

class TestBuildQuantity:
    def setup_method(self):
        State.reset()
        _reset_cache()

    def teardown_method(self):
        State.reset()
        _reset_cache()

    def _setup_state(self):
        State.models_cfg = {
            "quantity": {
                "base_poisson_lambda": 1.7,
                "monthly_factors": _DEFAULTS["monthly_factors"],
                "noise_sigma": 0.12,
                "min_qty": 1,
                "max_qty": 8,
            },
        }

    def test_deterministic_with_same_seed(self):
        self._setup_state()
        dates = np.array(["2023-01-15", "2023-06-20", "2023-12-01"] * 100,
                         dtype="datetime64[D]")

        qty1 = build_quantity(np.random.default_rng(42), dates)
        qty2 = build_quantity(np.random.default_rng(42), dates)

        np.testing.assert_array_equal(qty1, qty2)

    def test_different_seed_differs(self):
        self._setup_state()
        dates = np.array(["2023-01-15"] * 1000, dtype="datetime64[D]")

        qty1 = build_quantity(np.random.default_rng(42), dates)
        qty2 = build_quantity(np.random.default_rng(99), dates)

        # Extremely unlikely to be identical with 1000 draws
        assert not np.array_equal(qty1, qty2)

    def test_output_within_bounds(self):
        self._setup_state()
        dates = np.array(["2023-03-15"] * 5000, dtype="datetime64[D]")

        qty = build_quantity(np.random.default_rng(42), dates)

        assert qty.min() >= 1
        assert qty.max() <= 8

    def test_output_dtype_is_int(self):
        self._setup_state()
        dates = np.array(["2023-03-15"] * 100, dtype="datetime64[D]")

        qty = build_quantity(np.random.default_rng(42), dates)

        assert np.issubdtype(qty.dtype, np.integer)

    def test_empty_dates(self):
        self._setup_state()
        dates = np.array([], dtype="datetime64[D]")

        qty = build_quantity(np.random.default_rng(42), dates)

        assert qty.shape == (0,)

    def test_zero_noise(self):
        """With noise_sigma=0, output should still be valid."""
        State.models_cfg = {
            "quantity": {
                "base_poisson_lambda": 2.0,
                "monthly_factors": [1.0] * 12,
                "noise_sigma": 0.0,
                "min_qty": 1,
                "max_qty": 10,
            },
        }
        dates = np.array(["2023-06-15"] * 500, dtype="datetime64[D]")

        qty = build_quantity(np.random.default_rng(42), dates)

        assert qty.min() >= 1
        assert qty.max() <= 10
