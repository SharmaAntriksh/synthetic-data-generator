"""Tests verifying deterministic output (same seed = same result).

These tests confirm the idempotency guarantee: identical config + seed
should produce byte-identical outputs across runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.dimensions.geography import FALLBACK_ROWS, build_dim_geography
from src.engine.config.config_schema import AppConfig


# ===================================================================
# Geography determinism
# ===================================================================

class TestGeographyDeterminism:
    def _cfg(self, currencies):
        return AppConfig.model_validate({
            "geography": {},
            "exchange_rates": {"currencies": currencies},
        })

    def test_same_config_same_output(self):
        cfg = self._cfg(["USD", "EUR", "GBP", "INR"])

        df1 = build_dim_geography(cfg)
        df2 = build_dim_geography(cfg)

        pd.testing.assert_frame_equal(df1, df2)

    def test_currency_order_independent(self):
        """Different ordering of currencies should produce same rows."""
        df1 = build_dim_geography(self._cfg(["USD", "EUR", "GBP"]))
        df2 = build_dim_geography(self._cfg(["GBP", "USD", "EUR"]))

        # Same rows but potentially different order — sort and compare
        df1_sorted = df1.sort_values("GeographyKey").reset_index(drop=True)
        df2_sorted = df2.sort_values("GeographyKey").reset_index(drop=True)

        pd.testing.assert_frame_equal(df1_sorted, df2_sorted)


# ===================================================================
# Numpy RNG determinism
# ===================================================================

class TestRngDeterminism:
    """Verify that numpy's Generator produces identical sequences with same seed."""

    def test_same_seed_same_poisson(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        a = rng1.poisson(1.7, 10000)
        b = rng2.poisson(1.7, 10000)

        np.testing.assert_array_equal(a, b)

    def test_same_seed_same_lognormal(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        a = rng1.lognormal(0.0, 0.12, 10000)
        b = rng2.lognormal(0.0, 0.12, 10000)

        np.testing.assert_array_equal(a, b)

    def test_different_seed_differs(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(99)

        a = rng1.poisson(1.7, 10000)
        b = rng2.poisson(1.7, 10000)

        assert not np.array_equal(a, b)


# ===================================================================
# Pricing helper determinism
# ===================================================================

class TestPricingDeterminism:
    """Verify pricing helpers are deterministic given same inputs."""

    def test_quantize_deterministic(self):
        from src.facts.sales.sales_models.pricing_pipeline import _quantize
        x = np.array([47.3, 123.7, 999.1])
        step = np.array([10.0, 50.0, 100.0])

        r1 = _quantize(x, step, "floor")
        r2 = _quantize(x, step, "floor")

        np.testing.assert_array_equal(r1, r2)

    def test_choose_step_deterministic(self):
        from src.facts.sales.sales_models.pricing_pipeline import _choose_step
        band_max = np.array([100.0, 500.0, 1e18])
        band_step = np.array([5.0, 10.0, 50.0])
        prices = np.array([50.0, 250.0, 1000.0, 42.0, 499.9])

        r1 = _choose_step(prices, band_max, band_step)
        r2 = _choose_step(prices, band_max, band_step)

        np.testing.assert_array_equal(r1, r2)
