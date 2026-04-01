"""Tests for the geography dimension."""
from __future__ import annotations

import pytest
import pandas as pd

from src.dimensions.geography import (
    FALLBACK_ROWS,
    OUTPUT_COLS,
    build_dim_geography,
    normalize_geography_config,
)
from src.engine.config.config_schema import AppConfig
from src.exceptions import ConfigError, DimensionError


# ===================================================================
# FALLBACK_ROWS integrity
# ===================================================================

class TestFallbackRows:
    def test_all_rows_are_5_tuples(self):
        for i, row in enumerate(FALLBACK_ROWS):
            assert len(row) == 5, f"Row {i} has {len(row)} elements, expected 5"

    def test_no_duplicate_city_state_country(self):
        combos = [(r[0], r[1], r[2]) for r in FALLBACK_ROWS]

        assert len(combos) == len(set(combos)), "Duplicate city+state+country found"

    def test_all_continents_present(self):
        continents = {r[3] for r in FALLBACK_ROWS}
        expected = {"North America", "Europe", "Asia", "Africa", "Oceania",
                    "South America", "Middle East"}

        assert expected.issubset(continents)

    def test_iso_codes_are_uppercase_3char(self):
        for row in FALLBACK_ROWS:
            iso = row[4]
            assert iso == iso.upper(), f"ISO code {iso} should be uppercase"
            assert len(iso) == 3, f"ISO code {iso} should be 3 characters"


# ===================================================================
# build_dim_geography
# ===================================================================

class TestBuildDimGeography:
    def _cfg(self, currencies):
        return AppConfig.model_validate({
            "geography": {},
            "exchange_rates": {"currencies": currencies},
        })

    def test_basic_output(self):
        cfg = self._cfg(["USD", "EUR", "GBP", "INR", "CAD"])

        df = build_dim_geography(cfg)

        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == OUTPUT_COLS
        assert len(df) > 0

    def test_geography_key_sequential(self):
        cfg = self._cfg(["USD", "EUR"])

        df = build_dim_geography(cfg)
        expected_keys = list(range(1, len(df) + 1))

        assert list(df["GeographyKey"]) == expected_keys

    def test_filters_by_currency(self):
        cfg = self._cfg(["USD"])

        df = build_dim_geography(cfg)

        assert set(df["ISOCode"].unique()) == {"USD"}

    def test_multiple_currencies(self):
        cfg = self._cfg(["USD", "GBP"])

        df = build_dim_geography(cfg)

        assert set(df["ISOCode"].unique()) == {"USD", "GBP"}

    def test_no_matching_currency_raises(self):
        # base_currency (USD) is always included, so use a non-USD base
        # with a currency that has no geography rows
        cfg = AppConfig.model_validate({
            "geography": {},
            "exchange_rates": {"from_currencies": ["XYZ"], "to_currencies": ["XYZ"], "base_currency": "XYZ"},
        })

        with pytest.raises(DimensionError, match="No geography rows remain"):
            build_dim_geography(cfg)

    def test_deterministic(self):
        cfg = self._cfg(["USD", "EUR", "GBP"])

        df1 = build_dim_geography(cfg)
        df2 = build_dim_geography(cfg)

        pd.testing.assert_frame_equal(df1, df2)


# ===================================================================
# normalize_geography_config
# ===================================================================

class TestNormalizeGeographyConfig:
    def test_empty_config_returns_with_override(self):
        result = normalize_geography_config({})

        assert "override" in result
        assert result["override"]["seed"] is None
        assert isinstance(result["override"]["dates"], dict)
        assert "paths" not in result["override"]

    def test_override_seed_coerced_to_int(self):
        result = normalize_geography_config({"override": {"seed": "42"}})

        assert result["override"]["seed"] == 42

    def test_non_dict_override_raises(self):
        with pytest.raises(ConfigError, match="must be a mapping"):
            normalize_geography_config({"override": "bad"})
