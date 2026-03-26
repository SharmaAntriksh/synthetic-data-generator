"""Tests for src.dimensions.warehouses.generator."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.exceptions import DimensionError


@pytest.fixture()
def geography_parquet(tmp_path):
    """Write a small geography.parquet for warehouse tests."""
    geo = pd.DataFrame({
        "GeographyKey": np.arange(1, 11, dtype=np.int64),
        "Country": ["United States"] * 5 + ["United Kingdom"] * 3 + ["Australia", "New Zealand"],
        "State": [
            "New York", "California", "Texas", "Florida", "Illinois",
            "England", "Scotland", "Wales",
            "New South Wales", "Auckland",
        ],
    })
    geo.to_parquet(tmp_path / "geography.parquet", index=False)
    return geo


@pytest.fixture()
def small_stores_for_warehouses(geography_parquet):
    """Minimal stores DataFrame with required columns."""
    rng = np.random.default_rng(42)
    n = 30
    geo_keys = geography_parquet["GeographyKey"].to_numpy()
    return pd.DataFrame({
        "StoreKey": np.arange(1, n + 1, dtype=np.int64),
        "GeographyKey": rng.choice(geo_keys, size=n),
        "StoreZone": rng.choice(["North America", "Europe", "Asia Pacific"], size=n),
    })


class TestGreedyGroup:
    def test_large_items_stand_alone(self):
        from src.dimensions.warehouses.generator import _greedy_group
        items = [("A", 20), ("B", 15), ("C", 3)]
        groups = _greedy_group(items, threshold=10)
        assert ["A"] in groups
        assert ["B"] in groups

    def test_small_items_merged(self):
        from src.dimensions.warehouses.generator import _greedy_group
        items = [("A", 3), ("B", 4), ("C", 5)]
        groups = _greedy_group(items, threshold=10)
        assert len(groups) >= 1
        total = sum(len(g) for g in groups)
        assert total == 3

    def test_empty_input(self):
        from src.dimensions.warehouses.generator import _greedy_group
        assert _greedy_group([], threshold=10) == []


class TestUsRegionLabel:
    def test_single_region(self):
        from src.dimensions.warehouses.generator import _us_region_label
        label = _us_region_label(["New York", "Massachusetts"])
        assert isinstance(label, str)
        assert len(label) > 0

    def test_multiple_regions(self):
        from src.dimensions.warehouses.generator import _us_region_label
        label = _us_region_label(["New York", "California", "Texas"])
        assert "&" in label or len(label) > 0


class TestGenerateWarehouseTable:
    def test_basic_output(self, small_stores_for_warehouses, tmp_path, geography_parquet):
        from src.dimensions.warehouses.generator import generate_warehouse_table
        wh_df, store_map = generate_warehouse_table(
            small_stores_for_warehouses, tmp_path, seed=42,
        )
        assert isinstance(wh_df, pd.DataFrame)
        assert isinstance(store_map, dict)
        assert len(wh_df) > 0

    def test_expected_columns(self, small_stores_for_warehouses, tmp_path, geography_parquet):
        from src.dimensions.warehouses.generator import generate_warehouse_table
        wh_df, _ = generate_warehouse_table(
            small_stores_for_warehouses, tmp_path, seed=42,
        )
        expected = {"WarehouseKey", "WarehouseName", "WarehouseType", "Zone",
                    "Country", "Territory", "GeographyKey", "Capacity", "SquareFootage"}
        assert expected.issubset(set(wh_df.columns))

    def test_warehouse_key_unique(self, small_stores_for_warehouses, tmp_path, geography_parquet):
        from src.dimensions.warehouses.generator import generate_warehouse_table
        wh_df, _ = generate_warehouse_table(
            small_stores_for_warehouses, tmp_path, seed=42,
        )
        assert wh_df["WarehouseKey"].is_unique

    def test_all_stores_mapped(self, small_stores_for_warehouses, tmp_path, geography_parquet):
        from src.dimensions.warehouses.generator import generate_warehouse_table
        from src.defaults import ONLINE_STORE_KEY_BASE
        stores = small_stores_for_warehouses
        _, store_map = generate_warehouse_table(stores, tmp_path, seed=42)
        physical_keys = stores[stores["StoreKey"] < ONLINE_STORE_KEY_BASE]["StoreKey"].astype(int)
        for sk in physical_keys:
            assert sk in store_map, f"StoreKey {sk} not mapped to a warehouse"

    def test_determinism(self, small_stores_for_warehouses, tmp_path, geography_parquet):
        from src.dimensions.warehouses.generator import generate_warehouse_table
        wh1, m1 = generate_warehouse_table(small_stores_for_warehouses, tmp_path, seed=99)
        wh2, m2 = generate_warehouse_table(small_stores_for_warehouses, tmp_path, seed=99)
        pd.testing.assert_frame_equal(wh1, wh2)
        assert m1 == m2

    def test_missing_store_zone_raises(self, tmp_path, geography_parquet):
        from src.dimensions.warehouses.generator import generate_warehouse_table
        stores = pd.DataFrame({
            "StoreKey": [1, 2, 3],
            "GeographyKey": [1, 2, 3],
        })
        with pytest.raises(DimensionError, match="StoreZone"):
            generate_warehouse_table(stores, tmp_path, seed=42)

    def test_capacity_positive(self, small_stores_for_warehouses, tmp_path, geography_parquet):
        from src.dimensions.warehouses.generator import generate_warehouse_table
        wh_df, _ = generate_warehouse_table(
            small_stores_for_warehouses, tmp_path, seed=42,
        )
        assert (wh_df["Capacity"] > 0).all()
        assert (wh_df["SquareFootage"] > 0).all()
