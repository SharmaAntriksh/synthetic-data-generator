"""Tests for the dimension version store."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.versioning.version_store import (
    _compute_hash,
    _version_file,
    delete_version,
    load_version,
    save_version,
    should_regenerate,
)


# ===================================================================
# _compute_hash
# ===================================================================

class TestComputeHash:
    def test_deterministic(self):
        h1 = _compute_hash({"a": 1, "b": 2})
        h2 = _compute_hash({"a": 1, "b": 2})

        assert h1 == h2

    def test_key_order_independent(self):
        h1 = _compute_hash({"a": 1, "b": 2})
        h2 = _compute_hash({"b": 2, "a": 1})

        assert h1 == h2

    def test_different_values_differ(self):
        h1 = _compute_hash({"a": 1})
        h2 = _compute_hash({"a": 2})

        assert h1 != h2

    def test_nested_dict(self):
        h1 = _compute_hash({"a": {"b": 1}})
        h2 = _compute_hash({"a": {"b": 1}})

        assert h1 == h2

    def test_string_input(self):
        h = _compute_hash("simple string")

        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex

    def test_list_input(self):
        h = _compute_hash([1, 2, 3])

        assert isinstance(h, str)


# ===================================================================
# should_regenerate
# ===================================================================

class TestShouldRegenerate:
    def test_missing_parquet_returns_true(self, tmp_path):
        missing = tmp_path / "nonexistent.parquet"

        assert should_regenerate("test_dim", {"key": "val"}, missing) is True

    def test_missing_version_file_returns_true(self, tmp_path):
        parquet = tmp_path / "test.parquet"
        parquet.write_text("fake parquet data")

        # No version file exists for this dimension
        assert should_regenerate("__test_no_version__", {"key": "val"}, parquet) is True

    def test_changed_config_returns_true(self, tmp_path):
        parquet = tmp_path / "test.parquet"
        parquet.write_text("fake")
        dim_name = "__test_changed_cfg__"
        save_version(dim_name, {"version": 1}, parquet)

        try:
            assert should_regenerate(dim_name, {"version": 2}, parquet) is True
        finally:
            delete_version(dim_name)

    def test_unchanged_config_returns_false(self, tmp_path):
        parquet = tmp_path / "test.parquet"
        parquet.write_text("fake")
        dim_name = "__test_unchanged_cfg__"
        cfg = {"version": 1, "setting": "abc"}
        save_version(dim_name, cfg, parquet)

        try:
            assert should_regenerate(dim_name, cfg, parquet) is False
        finally:
            delete_version(dim_name)


# ===================================================================
# save_version / load_version / delete_version
# ===================================================================

class TestVersionLifecycle:
    def test_save_and_load(self, tmp_path):
        parquet = tmp_path / "dim.parquet"
        parquet.write_text("fake")
        dim_name = "__test_save_load__"
        save_version(dim_name, {"x": 42}, parquet)

        try:
            loaded = load_version(dim_name)

            assert loaded is not None
            assert "config_hash" in loaded
            assert loaded["config_hash"] == _compute_hash({"x": 42})
        finally:
            delete_version(dim_name)

    def test_delete_existing(self, tmp_path):
        parquet = tmp_path / "dim.parquet"
        parquet.write_text("fake")
        dim_name = "__test_delete__"
        save_version(dim_name, {"x": 1}, parquet)

        assert delete_version(dim_name) is True
        assert load_version(dim_name) is None

    def test_delete_nonexistent(self):
        assert delete_version("__test_nonexistent_dim__") is False

    def test_load_nonexistent(self):
        assert load_version("__test_nonexistent_dim__") is None
