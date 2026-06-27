"""Tests for per-store demand weighting in sales store-sampling (Part 1).

`sales.store_demand_weight` makes the store sampler draw orders in proportion to
a per-store weight, so bigger / higher-revenue stores sell more. None (default)
keeps the legacy uniform behavior.
"""
from __future__ import annotations

import collections

import numpy as np
import pytest

from src.facts.sales.sales_logic.chunk_builder import _weighted_pick


def _dense(weights_by_key: dict[int, float], n: int) -> np.ndarray:
    w = np.ones(n, dtype=np.float64)
    for k, v in weights_by_key.items():
        w[k] = v
    return w


class TestWeightedPick:
    def test_frequencies_track_weights(self):
        rng = np.random.default_rng(0)
        pool = np.array([10, 20, 30], dtype=np.int32)
        w = _dense({10: 1.0, 20: 2.0, 30: 4.0}, 31)
        picks = _weighted_pick(rng, pool, w, 300_000)
        c = collections.Counter(picks.tolist())
        f10, f20, f30 = c[10], c[20], c[30]
        assert abs(f20 / f10 - 2.0) < 0.1, (f10, f20, f30)
        assert abs(f30 / f10 - 4.0) < 0.15, (f10, f20, f30)
        assert set(c) == {10, 20, 30}  # never picks outside the pool

    def test_deterministic(self):
        pool = np.array([1, 2, 3, 4], dtype=np.int32)
        w = _dense({1: 1.0, 2: 3.0, 3: 0.5, 4: 2.0}, 5)
        a = _weighted_pick(np.random.default_rng(7), pool, w, 5000)
        b = _weighted_pick(np.random.default_rng(7), pool, w, 5000)
        assert np.array_equal(a, b)

    def test_equal_weights_are_uniform(self):
        # All-ones weights = the unconfigured default after the sampler collapse;
        # must reproduce uniform sampling (no separate uniform code path).
        rng = np.random.default_rng(2)
        pool = np.array([5, 6, 7], dtype=np.int32)
        w = _dense({5: 1.0, 6: 1.0, 7: 1.0}, 8)
        picks = _weighted_pick(rng, pool, w, 60_000)
        c = collections.Counter(picks.tolist())
        for k in (5, 6, 7):
            assert abs(c[k] / 60_000 - 1 / 3) < 0.03

    def test_zero_weight_total_falls_back_to_uniform(self):
        # all-zero weights -> uniform pick over the pool (no divide-by-zero)
        rng = np.random.default_rng(1)
        pool = np.array([5, 6, 7], dtype=np.int32)
        w = _dense({5: 0.0, 6: 0.0, 7: 0.0}, 8)
        picks = _weighted_pick(rng, pool, w, 60_000)
        c = collections.Counter(picks.tolist())
        # roughly uniform thirds
        for k in (5, 6, 7):
            assert abs(c[k] / 60_000 - 1 / 3) < 0.03

    def test_empty_pool_and_zero_k(self):
        w = _dense({}, 4)
        assert _weighted_pick(np.random.default_rng(0), np.array([], dtype=np.int32), w, 10).size == 0
        assert _weighted_pick(np.random.default_rng(0), np.array([1, 2], dtype=np.int32), w, 0).size == 0


class TestStoreDemandWeightConfig:
    """sales.store_demand_weight is a typed sub-model: valid shapes parse and
    stay dict-accessible; typos in the two top-level keys are rejected rather
    than silently ignored."""

    def test_valid_config_parses_and_is_dict_accessible(self):
        from src.engine.config.config_schema import SalesConfig, StoreDemandWeightConfig

        s = SalesConfig.model_validate(
            {"store_demand_weight": {"by_type": {"Hypermarket": 6.0}, "revenue_class": {"A": 1.5}}}
        )
        assert isinstance(s.store_demand_weight, StoreDemandWeightConfig)
        # _load_stores reads via .get() — must keep working on the model.
        assert s.store_demand_weight.get("by_type") == {"Hypermarket": 6.0}
        assert s.store_demand_weight.get("revenue_class") == {"A": 1.5}

    def test_default_is_none(self):
        from src.engine.config.config_schema import SalesConfig

        assert SalesConfig().store_demand_weight is None

    def test_top_level_key_typo_is_rejected(self):
        from pydantic import ValidationError

        from src.engine.config.config_schema import SalesConfig

        with pytest.raises(ValidationError):
            SalesConfig.model_validate({"store_demand_weight": {"by_typ": {"Hypermarket": 6.0}}})
