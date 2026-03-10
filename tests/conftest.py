"""Shared fixtures for the test suite."""
from __future__ import annotations

import pytest


# ── pytest plugin: visual separators between test classes ──────────


class GroupSeparatorPlugin:
    """Prints a separator line when the test class changes in verbose mode."""

    def __init__(self):
        self._prev_group = None
        self._tw = None
        self._first_items = set()

    def pytest_collection_modifyitems(self, items):
        """Pre-compute which items are the first in their class group."""
        seen = set()
        for item in items:
            parts = item.nodeid.split("::")
            group = "::".join(parts[:2]) if len(parts) >= 3 else parts[0]
            if group not in seen:
                seen.add(group)
                self._first_items.add(item.nodeid)

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_logreport(self, report):
        if report.when != "call" or self._tw is None:
            return
        parts = report.nodeid.split("::")
        group = "::".join(parts[:2]) if len(parts) >= 3 else parts[0]
        if group != self._prev_group:
            self._prev_group = group

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtestloop(self, session):
        reporter = session.config.pluginmanager.get_plugin("terminalreporter")
        if reporter:
            self._tw = reporter._tw

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_logstart(self, nodeid, location):
        if self._tw is None:
            return
        if nodeid in self._first_items:
            parts = nodeid.split("::")
            label = parts[1] if len(parts) >= 3 else parts[0].split("/")[-1]
            self._tw.line()
            self._tw.sep("─", label, bold=True)


def pytest_configure(config):
    config.pluginmanager.register(GroupSeparatorPlugin(), "group-separator")


@pytest.fixture
def minimal_pipeline_cfg() -> dict:
    """Minimal valid config.yaml-style dict that passes normalize_defaults."""
    return {
        "defaults": {
            "seed": 42,
            "dates": {"start": "2023-01-01", "end": "2025-12-31"},
        },
        "scale": {
            "sales_rows": 1000,
            "customers": 100,
            "stores": 5,
        },
        "sales": {
            "file_format": "parquet",
            "total_rows": 1000,
            "skip_order_cols": False,
            "parquet_folder": "./data/parquet_dims",
            "out_folder": "./data/fact_out",
            "delta_output_folder": "./data/fact_out/delta",
        },
        "customers": {
            "total_customers": 100,
            "region_mix": {"US": 60, "EU": 30, "India": 10},
        },
        "exchange_rates": {
            "currencies": ["USD", "EUR", "GBP", "INR", "CAD"],
            "base_currency": "USD",
        },
        "geography": {},
    }


@pytest.fixture
def minimal_models_cfg() -> dict:
    """Minimal valid models.yaml-style dict."""
    return {
        "models": {
            "quantity": {
                "base_poisson_lambda": 1.7,
                "monthly_factors": [0.99, 0.98, 1.00, 1.00, 1.01, 1.02,
                                    1.02, 1.01, 1.00, 1.03, 1.06, 1.05],
                "noise_sigma": 0.12,
                "min_qty": 1,
                "max_qty": 8,
            },
            "pricing": {
                "inflation": {
                    "annual_rate": 0.05,
                    "month_volatility_sigma": 0.012,
                    "factor_clip": [1.00, 1.30],
                },
                "markdown": {
                    "enabled": True,
                    "max_pct_of_price": 0.50,
                    "ladder": [
                        {"kind": "none", "value": 0.0, "weight": 0.35},
                        {"kind": "amt", "value": 50.0, "weight": 0.22},
                        {"kind": "pct", "value": 0.10, "weight": 0.15},
                    ],
                },
                "appearance": {
                    "enabled": True,
                    "unit_price": {
                        "rounding": "floor",
                        "bands": [
                            {"max": 100, "step": 5},
                            {"max": 500, "step": 10},
                            {"max": 2000, "step": 50},
                        ],
                        "endings": [{"value": 0.99, "weight": 1.0}],
                    },
                },
            },
        },
    }
