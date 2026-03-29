"""Trend presets — the single source of business shape.

Each preset defines a complete business story: macro demand trajectory
(growth curve, seasonality, shocks), customer lifecycle (acquisition,
churn), and demand behavior (participation, discovery, seasonal spikes).

Usage in models.yaml:
    models:
      macro_demand:
        trend: steady-growth     # required preset name
        # Optional power-user overrides:
        # seasonality: summer-peak
        # row_share_of_growth: 0.70
        # year_level_factors:
        #   mode: "once"
        #   values: [1.0, 1.1, ...]

The preset is resolved into:
  - models_cfg.macro_demand  (MacroDemandConfig)
  - models_cfg.customers     (CustomersDemandConfig)
  - cfg.customers.lifecycle   (dict)

``resolve_trend_preset()`` is called from pipeline_runner.py and handles
all three injections in one pass.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

from src.engine.config.config_schema import CustomersDemandConfig, MacroDemandConfig
from src.exceptions import ConfigError
from src.utils.config_merge import pydantic_to_explicit_dict, deep_merge
from src.utils.logging_utils import info


# ================================================================
# Seasonality presets (12 monthly multipliers, mean ~ 1.0)
# ================================================================
# Index 0 = January, index 11 = December.

_SEASONALITY_PRESETS: Dict[str, List[float]] = {
    # Standard retail: post-holiday dip Jan/Feb, gradual rise, Nov/Dec spike
    "retail": [
        0.85, 0.80, 0.90, 0.92, 0.95, 0.98,
        1.00, 0.97, 0.95, 1.05, 1.30, 1.33,
    ],
    # Summer-dominant: Jun-Aug peak, winter trough
    "summer-peak": [
        0.85, 0.85, 0.90, 0.95, 1.05, 1.15,
        1.20, 1.18, 1.00, 0.95, 0.96, 0.96,
    ],
    # Q4-heavy: strong Oct-Dec ramp, weak first half
    "q4-heavy": [
        0.82, 0.82, 0.85, 0.88, 0.90, 0.92,
        0.95, 0.95, 1.00, 1.15, 1.35, 1.41,
    ],
    # Back-to-school: Jul-Sep spike, mild holiday bump
    "back-to-school": [
        0.90, 0.85, 0.88, 0.92, 0.95, 1.00,
        1.15, 1.25, 1.10, 0.95, 0.98, 1.07,
    ],
    # Flat: no seasonal pattern
    "flat": [
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    ],
}

VALID_SEASONALITY_PRESETS = frozenset(_SEASONALITY_PRESETS.keys())


# ================================================================
# Trend presets
# ================================================================
# Each preset is a complete business shape definition:
#   - macro:     year_level_factors, noise, shocks, seasonality
#   - lifecycle: acquisition curve, churn, initial customer spread
#   - customers: participation ratios, discovery, seasonal spikes
#
# year_level_factors are designed for 10-20 years; shorter/longer date
# ranges work via the existing "once" mode clamping logic.

_TREND_PRESETS: Dict[str, Dict[str, Any]] = {

    # ------------------------------------------------------------------
    # steady-growth — Mature retailer, 3-5% annual growth, predictable.
    # ------------------------------------------------------------------
    "steady-growth": {
        "base_level": 1.0,
        "yearly_growth": 0.0,
        "year_level_factors": {
            "mode": "once",
            "values": [1.0, 1.06, 1.12, 1.18, 1.24, 1.28, 1.34, 1.40, 1.48, 1.55,
                        1.63, 1.71, 1.80, 1.89, 1.99, 2.09, 2.19, 2.30, 2.42, 2.54],
        },
        "monthly_seasonality": "flat",
        "noise_std": 0.06,
        "shock_probability": 0.03,
        "shock_impact": [-0.08, 0.05],
        "row_share_of_growth": 0.70,
        "early_month_cap": {"enabled": False},
        "eligible_blend": 0.0,
        "bootstrap_months": 1,
        "max_distinct_ratio": 0.70,
        "min_distinct_customers": 250,
        "lifecycle": {
            "enable_churn": True,
            "base_monthly_churn": 0.003,
            "min_tenure_months": 12,
            "initial_active_customers": 0.50,
            "initial_spread_months": 36,
            "acquisition_curve": "logistic",
            "acquisition_params": {"midpoint": 0.45, "steepness": 4.0},
        },
        "customers": {
            "distinct_ratio": 0.65,
            "new_customer_share": 0.12,
            "max_new_fraction_per_month": 0.06,
            "cycle_amplitude": 0.15,
            "discovery_shape": 0.0,
            "participation_noise": 0.10,
            "seasonal_spikes": [
                {"month": 11, "boost": 0.25},
                {"month": 12, "boost": 0.20},
            ],
        },
    },

    # ------------------------------------------------------------------
    # strong-growth — Inverse of decline. Steady acceleration upward,
    #                 visible even after pricing inflation.
    # ------------------------------------------------------------------
    "strong-growth": {
        "base_level": 1.0,
        "yearly_growth": 0.0,
        "year_level_factors": {
            "mode": "once",
            "values": [0.16, 0.20, 0.25, 0.30, 0.37, 0.45, 0.55, 0.67, 0.82, 1.0,
                        1.22, 1.49, 1.82, 2.22, 2.71, 3.31, 4.04, 4.93, 6.02, 7.35],
        },
        "monthly_seasonality": "flat",
        "noise_std": 0.05,
        "shock_probability": 0.06,
        "shock_impact": [-0.05, 0.20],
        "row_share_of_growth": 1.0,
        "early_month_cap": {
            "enabled": True,
            "max_rows_per_customer": 20,
            "redistribute_excess": True,
        },
        "eligible_blend": 0.0,
        "bootstrap_months": 8,
        "max_distinct_ratio": 0.60,
        "min_distinct_customers": 150,
        "lifecycle": {
            "enable_churn": True,
            "base_monthly_churn": 0.015,
            "min_tenure_months": 6,
            "initial_active_customers": 0.15,
            "initial_spread_months": 48,
            "acquisition_curve": "logistic",
            "acquisition_params": {"midpoint": 0.55, "steepness": 6.0},
        },
        "customers": {
            "distinct_ratio": 0.50,
            "new_customer_share": 0.15,
            "max_new_fraction_per_month": 0.08,
            "cycle_amplitude": 0.35,
            "discovery_shape": 0.2,
            "participation_noise": 0.20,
            "seasonal_spikes": [
                {"month": 3,  "boost": 0.25},
                {"month": 7,  "boost": 0.20},
                {"month": 11, "boost": 0.50},
                {"month": 12, "boost": 0.40},
            ],
        },
    },

    # ------------------------------------------------------------------
    # gradual-growth — Startup maturing into stable business. S-curve
    #                  trajectory with visible acceleration then leveling.
    # ------------------------------------------------------------------
    "gradual-growth": {
        "base_level": 1.0,
        "yearly_growth": 0.0,
        "year_level_factors": {
            "mode": "once",
            "values": [1.0, 1.15, 1.35, 1.80, 2.00, 1.85, 2.10, 2.40, 2.30, 2.70,
                        2.55, 2.85, 3.10, 2.95, 3.25, 3.50, 3.35, 3.65, 3.90, 3.80],
        },
        "monthly_seasonality": "flat",
        "noise_std": 0.06,
        "shock_probability": 0.08,
        "shock_impact": [-0.15, 0.20],
        "row_share_of_growth": 0.70,
        "early_month_cap": {
            "enabled": True,
            "max_rows_per_customer": 25,
            "redistribute_excess": True,
        },
        "eligible_blend": 0.0,
        "bootstrap_months": 6,
        "max_distinct_ratio": 0.65,
        "min_distinct_customers": 200,
        "lifecycle": {
            "enable_churn": True,
            "base_monthly_churn": 0.012,
            "min_tenure_months": 6,
            "initial_active_customers": 0.30,
            "initial_spread_months": 36,
            "acquisition_curve": "logistic",
            "acquisition_params": {"midpoint": 0.45, "steepness": 5.0},
        },
        "customers": {
            "distinct_ratio": 0.55,
            "new_customer_share": 0.08,
            "max_new_fraction_per_month": 0.05,
            "cycle_amplitude": 0.40,
            "discovery_shape": 0.0,
            "participation_noise": 0.25,
            "seasonal_spikes": [
                {"month": 3,  "boost": 0.30},
                {"month": 7,  "boost": 0.25},
                {"month": 9,  "boost": 0.20},
                {"month": 11, "boost": 0.65},
                {"month": 12, "boost": 0.45},
            ],
        },
    },

    # ------------------------------------------------------------------
    # hockey-stick — Viral growth startup. Explosive years 4-6.
    # ------------------------------------------------------------------
    "hockey-stick": {
        "base_level": 1.0,
        "yearly_growth": 0.0,
        "year_level_factors": {
            "mode": "once",
            "values": [1.0, 1.15, 1.40, 2.00, 3.20, 4.80, 5.50, 6.00, 6.30, 6.80,
                        7.50, 8.30, 9.20, 10.2, 11.3, 12.5, 13.8, 15.3, 17.0, 18.8],
        },
        "monthly_seasonality": "flat",
        "noise_std": 0.07,
        "shock_probability": 0.10,
        "shock_impact": [-0.15, 0.25],
        "row_share_of_growth": 1.0,
        "early_month_cap": {
            "enabled": True,
            "max_rows_per_customer": 20,
            "redistribute_excess": True,
        },
        "eligible_blend": 0.0,
        "bootstrap_months": 8,
        "max_distinct_ratio": 0.55,
        "min_distinct_customers": 150,
        "lifecycle": {
            "enable_churn": True,
            "base_monthly_churn": 0.012,
            "min_tenure_months": 4,
            "initial_active_customers": 0.15,
            "initial_spread_months": 36,
            "acquisition_curve": "logistic",
            "acquisition_params": {"midpoint": 0.45, "steepness": 6.0},
        },
        "customers": {
            "distinct_ratio": 0.45,
            "new_customer_share": 0.15,
            "max_new_fraction_per_month": 0.08,
            "cycle_amplitude": 0.50,
            "discovery_shape": 0.3,
            "participation_noise": 0.30,
            "seasonal_spikes": [
                {"month": 3,  "boost": 0.40},
                {"month": 7,  "boost": 0.35},
                {"month": 9,  "boost": 0.30},
                {"month": 11, "boost": 0.80},
                {"month": 12, "boost": 0.60},
            ],
        },
    },

    # ------------------------------------------------------------------
    # decline — Dying brand, steady year-over-year erosion.
    #           Factors compensate for pricing inflation so decline
    #           is visible in revenue charts.
    # ------------------------------------------------------------------
    "decline": {
        "base_level": 1.0,
        "yearly_growth": 0.0,
        "year_level_factors": {
            "mode": "once",
            "values": [1.0, 0.82, 0.67, 0.55, 0.45, 0.37, 0.30, 0.25, 0.20, 0.16,
                        0.13, 0.11, 0.09, 0.07, 0.06, 0.05, 0.04, 0.03, 0.025, 0.02],
        },
        "monthly_seasonality": "flat",
        "noise_std": 0.05,
        "shock_probability": 0.06,
        "shock_impact": [-0.20, 0.05],
        "row_share_of_growth": 1.0,
        "early_month_cap": {"enabled": False},
        "eligible_blend": 0.0,
        "bootstrap_months": 3,
        "max_distinct_ratio": 0.75,
        "min_distinct_customers": 200,
        "lifecycle": {
            "enable_churn": True,
            "base_monthly_churn": 0.045,
            "min_tenure_months": 6,
            "initial_active_customers": 0.90,
            "initial_spread_months": 0,
            "acquisition_curve": "logistic",
            "acquisition_params": {"midpoint": 0.70, "steepness": 3.0},
        },
        "customers": {
            "distinct_ratio": 0.60,
            "new_customer_share": 0.03,
            "max_new_fraction_per_month": 0.02,
            "cycle_amplitude": 0.20,
            "discovery_shape": -0.3,
            "participation_noise": 0.15,
            "seasonal_spikes": [
                {"month": 11, "boost": 0.30},
                {"month": 12, "boost": 0.20},
            ],
        },
    },

    # ------------------------------------------------------------------
    # boom-and-bust — Rapid rise then collapse (fad product).
    # ------------------------------------------------------------------
    "boom-and-bust": {
        "base_level": 1.0,
        "yearly_growth": 0.0,
        "year_level_factors": {
            "mode": "repeat",
            "values": [1.0, 1.80, 3.50, 4.50, 2.50, 1.20, 0.65, 0.45, 0.35, 0.30],
        },
        "monthly_seasonality": "flat",
        "noise_std": 0.08,
        "shock_probability": 0.12,
        "shock_impact": [-0.25, 0.30],
        "row_share_of_growth": 1.0,
        "early_month_cap": {
            "enabled": True,
            "max_rows_per_customer": 25,
            "redistribute_excess": True,
        },
        "eligible_blend": 0.0,
        "bootstrap_months": 6,
        "max_distinct_ratio": 0.60,
        "min_distinct_customers": 200,
        "lifecycle": {
            "enable_churn": True,
            "base_monthly_churn": 0.040,
            "min_tenure_months": 3,
            "initial_active_customers": 0.35,
            "initial_spread_months": 12,
            "acquisition_curve": "logistic",
            "acquisition_params": {"midpoint": 0.25, "steepness": 10.0},
        },
        "customers": {
            "distinct_ratio": 0.50,
            "new_customer_share": 0.12,
            "max_new_fraction_per_month": 0.07,
            "cycle_amplitude": 0.45,
            "discovery_shape": 0.0,
            "participation_noise": 0.25,
            "seasonal_spikes": [
                {"month": 11, "boost": 0.40},
                {"month": 12, "boost": 0.30},
            ],
        },
    },

    # ------------------------------------------------------------------
    # recession-recovery — U-shape: deep decline years 2-4, recovery 5-8.
    #           Factors compensate for inflation so the dip is visible.
    # ------------------------------------------------------------------
    "recession-recovery": {
        "base_level": 1.0,
        "yearly_growth": 0.0,
        "year_level_factors": {
            "mode": "repeat",
            "values": [1.0, 0.72, 0.52, 0.65, 0.85, 0.80, 0.73, 0.66, 0.60, 0.55],
        },
        "monthly_seasonality": "flat",
        "noise_std": 0.06,
        "shock_probability": 0.08,
        "shock_impact": [-0.18, 0.10],
        "row_share_of_growth": 1.0,
        "early_month_cap": {"enabled": False},
        "eligible_blend": 0.0,
        "bootstrap_months": 3,
        "max_distinct_ratio": 0.70,
        "min_distinct_customers": 250,
        "lifecycle": {
            "enable_churn": True,
            "base_monthly_churn": 0.010,
            "min_tenure_months": 8,
            "initial_active_customers": 0.55,
            "initial_spread_months": 30,
            "acquisition_curve": "logistic",
            "acquisition_params": {"midpoint": 0.45, "steepness": 4.0},
        },
        "customers": {
            "distinct_ratio": 0.60,
            "new_customer_share": 0.06,
            "max_new_fraction_per_month": 0.04,
            "cycle_amplitude": 0.25,
            "discovery_shape": 0.0,
            "participation_noise": 0.15,
            "seasonal_spikes": [
                {"month": 11, "boost": 0.25},
                {"month": 12, "boost": 0.20},
            ],
        },
    },

    # ------------------------------------------------------------------
    # seasonal-dominant — Nearly flat trend, strong seasonal swings.
    #                     THIS is the only preset where seasonality
    #                     IS the story. ~2x peak-to-trough ratio.
    # ------------------------------------------------------------------
    "seasonal-dominant": {
        "base_level": 1.0,
        "yearly_growth": 0.0,
        "year_level_factors": {
            "mode": "repeat",
            "values": [1.0, 1.08, 0.92, 1.12, 0.96, 0.88, 1.05, 0.94, 1.10, 0.98],
        },
        # Smooth wave peaking in Nov, trough in Feb
        # Dec wraps gently toward Jan (1.12->0.94) to avoid sharp V-dips
        "monthly_seasonality": [
            0.94, 0.88, 0.90, 0.93, 0.97, 1.00,
            1.02, 1.01, 0.98, 1.06, 1.16, 1.12,
        ],
        "noise_std": 0.10,
        "shock_probability": 0.04,
        "shock_impact": [-0.10, 0.08],
        "row_share_of_growth": 1.0,
        "early_month_cap": {"enabled": False},
        "eligible_blend": 0.0,
        "bootstrap_months": 3,
        "max_distinct_ratio": 0.70,
        "min_distinct_customers": 250,
        "lifecycle": {
            "enable_churn": True,
            "base_monthly_churn": 0.005,
            "min_tenure_months": 10,
            "initial_active_customers": 0.60,
            "initial_spread_months": 24,
            "acquisition_curve": "logistic",
            "acquisition_params": {"midpoint": 0.40, "steepness": 4.0},
        },
        "customers": {
            "distinct_ratio": 0.65,
            "new_customer_share": 0.08,
            "max_new_fraction_per_month": 0.05,
            "cycle_amplitude": 0.30,
            "discovery_shape": 0.0,
            "participation_noise": 0.15,
            "seasonal_spikes": [
                {"month": 3,  "boost": 0.20},
                {"month": 7,  "boost": 0.25},
                {"month": 11, "boost": 0.55},
                {"month": 12, "boost": 0.40},
            ],
        },
    },

    # ------------------------------------------------------------------
    # plateau — Growth for 4 years, then revenue flatlines.
    #           Plateau-phase factors DECLINE to offset inflation so
    #           revenue stays flat rather than drifting up.
    # ------------------------------------------------------------------
    "plateau": {
        "base_level": 1.0,
        "yearly_growth": 0.0,
        "year_level_factors": {
            "mode": "repeat",
            "values": [1.0, 1.18, 1.40, 1.65, 1.51, 1.37, 1.24, 1.13, 1.03, 0.93],
        },
        "monthly_seasonality": "flat",
        "noise_std": 0.05,
        "shock_probability": 0.05,
        "shock_impact": [-0.12, 0.08],
        "row_share_of_growth": 0.70,
        "early_month_cap": {
            "enabled": True,
            "max_rows_per_customer": 25,
            "redistribute_excess": True,
        },
        "eligible_blend": 0.0,
        "bootstrap_months": 6,
        "max_distinct_ratio": 0.65,
        "min_distinct_customers": 200,
        "lifecycle": {
            "enable_churn": True,
            "base_monthly_churn": 0.010,
            "min_tenure_months": 6,
            "initial_active_customers": 0.35,
            "initial_spread_months": 30,
            "acquisition_curve": "logistic",
            "acquisition_params": {"midpoint": 0.35, "steepness": 5.0},
        },
        "customers": {
            "distinct_ratio": 0.55,
            "new_customer_share": 0.08,
            "max_new_fraction_per_month": 0.05,
            "cycle_amplitude": 0.25,
            "discovery_shape": 0.0,
            "participation_noise": 0.15,
            "seasonal_spikes": [
                {"month": 11, "boost": 0.30},
                {"month": 12, "boost": 0.25},
            ],
        },
    },

    # ------------------------------------------------------------------
    # volatile — Wild revenue swings year to year.
    #            Factors compensate for inflation so down-years actually
    #            show as dips, not just slower growth.
    # ------------------------------------------------------------------
    "volatile": {
        "base_level": 1.0,
        "yearly_growth": 0.0,
        "year_level_factors": {
            "mode": "repeat",
            "values": [1.0, 1.20, 0.70, 1.25, 0.68, 1.15, 0.73, 0.52, 1.05, 0.72],
        },
        "monthly_seasonality": "flat",
        "noise_std": 0.14,
        "shock_probability": 0.15,
        "shock_impact": [-0.30, 0.35],
        "row_share_of_growth": 1.0,
        "early_month_cap": {"enabled": False},
        "eligible_blend": 0.0,
        "bootstrap_months": 3,
        "max_distinct_ratio": 0.70,
        "min_distinct_customers": 250,
        "lifecycle": {
            "enable_churn": True,
            "base_monthly_churn": 0.010,
            "min_tenure_months": 6,
            "initial_active_customers": 0.55,
            "initial_spread_months": 30,
            "acquisition_curve": "logistic",
            "acquisition_params": {"midpoint": 0.40, "steepness": 5.0},
        },
        "customers": {
            "distinct_ratio": 0.60,
            "new_customer_share": 0.10,
            "max_new_fraction_per_month": 0.06,
            "cycle_amplitude": 0.40,
            "discovery_shape": 0.0,
            "participation_noise": 0.30,
            "seasonal_spikes": [
                {"month": 11, "boost": 0.30},
                {"month": 12, "boost": 0.25},
            ],
        },
    },

    # ------------------------------------------------------------------
    # double-dip — Two distinct revenue downturns with partial recovery.
    #              Factors compensate for inflation so both dips show.
    # ------------------------------------------------------------------
    "double-dip": {
        "base_level": 1.0,
        "yearly_growth": 0.0,
        "year_level_factors": {
            "mode": "repeat",
            "values": [1.0, 0.75, 0.54, 0.60, 0.62, 0.45, 0.33, 0.36, 0.38, 0.39],
        },
        "monthly_seasonality": "flat",
        "noise_std": 0.06,
        "shock_probability": 0.10,
        "shock_impact": [-0.22, 0.12],
        "row_share_of_growth": 1.0,
        "early_month_cap": {"enabled": False},
        "eligible_blend": 0.0,
        "bootstrap_months": 3,
        "max_distinct_ratio": 0.70,
        "min_distinct_customers": 250,
        "lifecycle": {
            "enable_churn": True,
            "base_monthly_churn": 0.030,
            "min_tenure_months": 5,
            "initial_active_customers": 0.85,
            "initial_spread_months": 0,
            "acquisition_curve": "logistic",
            "acquisition_params": {"midpoint": 0.65, "steepness": 3.0},
        },
        "customers": {
            "distinct_ratio": 0.58,
            "new_customer_share": 0.04,
            "max_new_fraction_per_month": 0.03,
            "cycle_amplitude": 0.20,
            "discovery_shape": -0.2,
            "participation_noise": 0.15,
            "seasonal_spikes": [
                {"month": 11, "boost": 0.25},
                {"month": 12, "boost": 0.20},
            ],
        },
    },

    # ------------------------------------------------------------------
    # new-market-entry — Near-zero start with a long "finding PMF" phase
    #                    (years 1-4), then accelerating growth once traction
    #                    hits. Different from hockey-stick: the flat start
    #                    is longer and the ramp is more gradual.
    # ------------------------------------------------------------------
    "new-market-entry": {
        "base_level": 1.0,
        "yearly_growth": 0.0,
        "year_level_factors": {
            "mode": "once",
            "values": [0.05, 0.06, 0.08, 0.12, 0.20, 0.35, 0.55, 0.80, 1.10, 1.45,
                        1.85, 2.30, 2.80, 3.35, 3.95, 4.60, 5.30, 6.05, 6.85, 7.70],
        },
        "monthly_seasonality": "flat",
        "noise_std": 0.08,
        "shock_probability": 0.08,
        "shock_impact": [-0.10, 0.25],
        "row_share_of_growth": 1.0,
        "early_month_cap": {
            "enabled": True,
            "max_rows_per_customer": 15,
            "redistribute_excess": True,
        },
        "eligible_blend": 0.0,
        "bootstrap_months": 10,
        "max_distinct_ratio": 0.50,
        "min_distinct_customers": 100,
        "lifecycle": {
            "enable_churn": True,
            "base_monthly_churn": 0.015,
            "min_tenure_months": 3,
            "initial_active_customers": 0.10,
            "initial_spread_months": 48,
            "acquisition_curve": "logistic",
            "acquisition_params": {"midpoint": 0.55, "steepness": 6.0},
        },
        "customers": {
            "distinct_ratio": 0.45,
            "new_customer_share": 0.18,
            "max_new_fraction_per_month": 0.10,
            "cycle_amplitude": 0.35,
            "discovery_shape": 0.3,
            "participation_noise": 0.25,
            "seasonal_spikes": [
                {"month": 11, "boost": 0.30},
                {"month": 12, "boost": 0.25},
            ],
        },
    },

    # ------------------------------------------------------------------
    # seasonal-with-growth — Visible retail seasonality combined with a
    #                        clear upward trend. The middle ground between
    #                        seasonal-dominant (flat trend) and growth
    #                        presets (muted seasons).
    # ------------------------------------------------------------------
    "seasonal-with-growth": {
        "base_level": 1.0,
        "yearly_growth": 0.0,
        "year_level_factors": {
            "mode": "repeat",
            "values": [1.0, 1.18, 1.38, 1.60, 1.85, 2.12, 2.42, 2.75, 3.10, 3.50],
        },
        "monthly_seasonality": "retail",
        "noise_std": 0.05,
        "shock_probability": 0.05,
        "shock_impact": [-0.10, 0.12],
        "row_share_of_growth": 0.70,
        "early_month_cap": {
            "enabled": True,
            "max_rows_per_customer": 25,
            "redistribute_excess": True,
        },
        "eligible_blend": 0.0,
        "bootstrap_months": 6,
        "max_distinct_ratio": 0.65,
        "min_distinct_customers": 200,
        "lifecycle": {
            "enable_churn": True,
            "base_monthly_churn": 0.010,
            "min_tenure_months": 6,
            "initial_active_customers": 0.30,
            "initial_spread_months": 36,
            "acquisition_curve": "logistic",
            "acquisition_params": {"midpoint": 0.50, "steepness": 5.0},
        },
        "customers": {
            "distinct_ratio": 0.55,
            "new_customer_share": 0.10,
            "max_new_fraction_per_month": 0.06,
            "cycle_amplitude": 0.35,
            "discovery_shape": 0.0,
            "participation_noise": 0.20,
            "seasonal_spikes": [
                {"month": 3,  "boost": 0.25},
                {"month": 7,  "boost": 0.20},
                {"month": 9,  "boost": 0.15},
                {"month": 11, "boost": 0.60},
                {"month": 12, "boost": 0.45},
            ],
        },
    },

    # ------------------------------------------------------------------
    # slow-decline — Gentle, steady erosion over time. Softer than
    #                "decline" — more like a brand slowly losing relevance
    #                rather than collapsing. ~10%/yr revenue drop.
    # ------------------------------------------------------------------
    "slow-decline": {
        "base_level": 1.0,
        "yearly_growth": 0.0,
        "year_level_factors": {
            "mode": "repeat",
            "values": [1.0, 0.91, 0.83, 0.75, 0.68, 0.62, 0.56, 0.51, 0.47, 0.42],
        },
        "monthly_seasonality": "flat",
        "noise_std": 0.04,
        "shock_probability": 0.04,
        "shock_impact": [-0.08, 0.06],
        "row_share_of_growth": 1.0,
        "early_month_cap": {"enabled": False},
        "eligible_blend": 0.0,
        "bootstrap_months": 3,
        "max_distinct_ratio": 0.70,
        "min_distinct_customers": 250,
        "lifecycle": {
            "enable_churn": True,
            "base_monthly_churn": 0.030,
            "min_tenure_months": 8,
            "initial_active_customers": 0.92,
            "initial_spread_months": 0,
            "acquisition_curve": "logistic",
            "acquisition_params": {"midpoint": 0.60, "steepness": 3.0},
        },
        "customers": {
            "distinct_ratio": 0.62,
            "new_customer_share": 0.05,
            "max_new_fraction_per_month": 0.03,
            "cycle_amplitude": 0.18,
            "discovery_shape": -0.15,
            "participation_noise": 0.12,
            "seasonal_spikes": [
                {"month": 11, "boost": 0.25},
                {"month": 12, "boost": 0.18},
            ],
        },
    },

    # ------------------------------------------------------------------
    # stagnation — Truly flat revenue from day one. Factors stay at 1.0
    #              because inflation naturally offsets the baseline demand
    #              decay from customer lifecycle churn, producing flat
    #              revenue. No growth, no decline.
    # ------------------------------------------------------------------
    "stagnation": {
        "base_level": 1.0,
        "yearly_growth": 0.0,
        "year_level_factors": {
            "mode": "repeat",
            "values": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        },
        "monthly_seasonality": "flat",
        "noise_std": 0.04,
        "shock_probability": 0.04,
        "shock_impact": [-0.08, 0.06],
        "row_share_of_growth": 1.0,
        "early_month_cap": {"enabled": False},
        "eligible_blend": 0.0,
        "bootstrap_months": 3,
        "max_distinct_ratio": 0.70,
        "min_distinct_customers": 250,
        "lifecycle": {
            "enable_churn": False,
            "base_monthly_churn": 0.0,
            "min_tenure_months": 0,
            "initial_active_customers": 1.0,
            "initial_spread_months": 0,
            "acquisition_curve": "logistic",
            "acquisition_params": {"midpoint": 0.50, "steepness": 2.0},
        },
        "customers": {
            "distinct_ratio": 0.80,
            "new_customer_share": 0.0,
            "max_new_fraction_per_month": 0.0,
            "cycle_amplitude": 0.0,
            "discovery_shape": 0.0,
            "participation_noise": 0.02,
            "seasonal_spikes": [],
        },
    },
}

VALID_TRENDS = frozenset(_TREND_PRESETS.keys())

# Backward compatibility: map deprecated customers.profile names to trend presets.
_PROFILE_TO_TREND: Dict[str, Optional[str]] = {
    "steady": "steady-growth",
    "gradual": "gradual-growth",
    "aggressive": "hockey-stick",
    "instant": "stagnation",
    "decline": "decline",
}


# ================================================================
# Helpers
# ================================================================

def _resolve_seasonality(preset: dict, overrides: dict) -> list:
    """Determine the final 12-value monthly seasonality array.

    Priority: explicit overrides > preset default.
    Accepts either a string name or a list of 12 floats.
    """
    # Check for user override first
    user_seasonality = overrides.get("seasonality")
    if user_seasonality is not None:
        raw = user_seasonality
    else:
        raw = preset.get("monthly_seasonality", "flat")

    if isinstance(raw, str):
        name = raw.strip().lower()
        if name not in _SEASONALITY_PRESETS:
            raise ConfigError(
                f"Unknown seasonality preset: {name!r}. "
                f"Valid presets: {', '.join(sorted(VALID_SEASONALITY_PRESETS))}"
            )
        return list(_SEASONALITY_PRESETS[name])

    if isinstance(raw, (list, tuple)):
        if len(raw) != 12:
            raise ConfigError(
                f"monthly_seasonality must have exactly 12 values, got {len(raw)}"
            )
        return [float(v) for v in raw]

    raise ConfigError(
        f"seasonality must be a preset name or list of 12 floats, got {type(raw).__name__}"
    )


def _apply_first_year_pct(fyr: float, lifecycle: dict, demand: CustomersDemandConfig) -> None:
    """Derive acquisition knobs from a single first_year_pct value.

    fyr = 0.30 means 30% of customers exist in year 1, 70% acquired later.
    fyr = 0.80 means 80% in year 1, only 20% acquired later.
    """
    remaining = 1.0 - fyr

    lifecycle["initial_active_customers"] = fyr
    lifecycle["initial_spread_months"] = 12

    demand.new_customer_share = round(0.04 + remaining * 0.12, 3)
    demand.max_new_fraction_per_month = round(0.02 + remaining * 0.06, 3)


# ================================================================
# Public API
# ================================================================

def get_trend_names() -> list[str]:
    """Return sorted list of available trend preset names."""
    return sorted(_TREND_PRESETS.keys())


def get_seasonality_names() -> list[str]:
    """Return sorted list of available seasonality preset names."""
    return sorted(_SEASONALITY_PRESETS.keys())


def get_trend_defaults(name: str) -> dict:
    """Return a copy of the named trend preset dict."""
    if name not in _TREND_PRESETS:
        raise ConfigError(f"Unknown trend preset: {name!r}")
    return copy.deepcopy(_TREND_PRESETS[name])


def resolve_trend_preset(
    models_cfg,
    *,
    cfg=None,
    profile_name: str | None = None,
) -> None:
    """Resolve trend preset and inject macro demand, lifecycle, and customers.

    Reads ``models_cfg.macro_demand.trend`` for the preset name.
    Falls back to ``cfg.customers.profile`` (deprecated) via
    ``_PROFILE_TO_TREND`` mapping.

    Injects into:
      - ``models_cfg.macro_demand``  — MacroDemandConfig
      - ``models_cfg.customers``     — CustomersDemandConfig
      - ``cfg.customers.lifecycle``  — dict

    Explicit YAML values override preset defaults via deep merge.

    Mutates *models_cfg* and *cfg* in place.
    """
    macro = getattr(models_cfg, "macro_demand", None)
    if macro is None:
        return

    # Extract explicitly-set user overrides (excludes Pydantic defaults)
    if hasattr(macro, "model_fields_set"):
        overrides = pydantic_to_explicit_dict(macro)
    else:
        overrides = dict(macro) if isinstance(macro, Mapping) else {}

    trend_name = overrides.get("trend")

    # Backward compat: map deprecated customers.profile to trend preset
    if trend_name is None and profile_name is not None:
        profile_key = str(profile_name).strip().lower()
        auto_trend = _PROFILE_TO_TREND.get(profile_key)
        if auto_trend is not None:
            trend_name = auto_trend
            info(f"customers.profile='{profile_key}' mapped to trend '{auto_trend}' "
                 f"(profile is deprecated, use macro_demand.trend instead)")

    if trend_name is None:
        return

    trend_name = str(trend_name).strip().lower()
    if trend_name not in _TREND_PRESETS:
        raise ConfigError(
            f"Unknown trend preset: {trend_name!r}. "
            f"Valid presets: {', '.join(sorted(VALID_TRENDS))}"
        )

    preset = copy.deepcopy(_TREND_PRESETS[trend_name])

    # --- Extract lifecycle and customers sub-dicts before macro validation ---
    preset_lifecycle = preset.pop("lifecycle", {})
    preset_customers = preset.pop("customers", {})

    # Resolve seasonality to a concrete 12-value list
    preset["monthly_seasonality"] = _resolve_seasonality(preset, overrides)

    # Remove intermediate keys not part of MacroDemandConfig fields
    overrides.pop("trend", None)
    overrides.pop("seasonality", None)

    # Deep-merge: preset as base, explicit YAML values on top
    merged = deep_merge(preset, overrides)

    # Ensure yearly_growth is 0 when using presets (trajectory via factors only)
    if "yearly_growth" not in overrides:
        merged["yearly_growth"] = 0.0

    # Ensure seasonality_amplitude is 0 when using monthly_seasonality
    if "seasonality_amplitude" not in overrides:
        merged["seasonality_amplitude"] = 0.0

    models_cfg.macro_demand = MacroDemandConfig.model_validate(merged)

    # --- Inject lifecycle into cfg.customers.lifecycle ---
    if cfg is not None:
        cust_cfg = getattr(cfg, "customers", None)
        if cust_cfg is not None:
            existing_lifecycle = getattr(cust_cfg, "lifecycle", None) or {}
            if hasattr(existing_lifecycle, "model_fields_set"):
                existing_lifecycle = pydantic_to_explicit_dict(existing_lifecycle)
            merged_lifecycle = deep_merge(preset_lifecycle, existing_lifecycle)
            cust_cfg.lifecycle = merged_lifecycle

    # --- Inject customers demand into models_cfg.customers ---
    existing_demand = models_cfg.get("customers", None) or {}
    if hasattr(existing_demand, "model_fields_set"):
        existing_demand = pydantic_to_explicit_dict(existing_demand)
    merged_demand = deep_merge(preset_customers, existing_demand)
    models_cfg.customers = CustomersDemandConfig.model_validate(merged_demand)

    # --- Apply first_year_pct override (after preset, before return) ---
    if cfg is not None:
        cust_cfg = getattr(cfg, "customers", None)
        if cust_cfg is not None:
            first_year_pct = getattr(cust_cfg, "first_year_pct", None)
            if first_year_pct is not None:
                fyr = float(first_year_pct)
                if not 0.05 <= fyr <= 1.0:
                    raise ConfigError(
                        f"customers.first_year_pct must be between 0.05 and 1.0, got {fyr}"
                    )
                _apply_first_year_pct(fyr, cust_cfg.lifecycle, models_cfg.customers)

    info(f"Trend preset: '{trend_name}' applied"
         + (f" (seasonality: {overrides.get('seasonality', 'from preset')})"
            if "seasonality" in overrides else ""))
