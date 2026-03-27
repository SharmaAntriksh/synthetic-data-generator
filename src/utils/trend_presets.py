"""Trend presets for macro demand shape.

Controls the overall chart shape — growth trajectory, seasonality pattern,
month-to-month volatility, and demand shocks. Decoupled from customer
profiles, which own only lifecycle and demand behavior.

Usage in models.yaml:
    models:
      macro_demand:
        trend: steady-growth     # required preset name
        # Optional power-user overrides:
        # seasonality: summer-peak
        # year_level_factors:
        #   mode: "once"
        #   values: [1.0, 1.1, ...]

The preset is resolved into a complete MacroDemandConfig by
``resolve_trend_preset()``, called from pipeline_runner.py after
customer profile resolution.
"""
from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

from src.engine.config.config_schema import MacroDemandConfig
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
# Each preset is a complete macro demand configuration.
# year_level_factors are designed for 10 years; shorter/longer date
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
        # No seasonality — steady business, all variation from noise
        "monthly_seasonality": "flat",
        "noise_std": 0.06,
        "shock_probability": 0.03,
        "shock_impact": [-0.08, 0.05],
        "row_share_of_growth": 0.5,
        "early_month_cap": {"enabled": False},
        "eligible_blend": 0.0,
        "bootstrap_months": 3,
        "max_distinct_ratio": 0.70,
        "min_distinct_customers": 250,
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
        "row_share_of_growth": 0.7,
        "early_month_cap": {
            "enabled": True,
            "max_rows_per_customer": 20,
            "redistribute_excess": True,
        },
        "eligible_blend": 0.0,
        "bootstrap_months": 8,
        "max_distinct_ratio": 0.60,
        "min_distinct_customers": 150,
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
        "row_share_of_growth": 0.5,
        "early_month_cap": {
            "enabled": True,
            "max_rows_per_customer": 25,
            "redistribute_excess": True,
        },
        "eligible_blend": 0.0,
        "bootstrap_months": 6,
        "max_distinct_ratio": 0.65,
        "min_distinct_customers": 200,
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
        "row_share_of_growth": 0.6,
        "early_month_cap": {
            "enabled": True,
            "max_rows_per_customer": 20,
            "redistribute_excess": True,
        },
        "eligible_blend": 0.0,
        "bootstrap_months": 8,
        "max_distinct_ratio": 0.55,
        "min_distinct_customers": 150,
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
        "row_share_of_growth": 0.7,
        "early_month_cap": {"enabled": False},
        "eligible_blend": 0.0,
        "bootstrap_months": 3,
        "max_distinct_ratio": 0.75,
        "min_distinct_customers": 200,
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
        "row_share_of_growth": 0.8,
        "early_month_cap": {
            "enabled": True,
            "max_rows_per_customer": 25,
            "redistribute_excess": True,
        },
        "eligible_blend": 0.0,
        "bootstrap_months": 6,
        "max_distinct_ratio": 0.60,
        "min_distinct_customers": 200,
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
        "row_share_of_growth": 0.5,
        "early_month_cap": {"enabled": False},
        "eligible_blend": 0.0,
        "bootstrap_months": 3,
        "max_distinct_ratio": 0.70,
        "min_distinct_customers": 250,
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
        # Dec wraps gently toward Jan (1.12→0.94) to avoid sharp V-dips
        "monthly_seasonality": [
            0.94, 0.88, 0.90, 0.93, 0.97, 1.00,
            1.02, 1.01, 0.98, 1.06, 1.16, 1.12,
        ],
        "noise_std": 0.10,
        "shock_probability": 0.04,
        "shock_impact": [-0.10, 0.08],
        "row_share_of_growth": 0.5,
        "early_month_cap": {"enabled": False},
        "eligible_blend": 0.0,
        "bootstrap_months": 3,
        "max_distinct_ratio": 0.70,
        "min_distinct_customers": 250,
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
        "row_share_of_growth": 0.5,
        "early_month_cap": {
            "enabled": True,
            "max_rows_per_customer": 25,
            "redistribute_excess": True,
        },
        "eligible_blend": 0.0,
        "bootstrap_months": 6,
        "max_distinct_ratio": 0.65,
        "min_distinct_customers": 200,
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
        "row_share_of_growth": 0.5,
        "early_month_cap": {"enabled": False},
        "eligible_blend": 0.0,
        "bootstrap_months": 3,
        "max_distinct_ratio": 0.70,
        "min_distinct_customers": 250,
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
        "row_share_of_growth": 0.5,
        "early_month_cap": {"enabled": False},
        "eligible_blend": 0.0,
        "bootstrap_months": 3,
        "max_distinct_ratio": 0.70,
        "min_distinct_customers": 250,
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
        "row_share_of_growth": 0.7,
        "early_month_cap": {
            "enabled": True,
            "max_rows_per_customer": 15,
            "redistribute_excess": True,
        },
        "eligible_blend": 0.0,
        "bootstrap_months": 10,
        "max_distinct_ratio": 0.50,
        "min_distinct_customers": 100,
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
        "row_share_of_growth": 0.5,
        "early_month_cap": {
            "enabled": True,
            "max_rows_per_customer": 25,
            "redistribute_excess": True,
        },
        "eligible_blend": 0.0,
        "bootstrap_months": 6,
        "max_distinct_ratio": 0.65,
        "min_distinct_customers": 200,
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
        "row_share_of_growth": 0.5,
        "early_month_cap": {"enabled": False},
        "eligible_blend": 0.0,
        "bootstrap_months": 3,
        "max_distinct_ratio": 0.70,
        "min_distinct_customers": 250,
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
        "row_share_of_growth": 0.5,
        "early_month_cap": {"enabled": False},
        "eligible_blend": 0.0,
        "bootstrap_months": 3,
        "max_distinct_ratio": 0.70,
        "min_distinct_customers": 250,
    },
}

VALID_TRENDS = frozenset(_TREND_PRESETS.keys())

# Auto-mapping: when a customer profile is set but no trend is specified,
# fall back to a default trend that matches the profile's historical
# macro demand shape.
_PROFILE_TO_TREND: Dict[str, Optional[str]] = {
    "steady": "steady-growth",
    "gradual": "gradual-growth",
    "aggressive": "hockey-stick",
    "instant": None,  # instant = flat, no trend needed
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
            raise ValueError(
                f"Unknown seasonality preset: {name!r}. "
                f"Valid presets: {', '.join(sorted(VALID_SEASONALITY_PRESETS))}"
            )
        return list(_SEASONALITY_PRESETS[name])

    if isinstance(raw, (list, tuple)):
        if len(raw) != 12:
            raise ValueError(
                f"monthly_seasonality must have exactly 12 values, got {len(raw)}"
            )
        return [float(v) for v in raw]

    raise ValueError(
        f"seasonality must be a preset name or list of 12 floats, got {type(raw).__name__}"
    )


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
        raise ValueError(f"Unknown trend preset: {name!r}")
    return copy.deepcopy(_TREND_PRESETS[name])


def resolve_trend_preset(
    models_cfg,
    *,
    profile_name: str | None = None,
) -> None:
    """Resolve trend preset and inject into models_cfg.macro_demand.

    If ``models_cfg.macro_demand.trend`` is set, uses that preset.
    Otherwise, auto-maps from ``profile_name`` if available.
    If neither is set, returns without changes (backward compatible).

    Explicit YAML values in macro_demand override preset defaults.

    Mutates *models_cfg* in place.
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

    # Auto-map from profile if no explicit trend
    if trend_name is None and profile_name is not None:
        profile_key = str(profile_name).strip().lower()
        auto_trend = _PROFILE_TO_TREND.get(profile_key)
        if auto_trend is not None:
            trend_name = auto_trend
            info(f"Trend auto-mapped from profile '{profile_key}' -> '{auto_trend}'")

    if trend_name is None:
        return

    trend_name = str(trend_name).strip().lower()
    if trend_name not in _TREND_PRESETS:
        raise ValueError(
            f"Unknown trend preset: {trend_name!r}. "
            f"Valid presets: {', '.join(sorted(VALID_TRENDS))}"
        )

    preset = copy.deepcopy(_TREND_PRESETS[trend_name])

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

    info(f"Trend preset: '{trend_name}' applied"
         + (f" (seasonality: {overrides.get('seasonality', 'from preset')})"
            if "seasonality" in overrides else ""))
