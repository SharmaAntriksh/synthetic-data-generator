"""Customer behavior profiles — DEPRECATED.

This module is superseded by the trend preset system in
``trend_presets.py``, which now owns both macro demand shape AND
customer lifecycle/demand behavior.

The ``customers.profile`` config key still works for backward
compatibility — it is mapped to a trend preset name via
``_PROFILE_TO_TREND`` in ``trend_presets.py``.

New code should use ``models.macro_demand.trend`` instead.
"""
from __future__ import annotations

from collections.abc import Mapping

from typing import Any, Dict, Optional, Tuple

from src.engine.config.config_schema import CustomersDemandConfig
from src.utils.config_merge import pydantic_to_explicit_dict, deep_merge
from src.utils.logging_utils import info
from src.utils.trend_presets import _apply_first_year_pct


# ================================================================
# Profile definitions
# ================================================================

_PROFILES: Dict[str, Dict[str, Any]] = {

    # ------------------------------------------------------------------
    # gradual — Realistic startup ramp (S-curve acquisition, visible churn,
    #           seasonal spikes). Good default for demo / Power BI datasets.
    # ------------------------------------------------------------------
    "gradual": {
        "lifecycle": {
            "enable_churn": True,
            "base_monthly_churn": 0.055,
            "min_tenure_months": 4,
            "initial_active_customers": 0.40,
            "initial_spread_months": 36,
            "acquisition_curve": "logistic",
            "acquisition_params": {"midpoint": 0.30, "steepness": 8.0},
        },
        "demand": {
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
    # steady — Mature business. Nearly all customers exist from the start,
    #          very low churn, mild seasonality, predictable month-to-month.
    #          Should produce a flat, stable customer count over time.
    # ------------------------------------------------------------------
    "steady": {
        "lifecycle": {
            "enable_churn": True,
            "base_monthly_churn": 0.003,
            "min_tenure_months": 12,
            "initial_active_customers": 0.95,
            "initial_spread_months": 1,
            "acquisition_curve": "logistic",
            "acquisition_params": {"midpoint": 0.40, "steepness": 4.0},
        },
        "demand": {
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
    # aggressive — Fast-growing company. Rapid customer acquisition,
    #              strong seasonal peaks, high variability month-to-month.
    # ------------------------------------------------------------------
    "aggressive": {
        "lifecycle": {
            "enable_churn": True,
            "base_monthly_churn": 0.040,
            "min_tenure_months": 3,
            "initial_active_customers": 0.30,
            "initial_spread_months": 12,
            "acquisition_curve": "logistic",
            "acquisition_params": {"midpoint": 0.25, "steepness": 12.0},
        },
        "demand": {
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
    # instant — All customers available from day one, no lifecycle drama.
    #           Flat participation, no ramp, no churn, no seasonality.
    #           Simplest output — useful for testing, teaching, or when
    #           you just want uniform data without lifecycle complexity.
    # ------------------------------------------------------------------
    "instant": {
        "lifecycle": {
            "enable_churn": False,
            "base_monthly_churn": 0.0,
            "min_tenure_months": 0,
            "initial_active_customers": 1.0,
            "initial_spread_months": 0,
            "acquisition_curve": "logistic",
            "acquisition_params": {"midpoint": 0.50, "steepness": 2.0},
        },
        "demand": {
            "distinct_ratio": 0.80,
            "new_customer_share": 0.0,
            "max_new_fraction_per_month": 0.0,
            "cycle_amplitude": 0.0,
            "discovery_shape": 0.0,
            "participation_noise": 0.02,
            "seasonal_spikes": [],
        },
    },

    # ------------------------------------------------------------------
    # decline — Shrinking business. Large existing customer base eroding
    #           over time. High churn, minimal new acquisition. Pairs
    #           naturally with the "decline" macro demand trend.
    # ------------------------------------------------------------------
    "decline": {
        "lifecycle": {
            "enable_churn": True,
            "base_monthly_churn": 0.045,
            "min_tenure_months": 6,
            "initial_active_customers": 0.90,
            "initial_spread_months": 3,
            "acquisition_curve": "logistic",
            "acquisition_params": {"midpoint": 0.70, "steepness": 3.0},
        },
        "demand": {
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
}

VALID_PROFILES = frozenset(_PROFILES.keys())


# ================================================================
# Resolver
# ================================================================



def resolve_customer_profile(
    cfg,
    models_cfg,
):
    """
    Resolve customer profile and inject values into cfg and models_cfg.

    Reads ``cfg["customers"]["profile"]``. If set:
      - Injects lifecycle defaults into ``cfg["customers"]["lifecycle"]``
      - Injects demand defaults into ``models_cfg["customers"]``

    Macro demand (chart shape) is handled separately by
    ``trend_presets.resolve_trend_preset()``.

    Explicit values already present in either location take priority
    over profile defaults.

    Returns:
        (cfg, models_cfg) — mutated in place and returned for convenience.
    """
    cust_cfg = getattr(cfg, "customers", None)
    if cust_cfg is None:
        return cfg, models_cfg

    profile_name = getattr(cust_cfg, "profile", None)
    if profile_name is None:
        return cfg, models_cfg

    profile_name = str(profile_name).strip().lower()
    if profile_name not in _PROFILES:
        raise ValueError(
            f"Unknown customer profile: {profile_name!r}. "
            f"Valid profiles: {', '.join(sorted(VALID_PROFILES))}"
        )

    profile = _PROFILES[profile_name]

    # --- Inject lifecycle into cfg.customers.lifecycle ---
    existing_lifecycle = getattr(cust_cfg, "lifecycle", None) or {}
    # Strip Pydantic defaults so only YAML-explicit values override profile defaults
    if hasattr(existing_lifecycle, "model_fields_set"):
        existing_lifecycle = pydantic_to_explicit_dict(existing_lifecycle)
    merged_lifecycle = deep_merge(profile["lifecycle"], existing_lifecycle)
    cust_cfg.lifecycle = merged_lifecycle  # noqa: E501

    # --- Inject demand into models_cfg.customers (Pydantic model) ---
    existing_demand = models_cfg.get("customers", None) or {}
    # Strip Pydantic defaults so only YAML-explicit values override profile defaults
    if hasattr(existing_demand, "model_fields_set"):
        existing_demand = pydantic_to_explicit_dict(existing_demand)
    merged_demand = deep_merge(profile["demand"], existing_demand)
    models_cfg.customers = CustomersDemandConfig.model_validate(merged_demand)

    # --- Apply first_year_pct override (after profile, before return) ---
    first_year_pct = getattr(cust_cfg, "first_year_pct", None)
    if first_year_pct is not None:
        fyr = float(first_year_pct)
        if not 0.05 <= fyr <= 1.0:
            raise ValueError(
                f"customers.first_year_pct must be between 0.05 and 1.0, got {fyr}"
            )
        _apply_first_year_pct(fyr, cust_cfg.lifecycle, models_cfg.customers)

    info(f"Customer profile: '{profile_name}' applied"
         + (f" (first_year_pct={first_year_pct})" if first_year_pct is not None else ""))

    return cfg, models_cfg


def get_profile_names() -> list[str]:
    """Return sorted list of available profile names."""
    return sorted(_PROFILES.keys())


def get_profile_defaults(name: str) -> Optional[Dict[str, Any]]:
    """Return the raw profile definition, or None if not found."""
    return _PROFILES.get(name)
