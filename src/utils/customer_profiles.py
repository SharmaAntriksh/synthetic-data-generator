"""Customer behavior profiles.

Replaces ~30 knobs across config.yaml (customers.lifecycle) and
models.yaml (models.customers + models.macro_demand) with a single
``profile`` key.

Usage in config.yaml:
    customers:
      profile: gradual     # gradual | steady | aggressive | instant

The profile is resolved into three parts:
  - lifecycle:      injected into cfg.customers.lifecycle (dict)
  - demand:         injected into models_cfg.customers (CustomersDemandConfig)
  - macro_demand:   injected into models_cfg.macro_demand (MacroDemandConfig)

Any explicit overrides in config.yaml or models.yaml take priority
over profile defaults (merge, not replace).
"""
from __future__ import annotations

from collections.abc import Mapping

from typing import Any, Dict, Optional, Tuple

from src.engine.config.config_schema import CustomersDemandConfig, MacroDemandConfig
from src.utils.logging_utils import info


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
        "macro_demand": {
            "base_level": 1.0,
            "yearly_growth": 0.0,
            "seasonality_amplitude": 0.15,
            "seasonality_phase": 0.0,
            "noise_std": 0.04,
            "eligible_blend": 0.0,
            "year_level_factors": {
                "mode": "once",
                #         Y0   Y1    Y2    Y3    Y4    Y5    Y6    Y7    Y8    Y9
                "values": [1.0, 1.15, 1.35, 1.8, 2.0, 1.85, 2.1, 2.4, 2.3, 2.7],
            },
            "row_share_of_growth": 0.5,
            "shock_probability": 0.08,
            "shock_impact": [-0.15, 0.20],
            "early_month_cap": {
                "enabled": True,
                "max_rows_per_customer": 25,
                "redistribute_excess": True,
            },
        },
    },

    # ------------------------------------------------------------------
    # steady — Mature business. Most customers exist from early on,
    #          low churn, mild seasonality, predictable month-to-month.
    # ------------------------------------------------------------------
    "steady": {
        "lifecycle": {
            "enable_churn": True,
            "base_monthly_churn": 0.005,
            "min_tenure_months": 8,
            "initial_active_customers": 0.30,
            "initial_spread_months": 12,
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
        "macro_demand": {
            "base_level": 1.0,
            "yearly_growth": 0.0,
            "seasonality_amplitude": 0.08,
            "seasonality_phase": 0.0,
            "noise_std": 0.02,
            "eligible_blend": 0.0,
            "year_level_factors": {
                "mode": "once",
                #         Y0   Y1    Y2    Y3    Y4    Y5    Y6    Y7    Y8    Y9
                "values": [1.0, 1.03, 1.06, 1.10, 1.13, 1.10, 1.14, 1.17, 1.15, 1.20],
            },
            "row_share_of_growth": 0.5,
            "shock_probability": 0.03,
            "shock_impact": [-0.08, 0.05],
            "early_month_cap": {
                "enabled": False,
            },
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
        "macro_demand": {
            "base_level": 1.0,
            "yearly_growth": 0.0,
            "seasonality_amplitude": 0.25,
            "seasonality_phase": 0.0,
            "noise_std": 0.06,
            "eligible_blend": 0.0,
            "year_level_factors": {
                "mode": "once",
                #         Y0   Y1   Y2   Y3   Y4   Y5   Y6   Y7   Y8    Y9
                "values": [1.0, 1.3, 1.7, 2.8, 3.5, 3.0, 3.8, 4.5, 4.2, 5.0],
            },
            "row_share_of_growth": 0.6,
            "shock_probability": 0.12,
            "shock_impact": [-0.20, 0.30],
            "early_month_cap": {
                "enabled": True,
                "max_rows_per_customer": 20,
                "redistribute_excess": True,
            },
        },
    },

    # ------------------------------------------------------------------
    # instant — All customers available from day one, no lifecycle drama.
    #           Flat participation, no ramp, no churn. Simplest output —
    #           useful for testing or when you just want uniform data.
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
            "distinct_ratio": 0.60,
            "new_customer_share": 0.0,
            "max_new_fraction_per_month": 0.0,
            "cycle_amplitude": 0.0,
            "discovery_shape": 0.0,
            "participation_noise": 0.05,
            "seasonal_spikes": [],
        },
        "macro_demand": {
            "base_level": 1.0,
            "yearly_growth": 0.0,
            "seasonality_amplitude": 0.0,
            "seasonality_phase": 0.0,
            "noise_std": 0.02,
            "eligible_blend": 0.0,
            "year_level_factors": {
                "mode": "once",
                "values": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            },
            "row_share_of_growth": 0.5,
            "shock_probability": 0.0,
            "shock_impact": [-0.10, 0.10],
            "early_month_cap": {
                "enabled": False,
            },
        },
    },
}

VALID_PROFILES = frozenset(_PROFILES.keys())


# ================================================================
# Resolver
# ================================================================

def _pydantic_to_explicit_dict(obj) -> dict:
    """Convert a Pydantic model to a dict of only explicitly-set fields (recursive).

    Pydantic models expose ALL fields (including defaults) when iterated.
    This strips defaults so that only YAML-explicit values act as overrides
    in ``_deep_merge``, preventing Pydantic's zero-defaults from clobbering
    profile-injected values.
    """
    fields_set = getattr(obj, "model_fields_set", None)
    if fields_set is None:
        # Not a Pydantic model — return as-is (dict or similar)
        return dict(obj) if isinstance(obj, Mapping) else {}
    out = {}
    for k in fields_set:
        v = getattr(obj, k)
        # Recurse into nested Pydantic sub-models
        if hasattr(v, "model_fields_set"):
            v = _pydantic_to_explicit_dict(v)
        out[k] = v
    return out


def _deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge overrides into base. Overrides win for non-dict values."""
    result = dict(base)
    for k, v in overrides.items():
        if isinstance(v, Mapping) and isinstance(result.get(k), Mapping):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _apply_first_year_pct(fyr: float, lifecycle: dict, demand: CustomersDemandConfig) -> None:
    """
    Derive acquisition knobs from a single first_year_pct value.

    fyr = 0.30 means 30% of customers exist in year 1, 70% acquired later.
    fyr = 0.80 means 80% in year 1, only 20% acquired later.

    Adjusts:
      lifecycle: initial_active_customers, initial_spread_months
      demand:    new_customer_share, max_new_fraction_per_month
    """
    remaining = 1.0 - fyr

    lifecycle["initial_active_customers"] = fyr
    lifecycle["initial_spread_months"] = 12

    demand.new_customer_share = round(0.04 + remaining * 0.12, 3)
    demand.max_new_fraction_per_month = round(0.02 + remaining * 0.06, 3)


def resolve_customer_profile(
    cfg,
    models_cfg,
):
    """
    Resolve customer profile and inject values into cfg and models_cfg.

    Reads ``cfg["customers"]["profile"]``. If set:
      - Injects lifecycle defaults into ``cfg["customers"]["lifecycle"]``
      - Injects demand defaults into ``models_cfg["customers"]``
      - Injects macro_demand defaults into ``models_cfg["macro_demand"]``

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
        existing_lifecycle = _pydantic_to_explicit_dict(existing_lifecycle)
    merged_lifecycle = _deep_merge(profile["lifecycle"], existing_lifecycle)
    cust_cfg.lifecycle = merged_lifecycle  # noqa: E501

    # --- Inject demand into models_cfg.customers (Pydantic model) ---
    existing_demand = models_cfg.get("customers", None) or {}
    # Strip Pydantic defaults so only YAML-explicit values override profile defaults
    if hasattr(existing_demand, "model_fields_set"):
        existing_demand = _pydantic_to_explicit_dict(existing_demand)
    merged_demand = _deep_merge(profile["demand"], existing_demand)
    models_cfg.customers = CustomersDemandConfig.model_validate(merged_demand)

    # --- Inject macro_demand into models_cfg.macro_demand (Pydantic model) ---
    existing_macro = models_cfg.get("macro_demand", None) or {}
    # Strip Pydantic defaults so only YAML-explicit values override profile defaults
    if hasattr(existing_macro, "model_fields_set"):
        existing_macro = _pydantic_to_explicit_dict(existing_macro)
    merged_macro = _deep_merge(profile["macro_demand"], existing_macro)
    models_cfg.macro_demand = MacroDemandConfig.model_validate(merged_macro)

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
