"""Subscription plan catalog — constants, expansion, and payment data."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from src.exceptions import ValidationError


# Billing-cycle discount rates (off monthly price)
_CYCLE_DISCOUNT = {
    "Monthly":     0.00,
    "Quarterly":   0.05,
    "Half-Yearly": 0.10,
    "Annual":      0.17,
}
_CYCLE_MONTHS = {
    "Monthly": 1, "Quarterly": 3, "Half-Yearly": 6, "Annual": 12,
}

# Category mapping: PlanType -> higher-level Category for analytics
_CATEGORY_MAP = {
    "Streaming":     "Entertainment",
    "Gaming":        "Entertainment",
    "Fitness":       "Health",
    "Cloud Storage": "Productivity",
    "Education":     "Productivity",
    "News & Media":  "Information",
    "Music":         "Entertainment",
    "Productivity":  "Productivity",
}

# Base plan definitions: (PlanName, PlanType, BaseMonthlyPrice, Tier, MaxUsers, HasFreeTrial, LaunchDayOffset)
_BASE_PLANS: List[Tuple[str, str, float, str, int, int, int]] = [
    # Streaming — Netflix
    ("Netflix",                 "Streaming",      15.49, "Standard", 2, 1,   0),
    ("Netflix Premium",         "Streaming",      22.99, "Premium",  4, 1,   0),
    # Music — Spotify
    ("Spotify",                 "Music",          10.99, "Standard", 1, 1,  30),
    ("Spotify Family",          "Music",          16.99, "Premium",  6, 0, 120),
    # Cloud storage — Dropbox
    ("Dropbox Plus",            "Cloud Storage",  11.99, "Standard", 1, 1,   0),
    ("Dropbox Business",        "Cloud Storage",  20.00, "Premium",  5, 0,  90),
    # Fitness — Peloton
    ("Peloton",                 "Fitness",        12.99, "Standard", 1, 1,  15),
    ("Peloton All-Access",      "Fitness",        44.00, "Premium",  2, 1, 300),
    # Gaming — Xbox Game Pass
    ("Xbox Game Pass",          "Gaming",         10.99, "Standard", 1, 1,  60),
    ("Xbox Game Pass Ultimate", "Gaming",         19.99, "Premium",  1, 1, 365),
    # News & media — NYT
    ("NYT Digital",             "News & Media",    5.00, "Basic",    1, 1,   0),
    ("NYT All Access",          "News & Media",   12.50, "Premium",  5, 0, 180),
    # Education — Coursera / LinkedIn Learning
    ("Coursera Plus",           "Education",      59.00, "Standard", 1, 1, 150),
    ("LinkedIn Learning",       "Education",      29.99, "Premium", 10, 0, 365),
    # Productivity — Microsoft 365
    ("Microsoft 365",           "Productivity",    6.99, "Standard", 1, 0,  45),
    ("Microsoft 365 Business",  "Productivity",   12.50, "Premium", 25, 0, 210),
]

# Which billing cycles each base plan supports
_PLAN_CYCLES: Dict[str, List[str]] = {
    "Netflix":                 ["Monthly", "Annual"],
    "Netflix Premium":         ["Monthly", "Quarterly", "Annual"],
    "Spotify":                 ["Monthly", "Quarterly"],
    "Spotify Family":          ["Monthly", "Annual"],
    "Dropbox Plus":            ["Monthly"],
    "Dropbox Business":        ["Monthly", "Quarterly", "Annual"],
    "Peloton":                 ["Monthly", "Quarterly"],
    "Peloton All-Access":      ["Monthly", "Half-Yearly", "Annual"],
    "Xbox Game Pass":          ["Monthly", "Quarterly"],
    "Xbox Game Pass Ultimate": ["Monthly", "Annual"],
    "NYT Digital":             ["Monthly"],
    "NYT All Access":          ["Monthly", "Annual"],
    "Coursera Plus":           ["Monthly", "Half-Yearly"],
    "LinkedIn Learning":       ["Annual"],
    "Microsoft 365":           ["Monthly", "Annual"],
    "Microsoft 365 Business":  ["Monthly", "Quarterly", "Annual"],
}


def _expand_catalog() -> List[Tuple]:
    """Expand base plans × billing cycles into the full catalog."""
    rows = []
    for name, ptype, mprice, tier, maxu, trial, launch in _BASE_PLANS:
        category = _CATEGORY_MAP.get(ptype, ptype)
        for cycle in _PLAN_CYCLES[name]:
            discount = _CYCLE_DISCOUNT[cycle]
            months = _CYCLE_MONTHS[cycle]
            cycle_price = round(mprice * months * (1 - discount), 2)
            annual_price = round(mprice * 12 * (1 - discount), 2)
            rows.append((
                name, ptype, category, cycle, months, mprice, discount,
                cycle_price, annual_price, tier, maxu, trial, launch,
            ))
    return rows


PLANS_CATALOG = _expand_catalog()

_PLAN_TYPE_WEIGHT = {
    "Streaming": 4.0,
    "Music": 3.5,
    "Cloud Storage": 3.0,
    "Gaming": 2.5,
    "News & Media": 2.0,
    "Fitness": 1.5,
    "Education": 1.5,
    "Productivity": 2.5,
}

PAYMENT_METHODS = ["Credit Card", "Debit Card", "PayPal", "Bank Transfer"]
_PAYMENT_WEIGHTS = np.array([0.45, 0.25, 0.20, 0.10])
if abs(_PAYMENT_WEIGHTS.sum() - 1.0) > 1e-9:
    raise ValidationError(
        f"_PAYMENT_WEIGHTS must sum to 1.0, got {_PAYMENT_WEIGHTS.sum()}"
    )

CANCELLATION_REASONS = [
    "Too Expensive", "Not Using", "Switched Competitor",
    "Missing Features", "Poor Service", "Other",
]
