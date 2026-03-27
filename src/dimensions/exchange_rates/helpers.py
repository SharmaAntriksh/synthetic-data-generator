"""Shared helpers for the exchange_rates package (currency + FX dims)."""
from __future__ import annotations

from typing import List

import pandas as pd

from src.defaults import (
    CURRENCY_BASE,
    CURRENCY_DECIMAL_PLACES,
    CURRENCY_DECIMAL_PLACES_DEFAULT,
    CURRENCY_NAME_MAP,
    CURRENCY_SYMBOL_MAP,
)
from src.exceptions import ConfigError, DimensionError
from src.utils.logging_utils import warn


# ---------------------------------------------------------
# Currency list normalization
# ---------------------------------------------------------

def normalize_currency_list(currencies: List[str]) -> List[str]:
    """Validate, dedupe, upper-case, and ensure base currency is present."""
    if not isinstance(currencies, list) or not currencies:
        raise DimensionError("currencies must be a non-empty list")

    normalized: List[str] = []
    seen: set[str] = set()

    for c in currencies:
        if not isinstance(c, str) or not c.strip():
            raise DimensionError(f"Invalid currency code: {c!r}")

        code = c.strip().upper()

        if len(code) != 3 or not code.isalpha():
            raise DimensionError(f"Currency code must be 3 letters (e.g. USD). Got: {c!r}")

        if code in seen:
            raise DimensionError(f"Duplicate currency code: {code}")

        seen.add(code)
        normalized.append(code)

    base = CURRENCY_BASE.upper()
    if base not in seen:
        normalized.insert(0, base)

    return normalized


def currency_name(code: str) -> str:
    """Look up currency name, warn if unknown."""
    name = CURRENCY_NAME_MAP.get(code)
    if name is None:
        warn(f"No currency name mapping for '{code}'; using code as name")
        return code
    return name


def currency_symbol(code: str) -> str:
    """Look up currency symbol, fall back to code if unknown."""
    return CURRENCY_SYMBOL_MAP.get(code, code)


def currency_decimal_places(code: str) -> int:
    """Return the standard number of decimal places for a currency."""
    return CURRENCY_DECIMAL_PLACES.get(code, CURRENCY_DECIMAL_PLACES_DEFAULT)


# ---------------------------------------------------------
# FX date resolution (always uses global dates)
# ---------------------------------------------------------

def resolve_fx_dates(cfg) -> tuple:
    """Return (start, end) date strings from ``cfg.defaults.dates``.

    FX dates always follow the global date window.  Raises
    :class:`~src.exceptions.ConfigError` if global dates are missing.
    """
    defaults = cfg.defaults if hasattr(cfg, "defaults") else None
    if defaults is not None and hasattr(defaults, "dates") and defaults.dates is not None:
        return defaults.dates.start, defaults.dates.end

    raise ConfigError(
        "exchange_rates: global defaults dates are missing "
        "(cfg.defaults.dates must be set)."
    )


def parse_fx_date(label: str, value):
    """Parse a date value into a ``datetime.date``, raising ConfigError on failure."""
    try:
        return pd.to_datetime(value, errors="raise").date()
    except (ValueError, TypeError) as exc:
        raise ConfigError(f"exchange_rates: invalid {label} date '{value}'") from exc
