"""
Shared config-parsing and date-generation helpers used by dimension generators
(employees, employee_store_assignments, stores, etc.).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from src.utils.logging_utils import warn


# ---------------------------------------------------------------------------
# Safe type coercion
# ---------------------------------------------------------------------------

def as_dict(x: Any) -> Dict[str, Any]:
    """Return *x* if it is a dict, else an empty dict."""
    return x if isinstance(x, dict) else {}


def int_or(value: Any, default: int) -> int:
    """Convert *value* to ``int``, falling back to *default* on failure."""
    try:
        if value is None or value == "":
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def float_or(value: Any, default: float) -> float:
    """Convert *value* to ``float``, falling back to *default* on failure."""
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def bool_or(value: Any, default: bool) -> bool:
    """Convert *value* to ``bool``, falling back to *default* on failure.

    Accepts common truthy/falsy string representations including
    ``"t"``/``"f"``, ``"on"``/``"off"``, ``"yes"``/``"no"`` and numpy
    integer types.
    """
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    try:
        import numpy as _np
        if isinstance(value, (int, float, _np.integer)):
            return bool(int(value))
    except ImportError:
        if isinstance(value, (int, float)):
            return bool(int(value))
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"true", "t", "1", "yes", "y", "on"}:
            return True
        if s in {"false", "f", "0", "no", "n", "off"}:
            return False
    return bool(default)


def str_or(v: Any, default: str) -> str:
    """Return *v* as a stripped string, or *default* if empty/None."""
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def range2(v: Any, default_lo: float, default_hi: float) -> Tuple[float, float]:
    """Parse a 2-element list/tuple ``[lo, hi]``; ensures *hi >= lo*."""
    lo = default_lo
    hi = default_hi
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        lo = float_or(v[0], default_lo)
        hi = float_or(v[1], default_hi)
    if hi < lo:
        lo, hi = hi, lo
    return lo, hi


# ---------------------------------------------------------------------------
# Seed resolution
# ---------------------------------------------------------------------------

def pick_seed_nested(
    cfg: Dict[str, Any],
    local_cfg: Dict[str, Any],
    fallback: int = 42,
) -> int:
    """Resolve seed: ``override.seed → local_cfg.seed → defaults.seed → fallback``.

    Also checks ``_defaults.seed`` for backward compatibility.
    """
    override = as_dict(local_cfg.get("override"))
    seed = override.get("seed")
    if seed is None:
        seed = local_cfg.get("seed")
    if seed is None:
        seed = as_dict(cfg.get("defaults")).get("seed")
    if seed is None:
        seed = as_dict(cfg.get("_defaults")).get("seed")
    return int_or(seed, fallback)


def pick_seed_flat(
    cfg: Dict[str, Any],
    local_cfg: Dict[str, Any],
    fallback: int = 42,
) -> int:
    """Resolve seed: ``local_cfg.seed → cfg.seed → fallback``."""
    if "seed" in local_cfg and local_cfg["seed"] is not None:
        return int_or(local_cfg["seed"], fallback)
    if "seed" in cfg and cfg["seed"] is not None:
        return int_or(cfg["seed"], fallback)
    return fallback


# ---------------------------------------------------------------------------
# Date window resolution
# ---------------------------------------------------------------------------

def parse_global_dates(
    cfg: Dict[str, Any],
    local_cfg: Dict[str, Any],
    *,
    allow_override: bool = False,
    dimension_name: str = "",
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Resolve dataset-wide date window.

    Primary source: ``defaults.dates.{start, end}``

    When *allow_override* is True the function first checks
    ``local_cfg.override.dates.{start, end}`` (testing escape-hatch).
    """
    if allow_override:
        ov = as_dict(as_dict(local_cfg.get("override")).get("dates"))
        if ov and ov.get("start") and ov.get("end"):
            gs = pd.to_datetime(ov["start"]).normalize()
            ge = pd.to_datetime(ov["end"]).normalize()
            if ge < gs:
                warn(
                    f"{dimension_name}: override dates swapped "
                    f"(start={ov['start']!r}, end={ov['end']!r})"
                )
                gs, ge = ge, gs
            return gs, ge

    dd = as_dict(as_dict(cfg.get("defaults")).get("dates"))
    if dd and dd.get("start") and dd.get("end"):
        gs = pd.to_datetime(dd["start"]).normalize()
        ge = pd.to_datetime(dd["end"]).normalize()
        if ge < gs:
            if dimension_name:
                warn(
                    f"{dimension_name}: defaults dates swapped "
                    f"(start={dd['start']!r}, end={dd['end']!r})"
                )
            gs, ge = ge, gs
        return gs, ge

    label = f" for {dimension_name}" if dimension_name else ""
    raise KeyError(
        f"defaults.dates.start and defaults.dates.end are required{label}."
    )


# ---------------------------------------------------------------------------
# Date generation
# ---------------------------------------------------------------------------

def rand_dates_between(
    rng: np.random.Generator,
    start: pd.Timestamp,
    end: pd.Timestamp,
    n: int,
) -> pd.Series:
    """Return *n* random dates in ``[start, end]`` as a ``datetime64[ns]`` Series."""
    start = pd.to_datetime(start).normalize()
    end = pd.to_datetime(end).normalize()
    if end < start:
        start, end = end, start
    start_i = int(start.value // 86_400_000_000_000)
    end_i = int(end.value // 86_400_000_000_000)
    days = rng.integers(start_i, end_i + 1, size=int(n), dtype=np.int64)
    dt = pd.to_datetime(days.astype("datetime64[D]")).normalize()
    return pd.Series(dt, dtype="datetime64[ns]")


def rand_single_date(
    rng: np.random.Generator,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.Timestamp:
    """Sample one date in ``[start, end]`` without creating a pandas Series."""
    start_i = int(start.value // 86_400_000_000_000)
    end_i = int(end.value // 86_400_000_000_000)
    day = int(rng.integers(start_i, end_i + 1))
    return pd.Timestamp(day, unit="D")


# ---------------------------------------------------------------------------
# Geography / region mapping
# ---------------------------------------------------------------------------

_AMERICAS_CODES = frozenset({"USD", "CAD", "MXN", "BRL", "ARS", "CLP", "COP", "PEN"})
_EUROPE_CODES = frozenset({"EUR", "GBP", "CHF", "SEK", "NOK", "DKK", "PLN", "CZK", "HUF", "RON"})
_APAC_CODES = frozenset({
    "AUD", "NZD", "JPY", "CNY", "HKD", "SGD",
    "KRW", "TWD", "THB", "IDR", "PHP", "VND", "MYR",
})


def region_from_iso_code(code: str, default_region: str = "US") -> str:
    """Map an ISO/currency code to a name-pool region identifier."""
    c = (code or "").strip().upper()
    if c in {"INR"}:
        return "IN"
    if c in _AMERICAS_CODES:
        return "US"
    if c in _EUROPE_CODES:
        return "EU"
    if c in _APAC_CODES:
        return "AS"
    return default_region
