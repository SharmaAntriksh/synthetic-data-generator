"""Shared SQL helpers used by both CREATE TABLE and BULK INSERT generators."""
from __future__ import annotations

from typing import Mapping, Optional


def sql_escape_literal(value: str) -> str:
    """Escape a string for use inside a single-quoted SQL literal."""
    return value.replace("'", "''")


def quote_ident(name: str) -> str:
    """Bracket-quote an identifier; escape closing brackets.

    Strips existing bracket or double-quote wrapping before quoting.
    """
    raw = str(name).strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    if raw.startswith('"') and raw.endswith('"'):
        raw = raw[1:-1]
    return f"[{raw.replace(']', ']]')}]"


# -------------------------------------------------------------------
# Config-driven feature flags (used to decide which SQL objects to emit)
# -------------------------------------------------------------------

def returns_enabled(cfg: Optional[Mapping]) -> bool:
    """Return True if returns are effectively enabled."""
    if cfg is None:
        return True
    returns_cfg = getattr(cfg, "returns", None)
    if isinstance(returns_cfg, Mapping) and isinstance(getattr(returns_cfg, "enabled", None), (bool, int)):
        if not bool(returns_cfg.enabled):
            return False
    sales_cfg = getattr(cfg, "sales", None)
    skip_order = bool(getattr(sales_cfg, "skip_order_cols", False) if sales_cfg else False)
    sales_output = str(getattr(sales_cfg, "sales_output", "sales") if sales_cfg else "sales").strip().lower()
    if skip_order and sales_output == "sales":
        return False
    return True


def budget_enabled(cfg: Optional[Mapping]) -> bool:
    """Return True if budget generation is enabled in config."""
    if cfg is None:
        return False
    budget_cfg = getattr(cfg, "budget", None)
    if budget_cfg is None or not isinstance(budget_cfg, Mapping):
        return False
    return bool(getattr(budget_cfg, "enabled", False))


def inventory_enabled(cfg: Optional[Mapping]) -> bool:
    """Return True if inventory snapshot generation is enabled in config."""
    if cfg is None:
        return False
    inv_cfg = getattr(cfg, "inventory", None)
    if inv_cfg is None or not isinstance(inv_cfg, Mapping):
        return False
    return bool(getattr(inv_cfg, "enabled", False))


def complaints_enabled(cfg: Optional[Mapping]) -> bool:
    """Return True if complaints generation is enabled in config."""
    if cfg is None:
        return False
    cc = getattr(cfg, "complaints", None)
    if cc is None or not isinstance(cc, Mapping):
        return False
    return bool(getattr(cc, "enabled", False))
