import re
from pathlib import Path
from typing import Optional

from src.facts.sales.output_paths import (
    TABLE_SALES,
    TABLE_SALES_ORDER_DETAIL,
    TABLE_SALES_ORDER_HEADER,
    TABLE_SALES_RETURN
)

# Canonical folder names for scratch + packaged outputs
_TABLE_DIR_MAP = {
    TABLE_SALES: "sales",
    TABLE_SALES_ORDER_DETAIL: "sales_order_detail",
    TABLE_SALES_ORDER_HEADER: "sales_order_header",
    TABLE_SALES_RETURN: "sales_return",
}


def get_first_existing_path(cfg: dict, keys: list[str]) -> Optional[Path]:
    """
    Resolve config/model yaml paths from cfg keys.
    Checks:
      1) absolute path
      2) cwd-relative
      3) repo-root-relative
    Returns first that exists, else None.
    """
    # paths.py lives at: <repo>/src/engine/packaging/paths.py
    # repo root is 3 parents above "packaging"
    project_root = Path(__file__).resolve().parents[3]

    def _resolve_existing(v: str) -> Optional[Path]:
        p = Path(str(v)).expanduser()
        if p.is_absolute() and p.exists():
            return p

        cwd_candidate = (Path.cwd() / p).resolve()
        if cwd_candidate.exists():
            return cwd_candidate

        repo_candidate = (project_root / p).resolve()
        if repo_candidate.exists():
            return repo_candidate

        return None

    for k in keys:
        v = cfg.get(k)
        if not v:
            continue
        resolved = _resolve_existing(v)
        if resolved:
            return resolved
    return None


def to_snake(s: str) -> str:
    s = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", s)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


def tables_from_sales_cfg(sales_cfg: dict, cfg: Optional[dict] = None) -> list[str]:
    sales_output = str(sales_cfg.get("sales_output", "sales")).lower().strip()
    if sales_output not in {"sales", "sales_order", "both"}:
        raise ValueError(f"Invalid sales_output: {sales_output}")

    tables: list[str] = []
    if sales_output in {"sales", "both"}:
        tables.append(TABLE_SALES)
    if sales_output in {"sales_order", "both"}:
        tables += [TABLE_SALES_ORDER_DETAIL, TABLE_SALES_ORDER_HEADER]

    # Returns (optional)
    returns_enabled = False
    if isinstance(cfg, dict):
        returns_cfg = cfg.get("returns") or {}
        returns_enabled = bool(returns_cfg.get("enabled", False))

    if returns_enabled:
        # Import locally to avoid hard dependency if returns isn’t wired in some branches yet
        from src.facts.sales.output_paths import TABLE_SALES_RETURN  # noqa: WPS433

        tables.append(TABLE_SALES_RETURN)

    return tables


def table_dir_name(table: str) -> str:
    return _TABLE_DIR_MAP.get(table, to_snake(table))
