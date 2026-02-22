from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Optional, Set

from src.utils.logging_utils import work, skip

# -----------------------------
# Small SQL helpers
# -----------------------------

def _sql_escape_literal(value: str) -> str:
    """Escape a string for use inside a single-quoted SQL literal."""
    return value.replace("'", "''")


def _quote_ident(part: str) -> str:
    """Bracket-quote an identifier; escape closing brackets."""
    raw = str(part).strip()
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1]
    if raw.startswith('"') and raw.endswith('"'):
        raw = raw[1:-1]
    return f"[{raw.replace(']', ']]')}]"


def _quote_table(name: str) -> str:
    """
    Quote a table name. Supports:
      - Sales
      - dbo.Sales
      - [dbo].[Sales]
    Does NOT attempt to parse exotic cases (dots inside brackets), which we don't use in this repo.
    """
    parts = [p for p in str(name).strip().split(".") if p]
    if not parts:
        raise ValueError("Empty table name.")
    if len(parts) > 3:
        raise ValueError(f"Too many name parts: {name!r}")
    return ".".join(_quote_ident(p) for p in parts)


# -----------------------------
# Table inference
# -----------------------------

_CHUNK_SUFFIX_RE = re.compile(r"(?:_chunk\d+|_part\d+)$", flags=re.IGNORECASE)

# facts/<folder>/*.csv -> canonical table
_FOLDER_TABLE_ALIASES: dict[str, str] = {
    # facts
    "sales": "Sales",
    "sales_order_header": "SalesOrderHeader",
    "sales_order_detail": "SalesOrderDetail",

    # returns
    "sales_return": "SalesReturn",
    "salesreturn": "SalesReturn",
    "returns": "SalesReturn",
}

def _infer_table_from_filename(csv_file: str) -> str:
    """
    Infer PascalCase table name from a chunked snake_case filename.
    Examples:
      sales_chunk0001.csv              -> Sales
      sales_order_detail_chunk0001.csv -> SalesOrderDetail
    """
    base = Path(csv_file).stem
    base = _CHUNK_SUFFIX_RE.sub("", base)
    return base.replace("_", " ").title().replace(" ", "")


def _allowed_lookup(allowed_tables: Optional[Set[str]]) -> Optional[dict[str, str]]:
    """Map lower(table) -> canonical(table) for case-insensitive membership checks."""
    if not allowed_tables:
        return None
    return {str(t).strip().lower(): str(t).strip() for t in allowed_tables}


def _pick_target_table(csv_path: Path, *, allowed_tables: Optional[Set[str]]) -> Optional[str]:
    """
    Pick target table for a CSV file.
    - Prefer folder mapping (facts/<folder>/file.csv)
    - Fallback to filename inference
    - Apply allowed_tables filter (case-insensitive)
    """
    allowed_map = _allowed_lookup(allowed_tables)

    folder_key = csv_path.parent.name.strip().lower()
    candidate = _FOLDER_TABLE_ALIASES.get(folder_key) or _infer_table_from_filename(csv_path.name)

    if allowed_map is None:
        return candidate

    return allowed_map.get(candidate.strip().lower())


# -----------------------------
# Config-driven allowlist
# -----------------------------

def _returns_enabled_from_cfg(cfg: Optional[Mapping]) -> bool:
    """
    Returns True if returns are enabled. Supports:
      - facts: ['sales','returns']
      - facts: { enabled: ['sales','returns'] }
      - facts: { returns: true/false }
      - facts: { enabled: { returns: true/false } }
    Default: True when unspecified.
    """
    if cfg is None:
        return True

    facts_cfg = cfg.get("facts")

    def _list_has_returns(v) -> bool:
        norm = {str(x).strip().lower() for x in (v or [])}
        return (
            "returns" in norm
            or "salesreturn" in norm
            or "sales_return" in norm
            or "salesreturns" in norm
            or "sales_returns" in norm
        )

    if isinstance(facts_cfg, list):
        return _list_has_returns(facts_cfg)

    if facts_cfg is None or not isinstance(facts_cfg, Mapping):
        return True

    if isinstance(facts_cfg.get("returns"), (bool, int)):
        return bool(facts_cfg.get("returns"))

    enabled_cfg = facts_cfg.get("enabled")
    if isinstance(enabled_cfg, list):
        return _list_has_returns(enabled_cfg)

    if isinstance(enabled_cfg, Mapping) and isinstance(enabled_cfg.get("returns"), (bool, int)):
        return bool(enabled_cfg.get("returns"))

    return True


def _allowed_fact_tables_from_cfg(cfg: Optional[Mapping]) -> Optional[Set[str]]:
    """
    Allowed fact tables for facts bulk insert.
    - sales.sales_output drives Sales vs SalesOrderHeader/Detail
    - returns flags (above) controls SalesReturn
    """
    if cfg is None:
        return None

    sales_cfg = cfg.get("sales") or {}
    mode = str(sales_cfg.get("sales_output", "sales")).lower().strip()
    if mode not in {"sales", "sales_order", "both"}:
        raise ValueError(f"Invalid sales.sales_output: {mode!r}. Expected sales|sales_order|both.")

    allowed: set[str] = set()
    if mode in {"sales", "both"}:
        allowed.add("Sales")
    if mode in {"sales_order", "both"}:
        allowed.add("SalesOrderHeader")
        allowed.add("SalesOrderDetail")

    if _returns_enabled_from_cfg(cfg):
        allowed.add("SalesReturn")

    return allowed


# -----------------------------
# Script generator
# -----------------------------

def _iter_csv_files(folder: Path, *, recursive: bool) -> Iterable[Path]:
    if recursive:
        yield from sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() == ".csv")
    else:
        yield from sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".csv")


def generate_bulk_insert_script(
    csv_folder,
    *,
    output_sql_file: str = "bulk_insert.sql",
    table_name: Optional[str] = None,
    mode: str = "legacy",            # "legacy" | "csv"
    first_row: int = 2,
    field_terminator: str = ",",
    row_terminator: str = "0x0a",
    codepage: str = "65001",
    recursive: bool = False,
    allowed_tables: Optional[Set[str]] = None,
) -> Optional[str]:
    """
    Generate a BULK INSERT script.
    - If table_name is None, table is inferred per file.
    - recursive=True is required for facts/<table>/*.csv layout.
    - allowed_tables filters inferred tables (case-insensitive).
    """
    csv_folder = Path(csv_folder)
    if not csv_folder.exists() or not csv_folder.is_dir():
        skip(f"CSV folder not found or not a directory: {csv_folder}")
        return None

    target_sql = Path(output_sql_file)
    target_sql.parent.mkdir(parents=True, exist_ok=True)

    csv_paths = list(_iter_csv_files(csv_folder, recursive=recursive))
    if not csv_paths:
        skip(f"No CSV files found in {csv_folder}. Skipping BULK INSERT script.")
        return None

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = [
        "-- Auto-generated BULK INSERT script",
        f"-- Generated on: {ts}",
        "-- NOTE: 'FROM <path>' is evaluated on the SQL Server host.",
        "SET NOCOUNT ON;",
        "",
    ]

    allowed_map = _allowed_lookup(allowed_tables)
    emitted = 0

    for csv_path in csv_paths:
        if table_name is not None:
            tgt = table_name
        else:
            tgt = _pick_target_table(csv_path, allowed_tables=allowed_tables)
            if tgt is None:
                continue

        # (Extra safety) if caller passes allowed_tables and explicit table_name, filter too.
        if allowed_map is not None and table_name is not None:
            if table_name.strip().lower() not in allowed_map:
                continue

        quoted_table = _quote_table(tgt)

        csv_full_path_sql = _sql_escape_literal(str(csv_path.resolve()))
        rel_hint = str(csv_path.relative_to(csv_folder)) if recursive else csv_path.name

        with_opts: list[str] = [f"FIRSTROW = {int(first_row)}"]

        if mode == "csv":
            # SQL Server 2017+
            with_opts.insert(0, "FORMAT = 'CSV'")
            with_opts.append(f"CODEPAGE = '{codepage}'")
            with_opts.append("TABLOCK")
        else:
            with_opts.append(f"FIELDTERMINATOR = '{_sql_escape_literal(field_terminator)}'")
            with_opts.append(f"ROWTERMINATOR = '{_sql_escape_literal(row_terminator)}'")
            with_opts.append(f"CODEPAGE = '{codepage}'")
            with_opts.append("TABLOCK")

        opts_sql = ",\n    ".join(with_opts)

        lines.append(f"-- Source file: {rel_hint}")
        lines.append(f"BULK INSERT {quoted_table}")
        lines.append(f"FROM '{csv_full_path_sql}'")
        lines.append("WITH (")
        lines.append(f"    {opts_sql}")
        lines.append(");")
        lines.append("")
        emitted += 1

    if emitted == 0:
        skip(f"No matching CSV files to emit (allowed_tables filter may have excluded all). Folder: {csv_folder}")
        return None

    target_sql.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    work(f"Wrote BULK INSERT script: {target_sql.name}")
    return str(target_sql)


def generate_dims_and_facts_bulk_insert_scripts(
    *,
    dims_folder,
    facts_folder,
    cfg,
    load_output_folder,
    dims_sql_name: str = "01_bulk_insert_dims.sql",
    facts_sql_name: str = "02_bulk_insert_facts.sql",
    dims_mode: str = "csv",
    facts_mode: str = "legacy",
    row_terminator: str = "0x0a",
) -> tuple[str, str]:
    """
    Convenience wrapper that ALWAYS writes exactly two files:
      01_bulk_insert_dims.sql
      02_bulk_insert_facts.sql

    - dims: flat folder
    - facts: recursive scan (facts/<table>/*.csv)
    """
    load_output_folder = Path(load_output_folder)
    load_output_folder.mkdir(parents=True, exist_ok=True)

    dims_sql = load_output_folder / dims_sql_name
    facts_sql = load_output_folder / facts_sql_name

    generate_bulk_insert_script(
        dims_folder,
        output_sql_file=str(dims_sql),
        mode=dims_mode,
        table_name=None,
        recursive=False,
    )

    allowed = _allowed_fact_tables_from_cfg(cfg)

    generate_bulk_insert_script(
        facts_folder,
        output_sql_file=str(facts_sql),
        mode=facts_mode,
        table_name=None,
        row_terminator=row_terminator,
        recursive=True,
        allowed_tables=allowed,
    )

    return str(dims_sql), str(facts_sql)
