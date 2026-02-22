import os
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Optional, Set

from src.utils.logging_utils import work, skip


def _sql_escape_literal(value: str) -> str:
    """Escape a string for use inside a single-quoted SQL literal."""
    return value.replace("'", "''")


def _split_multipart_name(name: str) -> list[str]:
    """
    Split a multipart SQL Server name on dots, but ignore dots inside
    [brackets] or "double quotes".
    Examples:
      dbo.Sales -> ["dbo", "Sales"]
      [dbo].[Sales] -> ["[dbo]", "[Sales]"]
      "dbo"."Sales" -> ['"dbo"', '"Sales"']
    """
    s = name.strip()
    if not s:
        return []

    parts: list[str] = []
    buf: list[str] = []
    in_brackets = False
    in_dquotes = False

    for ch in s:
        if ch == "[" and not in_dquotes:
            in_brackets = True
            buf.append(ch)
            continue
        if ch == "]" and in_brackets and not in_dquotes:
            in_brackets = False
            buf.append(ch)
            continue
        if ch == '"' and not in_brackets:
            in_dquotes = not in_dquotes
            buf.append(ch)
            continue

        if ch == "." and not in_brackets and not in_dquotes:
            part = "".join(buf).strip()
            if part:
                parts.append(part)
            buf = []
        else:
            buf.append(ch)

    tail = "".join(buf).strip()
    if tail:
        parts.append(tail)

    return parts


def _unquote_identifier(part: str) -> str:
    """Remove surrounding [] or "" if present; return raw identifier text."""
    p = part.strip()
    if len(p) >= 2 and p[0] == "[" and p[-1] == "]":
        return p[1:-1]
    if len(p) >= 2 and p[0] == '"' and p[-1] == '"':
        # SQL Server escapes a double quote inside quoted identifiers as ""
        return p[1:-1].replace('""', '"')
    return p


def _quote_identifier_sqlserver(part: str) -> str:
    """Bracket-quote an identifier part; escape closing brackets."""
    raw = _unquote_identifier(part)
    raw = raw.replace("]", "]]")
    return f"[{raw}]"


def _quote_multipart_name_sqlserver(name: str) -> str:
    """
    Quote a potentially multipart SQL Server object name.
    Accepts 1-3 parts (table, schema.table, db.schema.table).
    """
    parts = _split_multipart_name(name)
    if not parts:
        raise ValueError("Empty table name.")
    if len(parts) > 3:
        raise ValueError(f"Too many name parts in table name: {name!r}")
    return ".".join(_quote_identifier_sqlserver(p) for p in parts)


_CHUNK_SUFFIX_RE = re.compile(r"(?:_chunk\d+|_part\d+)$", flags=re.IGNORECASE)


def _infer_table_from_filename(csv_file: str) -> str:
    """
    Infer PascalCase table name from a chunked snake_case filename.

    Examples:
      sales_chunk0001.csv                 -> Sales
      sales_order_detail_chunk0001.csv    -> SalesOrderDetail
      sales_order_header_chunk0001.csv    -> SalesOrderHeader
    """
    base = Path(csv_file).stem
    base = _CHUNK_SUFFIX_RE.sub("", base)
    # snake_case -> PascalCase
    return base.replace("_", " ").title().replace(" ", "")
# Folder name -> canonical fact table name (PascalCase) for facts/<folder>/*.csv
_FOLDER_TABLE_ALIASES: dict[str, str] = {
    # facts
    "sales": "Sales",
    "sales_order_header": "SalesOrderHeader",
    "salesorderheader": "SalesOrderHeader",
    "sales_order_detail": "SalesOrderDetail",
    "salesorderdetail": "SalesOrderDetail",

    # returns
    "sales_return": "SalesReturn",
    "salesreturn": "SalesReturn",
    "returns": "SalesReturn",
    "sales_returns": "SalesReturn",
    "salesreturns": "SalesReturn",
}

def _allowed_lookup(allowed_tables: Optional[Set[str]]) -> Optional[dict[str, str]]:
    """Map lower(table) -> canonical(table) for case-insensitive membership checks."""
    if not allowed_tables:
        return None
    return {str(t).strip().lower(): t for t in allowed_tables}

def _pick_target_table(csv_path: Path, *, allowed_tables: Optional[Set[str]]) -> Optional[str]:
    """
    Determine the target table for a csv_path.
    Prefer folder-based mapping (facts/<table_folder>/file.csv), then fallback to filename inference.
    Applies allowed_tables filter case-insensitively.
    """
    allowed_map = _allowed_lookup(allowed_tables)

    # 1) folder-based mapping (best for files like sales_return_chunk0000.csv)
    folder_key = csv_path.parent.name.strip().lower()
    candidate = _FOLDER_TABLE_ALIASES.get(folder_key)

    # 2) fallback: filename inference
    if candidate is None:
        candidate = _infer_table_from_filename(csv_path.name)

    if allowed_map is None:
        return candidate

    return allowed_map.get(str(candidate).strip().lower())


def _returns_enabled_from_cfg(cfg: Optional[Mapping]) -> bool:
    """
    Enable SalesReturn BULK INSERT if returns are enabled in config.

    Supports:
      - facts: ['sales','returns']
      - facts: { enabled: ['sales','returns'] }
      - facts: { returns: true/false }
      - facts: { enabled: { returns: true/false } }

    If no explicit selection is present, defaults to True.
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

    # facts: [..]
    if isinstance(facts_cfg, list):
        return _list_has_returns(facts_cfg)

    # missing or non-dict -> default enable
    if facts_cfg is None or not isinstance(facts_cfg, Mapping):
        return True

    # facts: { returns: bool }
    if isinstance(facts_cfg.get("returns"), (bool, int)):
        return bool(facts_cfg.get("returns"))

    enabled_cfg = facts_cfg.get("enabled")

    # facts: { enabled: [..] }
    if isinstance(enabled_cfg, list):
        return _list_has_returns(enabled_cfg)

    # facts: { enabled: { returns: bool } }
    if isinstance(enabled_cfg, Mapping) and isinstance(enabled_cfg.get("returns"), (bool, int)):
        return bool(enabled_cfg.get("returns"))

    return True


def _allowed_fact_tables_from_cfg(cfg: Optional[Mapping]) -> Optional[Set[str]]:
    """
    Build the set of fact tables we want in the facts bulk insert script.

    - sales.sales_output drives Sales vs SalesOrderHeader/Detail
    - facts/returns flags (or facts.enabled list) control SalesReturn

    Returns None if cfg is None (meaning: don't filter; include all inferred tables).
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
def _iter_csv_files(csv_folder: Path, *, recursive: bool) -> Iterable[Path]:
    """Yield .csv files (case-insensitive on suffix) from a folder (optionally recursive)."""
    if recursive:
        yield from sorted(
            p for p in csv_folder.rglob("*")
            if p.is_file() and p.suffix.lower() == ".csv"
        )
    else:
        yield from sorted(
            p for p in csv_folder.iterdir()
            if p.is_file() and p.suffix.lower() == ".csv"
        )


def generate_bulk_insert_script(
    csv_folder,
    table_name=None,
    output_sql_file="bulk_insert.sql",
    field_terminator=",",
    row_terminator="0x0a",
    codepage="65001",
    mode="legacy",  # "legacy" | "csv"
    *,
    first_row=2,
    csv_field_quote=None,
    csv_row_terminator=None,
    csv_field_terminator=None,
    max_errors=None,
    error_file=None,
    keep_nulls=False,
    recursive: bool = False,
    allowed_tables: Optional[Set[str]] = None,
):
    """
    Generate a BULK INSERT SQL script for CSV files in a folder.

    Enhancements vs previous version:
      - recursive=True scans subfolders (needed for facts layout)
      - filename inference strips _chunkNNNN suffixes
      - allowed_tables lets you filter inserts conditionally (based on config)

    Notes:
    - BULK INSERT file paths are resolved on the SQL Server machine.
      If your SQL Server is remote, you likely need a UNC path (\\\\server\\share\\file.csv)
      or another server-accessible location.
    - mode="csv" requires SQL Server 2017+ (FORMAT='CSV').

    Backward-compat:
    - If output_sql_file is a bare filename (no directories), the script is written to:
        <csv_folder>/../load/<output_sql_file>
    - If output_sql_file includes a directory (relative or absolute), that path is honored.
    """

    csv_folder = Path(csv_folder)
    if not csv_folder.exists() or not csv_folder.is_dir():
        skip(f"CSV folder not found or not a directory: {csv_folder}")
        return None

    out_path = Path(output_sql_file)

    # Honor explicit paths; keep legacy behavior only for a bare filename.
    if out_path.is_absolute() or out_path.parent != Path("."):
        target_sql = out_path
    else:
        target_sql = csv_folder.parent / "load" / out_path.name

    target_sql.parent.mkdir(parents=True, exist_ok=True)

    csv_paths = list(_iter_csv_files(csv_folder, recursive=recursive))
    if not csv_paths:
        skip(f"No CSV files found in {csv_folder}. Skipping BULK INSERT script.")
        return None

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = [
        "-- Auto-generated BULK INSERT script",
        f"-- Generated on: {timestamp}",
        "-- NOTE: 'FROM <path>' is evaluated on the SQL Server host.",
        "SET NOCOUNT ON;",
        "",
    ]

    emitted = 0

    for csv_path in csv_paths:
        if table_name is not None:
            target_table = table_name
        else:
            target_table = _pick_target_table(csv_path, allowed_tables=allowed_tables)
            if target_table is None:
                # filtered out by allowed_tables (case-insensitive) or unknown mapping
                continue

        quoted_table = _quote_multipart_name_sqlserver(str(target_table))

        csv_full_path = str(csv_path.resolve())
        csv_full_path_sql = _sql_escape_literal(csv_full_path)

        with_options: list[str] = []
        with_options.append(f"FIRSTROW = {int(first_row)}")

        # Optional diagnostics
        if max_errors is not None:
            with_options.append(f"MAXERRORS = {int(max_errors)}")
        if error_file is not None:
            with_options.append(f"ERRORFILE = '{_sql_escape_literal(str(error_file))}'")

        if keep_nulls:
            with_options.append("KEEPNULLS")

        if mode == "csv":
            with_options.insert(0, "FORMAT = 'CSV'")
            with_options.append(f"CODEPAGE = '{codepage}'")

            if csv_field_quote is not None:
                with_options.append(f"FIELDQUOTE = '{_sql_escape_literal(str(csv_field_quote))}'")
            if csv_row_terminator is not None:
                with_options.append(f"ROWTERMINATOR = '{_sql_escape_literal(str(csv_row_terminator))}'")
            if csv_field_terminator is not None:
                with_options.append(f"FIELDTERMINATOR = '{_sql_escape_literal(str(csv_field_terminator))}'")

            with_options.append("TABLOCK")
        else:
            with_options.append(f"FIELDTERMINATOR = '{_sql_escape_literal(str(field_terminator))}'")
            with_options.append(f"ROWTERMINATOR = '{_sql_escape_literal(str(row_terminator))}'")
            with_options.append(f"CODEPAGE = '{codepage}'")
            with_options.append("TABLOCK")

        opts_sql = ",\n    ".join(with_options)

        rel_hint = str(csv_path.relative_to(csv_folder)) if recursive else csv_path.name

        stmt = f"""-- Source file: {rel_hint}
BULK INSERT {quoted_table}
FROM '{csv_full_path_sql}'
WITH (
    {opts_sql}
);
"""
        lines.append(stmt.strip())
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
):
    """
    Convenience wrapper that ALWAYS writes exactly two files:
      01_bulk_insert_dims.sql
      02_bulk_insert_facts.sql  (conditional by cfg['sales']['sales_output'])

    - dims: flat folder
    - facts: recursive scan (handles facts/<table>/*.csv layout)
    """
    load_output_folder = Path(load_output_folder)
    load_output_folder.mkdir(parents=True, exist_ok=True)

    dims_sql = load_output_folder / dims_sql_name
    facts_sql = load_output_folder / facts_sql_name

    generate_bulk_insert_script(
        csv_folder=str(dims_folder),
        table_name=None,
        output_sql_file=str(dims_sql),
        mode=dims_mode,
    )

    allowed_tables = _allowed_fact_tables_from_cfg(cfg)

    generate_bulk_insert_script(
        csv_folder=str(facts_folder),
        table_name=None,
        output_sql_file=str(facts_sql),
        mode=facts_mode,
        row_terminator=row_terminator,
        recursive=True,
        allowed_tables=allowed_tables,
    )

    return str(dims_sql), str(facts_sql)
