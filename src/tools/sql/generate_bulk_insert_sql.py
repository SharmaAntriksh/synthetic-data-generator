import os
from datetime import datetime
from pathlib import Path

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


def _infer_table_from_filename(csv_file: str) -> str:
    base = Path(csv_file).stem
    # e.g. sales_order_detail -> SalesOrderDetail
    return base.replace("_", " ").title().replace(" ", "")


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
):
    """
    Generate a BULK INSERT SQL script for all CSV files in a folder.

    Notes:
    - BULK INSERT file paths are resolved on the SQL Server machine.
      If your SQL Server is remote, you likely need a UNC path (\\\\server\\share\\file.csv)
      or another server-accessible location.
    - mode="csv" requires SQL Server 2017+ (FORMAT='CSV').

    Backward-compat:
    - If output_sql_file is a bare filename (no directories), the script is written to:
        <csv_folder>/../load/<output_sql_file>
      (same as the previous behavior).
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

    # Collect CSV files (stable order)
    csv_files = sorted(
        f for f in os.listdir(csv_folder)
        if f.lower().endswith(".csv")
    )

    if not csv_files:
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

    for csv_file in csv_files:
        inferred = _infer_table_from_filename(csv_file)
        target_table = table_name or inferred
        quoted_table = _quote_multipart_name_sqlserver(str(target_table))

        csv_full_path = str((csv_folder / csv_file).resolve())
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

            # Optional CSV knobs (left unset by default to preserve current output)
            if csv_field_quote is not None:
                with_options.append(f"FIELDQUOTE = '{_sql_escape_literal(str(csv_field_quote))}'")
            if csv_row_terminator is not None:
                with_options.append(f"ROWTERMINATOR = '{_sql_escape_literal(str(csv_row_terminator))}'")
            if csv_field_terminator is not None:
                with_options.append(f"FIELDTERMINATOR = '{_sql_escape_literal(str(csv_field_terminator))}'")

            with_options.append("TABLOCK")
        else:
            # Legacy: explicit terminators
            with_options.append(f"FIELDTERMINATOR = '{_sql_escape_literal(str(field_terminator))}'")
            with_options.append(f"ROWTERMINATOR = '{_sql_escape_literal(str(row_terminator))}'")
            with_options.append(f"CODEPAGE = '{codepage}'")
            with_options.append("TABLOCK")

        opts_sql = ",\n    ".join(with_options)

        stmt = f"""-- Source file: {csv_file}
BULK INSERT {quoted_table}
FROM '{csv_full_path_sql}'
WITH (
    {opts_sql}
);
"""
        lines.append(stmt.strip())
        lines.append("")  # spacing between statements

    target_sql.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    work(f"Wrote BULK INSERT script: {target_sql.name}")
    return str(target_sql)
