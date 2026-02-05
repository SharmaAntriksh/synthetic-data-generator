from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence, Tuple

from src.utils.static_schemas import (
    STATIC_SCHEMAS,
    get_sales_schema,
    get_dates_schema,
)
from src.utils.logging_utils import work

ColumnSpec = Sequence[Tuple[str, str]]


def _quote_ident(name: str) -> str:
    """SQL Server identifier quoting with [] escaping."""
    return f"[{name.replace(']', ']]')}]"


def _qualify(schema: str, table: str) -> str:
    """Return schema-qualified table name: [schema].[table]."""
    return f"{_quote_ident(schema)}.{_quote_ident(table)}"


def create_table_from_static_schema(
    table_name: str,
    cols: ColumnSpec,
    *,
    schema: str = "dbo",
    drop_existing: bool = True,
    include_go: bool = True,
) -> str:
    """
    Generate a CREATE TABLE statement from a (col, dtype) schema.
    - schema-qualifies table names ([dbo].[Table])
    - optionally drops existing table first (idempotent reruns)
    - optionally adds GO separators (friendlier to sqlcmd/SSMS)
    """
    fq_table = _qualify(schema, table_name)
    # For OBJECT_ID string literal, you must NOT use brackets inside the name.
    object_id_name = f"{schema}.{table_name}"

    lines: list[str] = []

    if drop_existing:
        lines.append(
            f"IF OBJECT_ID(N'{object_id_name}', N'U') IS NOT NULL\n"
            f"    DROP TABLE {fq_table};"
        )
        if include_go:
            lines.append("GO")

    lines.append(f"CREATE TABLE {fq_table} (")
    for col, dtype in cols:
        lines.append(f"    {_quote_ident(col)} {dtype},")
    if len(lines) > 1 and lines[-1].endswith(","):
        lines[-1] = lines[-1].rstrip(",")
    lines.append(");")

    if include_go:
        lines.append("GO")

    return "\n".join(lines)


def generate_all_create_tables(
    dim_folder,   # kept for backward compat (unused)
    fact_folder,  # kept for backward compat (unused)
    output_folder,
    cfg,
    skip_order_cols: bool = False,
    *,
    schema: str = "dbo",
    drop_existing: bool = True,
):
    """
    Writes:
      <output_folder>/schema/01_create_dimensions.sql
      <output_folder>/schema/02_create_facts.sql

    Improvements over the previous implementation:
    - deterministic table order (sorted)
    - schema-qualified tables ([dbo].)
    - idempotent reruns via DROP TABLE if exists (toggle via drop_existing)
    - GO separators for batching
    - file headers + SET NOCOUNT ON
    """

    # Args currently unused by design; keep them to avoid breaking callers.
    _ = dim_folder
    _ = fact_folder

    schema_dir = Path(output_folder) / "schema"
    schema_dir.mkdir(parents=True, exist_ok=True)

    dim_out_path = schema_dir / "01_create_dimensions.sql"
    fact_out_path = schema_dir / "02_create_facts.sql"

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = [
        "-- Auto-generated CREATE TABLE scripts",
        f"-- Generated on: {ts}",
        "SET NOCOUNT ON;",
        "",
    ]

    # -------------------------
    # Dimensions (excluding Sales)
    # -------------------------
    dim_scripts: list[str] = []
    for table_name in sorted(STATIC_SCHEMAS.keys()):
        if table_name == "Sales":
            continue

        if table_name == "Dates":
            # Dates schema must respect config
            dates_cfg = cfg.get("dates")
            if dates_cfg is None:
                raise KeyError("cfg['dates'] is required to generate Dates schema.")
            cols = get_dates_schema(dates_cfg)
        else:
            cols = STATIC_SCHEMAS[table_name]

        dim_scripts.append(
            create_table_from_static_schema(
                table_name,
                cols,
                schema=schema,
                drop_existing=drop_existing,
                include_go=True,
            )
        )

    dim_out_path.write_text(
        "\n".join(header) + "\n\n" + "\n\n".join(dim_scripts) + "\n",
        encoding="utf-8",
    )

    # -------------------------
    # Facts (Sales only)
    # -------------------------
    fact_scripts: list[str] = []
    sales_schema = get_sales_schema(skip_order_cols)
    fact_scripts.append(
        create_table_from_static_schema(
            "Sales",
            sales_schema,
            schema=schema,
            drop_existing=drop_existing,
            include_go=True,
        )
    )

    fact_out_path.write_text(
        "\n".join(header) + "\n\n" + "\n\n".join(fact_scripts) + "\n",
        encoding="utf-8",
    )

    work(f"Created {dim_out_path.name}")
    work(f"Created {fact_out_path.name}")

    return dim_out_path, fact_out_path
