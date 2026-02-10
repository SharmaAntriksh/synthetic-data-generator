from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence, Tuple

from src.utils.static_schemas import (
    STATIC_SCHEMAS,
    get_sales_schema,
    get_dates_schema,
)
from src.utils.logging_utils import work

ColumnSpec = Sequence[Tuple[str, str]]

# Fact table names (SQL tables should remain PascalCase)
TABLE_SALES = "Sales"
TABLE_SALES_ORDER_HEADER = "SalesOrderHeader"
TABLE_SALES_ORDER_DETAIL = "SalesOrderDetail"

# Tables that should be emitted in the Facts script (not Dimensions),
# even though they live inside STATIC_SCHEMAS.
_FACT_TABLE_NAMES = {
    TABLE_SALES,
    TABLE_SALES_ORDER_HEADER,
    TABLE_SALES_ORDER_DETAIL,
}


def _quote_ident(name: str) -> str:
    """SQL Server identifier quoting with [] escaping."""
    return f"[{name.replace(']', ']]')}]"


def _qualify(schema: str, table: str) -> str:
    """Return schema-qualified table name: [schema].[table]."""
    return f"{_quote_ident(schema)}.{_quote_ident(table)}"


def _sales_output_mode(cfg: Mapping) -> str:
    sales_cfg = cfg.get("sales") or {}
    mode = str(sales_cfg.get("sales_output", "sales")).lower().strip()
    if mode not in {"sales", "sales_order", "both"}:
        raise ValueError(f"Invalid sales.sales_output: {mode!r}. Expected sales|sales_order|both.")
    return mode


def _skip_order_cols(cfg: Mapping, default: bool) -> bool:
    sales_cfg = cfg.get("sales") or {}
    # config should win if present
    if "skip_order_cols" in sales_cfg:
        return bool(sales_cfg.get("skip_order_cols"))
    return bool(default)


def _require_static_schema(table_name: str) -> ColumnSpec:
    """
    Fetch schema from STATIC_SCHEMAS with a helpful error if missing.
    STATIC_SCHEMAS is the source of truth for all non-Sales facts.
    """
    try:
        return STATIC_SCHEMAS[table_name]
    except KeyError as e:
        raise KeyError(
            f"STATIC_SCHEMAS is missing a schema for '{table_name}'. "
            f"Add it to src/utils/static_schemas.py."
        ) from e


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
    if lines and lines[-1].endswith(","):
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

    Facts included are conditional on cfg['sales']['sales_output']:
      - "sales"       -> Sales
      - "sales_order" -> SalesOrderHeader + SalesOrderDetail
      - "both"        -> Sales + SalesOrderHeader + SalesOrderDetail

    Notes:
      - SQL table names are PascalCase.
      - Sales schema is generated via get_sales_schema() to support skip_order_cols.
      - SalesOrderHeader/Detail schemas come from STATIC_SCHEMAS (no hard-coded columns here).
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
    # Dimensions (exclude fact tables)
    # -------------------------
    dim_scripts: list[str] = []
    for table_name in sorted(STATIC_SCHEMAS.keys()):
        if table_name in _FACT_TABLE_NAMES:
            continue

        if table_name == "Dates":
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
    # Facts (single script)
    # -------------------------
    mode = _sales_output_mode(cfg)
    include_sales = mode in {"sales", "both"}
    include_sales_order = mode in {"sales_order", "both"}

    effective_skip_order_cols = _skip_order_cols(cfg, skip_order_cols)

    fact_scripts: list[str] = []

    # Sales
    if include_sales:
        sales_schema = get_sales_schema(effective_skip_order_cols)
        fact_scripts.append(
            create_table_from_static_schema(
                TABLE_SALES,
                sales_schema,
                schema=schema,
                drop_existing=drop_existing,
                include_go=True,
            )
        )

    # SalesOrderHeader / SalesOrderDetail (no hard-coded columns; use STATIC_SCHEMAS)
    if include_sales_order:
        header_schema = _require_static_schema(TABLE_SALES_ORDER_HEADER)
        detail_schema = _require_static_schema(TABLE_SALES_ORDER_DETAIL)

        # Order-level first, then line-level
        fact_scripts.append(
            create_table_from_static_schema(
                TABLE_SALES_ORDER_HEADER,
                header_schema,
                schema=schema,
                drop_existing=drop_existing,
                include_go=True,
            )
        )
        fact_scripts.append(
            create_table_from_static_schema(
                TABLE_SALES_ORDER_DETAIL,
                detail_schema,
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
