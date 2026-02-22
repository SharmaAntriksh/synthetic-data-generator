from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence, Tuple

from src.utils.static_schemas import STATIC_SCHEMAS, get_dates_schema, get_sales_schema
from src.utils.logging_utils import work

ColumnSpec = Sequence[Tuple[str, str]]

# Fact table names (SQL tables should remain PascalCase)
TABLE_SALES = "Sales"
TABLE_SALES_ORDER_HEADER = "SalesOrderHeader"
TABLE_SALES_ORDER_DETAIL = "SalesOrderDetail"
TABLE_SALES_RETURN = "SalesReturn"

# Tables that should be emitted in the Facts script (not Dimensions),
# even though they live inside STATIC_SCHEMAS.
_FACT_TABLE_NAMES = {
    TABLE_SALES,
    TABLE_SALES_ORDER_HEADER,
    TABLE_SALES_ORDER_DETAIL,
    TABLE_SALES_RETURN,
}


def _quote_ident(name: str) -> str:
    return f"[{str(name).replace(']', ']]')}]"


def _qualify(schema: str, table: str) -> str:
    return f"{_quote_ident(schema)}.{_quote_ident(table)}"


def _sales_output_mode(cfg: Mapping) -> str:
    mode = str((cfg.get("sales") or {}).get("sales_output", "sales")).lower().strip()
    if mode not in {"sales", "sales_order", "both"}:
        raise ValueError(f"Invalid sales.sales_output: {mode!r}. Expected sales|sales_order|both.")
    return mode


def _skip_order_cols(cfg: Mapping, default: bool) -> bool:
    sales_cfg = cfg.get("sales") or {}
    return bool(sales_cfg.get("skip_order_cols", default))


def _returns_enabled(cfg: Mapping) -> bool:
    """
    Determine whether SalesReturn should be emitted.

    Supported config shapes (any one):
      - facts: ['sales','returns', ...]
      - facts.enabled: ['sales','returns', ...]
      - facts.returns: true|false
      - facts.enabled.returns: true|false

    Semantics:
      - If an explicit enabled LIST is provided, returns must be present to enable SalesReturn.
      - If no explicit selection is provided, defaults to True.
    """
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


def _require_static_schema(table_name: str) -> ColumnSpec:
    try:
        return STATIC_SCHEMAS[table_name]
    except KeyError as e:
        raise KeyError(
            f"STATIC_SCHEMAS is missing a schema for '{table_name}'. "
            f"Add it to src/utils/static_schemas.py."
        ) from e


def create_table_from_schema(
    table_name: str,
    cols: ColumnSpec,
    *,
    schema: str = "dbo",
    drop_existing: bool = True,
    include_go: bool = True,
) -> str:
    fq_table = _qualify(schema, table_name)
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
    if lines[-1].endswith(","):
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

    Schemas are sourced from:
      - STATIC_SCHEMAS for everything except Sales and Dates
      - get_sales_schema(skip_order_cols) for Sales
      - get_dates_schema(cfg['dates']) for Dates
    """
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

    # Dimensions
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
            create_table_from_schema(
                table_name,
                cols,
                schema=schema,
                drop_existing=drop_existing,
                include_go=True,
            )
        )

    dim_out_path.write_text("\n".join(header) + "\n\n" + "\n\n".join(dim_scripts) + "\n", encoding="utf-8")

    # Facts
    mode = _sales_output_mode(cfg)
    include_sales = mode in {"sales", "both"}
    include_sales_order = mode in {"sales_order", "both"}
    include_returns = _returns_enabled(cfg)

    eff_skip_order_cols = _skip_order_cols(cfg, skip_order_cols)

    fact_scripts: list[str] = []

    if include_sales:
        fact_scripts.append(
            create_table_from_schema(
                TABLE_SALES,
                get_sales_schema(eff_skip_order_cols),
                schema=schema,
                drop_existing=drop_existing,
                include_go=True,
            )
        )

    if include_sales_order:
        fact_scripts.append(
            create_table_from_schema(
                TABLE_SALES_ORDER_HEADER,
                _require_static_schema(TABLE_SALES_ORDER_HEADER),
                schema=schema,
                drop_existing=drop_existing,
                include_go=True,
            )
        )
        fact_scripts.append(
            create_table_from_schema(
                TABLE_SALES_ORDER_DETAIL,
                _require_static_schema(TABLE_SALES_ORDER_DETAIL),
                schema=schema,
                drop_existing=drop_existing,
                include_go=True,
            )
        )

    # SalesReturn placement:
    # - If sales_order tables exist: emit after detail
    # - Else: emit after Sales (if present) or as the only fact
    if include_returns and (include_sales or include_sales_order):
        fact_scripts.append(
            create_table_from_schema(
                TABLE_SALES_RETURN,
                _require_static_schema(TABLE_SALES_RETURN),
                schema=schema,
                drop_existing=drop_existing,
                include_go=True,
            )
        )

    fact_out_path.write_text("\n".join(header) + "\n\n" + "\n\n".join(fact_scripts) + "\n", encoding="utf-8")

    work(f"Created {dim_out_path.name}")
    work(f"Created {fact_out_path.name}")
    return dim_out_path, fact_out_path
