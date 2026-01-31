import os
from pathlib import Path

from src.utils.static_schemas import (
    STATIC_SCHEMAS,
    get_sales_schema,
    get_dates_schema,
)
from src.utils.logging_utils import work


def create_table_from_static_schema(table_name, cols):
    lines = [f"CREATE TABLE [{table_name}] ("]
    for col, dtype in cols:
        lines.append(f"    [{col}] {dtype},")
    lines[-1] = lines[-1].rstrip(",")
    lines.append(");")
    return "\n".join(lines)


def generate_all_create_tables(
    dim_folder,
    fact_folder,
    output_folder,
    cfg,
    skip_order_cols=False,
):
    schema_dir = Path(output_folder) / "schema"
    schema_dir.mkdir(parents=True, exist_ok=True)

    dim_out_path  = schema_dir / "01_create_dimensions.sql"
    fact_out_path = schema_dir / "02_create_facts.sql"

    dim_scripts = []
    fact_scripts = []

    # -------------------------
    # Dimensions
    # -------------------------
    for table_name, cols in STATIC_SCHEMAS.items():
        if table_name == "Sales":
            continue

        # 🔑 Dates schema must respect config
        if table_name == "Dates":
            dates_cols = get_dates_schema(cfg["dates"])
            dim_scripts.append(
                create_table_from_static_schema("Dates", dates_cols)
            )
        else:
            dim_scripts.append(
                create_table_from_static_schema(table_name, cols)
            )

    # -------------------------
    # Sales Fact Table
    # -------------------------
    sales_schema = get_sales_schema(skip_order_cols)
    fact_scripts.append(
        create_table_from_static_schema("Sales", sales_schema)
    )

    # -------------------------
    # Write outputs
    # -------------------------
    with open(dim_out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(dim_scripts))

    with open(fact_out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(fact_scripts))

    work(f"Created {Path(dim_out_path).name}")
    work(f"Created {Path(fact_out_path).name}")

    return dim_out_path, fact_out_path
