import os
from src.static_schemas import STATIC_SCHEMAS


def create_table_from_static_schema(table_name, cols):
    """
    Build a CREATE TABLE SQL script from a static schema definition.
    cols = list of (column_name, SQL_datatype)
    """
    lines = [f"CREATE TABLE [{table_name}] ("]
    for col, dtype in cols:
        lines.append(f"    [{col}] {dtype},")
    lines[-1] = lines[-1].rstrip(",")  # remove trailing comma
    lines.append(");")
    return "\n".join(lines)


def generate_all_create_tables(dim_folder, fact_folder, output_folder):
    """
    Generate CREATE TABLE scripts for:
      - All dimension tables found in dim_folder (CSV only)
      - The Sales fact table (always)

    Outputs:
      create_dimensions.sql
      create_facts.sql
    """
    os.makedirs(output_folder, exist_ok=True)

    dim_out_path = os.path.join(output_folder, "create_dimensions.sql")
    fact_out_path = os.path.join(output_folder, "create_facts.sql")

    dim_scripts = []
    fact_scripts = []

    # ------------------------------------------------------------
    # Dimension Tables (static schemas)
    # ------------------------------------------------------------
    # Dimensions are ALWAYS CSV at this stage (per main.py logic)
    for fname in sorted(os.listdir(dim_folder)):
        if not fname.lower().endswith(".csv"):
            continue  # skip parquet / other files safely

        table_name = os.path.splitext(fname)[0].capitalize()

        # Only generate scripts for tables present in STATIC_SCHEMAS
        if table_name in STATIC_SCHEMAS:
            dim_scripts.append(
                create_table_from_static_schema(table_name, STATIC_SCHEMAS[table_name])
            )

    # ------------------------------------------------------------
    # Fact table (Sales) - ALWAYS ONLY ONE
    # ------------------------------------------------------------
    fact_scripts.append(
        create_table_from_static_schema("Sales", STATIC_SCHEMAS["Sales"])
    )

    # ------------------------------------------------------------
    # Write final SQL files
    # ------------------------------------------------------------
    with open(dim_out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(dim_scripts))

    with open(fact_out_path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(fact_scripts))

    print("✔ CREATE TABLE scripts generated:")
    print(f"  - {dim_out_path}")
    print(f"  - {fact_out_path}")

    return dim_out_path, fact_out_path
