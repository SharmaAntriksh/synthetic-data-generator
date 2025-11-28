import os
from src.utils.static_schemas import STATIC_SCHEMAS, get_sales_schema


def create_table_from_static_schema(table_name, cols):
    lines = [f"CREATE TABLE [{table_name}] ("]
    for col, dtype in cols:
        lines.append(f"    [{col}] {dtype},")
    lines[-1] = lines[-1].rstrip(",")
    lines.append(");")
    return "\n".join(lines)


def generate_all_create_tables(dim_folder, fact_folder, output_folder, skip_order_cols=False):
    os.makedirs(output_folder, exist_ok=True)

    dim_out_path = os.path.join(output_folder, "create_dimensions.sql")
    fact_out_path = os.path.join(output_folder, "create_facts.sql")

    dim_scripts = []
    fact_scripts = []

    # -------------------------
    # Dimensions
    # -------------------------
    for fname in sorted(os.listdir(dim_folder)):
        if not fname.lower().endswith(".csv"):
            continue

        base = os.path.splitext(fname)[0]
        table_name = base.replace("_", " ").title().replace(" ", "_")

        if table_name in STATIC_SCHEMAS:
            dim_scripts.append(
                create_table_from_static_schema(table_name, STATIC_SCHEMAS[table_name])
            )

    # -------------------------
    # Sales Fact Table
    # -------------------------
    # main.py will pass the correct flag
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

    print("✔ CREATE TABLE scripts generated:")
    print(f"  - {dim_out_path}")
    print(f"  - {fact_out_path}")

    return dim_out_path, fact_out_path
