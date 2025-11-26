import os
from src.static_schemas import STATIC_SCHEMAS

def create_table_from_static_schema(table_name, cols):
    lines = [f"CREATE TABLE [{table_name}] ("]
    for col, dtype in cols:
        lines.append(f"    [{col}] {dtype},")
    lines[-1] = lines[-1].rstrip(",")
    lines.append(");")
    return "\n".join(lines)


def generate_all_create_tables(dim_folder, fact_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    dim_out = os.path.join(output_folder, "create_dimensions.sql")
    fact_out = os.path.join(output_folder, "create_facts.sql")

    dim_scripts = []
    fact_scripts = []

    # ---- DIMENSION TABLES (static) ----
    for f in sorted(os.listdir(dim_folder)):
        if f.lower().endswith(".csv"):
            name = os.path.splitext(f)[0].capitalize()
            if name in STATIC_SCHEMAS:
                dim_scripts.append(create_table_from_static_schema(
                    name, STATIC_SCHEMAS[name]
                ))

    # ---- FACT TABLE (only ONE static schema) ----
    fact_scripts.append(create_table_from_static_schema(
        "Sales", STATIC_SCHEMAS["Sales"]
    ))

    # write files
    with open(dim_out, "w", encoding="utf-8") as f:
        f.write("\n\n".join(dim_scripts))

    with open(fact_out, "w", encoding="utf-8") as f:
        f.write("\n\n".join(fact_scripts))

    print("✔ CREATE TABLE scripts generated:")
    print(f"  - {dim_out}")
    print(f"  - {fact_out}")

    return dim_out, fact_out
