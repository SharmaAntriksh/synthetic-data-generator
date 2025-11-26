import os
import pandas as pd
import keyword


# Mapping pandas dtype → SQL Server datatype
SQL_TYPE_MAP = {
    "int64": "INT",
    "float64": "FLOAT",
    "object": "VARCHAR(500)",
    "datetime64[ns]": "DATE",
    "bool": "BIT"
}


def infer_sql_type(col_series):
    """
    Smarter SQL type inference:
    - Preserve leading zeros → VARCHAR
    - Mixed numeric/text → VARCHAR
    - Pure integer → INT
    - Pure float → FLOAT
    - Dates → DATE
    """
    # Convert to string for safe inspection
    sample_values = col_series.dropna().astype(str).head(20)

    # 1. Detect leading zeros (e.g., "01", "007")
    if any(v.startswith("0") and v != "0" for v in sample_values):
        return "VARCHAR(500)"

    # 2. Mixed types → VARCHAR
    if any(not v.isdigit() for v in sample_values):
        # could be float, date, text - fallback VARCHAR
        return "VARCHAR(500)"

    # 3. Pure integer
    if col_series.dtype == "int64":
        return "INT"

    # 4. Pure float
    if col_series.dtype == "float64":
        return "FLOAT"

    # 5. Date (format must look like YYYY-MM-DD)
    if all(len(v) == 10 and v[4] == "-" and v[7] == "-" for v in sample_values):
        return "DATE"

    # fallback
    return "VARCHAR(500)"



def create_table_script_from_csv(csv_path, table_name):
    df = pd.read_csv(csv_path, nrows=500)

    # Clean BOM if present
    df.columns = [c.replace("\ufeff", "") for c in df.columns]

    cols = df.columns.tolist()

    lines = [f"CREATE TABLE {table_name} ("]

    for col in cols:
        sql_col = f"[{col}]"
        dtype = infer_sql_type(df[col])   # ✔ FIXED
        lines.append(f"    {sql_col} {dtype},")
    
    lines[-1] = lines[-1].rstrip(",")

    lines.append(");")
    return "\n".join(lines)



def generate_all_create_tables(dim_folder, fact_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    dim_out = os.path.join(output_folder, "create_dimensions.sql")
    fact_out = os.path.join(output_folder, "create_facts.sql")

    dim_scripts = []
    fact_scripts = []

    # ---------- DIMENSIONS ----------
    for f in sorted(os.listdir(dim_folder)):
        if f.endswith(".csv"):
            table = os.path.splitext(f)[0].capitalize()
            csv_path = os.path.join(dim_folder, f)
            script = create_table_script_from_csv(csv_path, table)
            dim_scripts.append(script + "\n")

    # ---------- FACTS ----------
    for f in sorted(os.listdir(fact_folder)):
        if f.endswith(".csv"):
            table = os.path.splitext(f)[0].capitalize()
            csv_path = os.path.join(fact_folder, f)
            script = create_table_script_from_csv(csv_path, table)
            fact_scripts.append(script + "\n")

    # Write output files
    with open(dim_out, "w", encoding="utf-8") as f:
        f.write("\n".join(dim_scripts))

    with open(fact_out, "w", encoding="utf-8") as f:
        f.write("\n".join(fact_scripts))

    print("✔ CREATE TABLE scripts generated:")
    print(f"  - {dim_out}")
    print(f"  - {fact_out}")
