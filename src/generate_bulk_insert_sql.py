import os
from datetime import datetime


def generate_bulk_insert_script(
    csv_folder,
    table_name=None,
    output_sql_file="bulk_insert.sql",
    field_terminator=",",
    row_terminator="\\n",
    codepage="65001",
):
    """
    Generate a BULK INSERT SQL script for all CSV files in a folder.
    This version uses CSV mode (no XML format files).
    """

    csv_files = sorted(
        f for f in os.listdir(csv_folder)
        if f.lower().endswith(".csv")
    )

    if not csv_files:
        print(f"No CSV files found in {csv_folder}. Skipping BULK INSERT script.")
        return None

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"-- Auto-generated BULK INSERT script",
        f"-- Generated on: {timestamp}",
        ""
    ]

    for csv_file in csv_files:

        # FIX: Force table name when provided (Sales), otherwise infer (Dimensions)
        if table_name is not None:
            inferred_table = table_name
        else:
            inferred_table = os.path.splitext(csv_file)[0].capitalize()

        csv_full_path = os.path.abspath(os.path.join(csv_folder, csv_file))

        stmt = f"""
BULK INSERT {inferred_table}
FROM '{csv_full_path}'
WITH (
    FORMAT = 'CSV',
    FIRSTROW = 2,
    FIELDTERMINATOR = '{field_terminator}',
    ROWTERMINATOR = '{row_terminator}',
    CODEPAGE = '{codepage}',
    TABLOCK
);
"""
        lines.append(stmt.strip())


    # Write the script
    with open(output_sql_file, "w", encoding="utf-8") as out:
        out.write("\n\n".join(lines))

    print(f"CSV Bulk Insert SQL created at: {output_sql_file}")
    return output_sql_file
