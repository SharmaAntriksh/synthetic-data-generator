import os
from datetime import datetime
from pathlib import Path
from src.utils.logging_utils import work, skip


def generate_bulk_insert_script(
    csv_folder,
    table_name=None,
    output_sql_file="bulk_insert.sql",
    field_terminator=",",
    row_terminator="0x0a",
    codepage="65001",
    mode="legacy",   # "legacy" | "csv"
):
    """
    Generate a BULK INSERT SQL script for all CSV files in a folder.
    """

    csv_folder = Path(csv_folder)

    output_sql_file = Path(output_sql_file)

    # If caller provides a path, use it as the anchor
    if output_sql_file.is_absolute():
        load_dir = output_sql_file.parent
    else:
        # Fallback (legacy behavior)
        load_dir = csv_folder.parent / "load"
        output_sql_file = load_dir / output_sql_file

    load_dir.mkdir(parents=True, exist_ok=True)

    # Collect CSV files
    csv_files = sorted(
        f for f in os.listdir(csv_folder)
        if f.lower().endswith(".csv")
    )

    if not csv_files:
        skip(f"No CSV files found in {csv_folder}. Skipping BULK INSERT script.")
        return None

    # Script header
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "-- Auto-generated BULK INSERT script",
        f"-- Generated on: {timestamp}",
        ""
    ]

    # Build BULK INSERT statements (FIXED)
    for csv_file in csv_files:
        base = os.path.splitext(csv_file)[0]

        inferred_table = (
            table_name
            or base.replace("_", " ").title().replace(" ", "")
        )

        csv_full_path = os.path.abspath(csv_folder / csv_file)

        if mode == "csv":
            stmt = f"""
BULK INSERT {inferred_table}
FROM '{csv_full_path}'
WITH (
    FORMAT = 'CSV',
    FIRSTROW = 2,
    CODEPAGE = '{codepage}',
    TABLOCK
);
"""
        else:  # legacy
            stmt = f"""
BULK INSERT {inferred_table}
FROM '{csv_full_path}'
WITH (
    FIRSTROW = 2,
    FIELDTERMINATOR = '{field_terminator}',
    ROWTERMINATOR = '{row_terminator}',
    CODEPAGE = '{codepage}',
    TABLOCK
);
"""

        lines.append(stmt.strip())

    # Write final SQL script
    with open(output_sql_file, "w", encoding="utf-8") as out:
        out.write("\n\n".join(lines))

    work(f"Wrote BULK INSERT script: {Path(output_sql_file).name}")
    return output_sql_file
