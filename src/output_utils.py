import os
import shutil
import pandas as pd
from datetime import datetime
import csv

def clear_folder(path):
    """
    Ensure `path` exists and is empty.
    - If missing: creates it.
    - If present: deletes all files and subfolders inside it.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        return

    for name in os.listdir(path):
        fp = os.path.join(path, name)
        try:
            if os.path.islink(fp) or os.path.isfile(fp):
                os.remove(fp)
            elif os.path.isdir(fp):
                shutil.rmtree(fp)
        except Exception as e:
            print(f"Warning: failed to remove {fp}: {e}")


def format_number_short(n):
    if n >= 1_000_000_000:
        return f"{round(n/1_000_000_000)}B"
    if n >= 1_000_000:
        return f"{round(n/1_000_000)}M"
    if n >= 1_000:
        return f"{round(n/1_000)}K"
    return str(n)


def convert_parquet_dims_to_csv(parquet_dims_folder, output_dims_folder):
    """Convert all parquet dimension files into CSV format."""
    os.makedirs(output_dims_folder, exist_ok=True)

    for f in os.listdir(parquet_dims_folder):
        if f.endswith(".parquet"):
            src = os.path.join(parquet_dims_folder, f)
            df = pd.read_parquet(src)
            out_name = f.replace(".parquet", ".csv")
            dst = os.path.join(output_dims_folder, out_name)
            df.to_csv(dst, index=False, encoding="utf-8", quoting=csv.QUOTE_ALL)


def create_final_output_folder(parquet_dims, fact_folder, file_format):
    """
    Create the final packaged dataset folder inside ./generated_datasets/
    Includes:
        dims/  (csv or parquet)
        facts/ (csv or parquet)
    Returns:
        final folder path
    """

    # ------------------------------
    # Count customer rows
    # ------------------------------
    cust_path = os.path.join(parquet_dims, "customers.parquet")
    customers_df = pd.read_parquet(cust_path, columns=["CustomerKey"])
    customer_rows = len(customers_df)

    # ------------------------------
    # Count sales rows
    # ------------------------------
    sales_rows = 0
    for f in os.listdir(fact_folder):
        fp = os.path.join(fact_folder, f)
        if file_format == "csv" and f.endswith(".csv"):
            sales_rows += sum(1 for _ in open(fp, "r", encoding="utf-8")) - 1
        elif file_format == "parquet" and f.endswith(".parquet"):
            sales_rows += len(pd.read_parquet(fp))

    cust_short = format_number_short(customer_rows)
    sales_short = format_number_short(sales_rows)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # ------------------------------
    # SAFE folder name in generated_datasets/
    # ------------------------------
    base_output_dir = "./generated_datasets"
    os.makedirs(base_output_dir, exist_ok=True)

    folder_name = os.path.join(
        base_output_dir,
        f"Customer {cust_short} - Sales {sales_short} - {timestamp}"
    )
    os.makedirs(folder_name, exist_ok=True)

    # ------------------------------
    # Create dims/ and facts/ subfolders
    # ------------------------------
    dim_out = os.path.join(folder_name, "dims")
    fact_out_final = os.path.join(folder_name, "facts")
    os.makedirs(dim_out, exist_ok=True)
    os.makedirs(fact_out_final, exist_ok=True)

    # ------------------------------
    # Dimensions: convert or copy
    # ------------------------------
    if file_format == "csv":
        convert_parquet_dims_to_csv(parquet_dims, dim_out)
    else:
        # Copy Parquet dims
        for f in os.listdir(parquet_dims):
            shutil.copy2(os.path.join(parquet_dims, f), os.path.join(dim_out, f))

    # ------------------------------
    # Facts: copy CSV or Parquet files
    # ------------------------------
    for f in os.listdir(fact_folder):
        shutil.copy2(os.path.join(fact_folder, f), os.path.join(fact_out_final, f))

    return folder_name
