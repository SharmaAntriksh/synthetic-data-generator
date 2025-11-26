from pathlib import Path
import shutil
import pandas as pd
from datetime import datetime
import csv


# ============================================================
# Folder Helpers
# ============================================================

def clear_folder(path: str | Path) -> None:
    """
    Ensure `path` exists and is empty.
    Deletes all files/subfolders inside it.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    for child in p.iterdir():
        try:
            if child.is_file() or child.is_symlink():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
        except Exception as e:
            print(f"Warning: failed to remove {child}: {e}")


# ============================================================
# Formatting Helpers
# ============================================================

def format_number_short(n: int) -> str:
    """Format large numbers: 1500 -> 1K, 2M, 3B, etc."""
    if n >= 1_000_000_000:
        return f"{n // 1_000_000_000}B"
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)


# ============================================================
# Dimension Conversion
# ============================================================

def convert_parquet_dims_to_csv(parquet_dims_folder: str | Path,
                                output_dims_folder: str | Path) -> None:
    """
    Convert all .parquet dimension files into CSV format.
    """
    src = Path(parquet_dims_folder)
    dst = Path(output_dims_folder)
    dst.mkdir(parents=True, exist_ok=True)

    for f in src.glob("*.parquet"):
        df = pd.read_parquet(f)
        df.to_csv(
            dst / (f.stem + ".csv"),
            index=False,
            encoding="utf-8",
            quoting=csv.QUOTE_ALL
        )


# ============================================================
# Counting Helpers
# ============================================================

def count_rows_csv(path: Path) -> int:
    """
    Efficient CSV row counter ignoring header.
    """
    with path.open("r", encoding="utf-8", newline="") as f:
        next(f, None)  # skip header
        return sum(1 for _ in f)


def count_rows_parquet(path: Path) -> int:
    """Load parquet and return row count."""
    return len(pd.read_parquet(path))


# ============================================================
# Final Output Folder Creator
# ============================================================

def create_final_output_folder(parquet_dims: str | Path,
                               fact_folder: str | Path,
                               file_format: str) -> Path:
    """
    Create final packaged dataset folder under ./generated_datasets/
    Structure:
        dims/  (csv or parquet)
        facts/ (csv or parquet)
    """
    parquet_dims = Path(parquet_dims)
    fact_folder = Path(fact_folder)

    # --------------------------------------------------------
    # Count Customer Rows
    # --------------------------------------------------------
    cust_path = parquet_dims / "customers.parquet"
    customer_rows = count_rows_parquet(cust_path)

    # --------------------------------------------------------
    # Count Sales Rows
    # --------------------------------------------------------
    if file_format == "csv":
        fact_files = list(fact_folder.glob("*.csv"))
        sales_rows = sum(count_rows_csv(f) for f in fact_files)
    else:
        fact_files = list(fact_folder.glob("*.parquet"))
        sales_rows = sum(count_rows_parquet(f) for f in fact_files)

    # --------------------------------------------------------
    # Naming for final folder
    # --------------------------------------------------------
    cust_short = format_number_short(customer_rows)
    sales_short = format_number_short(sales_rows)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    base_output_dir = Path("./generated_datasets")
    base_output_dir.mkdir(exist_ok=True)

    final_folder = base_output_dir / f"Customer_{cust_short}__Sales_{sales_short}__{timestamp}"
    final_folder.mkdir(exist_ok=True)

    # Prepare subfolders
    dims_out = final_folder / "dims"
    dims_out.mkdir(exist_ok=True)

    facts_out = final_folder / "facts"
    facts_out.mkdir(exist_ok=True)

    # --------------------------------------------------------
    # Dimension Files
    # --------------------------------------------------------
    if file_format == "csv":
        convert_parquet_dims_to_csv(parquet_dims, dims_out)
    else:
        for f in parquet_dims.glob("*.parquet"):
            shutil.copy2(f, dims_out / f.name)

    # --------------------------------------------------------
    # Fact Files
    # --------------------------------------------------------
    for f in fact_files:
        shutil.copy2(f, facts_out / f.name)

    return final_folder
