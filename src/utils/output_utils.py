from pathlib import Path
import shutil
import pandas as pd
from datetime import datetime
import csv
import pyarrow.parquet as pq

# Delta Lake availability check
try:
    from deltalake import write_deltalake
    DELTA_AVAILABLE = True
except Exception:
    DELTA_AVAILABLE = False


# ============================================================
# Folder Helpers
# ============================================================

def clear_folder(path: str | Path) -> None:
    """Ensure the folder exists and is empty."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    for child in p.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


# ============================================================
# Formatting Helpers
# ============================================================

def format_number_short(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n // 1_000_000_000}B"
    if n >= 1_000_000:
        return f"{n // 1_000_000}M"
    if n >= 1_000:
        return f"{n // 1_000}K"
    return str(n)


# ============================================================
# Counting Helpers
# ============================================================

def count_rows_csv(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        next(f, None)
        return sum(1 for _ in f)


def count_rows_parquet(path: Path) -> int:
    return len(pd.read_parquet(path))


# ============================================================
# Final Output Folder Creator
# ============================================================

def create_final_output_folder(parquet_dims: str | Path,
                               fact_folder: str | Path,
                               file_format: str) -> Path:

    parquet_dims = Path(parquet_dims)
    fact_folder = Path(fact_folder)

    # --------------------------------------------------------
    # Count Customer Rows
    # --------------------------------------------------------
    cust_path = parquet_dims / "customers.parquet"
    customer_rows = count_rows_parquet(cust_path)

    # --------------------------------------------------------
    # Count Sales Rows (Delta-aware)
    # --------------------------------------------------------
    if file_format == "csv":
        fact_files = list(fact_folder.glob("*.csv"))
        sales_rows = sum(count_rows_csv(f) for f in fact_files)

    elif file_format == "deltaparquet":
        delta_dir = fact_folder / "delta"
        part_files = list(delta_dir.glob("*.parquet"))
        sales_rows = sum(count_rows_parquet(f) for f in part_files)

    else:   # parquet mode
        fact_files = list(fact_folder.glob("*.parquet"))
        sales_rows = sum(count_rows_parquet(f) for f in fact_files)

    # --------------------------------------------------------
    # Naming
    # --------------------------------------------------------
    cust_short = format_number_short(customer_rows)
    sales_short = format_number_short(sales_rows)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    base_output_dir = Path("./generated_datasets")
    base_output_dir.mkdir(exist_ok=True)

    final_folder = base_output_dir / f"Customer_{cust_short}__Sales_{sales_short}__{timestamp}"
    final_folder.mkdir(exist_ok=True)

    dims_out = final_folder / "dims"
    dims_out.mkdir(exist_ok=True)

    facts_out = final_folder / "facts"
    facts_out.mkdir(exist_ok=True)

    # --------------------------------------------------------
    # DIMENSIONS
    # --------------------------------------------------------
    dim_files = list(parquet_dims.glob("*.parquet"))
    
    dim_files = [
        f for f in parquet_dims.glob("*.parquet")
        if f.stem.lower() not in {"geography_source", "worldcities"}  # skip raw source
    ]

    if file_format == "csv":
        # Convert dims to CSV
        for f in dim_files:
            df = pd.read_parquet(f)
            df.to_csv(
                dims_out / (f.stem + ".csv"),
                index=False,
                encoding="utf-8",
                quoting=csv.QUOTE_ALL
            )
    else:
        # Parquet and DeltaParquet
        for f in dim_files:
            if file_format == "deltaparquet":
                if not DELTA_AVAILABLE:
                    raise RuntimeError("DeltaParquet mode selected but 'deltalake' is not installed.")
                table = pq.read_table(f)
                dim_delta_out = dims_out / f.stem
                write_deltalake(str(dim_delta_out), table, mode="overwrite")
            else:
                shutil.copy2(f, dims_out / f.name)

    # --------------------------------------------------------
    # FACT FILES
    # --------------------------------------------------------

    if file_format == "deltaparquet":
        # Delta-only output
        delta_src = fact_folder / "delta"
        if not delta_src.exists():
            raise RuntimeError("Delta folder is missing in fact_folder for deltaparquet mode.")

        delta_dest = facts_out / "sales"
        shutil.copytree(delta_src, delta_dest, dirs_exist_ok=True)

        return final_folder

    # CSV or Parquet mode
    fact_files = list(fact_folder.glob("*.parquet")) + list(fact_folder.glob("*.csv"))

    # Copy fact files
    for f in fact_files:
        shutil.copy2(f, facts_out / f.name)

    # CSV conversion for parquet facts
    if file_format == "csv":
        # Copy CSV files produced directly by workers
        for f in fact_folder.glob("*.csv"):
            shutil.copy2(f, facts_out / f.name)


    # Delta dual-write (parquet + delta)
    delta_src = fact_folder / "delta"
    if file_format == "parquet" and delta_src.exists() and any(f.name.startswith("part-") for f in delta_src.glob("*.parquet")):
        delta_dest = facts_out / "sales"
        shutil.copytree(delta_src, delta_dest, dirs_exist_ok=True)

    return final_folder
