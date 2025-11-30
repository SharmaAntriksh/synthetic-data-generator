from pathlib import Path
import shutil
import pandas as pd
from datetime import datetime
import csv
import pyarrow.parquet as pq


# ============================================================
# Optional Delta Support
# ============================================================
try:
    from deltalake import write_deltalake
    DELTA_AVAILABLE = True
except Exception:
    DELTA_AVAILABLE = False


# ============================================================
# Folder Utilities
# ============================================================
def clear_folder(path: str | Path) -> None:
    """Ensure folder exists and is empty."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    for child in p.iterdir():
        if child.is_file() or child.is_symlink():
            child.unlink()
        else:
            shutil.rmtree(child)


# ============================================================
# Helpers
# ============================================================
def format_number_short(n: int) -> str:
    """Short numeric formatter: 12000 → 12K."""
    if n >= 1_000_000_000: return f"{n // 1_000_000_000}B"
    if n >= 1_000_000:     return f"{n // 1_000_000}M"
    if n >= 1_000:         return f"{n // 1_000}K"
    return str(n)


def count_rows_csv(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        next(f, None)
        return sum(1 for _ in f)


def count_rows_parquet(path: Path) -> int:
    """Count rows without loading the data (zero memory)."""
    pf = pq.ParquetFile(path)
    return pf.metadata.num_rows


# ============================================================
# Final Output Folder Builder
# ============================================================
def create_final_output_folder(parquet_dims: str | Path,
                               fact_folder: str | Path,
                               file_format: str,
                               sales_rows_expected: int = None) -> Path:


    parquet_dims = Path(parquet_dims)
    fact_folder = Path(fact_folder)

    # --------------------------------------------------------
    # Count Customers
    # --------------------------------------------------------
    customer_rows = count_rows_parquet(parquet_dims / "customers.parquet")

# --------------------------------------------------------
    # Sales row count (use config.json instead of counting)
    # --------------------------------------------------------
    if sales_rows_expected is None:
        raise RuntimeError("sales_rows_expected must be provided to create_final_output_folder.")

    sales_rows = sales_rows_expected

    # --------------------------------------------------------
    # Name Output Folder
    # --------------------------------------------------------
    cust_short = format_number_short(customer_rows)
    sales_short = format_number_short(sales_rows)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    final_root = Path("./generated_datasets")
    final_root.mkdir(exist_ok=True)

    final_folder = final_root / f"Customer_{cust_short}__Sales_{sales_short}__{timestamp}"
    final_folder.mkdir(exist_ok=True)

    dims_out = final_folder / "dims"
    dims_out.mkdir(exist_ok=True)

    facts_out = final_folder / "facts"
    facts_out.mkdir(exist_ok=True)

    # --------------------------------------------------------
    # DIMENSIONS
    # --------------------------------------------------------
    dim_files = [
        f for f in parquet_dims.glob("*.parquet")
        if f.stem.lower() not in {"geography_source", "worldcities"}
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
        # Parquet or DeltaParquet
        for f in dim_files:
            if file_format == "deltaparquet":
                if not DELTA_AVAILABLE:
                    raise RuntimeError("DeltaParquet mode selected but 'deltalake' is not installed.")
                table = pq.read_table(f)
                write_deltalake(str(dims_out / f.stem), table, mode="overwrite")
            else:
                shutil.copy2(f, dims_out / f.name)

    # --------------------------------------------------------
    # FACT FILES
    # --------------------------------------------------------

    # Delta-only mode
    if file_format == "deltaparquet":

        # 1) Try modern sales.py output: fact_folder/sales/delta
        delta_src = fact_folder / "sales" / "delta"

        # 2) Fallback: fact_folder/delta
        if not delta_src.exists():
            delta_src = fact_folder / "delta"

        # 3) Final fallback: any delta subfolder under fact_folder
        if not delta_src.exists():
            for child in fact_folder.iterdir():
                if child.is_dir() and (child / "_delta_log").exists():
                    delta_src = child
                    break

        # 4) If still not found → real error
        if not delta_src.exists():
            raise RuntimeError("Delta output folder missing for deltaparquet mode.")

        shutil.copytree(delta_src, facts_out / "sales", dirs_exist_ok=True)
        return final_folder


    # CSV + PARQUET modes
    for f in fact_folder.glob("*.*"):
        if f.suffix.lower() in {".csv", ".parquet"}:
            shutil.copy2(f, facts_out / f.name)

    # If also delta was written alongside parquet
    delta_src = fact_folder / "delta"
    if file_format == "parquet" and delta_src.exists():
        if any(f.name.startswith("part-") for f in delta_src.glob("*.parquet")):
            shutil.copytree(delta_src, facts_out / "sales", dirs_exist_ok=True)

    return final_folder
