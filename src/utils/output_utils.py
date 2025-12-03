import os
import shutil
import time
from pathlib import Path
import pyarrow.parquet as pq
from src.utils.logging_utils import stage, done, info
from datetime import datetime


# ============================================================
# Helpers
# ============================================================

def format_number_short(n: int) -> str:
    if n >= 1_000_000_000: return f"{n // 1_000_000_000}B"
    if n >= 1_000_000:     return f"{n // 1_000_000}M"
    if n >= 1_000:         return f"{n // 1_000}K"
    return str(n)


def create_final_output_folder(
    final_folder_root: Path,
    parquet_dims: Path,
    fact_folder: Path,
    sales_cfg: dict,
    file_format: str,
    sales_rows_expected: int,
    cfg: dict
):
    """
    Packs cleaned dimension + fact data according to config rules.
    """
    stage("Creating Final Output Folder")

    # ---------------------------
    # Build human-readable timestamp   (Windows-safe)
    # ---------------------------
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %I_%M_%S %p")  # 12-hour format, AM/PM, safe

    # ---------------------------
    # Short counts
    # ---------------------------
    customer_total = cfg["customers"]["total_customers"]
    sales_total = sales_cfg["total_rows"]

    cust_short = format_number_short(customer_total)
    sales_short = format_number_short(sales_total)

    # ---------------------------
    # Format label
    # ---------------------------
    fmt_label = {
        "deltaparquet": "DeltaParquet",
        "parquet": "Parquet",
        "csv": "CSV",
    }.get(file_format.lower(), file_format)

    # ---------------------------
    # Final folder naming
    # ---------------------------
    dataset_name = (
        f"{timestamp} "
        f"Customers {cust_short} "
        f"Sales {sales_short} "
        f"{fmt_label}"
    )

    final_folder = final_folder_root / dataset_name
    dims_out = final_folder / "dimensions"
    facts_out = final_folder / "facts"

    # --------------------------------------------------------
    # CLEAN THIS DATASET FOLDER ONLY
    # --------------------------------------------------------
    if final_folder.exists():
        shutil.rmtree(final_folder, ignore_errors=True)
    final_folder.mkdir(parents=True, exist_ok=True)

    dims_out.mkdir(parents=True, exist_ok=True)
    facts_out.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # DIMENSIONS HANDLING
    # --------------------------------------------------------
    ff = file_format.lower()

    if ff == "parquet":
        # Copy dimension parquet files
        for f in parquet_dims.glob("*.parquet"):
            shutil.copy2(f, dims_out / f.name)

    elif ff == "csv":
        # Convert parquet → CSV
        import pandas as pd
        for f in parquet_dims.glob("*.parquet"):
            df = pd.read_parquet(f)
            csv_path = dims_out / (f.stem + ".csv")
            df.to_csv(csv_path, index=False)

    elif ff == "deltaparquet":
        # Convert parquet → Delta table
        from deltalake import write_deltalake
        for f in parquet_dims.glob("*.parquet"):
            dim_name = f.stem
            delta_out = dims_out / dim_name
            delta_out.mkdir(parents=True, exist_ok=True)
            table = pq.read_table(f)

            write_deltalake(
                str(delta_out),
                table,
                mode="overwrite"
            )
    else:
        raise ValueError(f"Unknown file_format: {file_format}")

    # --------------------------------------------------------
    # FACT HANDLING
    # --------------------------------------------------------
    sales_target = facts_out / "sales"

    # Always clean sales output first
    if sales_target.exists():
        shutil.rmtree(sales_target, ignore_errors=True)

    # ---------------------------
    # DELTA PARQUET MODE
    # ---------------------------
    if ff == "deltaparquet":

        # Locate Delta output folder
        delta_src = None

        # config override
        cfg_delta = sales_cfg.get("delta_output_folder")
        if cfg_delta:
            d = Path(cfg_delta).expanduser().resolve()
            if d.exists():
                delta_src = d

        # fallback to internal
        if delta_src is None:
            fb = fact_folder / "delta"
            if fb.exists():
                delta_src = fb

        if delta_src is None:
            raise RuntimeError("DeltaParquet output folder not found!")

        # Copy REAL delta table only
        shutil.copytree(
            delta_src,
            sales_target,
            ignore=shutil.ignore_patterns("_tmp_parts*", "tmp*", "*_tmp*")
        )

        done("Creating Final Output Folder")
        return final_folder

    # --------------------------------------------------------
    # PARQUET MODE
    # --------------------------------------------------------
    if ff == "parquet":
        partitioned_sales = fact_folder / "sales"

        if partitioned_sales.exists():
            shutil.copytree(
                partitioned_sales,
                sales_target,
                ignore=shutil.ignore_patterns("_tmp_parts*", "tmp*", "*_tmp*")
            )

        done("Creating Final Output Folder")
        return final_folder

    # --------------------------------------------------------
    # CSV MODE
    # --------------------------------------------------------
    if ff == "csv":
        partitioned_sales = fact_folder / "sales"
        import pandas as pd

        for file in partitioned_sales.rglob("*.parquet"):
            rel = file.relative_to(partitioned_sales)
            out_file = sales_target / rel.with_suffix(".csv")
            out_file.parent.mkdir(parents=True, exist_ok=True)

            df = pd.read_parquet(file)
            df.to_csv(out_file, index=False)

        done("Creating Final Output Folder")
        return final_folder

    raise ValueError(f"Unknown file_format: {file_format}")
