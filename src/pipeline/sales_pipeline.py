import time
from pathlib import Path
from contextlib import contextmanager

from src.facts.sales import generate_sales_fact
from src.utils.output_utils import clear_folder


@contextmanager
def stage(label: str):
    print(f"\n=== {label}... ===")
    t = time.time()
    yield
    print(f"\n✔ {label} completed in {time.time() - t:.2f} seconds")


def run_sales_pipeline(sales_cfg, fact_out: Path):
    """
    Handles: clearing output folder, adjusting CSV mode,
    setting delta defaults, and calling generate_sales_fact().
    """

    with stage("Generating Sales"):
        # Clear fact folder before generating new data
        clear_folder(fact_out)

        # CSV mode disables parquet/delta related options
        if sales_cfg.get("file_format") == "csv":
            sales_cfg["merge_parquet"] = False
            sales_cfg["delete_chunks"] = False
            sales_cfg["write_pyarrow"] = False
            sales_cfg.setdefault("write_delta", False)

        # Ensure delta folder exists if delta output is requested
        sales_cfg.setdefault(
            "delta_output_folder",
            str(Path(sales_cfg["out_folder"]) / "delta")
        )

        # Run actual fact generation
        generate_sales_fact(
            parquet_folder=sales_cfg["parquet_folder"],
            out_folder=sales_cfg["out_folder"],

            total_rows=sales_cfg["total_rows"],
            chunk_size=sales_cfg["chunk_size"],

            start_date=sales_cfg["start_date"],
            end_date=sales_cfg["end_date"],

            delete_chunks=sales_cfg.get("delete_chunks", False),
            heavy_pct=sales_cfg.get("heavy_pct", 5),
            heavy_mult=sales_cfg.get("heavy_mult", 5),
            seed=sales_cfg.get("seed", 42),

            file_format=sales_cfg["file_format"],
            row_group_size=sales_cfg.get("row_group_size"),
            compression=sales_cfg.get("compression"),
            merge_parquet=sales_cfg.get("merge_parquet"),
            merged_file=sales_cfg.get("merged_file"),

            workers=sales_cfg.get("workers"),
            write_pyarrow=sales_cfg.get("write_pyarrow", True),
            tune_chunk=sales_cfg.get("tune_chunk", False),

            # DELTA SUPPORT
            write_delta=sales_cfg.get("write_delta", False),
            delta_output_folder=sales_cfg.get("delta_output_folder"),
            skip_order_cols=sales_cfg.get("skip_order_cols", False),
        )

    return fact_out
