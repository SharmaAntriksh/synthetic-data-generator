from datetime import datetime
from src.pipeline.config_loader import load_config
from src.pipeline.config import validate_config, prepare_paths
from src.pipeline.dimensions import generate_dimensions
from src.pipeline.sales_pipeline import run_sales_pipeline
from src.pipeline.packaging import package_output
from src.utils.logging_utils import done, human_duration
from src.utils.versioning_validation import validate_all_dimensions
import json

def main():
    start_time = datetime.now()

    # ------------------------------------------------------------
    # Load & validate config
    # ------------------------------------------------------------
    with open("config.json", "r") as f:
        raw_cfg = json.load(f)

    cfg = load_config(raw_cfg)
    sales_cfg = cfg["sales"]

    validate_config(
        sales_cfg,
        "sales",
        ["total_rows", "chunk_size", "file_format"]
    )

    # ------------------------------------------------------------
    # Prepare output paths
    # ------------------------------------------------------------
    parquet_dims, fact_out = prepare_paths(sales_cfg)

    # ------------------------------------------------------------
    # Versioning validation
    # ------------------------------------------------------------
    dimension_names = [
        "geography", "customers", "promotions",
        "stores", "dates", "currency", "exchange_rates",
    ]
    validate_all_dimensions(cfg, parquet_dims, dimension_names)

    # ------------------------------------------------------------
    # Generate dimensions (dependency-aware)
    # ------------------------------------------------------------
    generate_dimensions(cfg, parquet_dims)

    # ------------------------------------------------------------
    # Generate Sales Fact
    # ------------------------------------------------------------
    run_sales_pipeline(sales_cfg, fact_out, parquet_dims, cfg)

    # ------------------------------------------------------------
    # Package output
    # ------------------------------------------------------------
    package_output(cfg, sales_cfg, parquet_dims, fact_out)

    # ------------------------------------------------------------
    # Runtime summary
    # ------------------------------------------------------------
    total_seconds = (datetime.now() - start_time).total_seconds()
    done(f"Total runtime: {total_seconds:.2f} seconds ({human_duration(total_seconds)})")


if __name__ == "__main__":
    main()
