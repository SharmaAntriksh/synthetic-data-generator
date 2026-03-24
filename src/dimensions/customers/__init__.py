"""Customer dimension package.

Public API:
  - run_customers(cfg, parquet_folder)
  - generate_synthetic_customers(cfg, parquet_dims_folder)
  - run_subscriptions(cfg, parquet_folder)
"""
from src.dimensions.customers.generator import (  # noqa: F401
    generate_synthetic_customers,
    run_customers,
)
from src.dimensions.customers.subscriptions import run_subscriptions  # noqa: F401
