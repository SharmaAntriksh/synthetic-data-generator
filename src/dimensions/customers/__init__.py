"""Customer dimension package — split from monolithic customers.py.

Public API (re-exported for backward compatibility):
  - generate_synthetic_customers(cfg, parquet_dims_folder)
  - run_customers(cfg, parquet_folder)
"""
from src.dimensions.customers.generator import (  # noqa: F401
    generate_synthetic_customers,
    run_customers,
)
