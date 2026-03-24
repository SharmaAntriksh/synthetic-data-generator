"""Budget fact package.

Public API:
  - run_budget_pipeline(accumulator, parquet_dims, fact_out, cfg, file_format)
"""
from .runner import run_budget_pipeline  # noqa: F401
