"""Wishlists fact package.

Public API:
  - run_wishlist_pipeline(accumulator, parquet_dims, fact_out, cfg, file_format)
"""
from .runner import run_wishlist_pipeline

__all__ = ["run_wishlist_pipeline"]
