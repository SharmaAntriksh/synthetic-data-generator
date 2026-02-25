"""
Sales fact package.

Keep this module import-light to avoid circular imports during packaging/path discovery.
"""
from __future__ import annotations

__all__ = ["generate_sales_fact"]


def generate_sales_fact(*args, **kwargs):
    from .sales import generate_sales_fact as _impl
    return _impl(*args, **kwargs)