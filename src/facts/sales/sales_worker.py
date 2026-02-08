"""
Backward-compatible shim.

The worker implementation lives under src.facts.sales.worker.*.
Keep this module stable so older imports still work:
  from src.facts.sales.sales_worker import init_sales_worker, _worker_task
"""

from __future__ import annotations

from .worker.init import init_sales_worker
from .worker.task import _worker_task

__all__ = ["init_sales_worker", "_worker_task"]
