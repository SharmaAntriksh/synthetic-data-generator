"""Inventory snapshot fact table generation.

Generates monthly periodic snapshots at the (ProductKey, StoreKey) grain,
simulating realistic stock levels driven by actual sales demand and
ProductProfile replenishment attributes.
"""
from __future__ import annotations

from .accumulator import InventoryAccumulator
from .micro_agg import micro_aggregate_inventory
from .runner import run_inventory_pipeline

__all__ = [
    "InventoryAccumulator",
    "micro_aggregate_inventory",
    "run_inventory_pipeline",
]
