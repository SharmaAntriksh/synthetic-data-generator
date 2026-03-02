"""Budget pipeline runner.

Called from sales_runner.py after sales generation completes.
Uses the BudgetAccumulator that was populated during sales generation.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.logging_utils import stage, info, done

from .accumulator import BudgetAccumulator
from .engine import load_budget_config, _compute_yearly_budget


def run_budget_pipeline(
    *,
    accumulator: BudgetAccumulator,
    parquet_dims: Path,
    fact_out: Path,
    cfg: Dict[str, Any],
    file_format: str = "parquet",
) -> Optional[Dict[str, Any]]:
    """
    Generate budget fact tables from streaming-aggregated sales actuals.

    Args:
        accumulator:   BudgetAccumulator populated during sales generation
        parquet_dims:  path to generated dimension parquets
        fact_out:      path to write budget fact parquets
        cfg:           full config dict
        file_format:   "csv" | "parquet" | "deltaparquet"

    Returns:
        summary dict or None if budget is disabled / no data
    """
    bcfg = load_budget_config(cfg)
    if not bcfg.enabled:
        info("Budget generation: disabled in config")
        return None

    if not accumulator.has_data:
        info("Budget generation: no sales actuals accumulated, skipping")
        return None

    stage("Generating Budget")
    t0 = time.time()

    # ---- Finalize actuals from accumulated micro-aggregates ----
    actuals_monthly = accumulator.finalize_sales()

    info(f"Budget actuals: {len(actuals_monthly):,} monthly grain rows "
         f"({actuals_monthly['Year'].nunique()} years × "
         f"{actuals_monthly['Country'].nunique()} countries × "
         f"{actuals_monthly['Category'].nunique()} categories)")

    # ---- Compute yearly budget only ----
    actuals_annual = actuals_monthly.groupby(
        ["Country", "Category", "Year"], as_index=False
    ).agg(SalesAmount=("SalesAmount", "sum"), SalesQuantity=("SalesQuantity", "sum"))

    yearly = _compute_yearly_budget(actuals_annual, bcfg)

    # ---- Write output ----
    budget_out = fact_out / "budget"
    budget_out.mkdir(parents=True, exist_ok=True)

    _write_budget(yearly, budget_out, "budget_yearly", file_format)

    elapsed = time.time() - t0
    done(f"Budget completed in {elapsed:.1f}s ({len(yearly):,} yearly rows)")

    return {
        "yearly_rows": len(yearly),
        "elapsed_sec": elapsed,
    }


def _write_budget(df: pd.DataFrame, out_dir: Path, name: str, file_format: str) -> None:
    """Write a budget DataFrame in the requested format (parquet, csv, or delta)."""
    table = pa.Table.from_pandas(df, preserve_index=False)

    if file_format == "deltaparquet":
        # Write as a Delta table directory: out_dir/<name>/
        delta_dir = out_dir / name
        delta_dir.mkdir(parents=True, exist_ok=True)
        try:
            from deltalake import write_deltalake
        except ImportError:
            from deltalake.writer import write_deltalake
        write_deltalake(str(delta_dir), table, mode="overwrite")
        info(f"  Wrote {name}: {len(df):,} rows -> {delta_dir.name}/ (delta)")
        return

    # Parquet (always written for parquet and csv formats)
    parquet_path = out_dir / f"{name}.parquet"
    pq.write_table(
        table, str(parquet_path),
        compression="snappy",
        row_group_size=500_000,
        use_dictionary=True,
    )

    if file_format == "csv":
        csv_path = out_dir / f"{name}.csv"
        df.to_csv(str(csv_path), index=False)
        info(f"  Wrote {name}: {len(df):,} rows -> {parquet_path.name}, {csv_path.name}")
    else:
        info(f"  Wrote {name}: {len(df):,} rows -> {parquet_path.name}")
