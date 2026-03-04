"""Budget pipeline runner.

Called from sales_runner.py after sales generation completes.
Uses the BudgetAccumulator that was populated during sales generation.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.utils.logging_utils import stage, info, done, short_path

from .accumulator import BudgetAccumulator
from .engine import load_budget_config, compute_budget


def run_budget_pipeline(
    *,
    accumulator: BudgetAccumulator,
    parquet_dims: Path,
    fact_out: Path,
    cfg: Dict[str, Any],
    file_format: str = "parquet",
) -> Optional[Dict[str, Any]]:
    """
    Generate budget_yearly fact table from streaming-aggregated sales actuals.

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

    t0 = time.time()

    # ---- Finalize actuals from accumulated micro-aggregates ----
    actuals_monthly = accumulator.finalize_sales()

    info(f"Budget actuals: {len(actuals_monthly):,} monthly grain rows "
         f"({actuals_monthly['Year'].nunique()} years × "
         f"{actuals_monthly['Country'].nunique()} countries × "
         f"{actuals_monthly['Category'].nunique()} categories)")

    # ---- Compute budget tables ----
    yearly, monthly = compute_budget(
        actuals_monthly=actuals_monthly,
        bcfg=bcfg,
    )

    # ---- Write output ----
    budget_out = fact_out / "budget"
    budget_out.mkdir(parents=True, exist_ok=True)

    _write_budget(yearly, budget_out, "budget_yearly", file_format)
    _write_budget(monthly, budget_out, "budget_monthly", file_format)

    elapsed = time.time() - t0
    yearly_rows = len(yearly)
    monthly_rows = len(monthly)

    return {
        "yearly_rows": yearly_rows,
        "monthly_rows": monthly_rows,
        "elapsed_sec": elapsed,
    }


def _write_budget(df: pd.DataFrame, out_dir: Path, name: str, file_format: str) -> None:
    """Write a budget DataFrame in the requested format (parquet, csv, or delta)."""
    table = pa.Table.from_pandas(df, preserve_index=False)

    if file_format == "deltaparquet":
        delta_dir = out_dir / name
        delta_dir.mkdir(parents=True, exist_ok=True)
        try:
            from deltalake import write_deltalake
        except ImportError:
            from deltalake.writer import write_deltalake
        write_deltalake(str(delta_dir), table, mode="overwrite")
        info(f"Wrote {name}: {len(df):,} rows -> {short_path(delta_dir)}/")
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
        csv_df = _prepare_budget_csv(df, name)
        csv_df.to_csv(str(csv_path), index=False, float_format="%.6f")
        info(f"Wrote {name}: {len(df):,} rows -> {short_path(csv_path)}")
    else:
        info(f"Wrote {name}: {len(df):,} rows -> {short_path(parquet_path)}")


# ----------------------------------------------------------------
# CSV preparation: column selection + type/precision alignment
# ----------------------------------------------------------------

_BUDGET_CSV_COLUMNS: dict[str, list[str]] = {
    "budget_yearly": [
        "Country", "Category", "BudgetYear", "Scenario",
        "BudgetGrowthPct", "BudgetSalesAmount", "BudgetSalesQuantity",
        "BudgetMethod",
    ],
    "budget_monthly": [
        "Country", "Category", "BudgetYear", "BudgetMonthStart", "Scenario",
        "BudgetAmount", "BudgetQuantity", "BudgetMethod",
    ],
}

_BUDGET_CSV_ROUND: dict[str, int] = {
    "BudgetGrowthPct": 6,
    "BudgetSalesAmount": 2,
    "BudgetSalesQuantity": 2,
    "BudgetAmount": 2,
    "BudgetQuantity": 2,
}

_BUDGET_CSV_INT_COLS: tuple[str, ...] = ("BudgetYear",)


def _prepare_budget_csv(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Select schema columns, round decimals, and cast integers for clean CSV output."""
    expected_cols = _BUDGET_CSV_COLUMNS.get(name)
    if expected_cols is not None:
        out = df.copy()
        for col in expected_cols:
            if col not in out.columns:
                out[col] = None
        out = out[expected_cols]
    else:
        out = df.copy()

    for col, dp in _BUDGET_CSV_ROUND.items():
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").round(dp)

    for col in _BUDGET_CSV_INT_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)

    return out
