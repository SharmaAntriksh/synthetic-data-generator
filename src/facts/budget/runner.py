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
    returns_annual = accumulator.finalize_returns()

    info(f"Budget actuals: {len(actuals_monthly):,} monthly grain rows "
         f"({actuals_monthly['Year'].nunique()} years × "
         f"{actuals_monthly['Country'].nunique()} countries × "
         f"{actuals_monthly['Category'].nunique()} categories)")

    # ---- Resolve dimension inputs for channel/month allocation + FX ----
    exchange_rates_path = parquet_dims / "exchange_rates.parquet"
    country_to_currency = _load_country_to_currency(parquet_dims)

    # ---- Compute all budget stages ----
    yearly, channel_month, fx_budget = compute_budget(
        actuals_monthly=actuals_monthly,
        returns_annual=returns_annual,
        exchange_rates_path=exchange_rates_path,
        country_to_currency=country_to_currency,
        country_labels=accumulator._country_labels,
        bcfg=bcfg,
    )

    # ---- Write output ----
    budget_out = fact_out / "budget"
    budget_out.mkdir(parents=True, exist_ok=True)

    _write_budget(yearly, budget_out, "budget_yearly", file_format)

    if channel_month is not None and len(channel_month) > 0:
        _write_budget(channel_month, budget_out, "budget_channel_month", file_format)

    if fx_budget is not None and len(fx_budget) > 0:
        _write_budget(fx_budget, budget_out, "budget_channel_month_fx", file_format)

    elapsed = time.time() - t0

    yearly_rows = len(yearly)
    cm_rows = len(channel_month) if channel_month is not None else 0
    fx_rows = len(fx_budget) if fx_budget is not None else 0

    done(f"Budget completed in {elapsed:.1f}s "
         f"({yearly_rows:,} yearly, {cm_rows:,} channel-month, {fx_rows:,} FX rows)")

    return {
        "yearly_rows": yearly_rows,
        "channel_month_rows": cm_rows,
        "fx_rows": fx_rows,
        "elapsed_sec": elapsed,
    }


def _load_country_to_currency(parquet_dims: Path) -> Dict[str, str]:
    """
    Build Country -> ISOCode (currency) mapping from the Geography dimension.

    Falls back to empty dict if the parquet is missing or lacks the columns,
    which causes the FX stage to be skipped gracefully.
    """
    geo_path = parquet_dims / "geography.parquet"
    if not geo_path.exists():
        info("Budget: geography.parquet not found, FX conversion will be skipped")
        return {}

    try:
        geo = pd.read_parquet(geo_path, columns=["Country", "ISOCode"])
        geo = geo.dropna(subset=["Country", "ISOCode"]).drop_duplicates("Country")
        return dict(zip(geo["Country"], geo["ISOCode"]))
    except Exception as exc:
        info(f"Budget: could not load country->currency mapping: {exc}")
        return {}


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
