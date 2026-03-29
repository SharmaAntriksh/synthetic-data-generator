"""Currency dimension generator.

Produces ``currency.parquet`` with columns:
  CurrencyKey, CurrencyCode, CurrencyName, CurrencySymbol, DecimalPlaces
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd

from src.utils.logging_utils import info, skip, stage
from src.utils.output_utils import write_parquet_with_date32
from src.versioning.version_store import should_regenerate, save_version

from .helpers import (
    normalize_currency_list,
    currency_name,
    currency_symbol,
    currency_decimal_places,
)


# ---------------------------------------------------------
# Core builder
# ---------------------------------------------------------

def build_dim_currency(currencies: List[str]) -> pd.DataFrame:
    """Build the currency dimension DataFrame from a normalized currency list."""
    currencies = normalize_currency_list(currencies)
    return pd.DataFrame({
        "CurrencyKey": pd.RangeIndex(1, len(currencies) + 1).astype("int32"),
        "CurrencyCode": currencies,
        "CurrencyName": [currency_name(c) for c in currencies],
        "CurrencySymbol": [currency_symbol(c) for c in currencies],
        "DecimalPlaces": [currency_decimal_places(c) for c in currencies],
    })


# ---------------------------------------------------------
# Pipeline wrapper
# ---------------------------------------------------------

def run_currency(cfg, parquet_folder: Path) -> None:
    """Generate and write the currency dimension.

    Currency list is sourced from ``cfg.currency.currencies`` when set
    explicitly, otherwise derived from the union of
    ``cfg.exchange_rates.from_currencies`` and
    ``cfg.exchange_rates.to_currencies``.
    """
    from src.engine.config.config_schema import CurrencyConfig

    cur_cfg = cfg.currency or CurrencyConfig()
    fx_cfg = cfg.exchange_rates
    out_path = Path(parquet_folder) / "currency.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine effective currency list
    if cur_cfg.currencies:
        currencies = normalize_currency_list(cur_cfg.currencies)
    else:
        raw = list(dict.fromkeys(
            list(fx_cfg.from_currencies or []) + list(fx_cfg.to_currencies or [])
        ))
        currencies = normalize_currency_list(raw or ["USD"])

    version_cfg = {
        "currencies": currencies,
        "base_currency": (fx_cfg.base_currency or "").upper(),
    }

    if not should_regenerate("currency", version_cfg, out_path):
        skip("Currency up-to-date")
        return

    compression = cur_cfg.parquet_compression
    compression_level = cur_cfg.parquet_compression_level
    force_date32 = bool(cur_cfg.force_date32)

    with stage("Generating Currency"):
        df = build_dim_currency(currencies)

        write_parquet_with_date32(
            df,
            out_path,
            cast_all_datetime=False,
            compression=str(compression),
            compression_level=(int(compression_level) if compression_level is not None else None),
            force_date32=force_date32,
        )

    save_version("currency", version_cfg, out_path)
    info(f"Currency dimension written: {out_path.name}")
