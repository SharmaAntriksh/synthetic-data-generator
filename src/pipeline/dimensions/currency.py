# ---------------------------------------------------------
#  CURRENCY DIMENSION (FIXED VERSIONING + DATE-AWARE)
# ---------------------------------------------------------

import pandas as pd
from pathlib import Path

from src.utils.logging_utils import info, skip, stage
from src.pipeline.versioning import should_regenerate, save_version

CURRENCY_NAME_MAP = {
    "USD": "US Dollar",
    "EUR": "Euro",
    "INR": "Indian Rupee",
    "GBP": "British Pound",
    "AUD": "Australian Dollar",
    "CAD": "Canadian Dollar",
    "CNY": "Chinese Yuan",
    "JPY": "Japanese Yen",
    "NZD": "New Zealand Dollar",
    "CHF": "Swiss Franc",
    "SEK": "Swedish Krona",
    "NOK": "Norwegian Krone",
    "SGD": "Singapore Dollar",
    "HKD": "Hong Kong Dollar",
    "KRW": "Korean Won",
    "ZAR": "South African Rand",
}


def generate_currency_dimension(currencies):
    df = pd.DataFrame({
        "CurrencyKey": range(1, len(currencies) + 1),
        "ISOCode": currencies,
        "CurrencyName": [CURRENCY_NAME_MAP.get(c, c) for c in currencies],
    })
    return df


def run_currency(cfg, parquet_folder: Path):
    """
    Currency dimension should depend ONLY on:
      - exchange_rates section
      - defaults.dates (ONLY if use_global_dates = true)
    """

    out_path = parquet_folder / "currency.parquet"

    ex_cfg = cfg["exchange_rates"]

    # --------------------------
    # Determine which date window we depend on
    # --------------------------
    if ex_cfg.get("use_global_dates", True):
        defaults_dates = (
            cfg.get("defaults", {}).get("dates")
            or cfg.get("_defaults", {}).get("dates")
        )
        date_dependency = defaults_dates
    else:
        date_dependency = ex_cfg.get("override", {}).get("dates", {})

    # --------------------------
    # Versioning: 
    # Use only exchange_rates section + date dependency
    # --------------------------
    version_cfg = {
        **ex_cfg,
        "effective_dates": date_dependency
    }

    if not should_regenerate("currency", version_cfg, out_path):
        skip("Currency up-to-date; skipping.")
        return

    # --------------------------
    # Build dimension
    # --------------------------
    currencies = ex_cfg["currencies"]

    with stage("Generating Currency"):
        df = generate_currency_dimension(currencies)
        df.to_parquet(out_path, index=False)

    save_version("currency", version_cfg, out_path)
    info(f"Currency dimension written → {out_path}")
