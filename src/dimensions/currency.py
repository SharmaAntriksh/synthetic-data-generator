from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.utils.logging_utils import info, skip, stage
from src.utils.output_utils import write_parquet_with_date32
from src.versioning.version_store import should_regenerate, save_version


from src.defaults import CURRENCY_NAME_MAP


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def _require_section(cfg: Dict, name: str) -> Dict:
    if not isinstance(cfg, Mapping):
        raise TypeError("cfg must be a dict")
    section = getattr(cfg, name, None) if hasattr(cfg, name) else (cfg[name] if isinstance(cfg, dict) and name in cfg else None)
    if section is None or not isinstance(section, Mapping):
        raise KeyError(f"Missing required config section: '{name}'")
    return section


def _get_effective_dates(cfg: Dict, ex_cfg: Dict) -> Dict:
    """
    Currency dimension itself doesn't *need* dates, but you intentionally
    tie versioning to the effective FX window, so we preserve that behavior.
    """
    use_global = bool(getattr(ex_cfg, "use_global_dates", True))

    if use_global:
        defaults_dates = (
            getattr(cfg.defaults, "dates", None) if hasattr(cfg, "defaults") else None
        ) or (
            getattr(getattr(cfg, "_defaults", None), "dates", None)
        ) or {}
        return defaults_dates if isinstance(defaults_dates, Mapping) else {}
    else:
        override = getattr(ex_cfg, "override", None) or {}
        if isinstance(override, dict):
            override_dates = override.get("dates", None) or {}
        else:
            override_dates = getattr(override, "dates", None) or {}
        return override_dates if isinstance(override_dates, Mapping) else {}


def _normalize_currency_list(currencies: List[str]) -> List[str]:
    if not isinstance(currencies, list) or not currencies:
        raise ValueError("exchange_rates.currencies must be a non-empty list")

    normalized: List[str] = []
    seen = set()

    for c in currencies:
        if not isinstance(c, str) or not c.strip():
            raise ValueError(f"Invalid currency code in exchange_rates.currencies: {c!r}")

        code = c.strip().upper()

        # Light validation (ISO-4217 is 3 letters; keep it strict to avoid junk)
        if len(code) != 3 or not code.isalpha():
            raise ValueError(f"Currency code must be 3 letters (e.g. USD). Got: {c!r}")

        if code in seen:
            raise ValueError(f"Duplicate currency code in exchange_rates.currencies: {code}")

        seen.add(code)
        normalized.append(code)

    return normalized


def build_dim_currency(currencies: List[str]) -> pd.DataFrame:
    currencies = _normalize_currency_list(currencies)

    df = pd.DataFrame(
        {
            "CurrencyKey": pd.RangeIndex(start=1, stop=len(currencies) + 1, step=1, name=None).astype("int64"),
            "ToCurrency": currencies,
            "CurrencyName": [CURRENCY_NAME_MAP.get(c, c) for c in currencies],
        }
    )
    return df


# ---------------------------------------------------------
# Pipeline wrapper
# ---------------------------------------------------------

def run_currency(cfg: Dict, parquet_folder: Path) -> None:
    """
    Versioning dependencies (minimal, intentional):
      - exchange_rates.currencies (+ base_currency/use_global_dates)
      - effective FX date window (global dates if use_global_dates=true, else override.dates)
    """
    ex_cfg = _require_section(cfg, "exchange_rates")
    # currency section is optional — only holds parquet knobs
    from src.engine.config.config_schema import CurrencyConfig
    cur_cfg = cfg.currency or CurrencyConfig()

    out_path = Path(parquet_folder) / "currency.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    currencies = _normalize_currency_list(ex_cfg.currencies)
    effective_dates = _get_effective_dates(cfg, ex_cfg)

    # Minimal version_cfg: avoids regen churn when unrelated exchange_rates keys change
    version_cfg = {
        "currencies": currencies,
        "base_currency": (getattr(ex_cfg, "base_currency", "") or "").upper(),
        "use_global_dates": bool(getattr(ex_cfg, "use_global_dates", True)),
        "effective_dates": effective_dates,
    }

    if not should_regenerate("currency", version_cfg, out_path):
        skip("Currency up-to-date")
        return

    # Optional parquet knobs (consistent with other dims)
    compression = cur_cfg.parquet_compression
    compression_level = cur_cfg.parquet_compression_level
    force_date32 = bool(cur_cfg.force_date32)

    with stage("Generating Currency"):
        df = build_dim_currency(currencies)

        # Currency has no datetime cols; this will just write parquet normally,
        # but keeps a consistent write path across dims.
        write_parquet_with_date32(
            df,
            out_path,
            cast_all_datetime=False,
            compression=str(compression),
            compression_level=(int(compression_level) if compression_level is not None else None),
            force_date32=force_date32,
        )

    save_version("currency", version_cfg, out_path)
    info(f"Currency dimension written: {out_path}")
