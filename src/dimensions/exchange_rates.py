import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.logging_utils import info, skip, stage
from src.versioning.version_store import should_regenerate, save_version
from src.integrations.fx_yahoo import build_or_update_fx
from src.defaults import CURRENCY_BASE
from src.exceptions import ConfigError


# ---------------------------------------------------------
# Resolve FX date range
# ---------------------------------------------------------
def resolve_fx_dates(fx_cfg, global_defaults):
    """
    Determine the FX effective date range.

    Rules:
    - If use_global_dates = true → use global defaults
    - If use_global_dates = false → use section dates, then apply override
    """
    if fx_cfg.use_global_dates:
        if not global_defaults or "start" not in global_defaults or "end" not in global_defaults:
            raise ValueError("use_global_dates=true but global defaults dates are missing (cfg.defaults.dates or cfg._defaults.dates).")
        return global_defaults["start"], global_defaults["end"]

    base_dates = getattr(fx_cfg, "dates", {}) or {}
    start = base_dates.get("start") if isinstance(base_dates, dict) else getattr(base_dates, "start", None)
    end = base_dates.get("end") if isinstance(base_dates, dict) else getattr(base_dates, "end", None)

    override = getattr(fx_cfg, "override", {}) or {}
    override_dates = override.get("dates", {}) if isinstance(override, dict) else getattr(override, "dates", {})
    if isinstance(override_dates, dict):
        start = override_dates.get("start", start)
        end = override_dates.get("end", end)

    if start is None or end is None:
        raise ValueError("FX dates could not be resolved (missing start/end).")

    return start, end


# ---------------------------------------------------------
# Main pipeline wrapper
# ---------------------------------------------------------
def run_exchange_rates(cfg, parquet_folder: Path):
    """
    Exchange Rates dimension:
    - regenerates only when FX-related cfg or date window changes
    - writes parquet: Date, FromCurrency, ToCurrency, Rate

    Invariant (per fx_yahoo.py):
      FromCurrency = USD
      Rate = units of ToCurrency per 1 USD  (USD -> Curr)
    """
    out_path = parquet_folder / "exchange_rates.parquet"
    fx_cfg = cfg.exchange_rates

    # GLOBAL DEFAULTS (supports both `defaults` and `_defaults`)
    defaults = cfg.defaults if hasattr(cfg, "defaults") else getattr(cfg, "_defaults", None)
    if defaults is not None and hasattr(defaults, "dates") and defaults.dates is not None:
        dates_obj = defaults.dates
        global_defaults = {"start": dates_obj.start, "end": dates_obj.end}
    else:
        global_defaults = None

    # Resolve effective FX date window
    start_str, end_str = resolve_fx_dates(fx_cfg, global_defaults)
    def _parse_fx_date(label: str, value) -> "datetime.date":
        try:
            return pd.to_datetime(value, errors="raise").date()
        except (ValueError, TypeError) as exc:
            raise ConfigError(f"exchange_rates: invalid {label} date '{value}'") from exc

    start = _parse_fx_date("start", start_str)
    end = _parse_fx_date("end", end_str)

    currencies = fx_cfg.currencies
    base = fx_cfg.base_currency
    master = fx_cfg.master_file
    annual_drift = fx_cfg.future_annual_drift

    # Enforce current supported invariant
    if base != CURRENCY_BASE:
        raise ValueError(
            f"Only base_currency='{CURRENCY_BASE}' is supported currently. Got base_currency={base!r}. "
            f"fx_yahoo master is stored as {CURRENCY_BASE} -> Curr."
        )

    # Minimal config for versioning (dimension-only)
    minimal_cfg = {
        "currencies": currencies,
        "base": base,
        "master_file": master,
        "use_global_dates": fx_cfg.use_global_dates,
        "start": start_str,
        "end": end_str,
        "future_annual_drift": annual_drift,
    }

    if not should_regenerate("exchange_rates", minimal_cfg, out_path):
        skip("Exchange Rates up-to-date")
        return

    # Ensure master directory exists (important if file was deleted/untracked)
    master_path = Path(master).expanduser()
    master_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Update/build FX master (USD -> Curr, missing days filled)
    with stage("Updating FX Master"):
        master_fx = build_or_update_fx(start, end, str(master_path), currencies=currencies, annual_drift=annual_drift)

    # Normalize Date once
    master_fx = master_fx.copy()
    master_fx["Date"] = pd.to_datetime(master_fx["Date"], errors="raise").dt.date

    # Step 2: Slice master (USD -> currencies) and apply date window
    df = master_fx[
        (master_fx["FromCurrency"] == CURRENCY_BASE) &
        (master_fx["ToCurrency"].isin(currencies)) &
        (master_fx["Date"] >= start) &
        (master_fx["Date"] <= end)
    ].copy()

    # Basic integrity checks
    if df.empty:
        raise ValueError("Exchange rates slice is empty after filtering. Check currencies/date window/master_file.")

    # Keep stable ordering for reproducibility
    df = df[["Date", "FromCurrency", "ToCurrency", "Rate"]]
    df = df.sort_values(["Date", "FromCurrency", "ToCurrency"]).reset_index(drop=True)

    # Validate rates
    if not np.isfinite(df["Rate"]).all():
        raise ValueError("Invalid FX rate: non-finite values found (NaN/inf).")

    if (df["Rate"] <= 0).any():
        raise ValueError("Invalid FX rate: non-positive values found.")

    # Write output + version stamp
    parquet_folder.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    save_version("exchange_rates", minimal_cfg, out_path)
    info(f"Exchange Rates dimension written: {out_path.name}")
