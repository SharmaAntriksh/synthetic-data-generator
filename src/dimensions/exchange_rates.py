import numpy as np
import pandas as pd
from pathlib import Path

from src.utils.logging_utils import info, skip, stage
from src.versioning.version_store import should_regenerate, save_version
from src.integrations.fx_yahoo import build_or_update_fx


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
    if fx_cfg.get("use_global_dates", False):
        if not global_defaults or "start" not in global_defaults or "end" not in global_defaults:
            raise ValueError("use_global_dates=true but global defaults dates are missing (cfg.defaults.dates or cfg._defaults.dates).")
        return global_defaults["start"], global_defaults["end"]

    base_dates = fx_cfg.get("dates", {})
    start = base_dates.get("start")
    end = base_dates.get("end")

    override_dates = fx_cfg.get("override", {}).get("dates", {})
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
    fx_cfg = cfg["exchange_rates"]

    # GLOBAL DEFAULTS (supports both `defaults` and `_defaults`)
    global_defaults = (
        cfg.get("defaults", {}).get("dates")
        or cfg.get("_defaults", {}).get("dates")
    )

    # Resolve effective FX date window
    start_str, end_str = resolve_fx_dates(fx_cfg, global_defaults)
    start = pd.to_datetime(start_str, errors="raise").date()
    end = pd.to_datetime(end_str, errors="raise").date()

    currencies = fx_cfg["currencies"]
    base = fx_cfg["base_currency"]
    master = fx_cfg["master_file"]

    # Enforce current supported invariant
    if base != "USD":
        raise ValueError(
            f"Only base_currency='USD' is supported currently. Got base_currency={base!r}. "
            "fx_yahoo master is stored as USD -> Curr."
        )

    # Minimal config for versioning (dimension-only)
    minimal_cfg = {
        "currencies": currencies,
        "base": base,
        "master_file": master,
        "use_global_dates": fx_cfg.get("use_global_dates", False),
        "start": start_str,
        "end": end_str,
    }

    force = fx_cfg.get("_force_regenerate", False)

    if not force and not should_regenerate("exchange_rates", minimal_cfg, out_path):
        skip("Exchange Rates up-to-date; skipping.")
        return

    # Ensure master directory exists (important if file was deleted/untracked)
    master_path = Path(master).expanduser()
    master_path.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: Update/build FX master (USD -> Curr, missing days filled)
    with stage("Updating FX Master"):
        master_fx = build_or_update_fx(start, end, str(master_path), currencies=currencies)

    # Normalize Date once
    master_fx = master_fx.copy()
    master_fx["Date"] = pd.to_datetime(master_fx["Date"], errors="raise").dt.date

    # Step 2: Slice master (USD -> currencies) and apply date window
    df = master_fx[
        (master_fx["FromCurrency"] == "USD") &
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
    info(f"Exchange Rates dimension written: {out_path}")
