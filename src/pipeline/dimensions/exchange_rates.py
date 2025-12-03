import pandas as pd
from pathlib import Path

from src.utils.logging_utils import info, skip, stage
from src.pipeline.versioning import should_regenerate, save_version
from src.services.fx_yahoo import build_or_update_fx


# ---------------------------------------------------------
# Resolve FX date range (corrected)
# ---------------------------------------------------------

def resolve_fx_dates(fx_cfg, global_defaults):
    """
    Determine the FX effective date range.

    Correct rules:
    - If use_global_dates = true → use TRUE global defaults (cfg["_defaults"])
      regardless of overrides or section dates.
    - If use_global_dates = false → use merged section defaults, then apply override.
    """

    # -----------------------------------------------------
    # Case 1: global dates explicitly requested
    # -----------------------------------------------------
    if fx_cfg.get("use_global_dates", False):
        return global_defaults["start"], global_defaults["end"]

    # -----------------------------------------------------
    # Case 2: section-level defaults (already merged)
    # -----------------------------------------------------
    base_dates = fx_cfg.get("dates", {})
    start = base_dates.get("start")
    end   = base_dates.get("end")

    # -----------------------------------------------------
    # Apply overrides
    # -----------------------------------------------------
    override_dates = fx_cfg.get("override", {}).get("dates", {})
    start = override_dates.get("start", start)
    end   = override_dates.get("end", end)

    return start, end


# ---------------------------------------------------------
# Main pipeline wrapper
# ---------------------------------------------------------

def run_exchange_rates(cfg, parquet_folder: Path):
    """
    Generate Exchange Rates dimension using Yahoo Finance.
    Corrected to respect use_global_dates flag.
    """

    out_path = parquet_folder / "exchange_rates.parquet"

    if not should_regenerate("exchange_rates", cfg, out_path):
        skip("Exchange Rates up-to-date; skipping.")
        return

    fx_cfg = cfg["exchange_rates"]

    # ---------------------------------------------------------
    # Retrieve TRUE global defaults (not merged)
    # ---------------------------------------------------------
    global_defaults = cfg["_defaults"]["dates"]

    # ---------------------------------------------------------
    # Resolve effective date range
    # ---------------------------------------------------------
    start_str, end_str = resolve_fx_dates(fx_cfg, global_defaults)

    start = pd.to_datetime(start_str).date()
    end   = pd.to_datetime(end_str).date()

    currencies = fx_cfg["currencies"]
    base       = fx_cfg["base_currency"]
    master     = fx_cfg["master_file"]

    # ---------------------------------------------------------
    # Step 1: Update or build master FX file from Yahoo Finance
    # ---------------------------------------------------------
    with stage("Updating FX Master"):
        master_fx = build_or_update_fx(start, end, master, currencies=currencies)

    # Ensure Date is normalized before slicing
    master_fx["Date"] = pd.to_datetime(master_fx["Date"], errors = 'coerce', format = '%Y-%m-%d').dt.date

    # ---------------------------------------------------------
    # Step 2: Slice master FX file based on resolved dates
    # ---------------------------------------------------------
    with stage("Generating Exchange Rates"):
        df = master_fx[
            (master_fx["ToCurrency"].isin(currencies)) &
            (master_fx["FromCurrency"] == base) &
            (master_fx["Date"] >= start) &
            (master_fx["Date"] <= end)
        ].reset_index(drop=True)
        
        df = df[["Date", "FromCurrency", "ToCurrency", "Rate"]]
        df.to_parquet(out_path, index=False)

    save_version("exchange_rates", cfg, out_path)
    info(f"Exchange Rates dimension written → {out_path}")
