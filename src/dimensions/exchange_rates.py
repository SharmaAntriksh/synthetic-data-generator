import pandas as pd
from pathlib import Path

from src.utils.logging_utils import info, skip, stage
from src.versioning.version_store import should_regenerate, save_version
from src.integrations.fx_yahoo import build_or_update_fx


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
    Exchange Rates dimension:
    - respects use_global_dates=true/false
    - regenerates only when FX-related cfg or date window changes
    """

    out_path = parquet_folder / "exchange_rates.parquet"
    fx_cfg = cfg["exchange_rates"]

    # ---------------------------------------------------------
    # GLOBAL DEFAULTS (supports both `defaults` and `_defaults`)
    # ---------------------------------------------------------
    global_defaults = (
        cfg.get("defaults", {}).get("dates")
        or cfg.get("_defaults", {}).get("dates")
    )

    # ---------------------------------------------------------
    # Resolve effective FX date window
    # ---------------------------------------------------------
    start_str, end_str = resolve_fx_dates(fx_cfg, global_defaults)
    start = pd.to_datetime(start_str).date()
    end   = pd.to_datetime(end_str).date()

    # ---------------------------------------------------------
    # Minimal config for versioning (dimension-only)
    # ---------------------------------------------------------
    minimal_cfg = {
        "currencies": fx_cfg.get("currencies"),
        "base": fx_cfg.get("base_currency"),
        "use_global_dates": fx_cfg.get("use_global_dates", False),
        "start": start_str,
        "end": end_str,
    }

    force = fx_cfg.get("_force_regenerate", False)

    if not force and not should_regenerate("exchange_rates", minimal_cfg, out_path):
        skip("Exchange Rates up-to-date; skipping.")
        return

    # ---------------------------------------------------------
    # Pull parameters
    # ---------------------------------------------------------
    currencies = fx_cfg["currencies"]
    base       = fx_cfg["base_currency"]
    master     = fx_cfg["master_file"]

    # ---------------------------------------------------------
    # Step 1: Update or build FX master
    # ---------------------------------------------------------
    with stage("Updating FX Master"):
        master_fx = build_or_update_fx(start, end, master, currencies=currencies)

    master_fx["Date"] = (
        pd.to_datetime(master_fx["Date"], errors="coerce", format="%Y-%m-%d").dt.date
    )

    # ---------------------------------------------------------
    # Step 2: Slice master based on final date window
    # ---------------------------------------------------------
    direct = master_fx[
        (master_fx["FromCurrency"] == base) &
        (master_fx["ToCurrency"].isin(currencies))
    ].copy()

    # Find missing currencies that lack From=USD rows
    missing = set(currencies) - set(direct["ToCurrency"].unique())

    if missing:
        # Look for inverted rows (target -> base) and invert
        inv = master_fx[
            (master_fx["ToCurrency"] == base) &
            (master_fx["FromCurrency"].isin(missing))
        ].copy()

        if not inv.empty:
            inv["Rate"] = 1.0 / inv["Rate"]
            inv = inv.rename(columns={
                "FromCurrency": "ToCurrency",
                "ToCurrency": "FromCurrency"
            })
            direct = pd.concat([direct, inv], ignore_index=True)

    # Apply date filter at the end
    direct["Date"] = pd.to_datetime(direct["Date"], errors="coerce").dt.date
    df = direct[
        (direct["Date"] >= start) &
        (direct["Date"] <= end)
    ].reset_index(drop=True)

    df = df[["Date", "FromCurrency", "ToCurrency", "Rate"]]
    df.to_parquet(out_path, index=False)

    if (df["Rate"] <= 0).any():
        raise ValueError("Invalid FX rate: non-positive values.")

    save_version("exchange_rates", minimal_cfg, out_path)
    info(f"Exchange Rates dimension written: {out_path}")
