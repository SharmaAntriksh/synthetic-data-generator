import pandas as pd
from pathlib import Path
import yfinance as yf
from datetime import timedelta

from src.utils.logging_utils import info

# ---------------------------------------------------------
# DEFAULT CURRENCY LIST
# ---------------------------------------------------------
CURRENCIES = [
    "EUR", "GBP", "INR", "AUD", "CAD", "CNY",
    "JPY", "SGD", "CHF", "ZAR", "HKD", "NZD", "SEK"
]

BASE = "USD"


# ---------------------------------------------------------
# Download single currency history (USD -> CUR)
# ---------------------------------------------------------
def download_history(currency, start_date, end_date):
    """
    Download historical FX data and return rates as:
      Rate = units of `currency` per 1 USD  (USD -> currency)

    Strategy:
    - Prefer Yahoo ticker: USD{CUR}=X  (usually already USD -> CUR)
    - Fallback to: {CUR}USD=X and invert
    """
    # Normalize input dates to Timestamps (safe for yfinance + pd.date_range)
    start_ts = pd.to_datetime(start_date).normalize()
    end_ts = pd.to_datetime(end_date).normalize()

    if currency == BASE:
        dates = pd.date_range(start=start_ts, end=end_ts, freq="D")
        return pd.DataFrame({"Date": dates, "Rate": 1.0})

    # Primary: USD -> CUR
    primary = f"{BASE}{currency}=X"
    fallback = f"{currency}{BASE}=X"  # may represent USD per 1 CUR, invert to get CUR per 1 USD

    def _download(ticker: str) -> pd.DataFrame:
        data = yf.download(
            ticker,
            start=start_ts - timedelta(days=3),
            end=end_ts + timedelta(days=3),
            auto_adjust=False,
            progress=False
        )
        return data

    invert = False
    data = _download(primary)

    if data.empty:
        data = _download(fallback)
        invert = True

    if data.empty:
        raise ValueError(f"No FX data found for {currency} (tried {primary} and {fallback})")

    # Flatten MultiIndex columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [
            "_".join([str(c) for c in col if c]).strip()
            for col in data.columns.values
        ]

    # Pick a Close-like column
    rate_col = None
    for col in data.columns:
        if str(col).lower().startswith("close"):
            rate_col = col
            break

    if rate_col is None:
        raise ValueError(f"No 'Close' column found for {currency}. Found: {list(data.columns)}")

    data = data[[rate_col]].rename(columns={rate_col: "Rate"}).reset_index()

    # Ensure Date is datetime64[ns]
    if "Date" in data.columns:
        data["Date"] = pd.to_datetime(data["Date"])
    else:
        # yfinance sometimes returns index under different name, but Date is typical
        data = data.rename(columns={data.columns[0]: "Date"})
        data["Date"] = pd.to_datetime(data["Date"])

    # Invert if fallback ticker used (to get USD -> CUR)
    if invert:
        if (data["Rate"] <= 0).any():
            raise ValueError(f"Non-positive FX rates encountered for {currency} before inversion.")
        data["Rate"] = 1.0 / data["Rate"]

    return data[["Date", "Rate"]]


# ---------------------------------------------------------
# Fill missing days safely
# ---------------------------------------------------------
def fill_missing_days(df, start_date, end_date):
    """
    Fill weekends/holidays with forward fill (previous day),
    then backfill for leading gaps, then default 1.0.
    """
    start_ts = pd.to_datetime(start_date).normalize()
    end_ts = pd.to_datetime(end_date).normalize()

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()

    full = pd.DataFrame({"Date": pd.date_range(start=start_ts, end=end_ts, freq="D")})
    merged = full.merge(df, on="Date", how="left")

    # Your requirement: missing days filled from previous day
    merged["Rate"] = merged["Rate"].ffill()
    merged["Rate"] = merged["Rate"].bfill().fillna(1.0)

    return merged


# ---------------------------------------------------------
# Build or update master FX store
# ---------------------------------------------------------
def build_or_update_fx(start_date, end_date, out_path, currencies=None):
    """
    Build or update a master FX file covering the date range.

    Stored invariant:
      FromCurrency = USD
      ToCurrency   = CUR
      Rate         = CUR per 1 USD (USD -> CUR)

    Date is kept as datetime64[ns] throughout.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    curr_list = currencies or CURRENCIES

    start_ts = pd.to_datetime(start_date).normalize()
    end_ts = pd.to_datetime(end_date).normalize()

    # Load existing master file (keep datetime64)
    if out_path.exists():
        master = pd.read_parquet(out_path)
        if not master.empty:
            master["Date"] = pd.to_datetime(master["Date"]).dt.normalize()
    else:
        master = pd.DataFrame(columns=["Date", "FromCurrency", "ToCurrency", "Rate"])

    updates = []

    for cur in curr_list:
        info(f"Updating FX for {cur}...")

        if master.empty:
            cur_existing_start = None
            cur_existing_end = None
        else:
            df_cur = master[master["ToCurrency"] == cur]
            if df_cur.empty:
                cur_existing_start = None
                cur_existing_end = None
            else:
                cur_existing_start = pd.to_datetime(df_cur["Date"].min()).normalize()
                cur_existing_end = pd.to_datetime(df_cur["Date"].max()).normalize()

        # Keep your original “download range” behavior (safe, not optimized)
        dl_start = start_ts if (cur_existing_start is None or start_ts < cur_existing_start) else cur_existing_start
        dl_end = end_ts if (cur_existing_end is None or end_ts > cur_existing_end) else cur_existing_end

        df_dl = download_history(cur, dl_start, dl_end)
        df_dl = fill_missing_days(df_dl, dl_start, dl_end)

        df_dl["FromCurrency"] = BASE
        df_dl["ToCurrency"] = cur

        updates.append(df_dl)

    updates_df = pd.concat(updates, ignore_index=True)

    if master.empty:
        master_updated = updates_df
    else:
        master_updated = pd.concat([master, updates_df], ignore_index=True)
        master_updated["Date"] = pd.to_datetime(master_updated["Date"]).dt.normalize()
        master_updated = (
            master_updated
            .drop_duplicates(subset=["Date", "FromCurrency", "ToCurrency"], keep="last")
            .sort_values(["Date", "FromCurrency", "ToCurrency"])
            .reset_index(drop=True)
        )

    master_updated.to_parquet(out_path, index=False)
    return master_updated
