# fx_yahoo.py
# Clean, stable FX history builder for USD base currency

import pandas as pd
import yfinance as yf
from pathlib import Path

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE = "USD"

CURRENCIES = [
    "USD", "EUR", "INR", "GBP", "AUD", "CAD",
    "CNY", "JPY", "NZD", "CHF", "SEK",
    "NOK", "SGD", "HKD", "KRW", "ZAR",
]

# ---------------------------------------------------------
# Yahoo ticker helper
# ---------------------------------------------------------

def ticker(cur):
    if cur == BASE:
        return None
    if cur == "INR":
        return "USDINR=X"        # INR is reversed in Yahoo
    return f"{cur}USD=X"         # EURUSD=X, GBPUSD=X, AUDUSD=X, etc.

# ---------------------------------------------------------
# Extract CLOSE prices safely from Yahoo DataFrame
# ---------------------------------------------------------

def extract_close(df):
    # Multi-index columns (new Yahoo format)
    if isinstance(df.columns, pd.MultiIndex):
        close_cols = [c for c in df.columns if c[0].lower() == "close"]
        if not close_cols:
            return None
        # Example: ("Close", "EURUSD=X")
        return df[close_cols[0]].dropna()

    # Single-index columns (old Yahoo format)
    if "Close" in df.columns:
        return df["Close"].dropna()

    return None

# ---------------------------------------------------------
# Download FX history for a single currency
# ---------------------------------------------------------

def download_history(cur, start, end):
    t = ticker(cur)
    if t is None:
        return pd.DataFrame()

    end_dt = pd.to_datetime(end) + pd.Timedelta(days=1)

    df = yf.download(
        t,
        start=str(start),
        end=end_dt.strftime("%Y-%m-%d"),
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if df is None or df.empty:
        return pd.DataFrame()

    close_series = extract_close(df)
    if close_series is None or close_series.empty:
        return pd.DataFrame()

    close_series.index = pd.to_datetime(close_series.index)

    rows = []
    for dt, rate in close_series.items():
        # ensure scalar + limit to 6 decimals
        rows.append({
            "Date": dt,
            "FromCurrency": BASE,
            "ToCurrency": cur,
            "ExchangeRate": round(float(rate), 8),
        })

    return pd.DataFrame(rows)

# ---------------------------------------------------------
# Build or update full FX history master file
# ---------------------------------------------------------

def build_or_update_fx(start_date, end_date, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing file or start fresh
    if out_path.exists():
        print("Loading existing FX history:", out_path)
        fx = pd.read_parquet(out_path)
        fx["Date"] = pd.to_datetime(fx["Date"])
        last_date = fx["Date"].max().date()
    else:
        print("No existing FX history found. Creating new.")
        fx = pd.DataFrame(columns=["Date", "FromCurrency", "ToCurrency", "ExchangeRate"])
        last_date = pd.to_datetime(start_date).date() - pd.Timedelta(days=1)

    target_end = pd.to_datetime(end_date).date()

    if last_date >= target_end:
        print("FX already covers required range.")
        return fx

    start_dt = pd.to_datetime(last_date) + pd.Timedelta(days=1)
    start = start_dt.date()

    print(f"Updating FX: {start} → {target_end}")

    all_rows = []

    for cur in CURRENCIES:
        if cur == BASE:
            continue

        print("Downloading:", cur)
        df = download_history(cur, start, target_end)
        if not df.empty:
            all_rows.append(df)

    # Remove any DF that is completely empty or all NaN
    valid_rows = [df for df in all_rows if not df.empty]

    if valid_rows:
        new_fx = pd.concat(valid_rows, ignore_index=True)
    else:
        new_fx = pd.DataFrame(columns=["Date", "FromCurrency", "ToCurrency", "ExchangeRate"])


    # Add USD→USD rows
    if not new_fx.empty:
        dates = sorted(new_fx["Date"].unique())
        usd_rows = [{
            "Date": d,
            "FromCurrency": BASE,
            "ToCurrency": BASE,
            "ExchangeRate": 1.0
        } for d in dates]

        new_fx = pd.concat([new_fx, pd.DataFrame(usd_rows)], ignore_index=True)

    # Merge with previous data
    if new_fx.empty:
        pass  # nothing to add
    elif fx.empty:
        fx = new_fx.copy()
    else:
        fx = pd.concat([fx, new_fx], ignore_index=True)

    # ------------------------------------------------------------
    # Fill missing weekend/holiday dates with forward-fill (ffill)
    # ------------------------------------------------------------
    full_range = pd.date_range(
        start=fx["Date"].min(),
        end=fx["Date"].max(),
        freq="D"
    )

    filled_rows = []
    for cur in fx["ToCurrency"].unique():
        sub = fx[fx["ToCurrency"] == cur].set_index("Date")
        sub = sub.reindex(full_range)                 # add all missing days
        sub["FromCurrency"] = BASE
        sub["ToCurrency"] = cur
        sub["ExchangeRate"] = sub["ExchangeRate"].ffill()   # weekend/holo ffill
        filled_rows.append(sub.reset_index().rename(columns={"index": "Date"}))

    fx = pd.concat(filled_rows, ignore_index=True)

    # ------------------------------------------------------------
    # Save final sorted file
    # ------------------------------------------------------------
    fx.sort_values(["Date", "ToCurrency"], inplace=True)
    fx.to_parquet(out_path, index=False)

    print("Saved FX:", out_path)
    print("Rows:", len(fx))

    return fx
