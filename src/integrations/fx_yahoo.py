import hashlib

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import timedelta

from src.utils.logging_utils import info
from src.exceptions import DimensionError
from src.defaults import CURRENCY_BASE, CURRENCY_DEFAULT_LIST


# ---------------------------------------------------------
# Future-rate projection tuning
# ---------------------------------------------------------
# Each currency's future path is a seeded random walk whose average trend is the
# currency's own long-run drift (shrunk toward neutral and capped) and whose
# day-to-day jitter is that currency's own historical volatility. This keeps
# trends diverging per currency without letting any estimate run away.
_FX_DRIFT_WINDOW_YEARS = 10     # trailing history used to estimate the trend
_FX_DRIFT_SHRINK = 0.6          # trust 60% of the measured long-run trend
_FX_DRIFT_CAP = 0.06            # hard speed limit: +/-6%/yr on the trend
_FX_VOL_FLOOR = 2e-4            # min daily jitter (avoid dead-flat paths)
_FX_VOL_CAP = 0.04             # max daily jitter (avoid absurd swings)
_FX_MIN_HISTORY = 250           # rows needed to estimate drift from history


def _currency_seed(base_seed: int, cur: str) -> int:
    """Stable per-currency seed so each currency gets an independent but
    fully reproducible future path."""
    h = hashlib.sha256(f"{int(base_seed)}:{cur}".encode()).hexdigest()
    return int(h[:16], 16)


def _estimate_daily_vol(rates: pd.Series) -> float:
    """Per-currency daily volatility from historical day-to-day moves.

    The master is dense (weekends/holidays carried forward), which injects
    no-change days; those are dropped so the estimate reflects real trading-day
    moves, then clamped to a sane band.
    """
    vals = rates.to_numpy(dtype="float64")
    if vals.size < 2:
        return _FX_VOL_FLOOR
    logret = np.log(np.clip(vals[1:], 1e-12, None) / np.clip(vals[:-1], 1e-12, None))
    logret = logret[np.isfinite(logret)]
    moves = logret[logret != 0.0]
    if moves.size < 30:
        return _FX_VOL_FLOOR
    return float(np.clip(moves.std(), _FX_VOL_FLOOR, _FX_VOL_CAP))


def _estimate_annual_drift(dates: pd.Series, rates: pd.Series, *, fallback: float) -> float:
    """Long-run annual trend from a straight-line fit through the (log) history.

    A line fit over a long window is far steadier than comparing two endpoints
    (which flips sign depending on the start day). The slope is shrunk toward
    no-trend and capped so a noisy or extreme estimate cannot produce runaway
    future rates. Currencies with too little history fall back to *fallback*.
    """
    if len(rates) < _FX_MIN_HISTORY:
        return float(np.clip(fallback, -_FX_DRIFT_CAP, _FX_DRIFT_CAP))

    cutoff = dates.max() - pd.DateOffset(years=_FX_DRIFT_WINDOW_YEARS)
    mask = (dates >= cutoff).to_numpy()
    d = dates[mask]
    r = rates[mask]
    if len(r) < _FX_MIN_HISTORY:
        d, r = dates, rates

    t_years = (d.to_numpy() - d.to_numpy()[0]) / np.timedelta64(1, "D") / 365.25
    y = np.log(np.clip(r.to_numpy(dtype="float64"), 1e-12, None))
    slope = np.polyfit(t_years, y, 1)[0]          # change in log(rate) per year
    annual = float(np.expm1(slope)) * _FX_DRIFT_SHRINK
    return float(np.clip(annual, -_FX_DRIFT_CAP, _FX_DRIFT_CAP))


def _project_future_rates(future_dates, anchor_rate, mu_annual, sigma_d, rng):
    """Seeded random walk for future dates.

    Continuous with history (the first projected day is one step past the
    anchor), jitters by *sigma_d* on trading days, and stays flat on weekends to
    mirror real FX. The whole week's drift is spread across the ~5 trading days
    so the path's average trend is *mu_annual* (zeroing weekend drift outright
    would undershoot it by ~5/7). Always positive.
    """
    mu_d = np.log1p(mu_annual) / 365.25
    weekday = future_dates.dt.weekday.to_numpy() < 5
    n = len(future_dates)
    z = rng.standard_normal(n)
    # 5 trading days carry 7 days' worth of drift; weekends contribute nothing.
    mu_trading = mu_d * 7.0 / 5.0
    incr = np.where(weekday, (mu_trading - 0.5 * sigma_d ** 2) + sigma_d * z, 0.0)
    return anchor_rate * np.exp(np.cumsum(incr))


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
    # Imported lazily: yfinance (and its transitive deps) is heavy, and importing it
    # at module scope pulled it into every spawned sales worker via the dimensions
    # import chain. Workers never download FX, so defer the cost to actual use.
    import yfinance as yf

    # Normalize input dates to Timestamps (safe for yfinance + pd.date_range)
    start_ts = pd.to_datetime(start_date).normalize()
    end_ts = pd.to_datetime(end_date).normalize()

    if currency == CURRENCY_BASE:
        dates = pd.date_range(start=start_ts, end=end_ts, freq="D")
        return pd.DataFrame({"Date": dates, "Rate": 1.0})

    # Primary: USD -> CUR
    primary = f"{CURRENCY_BASE}{currency}=X"
    fallback = f"{currency}{CURRENCY_BASE}=X"  # may represent USD per 1 CUR, invert to get CUR per 1 USD

    def _download(ticker: str) -> pd.DataFrame:
        data = yf.download(
            ticker,
            start=start_ts - timedelta(days=3),
            end=end_ts + timedelta(days=3),
            auto_adjust=False,
            progress=False,
        )
        return data

    invert = False
    data = _download(primary)

    if data.empty:
        data = _download(fallback)
        invert = True

    if data.empty:
        raise DimensionError(f"No FX data found for {currency} (tried {primary} and {fallback})")

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
    Reindex *df* to every calendar day in ``[start_date, end_date]``, filling
    weekends/holidays by forward fill (then backfill leading gaps).

    ``df`` may (and from ``download_history`` does) contain a few buffer days of
    real data just outside the window; those are used as fill sources so a window
    that itself contains no trading day (e.g. a single weekend) still fills from
    the adjacent Friday. Only when there is no real rate anywhere in or around the
    window do we raise — refusing to emit a misleading 1.0 (par) rate.
    """
    start_ts = pd.to_datetime(start_date).normalize()
    end_ts = pd.to_datetime(end_date).normalize()

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    df = df.drop_duplicates(subset="Date", keep="last").set_index("Date").sort_index()

    window = pd.date_range(start=start_ts, end=end_ts, freq="D")

    # Fill across the union of real observations (incl. out-of-window buffer days)
    # and the window itself, so in-window weekend/holiday dates ffill/bfill from
    # the nearest real trading day even when the window has no trading day of its own.
    all_dates = df.index.union(pd.DatetimeIndex(window))
    rates = df["Rate"].reindex(all_dates).ffill().bfill()
    window_rates = rates.reindex(pd.DatetimeIndex(window)).to_numpy()

    if np.isnan(window_rates).any():
        # No real rate anywhere in or adjacent to this window (df empty/all-NaN).
        # A 1.0 fallback would silently corrupt the series with a fake 1:1 USD
        # parity, so fail instead.
        raise DimensionError(
            f"fill_missing_days: no usable rate data in or around window "
            f"{start_ts.date()} -> {end_ts.date()}. "
            "Refusing to fall back to a 1.0 par rate."
        )

    return pd.DataFrame({"Date": window, "Rate": window_rates})


# ---------------------------------------------------------
# Refresh master FX store to today
# ---------------------------------------------------------
def refresh_fx_master(out_path):
    """
    Top up the master FX file to today's date.

    Reads whatever currencies are already stored in the master and downloads
    only the gap from each currency's last recorded date to today.  No config
    or date-range arguments needed — the master is self-describing.

    Intended to be run on-demand (e.g. via --refresh-fx-master CLI flag) so
    the bundled file stays current without triggering a full pipeline run.
    """
    out_path = Path(out_path)
    if not out_path.exists():
        raise FileNotFoundError(f"FX master file not found: {out_path}")

    master = pd.read_parquet(out_path)
    if master.empty:
        raise ValueError("FX master file is empty — nothing to refresh.")

    master["Date"] = pd.to_datetime(master["Date"]).dt.normalize()
    today = pd.Timestamp.now().normalize()

    currencies_in_master = master["ToCurrency"].unique().tolist()
    updates = []

    for cur in currencies_in_master:
        df_cur = master[master["ToCurrency"] == cur]
        last_date = pd.to_datetime(df_cur["Date"].max()).normalize()

        if last_date >= today:
            info(f"FX for {cur} already up to date ({last_date.date()}); skipping.")
            continue

        gap_start = last_date + pd.Timedelta(days=1)
        info(f"Refreshing FX for {cur}: {gap_start.date()} -> {today.date()}")
        df_gap = download_history(cur, gap_start, today)
        df_gap = fill_missing_days(df_gap, gap_start, today)
        df_gap["FromCurrency"] = CURRENCY_BASE
        df_gap["ToCurrency"] = cur
        updates.append(df_gap)

    if not updates:
        info("FX master already up to date - nothing downloaded.")
        return master

    updates_df = pd.concat(updates, ignore_index=True)
    master_updated = pd.concat([master, updates_df], ignore_index=True)
    master_updated["Date"] = pd.to_datetime(master_updated["Date"]).dt.normalize()
    master_updated = (
        master_updated
        .drop_duplicates(subset=["Date", "FromCurrency", "ToCurrency"], keep="last")
        .sort_values(["Date", "FromCurrency", "ToCurrency"])
        .reset_index(drop=True)
    )

    master_updated.to_parquet(out_path, index=False)
    info(f"FX master refreshed -> {out_path.name}  ({len(master_updated)} rows)")
    return master_updated


# ---------------------------------------------------------
# Build or update master FX store
# ---------------------------------------------------------
def build_or_update_fx(start_date, end_date, out_path, currencies=None, annual_drift=0.02, seed=42):
    """
    Build or update a master FX file covering the date range.

    Stored invariant:
      FromCurrency = USD
      ToCurrency   = CUR
      Rate         = CUR per 1 USD (USD -> CUR)

    Date is kept as datetime64[ns] throughout.

    For dates beyond the last real data point (i.e. future dates), each currency
    is projected as a seeded random walk: its average trend is its own long-run
    drift (estimated from history, shrunk and capped), its day-to-day jitter is
    its own historical volatility, and weekends stay flat. *seed* makes the paths
    reproducible; *annual_drift* is the fallback trend for currencies with too
    little history to estimate one. Projected values are NOT written back to the
    master file.
    """
    if annual_drift <= -1.0:
        raise ValueError(
            f"annual_drift must be > -1.0 (got {annual_drift}); "
            "values <= -1 produce zero or negative projected rates."
        )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    curr_list = currencies if currencies is not None else CURRENCY_DEFAULT_LIST

    start_ts = pd.to_datetime(start_date).normalize()
    end_ts = pd.to_datetime(end_date).normalize()

    # Load existing master file (keep datetime64)
    if out_path.exists():
        master = pd.read_parquet(out_path)
        if not master.empty:
            master["Date"] = pd.to_datetime(master["Date"]).dt.normalize()
    else:
        master = pd.DataFrame(columns=["Date", "FromCurrency", "ToCurrency", "Rate"])

    today = pd.Timestamp.now().normalize()
    updates = []
    cached_currencies = []

    for cur in curr_list:
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

        # Determine which gaps need downloading (only fetch what isn't already in the master)
        gaps = []
        if cur_existing_start is None:
            # No existing data — download the full requested range (capped at today for real rates)
            gaps.append((start_ts, min(end_ts, today)))
        else:
            if start_ts < cur_existing_start:
                gaps.append((start_ts, cur_existing_start - pd.Timedelta(days=1)))
            if end_ts > cur_existing_end:
                gaps.append((cur_existing_end + pd.Timedelta(days=1), min(end_ts, today)))

        if not gaps:
            cached_currencies.append(cur)
        else:
            for gap_start, gap_end in gaps:
                if gap_start > gap_end:
                    continue
                info(f"Downloading FX for {cur}: {gap_start.date()} -> {gap_end.date()}")
                df_gap = download_history(cur, gap_start, gap_end)
                df_gap = fill_missing_days(df_gap, gap_start, gap_end)
                df_gap["FromCurrency"] = CURRENCY_BASE
                df_gap["ToCurrency"] = cur
                updates.append(df_gap)

    if cached_currencies:
        info(f"FX cached (already covered): {', '.join(cached_currencies)}")

    # Merge any new downloads into master and persist (only real data goes into the file)
    if updates:
        updates_df = pd.concat(updates, ignore_index=True)
        master_updated = pd.concat([master, updates_df], ignore_index=True) if not master.empty else updates_df
        master_updated["Date"] = pd.to_datetime(master_updated["Date"]).dt.normalize()
        master_updated = (
            master_updated
            .drop_duplicates(subset=["Date", "FromCurrency", "ToCurrency"], keep="last")
            .sort_values(["Date", "FromCurrency", "ToCurrency"])
            .reset_index(drop=True)
        )
        master_updated.to_parquet(out_path, index=False)
    else:
        master_updated = master.copy()

    # Build the return DataFrame for the full requested range.
    # Real-data gaps (weekends/holidays) are filled via ffill within the master.
    # Future dates (beyond the last real rate) are projected as a per-currency
    # seeded random walk. Projected values are NOT written back to the master.
    full_range = pd.date_range(start=start_ts, end=end_ts, freq="D")
    parts = []
    for cur in curr_list:
        df_cur = master_updated[master_updated["ToCurrency"] == cur].sort_values("Date").copy()
        if df_cur.empty:
            continue

        full = pd.DataFrame({"Date": full_range})
        merged = full.merge(df_cur, on="Date", how="left")

        # Fill weekends/holidays within the real data range
        merged["Rate"] = merged["Rate"].ffill().bfill()

        # Project future dates beyond the last real anchor
        anchor_date = df_cur["Date"].max()
        future_mask = merged["Date"] > anchor_date
        if future_mask.any():
            # Derive anchor_rate from the ffill'd merged frame when possible.
            # When the entire requested range is beyond anchor_date (e.g. a fully
            # future date window), historical within the merged frame is empty —
            # fall back to the last real rate from df_cur directly.
            historical = merged.loc[~future_mask, "Rate"]
            if not historical.empty and not historical.iloc[-1:].isna().all():
                anchor_rate = historical.iloc[-1]
            else:
                anchor_rate = df_cur["Rate"].iloc[-1]
            # Per-currency realistic projection: long-run drift (shrunk + capped)
            # plus that currency's own historical volatility, as an independent
            # seeded random walk. Trends diverge per currency and cross-pairs are
            # no longer dead-flat.
            mu_annual = _estimate_annual_drift(df_cur["Date"], df_cur["Rate"], fallback=annual_drift)
            sigma_d = _estimate_daily_vol(df_cur["Rate"])
            rng = np.random.default_rng(_currency_seed(seed, cur))
            merged.loc[future_mask, "Rate"] = _project_future_rates(
                merged.loc[future_mask, "Date"], anchor_rate, mu_annual, sigma_d, rng
            )

        merged["FromCurrency"] = CURRENCY_BASE
        merged["ToCurrency"] = cur
        parts.append(merged[["Date", "FromCurrency", "ToCurrency", "Rate"]])

    return pd.concat(parts, ignore_index=True) if parts else master_updated
