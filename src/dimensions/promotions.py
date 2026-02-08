# ---------------------------------------------------------
#  PROMOTIONS DIMENSION (PIPELINE READY – OPTIMIZED)
#  + Writes StartDate/EndDate as Arrow date32 (if pyarrow available)
#    so Power Query imports them as Date (not DateTime).
# ---------------------------------------------------------

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import warnings

from src.utils.logging_utils import info, skip, stage
from src.versioning.version_store import should_regenerate, save_version


# ---------------------------------------------------------
#  CONSTANTS
# ---------------------------------------------------------

PROMO_TYPES = {
    "Holiday": "Holiday Discount",
    "Seasonal": "Seasonal Discount",
    "Clearance": "Clearance",
    "Limited": "Limited Time",
    "NoDiscount": "No Discount",
}

CATEGORIES = ["Store", "Online", "Region"]

# name, start_mmdd, end_mmdd, discount_min, discount_max
HOLIDAYS: List[Tuple[str, str, str, float, float]] = [
    ("Black Friday",   "11-25", "11-30", 0.20, 0.70),
    ("Cyber Monday",   "11-28", "12-02", 0.15, 0.50),
    ("Christmas",      "12-10", "12-31", 0.20, 0.60),
    ("New Year",       "12-26", "01-05", 0.10, 0.40),
    ("Back-to-School", "07-01", "09-15", 0.05, 0.25),
    ("Easter",         "03-20", "04-10", 0.05, 0.30),
    ("Diwali",         "10-01", "11-15", 0.10, 0.50),
]

# Seasonal name -> (start_month, end_month) (end may wrap to next year)
SEASON_WINDOWS: Dict[str, Tuple[int, int]] = {
    "Spring Clearance": (2, 4),
    "Summer Sale": (5, 8),
    "Autumn Sale": (9, 10),
    "Winter Sale": (11, 1),           # wraps year
    "Mid-Season Discount": (3, 9),
}


# ---------------------------------------------------------
#  PARQUET WRITER: FORCE DATE TYPES FOR POWER QUERY
# ---------------------------------------------------------

def _infer_datetime_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]


def _write_parquet_with_date32(
    df: pd.DataFrame,
    out_path: Path,
    *,
    compression: str = "snappy",
    compression_level: Optional[int] = None,
    force_date32: bool = True,
) -> None:
    """
    Write Parquet so datetime64 columns are stored as Arrow date32.
    Power Query usually imports Arrow DATE as Date (not DateTime).

    Fallback:
      - If pyarrow is unavailable and force_date32 is True,
        convert datetime cols to python date objects (object dtype) just for writing.
    """
    dt_cols = _infer_datetime_cols(df)

    if not dt_cols:
        df.to_parquet(out_path, index=False)
        return

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception:
        if force_date32:
            df2 = df.copy()
            for c in dt_cols:
                df2[c] = pd.to_datetime(df2[c]).dt.date
            df2.to_parquet(out_path, index=False)
        else:
            df.to_parquet(out_path, index=False)
        return

    # Build Arrow table and cast datetime cols to date32
    table = pa.Table.from_pandas(df, preserve_index=False)

    fields = []
    dt_cols_set = set(dt_cols)
    for f in table.schema:
        if f.name in dt_cols_set:
            fields.append(pa.field(f.name, pa.date32()))
        else:
            fields.append(f)

    table = table.cast(pa.schema(fields), safe=False)

    kwargs = {"compression": compression}
    if compression_level is not None:
        kwargs["compression_level"] = compression_level

    pq.write_table(table, str(out_path), **kwargs)


# ---------------------------------------------------------
#  HELPERS
# ---------------------------------------------------------

def _mmdd(mmdd: str, year: int) -> pd.Timestamp:
    m, d = map(int, mmdd.split("-"))
    return pd.Timestamp(year=year, month=m, day=d)


def _valid_window(s: Optional[pd.Timestamp], e: Optional[pd.Timestamp]) -> bool:
    return s is not None and e is not None and s < e


def _build_year_windows(
    start: pd.Timestamp, end: pd.Timestamp
) -> Tuple[List[int], Dict[int, Tuple[pd.Timestamp, pd.Timestamp]]]:
    start = pd.to_datetime(start).normalize()
    end = pd.to_datetime(end).normalize()
    if end < start:
        raise ValueError(f"Promotions: end < start ({end.date()} < {start.date()})")

    years = list(range(start.year, end.year + 1))
    windows: Dict[int, Tuple[pd.Timestamp, pd.Timestamp]] = {}
    for y in years:
        ys = max(pd.Timestamp(f"{y}-01-01"), start)
        ye = min(pd.Timestamp(f"{y}-12-31"), end)
        windows[y] = (ys, ye)
    return years, windows


def _clamp_to_year_window(
    dt: pd.Timestamp, year_windows: Dict[int, Tuple[pd.Timestamp, pd.Timestamp]]
) -> Optional[pd.Timestamp]:
    ws_we = year_windows.get(int(dt.year))
    if not ws_we:
        return None
    ws, we = ws_we
    if dt < ws:
        return ws
    if dt > we:
        return we
    return dt


def _normalize_override_dates(promo_cfg: Dict) -> Dict:
    """
    Supports either:
      promotions.override: { dates: {start,end} }
    or:
      promotions.override: { start,end }
    """
    override = (promo_cfg or {}).get("override", {}) or {}
    if isinstance(override, dict) and isinstance(override.get("dates"), dict):
        return override["dates"] or {}
    return override if isinstance(override, dict) else {}


def _pick_seed(promo_cfg: Dict, default_seed: int = 42) -> int:
    override = (promo_cfg or {}).get("override", {}) or {}
    if not isinstance(override, dict):
        override = {}

    seed = (promo_cfg or {}).get("seed", None)
    if seed is None:
        seed = override.get("seed", None)
    if seed is None:
        seed = default_seed

    try:
        return int(seed)
    except (TypeError, ValueError):
        raise ValueError(f"Promotions: seed must be an integer, got {seed!r}")


def _discount(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.round(rng.uniform(lo, hi), 2))


def _category(rng: np.random.Generator) -> str:
    return str(rng.choice(CATEGORIES))


def _seasonal_start_date(rng: np.random.Generator, y: int, start_m: int, end_m: int) -> pd.Timestamp:
    # window wraps year
    if start_m <= end_m:
        m = int(rng.integers(start_m, end_m + 1))
        year_for_start = y
    else:
        months = list(range(start_m, 13)) + list(range(1, end_m + 1))
        m = int(rng.choice(months))
        year_for_start = y if m >= start_m else y + 1

    day = int(rng.integers(1, 25))
    return pd.Timestamp(year=year_for_start, month=m, day=day)


def _random_start_date(rng: np.random.Generator, y: int) -> pd.Timestamp:
    m = int(rng.integers(1, 13))
    d = int(rng.integers(1, 25))
    return pd.Timestamp(year=y, month=m, day=d)


# ---------------------------------------------------------
#  PROMOTION GENERATOR
# ---------------------------------------------------------

def generate_promotions_catalog(
    *,
    years: List[int],
    year_windows: Dict[int, Tuple[pd.Timestamp, pd.Timestamp]],
    num_seasonal: int = 20,
    num_clearance: int = 8,
    num_limited: int = 12,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Returns columns (UNCHANGED):
      PromotionKey, PromotionLabel, PromotionName, PromotionDescription,
      DiscountPct, PromotionType, PromotionCategory, StartDate, EndDate
    """
    if not years:
        raise ValueError("Promotions: No years provided.")
    if not year_windows:
        raise ValueError("Promotions: year_windows is empty.")

    rng = np.random.default_rng(int(seed))
    rows: List[Dict] = []

    # Holidays
    for y in years:
        for name, s_mmdd, e_mmdd, dmin, dmax in HOLIDAYS:
            s = _mmdd(s_mmdd, y)

            e_month = int(e_mmdd.split("-")[0])
            e_year = y + 1 if e_month < s.month else y
            e = _mmdd(e_mmdd, e_year)

            s = _clamp_to_year_window(s, year_windows)
            e = _clamp_to_year_window(e, year_windows)

            if not _valid_window(s, e):
                continue

            rows.append(
                {
                    "TypeGroup": "Holiday",
                    "SeasonType": name,
                    "Year": y,
                    "PromotionName": f"{name} {y}",
                    "PromotionDescription": f"{name} {y} Promotion",
                    "DiscountPct": _discount(rng, dmin, dmax),
                    "PromotionType": PROMO_TYPES["Holiday"],
                    "PromotionCategory": _category(rng),
                    "StartDate": pd.Timestamp(s),
                    "EndDate": pd.Timestamp(e),
                }
            )

    # Seasonal
    for _ in range(int(num_seasonal)):
        y = int(rng.choice(years))
        season_name = str(rng.choice(list(SEASON_WINDOWS.keys())))
        sm, em = SEASON_WINDOWS[season_name]

        start = _seasonal_start_date(rng, y, sm, em)
        end = start + timedelta(days=int(rng.integers(10, 60)))

        start = _clamp_to_year_window(start, year_windows)
        end = _clamp_to_year_window(end, year_windows)

        if not _valid_window(start, end):
            continue

        rows.append(
            {
                "TypeGroup": "Seasonal",
                "SeasonType": season_name,
                "Year": y,
                "PromotionName": None,
                "PromotionDescription": None,
                "DiscountPct": _discount(rng, 0.05, 0.30),
                "PromotionType": PROMO_TYPES["Seasonal"],
                "PromotionCategory": _category(rng),
                "StartDate": pd.Timestamp(start),
                "EndDate": pd.Timestamp(end),
            }
        )

    # Clearance
    for _ in range(int(num_clearance)):
        y = int(rng.choice(years))
        start = _random_start_date(rng, y)
        end = start + timedelta(days=int(rng.integers(3, 25)))

        start = _clamp_to_year_window(start, year_windows)
        end = _clamp_to_year_window(end, year_windows)

        if not _valid_window(start, end):
            continue

        rows.append(
            {
                "TypeGroup": "Clearance",
                "SeasonType": "Clearance",
                "Year": y,
                "PromotionName": None,
                "PromotionDescription": None,
                "DiscountPct": _discount(rng, 0.30, 0.70),
                "PromotionType": PROMO_TYPES["Clearance"],
                "PromotionCategory": _category(rng),
                "StartDate": pd.Timestamp(start),
                "EndDate": pd.Timestamp(end),
            }
        )

    # Limited
    for _ in range(int(num_limited)):
        y = int(rng.choice(years))
        start = _random_start_date(rng, y)
        end = start + timedelta(days=int(rng.integers(1, 15)))

        start = _clamp_to_year_window(start, year_windows)
        end = _clamp_to_year_window(end, year_windows)

        if not _valid_window(start, end):
            continue

        rows.append(
            {
                "TypeGroup": "Limited",
                "SeasonType": "Limited Time",
                "Year": y,
                "PromotionName": None,
                "PromotionDescription": None,
                "DiscountPct": _discount(rng, 0.05, 0.35),
                "PromotionType": PROMO_TYPES["Limited"],
                "PromotionCategory": _category(rng),
                "StartDate": pd.Timestamp(start),
                "EndDate": pd.Timestamp(end),
            }
        )

    if not rows:
        raise ValueError("Promotions: No promotions could be generated (check date windows and config).")

    df = pd.DataFrame(rows)

    # Stable ordering
    df = df.sort_values(["TypeGroup", "Year", "SeasonType", "StartDate"]).reset_index(drop=True)

    # Number + label non-holiday promos
    mask_non_holiday = df["TypeGroup"] != "Holiday"
    if mask_non_holiday.any():
        grp = df.loc[mask_non_holiday].groupby(["Year", "TypeGroup", "SeasonType"], sort=False)
        local_index = grp.cumcount() + 1
        df.loc[mask_non_holiday, "LocalIndex"] = local_index.to_numpy()

        df.loc[mask_non_holiday, "PromotionName"] = (
            df.loc[mask_non_holiday, "SeasonType"].astype(str)
            + " "
            + df.loc[mask_non_holiday, "Year"].astype(int).astype(str)
            + " #"
            + df.loc[mask_non_holiday, "LocalIndex"].astype(int).astype(str)
        )
        df.loc[mask_non_holiday, "PromotionDescription"] = (
            df.loc[mask_non_holiday, "SeasonType"].astype(str)
            + " for "
            + df.loc[mask_non_holiday, "Year"].astype(int).astype(str)
        )

    df.loc[~mask_non_holiday, "LocalIndex"] = None

    # Append "No Discount"
    min_year = min(years)
    max_year = max(years)
    no_discount = pd.DataFrame(
        [
            {
                "TypeGroup": "NoDiscount",
                "SeasonType": "NoDiscount",
                "Year": min_year,
                "PromotionName": "No Discount",
                "PromotionDescription": "No Discount",
                "DiscountPct": 0.0,
                "PromotionType": PROMO_TYPES["NoDiscount"],
                "PromotionCategory": "No Discount",
                "StartDate": pd.Timestamp(year_windows[min_year][0]),
                "EndDate": pd.Timestamp(year_windows[max_year][1]),
                "LocalIndex": None,
            }
        ]
    )

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=FutureWarning,
            message=r".*DataFrame concatenation with empty or all-NA entries.*",
        )
        final = pd.concat([df, no_discount], ignore_index=True)

    # Final ordering + keys
    final = final.sort_values("StartDate").reset_index(drop=True)
    final["PromotionKey"] = (final.index + 1).astype(np.int64)
    final["PromotionLabel"] = final["PromotionKey"]

    # Output schema (UNCHANGED)
    return final[
        [
            "PromotionKey",
            "PromotionLabel",
            "PromotionName",
            "PromotionDescription",
            "DiscountPct",
            "PromotionType",
            "PromotionCategory",
            "StartDate",
            "EndDate",
        ]
    ]


# ---------------------------------------------------------
#  PIPELINE ENTRYPOINT
# ---------------------------------------------------------

def run_promotions(cfg: Dict, parquet_folder: Path) -> None:
    out_path = parquet_folder / "promotions.parquet"

    if not isinstance(cfg, dict) or "promotions" not in cfg:
        raise KeyError("Missing required config section: 'promotions'")

    promo_cfg = cfg["promotions"] or {}
    defaults_dates = (cfg.get("defaults", {}) or {}).get("dates") or (cfg.get("_defaults", {}) or {}).get("dates")
    if not defaults_dates or "start" not in defaults_dates or "end" not in defaults_dates:
        raise ValueError("Promotions: missing defaults.dates.start/end (or _defaults.dates.start/end)")

    version_cfg = {**promo_cfg, "global_dates": defaults_dates}

    force = bool(promo_cfg.get("_force_regenerate", False))
    if not force and not should_regenerate("promotions", version_cfg, out_path):
        skip("Promotions up-to-date; skipping.")
        return

    override_dates = _normalize_override_dates(promo_cfg)

    if override_dates.get("start") and override_dates.get("end"):
        start = pd.to_datetime(override_dates["start"])
        end = pd.to_datetime(override_dates["end"])
    else:
        start = pd.to_datetime(defaults_dates["start"])
        end = pd.to_datetime(defaults_dates["end"])

    years, windows = _build_year_windows(start, end)
    seed = _pick_seed(promo_cfg)

    # Optional parquet settings (safe defaults)
    compression = promo_cfg.get("parquet_compression", "snappy")
    compression_level = promo_cfg.get("parquet_compression_level", None)
    force_date32 = bool(promo_cfg.get("force_date32", True))

    with stage("Generating Promotions"):
        df = generate_promotions_catalog(
            years=years,
            year_windows=windows,
            num_seasonal=int(promo_cfg.get("num_seasonal", 20)),
            num_clearance=int(promo_cfg.get("num_clearance", 8)),
            num_limited=int(promo_cfg.get("num_limited", 12)),
            seed=seed,
        )

        # Write with date32 for Power Query
        _write_parquet_with_date32(
            df,
            out_path,
            compression=str(compression),
            compression_level=(int(compression_level) if compression_level is not None else None),
            force_date32=force_date32,
        )

    save_version("promotions", version_cfg, out_path)
    info(f"Promotions dimension written: {out_path}")
