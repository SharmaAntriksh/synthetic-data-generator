# ---------------------------------------------------------
#  PROMOTIONS DIMENSION (PIPELINE READY)
# ---------------------------------------------------------

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from src.utils.logging_utils import info, fail, skip, stage
from src.versioning.version_store import should_regenerate, save_version


# ---------------------------------------------------------
#  ORIGINAL PROMOTION GENERATOR (unchanged)  :contentReference[oaicite:0]{index=0}
# ---------------------------------------------------------

def generate_promotions_catalog(
    years,
    year_windows,
    num_seasonal=20,
    num_clearance=8,
    num_limited=12,
    seed=42
):
    """
    Generate promotions using precise date windows per year.
    `years` = list of valid years
    `year_windows` = { year: (start_dt, end_dt) }
    """
    if not years:
        raise ValueError("Promotions: No years provided.")

    np.random.seed(seed)

    start_year = min(years)
    end_year = max(years)

    # ------------------------------------------------------
    # Helpers using full date windows
    # ------------------------------------------------------
    def clamp(dt):
        y = dt.year
        if y not in year_windows:
            return None
        ws, we = year_windows[y]
        if dt < ws:
            return ws
        if dt > we:
            return we
        return dt

    def in_range(s, e):
        return s is not None and e is not None

    def mmdd_to_date(mmdd, y):
        m, d = map(int, mmdd.split("-"))
        return datetime(y, m, d)

    HOLIDAYS = [
        ("Black Friday",   "11-25", "11-30", 3, 10, 0.20, 0.70),
        ("Cyber Monday",   "11-28", "12-02", 2, 6,  0.15, 0.50),
        ("Christmas",      "12-10", "12-31", 7, 30, 0.20, 0.60),
        ("New Year",       "12-26", "01-05", 4, 14, 0.10, 0.40),
        ("Back-to-School", "07-01", "09-15", 14, 60, 0.05, 0.25),
        ("Easter",         "03-20", "04-10", 7, 25, 0.05, 0.30),
        ("Diwali",         "10-01", "11-15", 7, 30, 0.10, 0.50),
        ("Summer Sale",    "05-01", "08-31", 10, 60, 0.05, 0.30),
        ("Spring Sale",    "02-15", "04-30", 7, 40, 0.05, 0.25),
    ]

    PROMO_TYPES = {
        "Holiday": "Holiday Discount",
        "Seasonal": "Seasonal Discount",
        "Clearance": "Clearance",
        "Limited": "Limited Time",
        "NoDiscount": "No Discount"
    }

    CATEGORIES = ["Store", "Online", "Region"]

    SEASONAL_NAMES = [
        "Spring Clearance",
        "Summer Sale",
        "Autumn Sale",
        "Winter Sale",
        "Mid-Season Discount",
    ]

    holiday = []

    for y in years:
        for name, s_mmdd, e_mmdd, min_d, max_d, dmin, dmax in HOLIDAYS:

            s = mmdd_to_date(s_mmdd, y)

            e_month = int(e_mmdd.split("-")[0])
            s_month = int(s_mmdd.split("-")[0])
            e_year = y + 1 if e_month < s_month else y
            e = mmdd_to_date(e_mmdd, e_year)

            span = (e - s).days
            p_start = s if span <= min_d else s + timedelta(days=np.random.randint(0, span - min_d + 1))
            p_end   = p_start + timedelta(days=np.random.randint(min_d, max_d + 1))

            p_start = clamp(p_start)
            p_end   = clamp(p_end)

            if not in_range(p_start, p_end):
                continue

            holiday.append({
                "TypeGroup": "Holiday",
                "SeasonType": name,
                "Year": y,
                "PromotionName": f"{name} {y} Promotion",
                "PromotionDescription": f"{name} {y} Holiday Discount",
                "DiscountPct": round(np.random.uniform(dmin, dmax), 2),
                "PromotionType": PROMO_TYPES["Holiday"],
                "PromotionCategory": np.random.choice(CATEGORIES),
                "StartDate": pd.Timestamp(p_start),
                "EndDate": pd.Timestamp(p_end),
            })

    # seasonal, clearance, limited
    def random_window(y, min_days, max_days):
        start = datetime(y, np.random.randint(1, 13), np.random.randint(1, 29))
        end   = start + timedelta(days=np.random.randint(min_days, max_days))
        return clamp(start), clamp(end)

    seasonal, clearance, limited = [], [], []

    for _ in range(num_seasonal):
        y = np.random.choice(years)
        stype = np.random.choice(SEASONAL_NAMES)
        s, e = random_window(y, 10, 60)
        if in_range(s, e):
            seasonal.append({
                "TypeGroup": "Seasonal",
                "SeasonType": stype,
                "Year": y,
                "StartDate": pd.Timestamp(s),
                "EndDate": pd.Timestamp(e),
                "DiscountPct": round(np.random.uniform(0.05, 0.30), 2),
                "PromotionType": PROMO_TYPES["Seasonal"],
                "PromotionCategory": np.random.choice(CATEGORIES),
            })

    for _ in range(num_clearance):
        y = np.random.choice(years)
        s, e = random_window(y, 3, 25)
        if in_range(s, e):
            clearance.append({
                "TypeGroup": "Clearance",
                "SeasonType": "Clearance",
                "Year": y,
                "StartDate": pd.Timestamp(s),
                "EndDate": pd.Timestamp(e),
                "DiscountPct": round(np.random.uniform(0.30, 0.70), 2),
                "PromotionType": PROMO_TYPES["Clearance"],
                "PromotionCategory": np.random.choice(CATEGORIES),
            })

    for _ in range(num_limited):
        y = np.random.choice(years)
        s, e = random_window(y, 1, 15)
        if in_range(s, e):
            limited.append({
                "TypeGroup": "Limited",
                "SeasonType": "Limited Time",
                "Year": y,
                "StartDate": pd.Timestamp(s),
                "EndDate": pd.Timestamp(e),
                "DiscountPct": round(np.random.uniform(0.05, 0.35), 2),
                "PromotionType": PROMO_TYPES["Limited"],
                "PromotionCategory": np.random.choice(CATEGORIES),
            })

    df = pd.DataFrame(holiday + seasonal + clearance + limited)

    numbered = []
    non_holiday = df[df["TypeGroup"] != "Holiday"]

    for (y, group_type, stype), group in non_holiday.groupby(["Year", "TypeGroup", "SeasonType"]):
        group = group.sort_values("StartDate").copy()
        group["LocalIndex"] = range(1, len(group) + 1)
        group["PromotionName"] = (
            group["SeasonType"] + " " + group["Year"].astype(str) + " #" + group["LocalIndex"].astype(str)
        )
        group["PromotionDescription"] = (
            group["SeasonType"] + " for " + group["Year"].astype(str)
        )
        numbered.append(group)

    holiday_df = df[df["TypeGroup"] == "Holiday"].copy()
    holiday_df["LocalIndex"] = None

    final = pd.concat([holiday_df] + numbered, ignore_index=True)

    final = pd.concat([
        final,
        pd.DataFrame([{
            "TypeGroup": "NoDiscount",
            "SeasonType": "NoDiscount",
            "Year": min(years),
            "PromotionName": "No Discount",
            "PromotionDescription": "No Discount",
            "DiscountPct": 0.00,
            "PromotionType": "No Discount",
            "PromotionCategory": "No Discount",
            "StartDate": pd.Timestamp(year_windows[min(years)][0]),
            "EndDate": pd.Timestamp(year_windows[max(years)][1]),
            "LocalIndex": None,
        }])
    ], ignore_index=True)

    final = final.sort_values("StartDate").reset_index(drop=True)
    final["PromotionKey"] = final.index + 1
    final["PromotionLabel"] = final["PromotionKey"]

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
            "EndDate"
        ]
    ]


# ---------------------------------------------------------
#  PIPELINE ENTRYPOINT
# ---------------------------------------------------------

def run_promotions(cfg, parquet_folder: Path):
    out_path = parquet_folder / "promotions.parquet"

    # ---------------------------------------------------------
    # Promotions depend on cfg["promotions"] + defaults dates
    # ---------------------------------------------------------
    promo_cfg = cfg["promotions"]
    defaults_dates = cfg.get("defaults", {}).get("dates") or cfg.get("_defaults", {}).get("dates")

    # Build versioning input = promotions section + global default dates
    version_cfg = {
        **promo_cfg,
        "global_dates": defaults_dates
    }

    # ---------------------------------------------------------
    # Regeneration check (correct)
    # ---------------------------------------------------------
    if not should_regenerate("promotions", version_cfg, out_path):
        skip("Promotions up-to-date; skipping.")
        return

    # ---------------------------------------------------------
    # Resolve start/end dates
    # ---------------------------------------------------------
    override_dates = promo_cfg.get("override", {}).get("dates")
    if override_dates and override_dates.get("start") and override_dates.get("end"):
        start = pd.to_datetime(override_dates["start"])
        end   = pd.to_datetime(override_dates["end"])
    else:
        start = pd.to_datetime(defaults_dates["start"])
        end   = pd.to_datetime(defaults_dates["end"])

    years = list(range(start.year, end.year + 1))

    windows = {}
    for y in years:
        ys = pd.Timestamp(f"{y}-01-01")
        ye = pd.Timestamp(f"{y}-12-31")
        ys = max(ys, start)
        ye = min(ye, end)
        windows[y] = (ys, ye)

    # ---------------------------------------------------------
    # Generate promotions
    # ---------------------------------------------------------
    with stage("Generating Promotions"):
        df = generate_promotions_catalog(
            years=years,
            year_windows=windows,
            num_seasonal=promo_cfg.get("num_seasonal", 20),
            num_clearance=promo_cfg.get("num_clearance", 8),
            num_limited=promo_cfg.get("num_limited", 12),
            seed=promo_cfg.get("override", {}).get("seed", 42)
        )
        df.to_parquet(out_path, index=False)

    save_version("promotions", version_cfg, out_path)
    info(f"Promotions dimension written → {out_path}")

