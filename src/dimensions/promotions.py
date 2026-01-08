# ---------------------------------------------------------
#  PROMOTIONS DIMENSION (PIPELINE READY – FIXED)
# ---------------------------------------------------------

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from src.utils.logging_utils import info, skip, stage
from src.versioning.version_store import should_regenerate, save_version


# ---------------------------------------------------------
#  PROMOTION GENERATOR (SEMANTIC + SCALABLE)
# ---------------------------------------------------------

def generate_promotions_catalog(
    years,
    year_windows,
    num_seasonal=20,
    num_clearance=8,
    num_limited=12,
    seed=42
):
    if not years:
        raise ValueError("Promotions: No years provided.")

    np.random.seed(seed)

    # -----------------------------------------------------
    # Helpers
    # -----------------------------------------------------
    def clamp(dt):
        ws, we = year_windows.get(dt.year, (None, None))
        if ws is None:
            return None
        return max(ws, min(dt, we))

    def valid(s, e):
        return s is not None and e is not None and s < e

    def mmdd(mmdd, y):
        m, d = map(int, mmdd.split("-"))
        return datetime(y, m, d)

    # -----------------------------------------------------
    # Promotion Metadata
    # -----------------------------------------------------
    PROMO_TYPES = {
        "Holiday": "Holiday Discount",
        "Seasonal": "Seasonal Discount",
        "Clearance": "Clearance",
        "Limited": "Limited Time",
        "NoDiscount": "No Discount"
    }

    CATEGORIES = ["Store", "Online", "Region"]

    # -----------------------------------------------------
    # HOLIDAYS (DETERMINISTIC – ONE PER YEAR)
    # -----------------------------------------------------
    HOLIDAYS = [
        ("Black Friday",   "11-25", "11-30", 0.20, 0.70),
        ("Cyber Monday",   "11-28", "12-02", 0.15, 0.50),
        ("Christmas",      "12-10", "12-31", 0.20, 0.60),
        ("New Year",       "12-26", "01-05", 0.10, 0.40),
        ("Back-to-School", "07-01", "09-15", 0.05, 0.25),
        ("Easter",         "03-20", "04-10", 0.05, 0.30),
        ("Diwali",         "10-01", "11-15", 0.10, 0.50),
    ]

    holiday_rows = []

    for y in years:
        for name, s_mmdd, e_mmdd, dmin, dmax in HOLIDAYS:
            s = mmdd(s_mmdd, y)
            e_month = int(e_mmdd.split("-")[0])
            e_year = y + 1 if e_month < s.month else y
            e = mmdd(e_mmdd, e_year)

            s, e = clamp(s), clamp(e)
            if not valid(s, e):
                continue

            holiday_rows.append({
                "TypeGroup": "Holiday",
                "SeasonType": name,
                "Year": y,
                "PromotionName": f"{name} {y}",
                "PromotionDescription": f"{name} {y} Promotion",
                "DiscountPct": round(np.random.uniform(dmin, dmax), 2),
                "PromotionType": PROMO_TYPES["Holiday"],
                "PromotionCategory": np.random.choice(CATEGORIES),
                "StartDate": pd.Timestamp(s),
                "EndDate": pd.Timestamp(e),
            })

    # -----------------------------------------------------
    # SEASONAL (BOUNDED RANDOMNESS)
    # -----------------------------------------------------
    SEASON_WINDOWS = {
        "Spring Clearance": (2, 4),
        "Summer Sale": (5, 8),
        "Autumn Sale": (9, 10),
        "Winter Sale": (11, 1),
        "Mid-Season Discount": (3, 9),
    }

    def seasonal_window(y, start_m, end_m):
        if start_m <= end_m:
            m = np.random.randint(start_m, end_m + 1)
            year = y
        else:
            m = np.random.choice([*range(start_m, 13), *range(1, end_m + 1)])
            year = y if m >= start_m else y + 1

        start = datetime(year, m, np.random.randint(1, 25))
        end = start + timedelta(days=np.random.randint(10, 60))
        return clamp(start), clamp(end)

    seasonal_rows = []

    for _ in range(num_seasonal):
        y = np.random.choice(years)
        name = np.random.choice(list(SEASON_WINDOWS.keys()))
        sm, em = SEASON_WINDOWS[name]
        s, e = seasonal_window(y, sm, em)
        if valid(s, e):
            seasonal_rows.append({
                "TypeGroup": "Seasonal",
                "SeasonType": name,
                "Year": y,
                "StartDate": pd.Timestamp(s),
                "EndDate": pd.Timestamp(e),
                "DiscountPct": round(np.random.uniform(0.05, 0.30), 2),
                "PromotionType": PROMO_TYPES["Seasonal"],
                "PromotionCategory": np.random.choice(CATEGORIES),
            })

    # -----------------------------------------------------
    # CLEARANCE / LIMITED (FREE RANDOMNESS)
    # -----------------------------------------------------
    def random_window(y, min_d, max_d):
        start = datetime(y, np.random.randint(1, 13), np.random.randint(1, 25))
        end = start + timedelta(days=np.random.randint(min_d, max_d))
        return clamp(start), clamp(end)

    clearance_rows, limited_rows = [], []

    for _ in range(num_clearance):
        y = np.random.choice(years)
        s, e = random_window(y, 3, 25)
        if valid(s, e):
            clearance_rows.append({
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
        if valid(s, e):
            limited_rows.append({
                "TypeGroup": "Limited",
                "SeasonType": "Limited Time",
                "Year": y,
                "StartDate": pd.Timestamp(s),
                "EndDate": pd.Timestamp(e),
                "DiscountPct": round(np.random.uniform(0.05, 0.35), 2),
                "PromotionType": PROMO_TYPES["Limited"],
                "PromotionCategory": np.random.choice(CATEGORIES),
            })

    # -----------------------------------------------------
    # FINAL ASSEMBLY
    # -----------------------------------------------------
    df = pd.DataFrame(
        holiday_rows + seasonal_rows + clearance_rows + limited_rows
    )

    numbered = []
    for (y, tg, st), g in df[df["TypeGroup"] != "Holiday"].groupby(
        ["Year", "TypeGroup", "SeasonType"]
    ):
        g = g.sort_values("StartDate").copy()
        g["LocalIndex"] = range(1, len(g) + 1)
        g["PromotionName"] = f"{st} {y} #" + g["LocalIndex"].astype(str)
        g["PromotionDescription"] = f"{st} for {y}"
        numbered.append(g)

    holidays = df[df["TypeGroup"] == "Holiday"].copy()
    holidays["LocalIndex"] = None

    final = pd.concat([holidays] + numbered, ignore_index=True)

    final = pd.concat([
        final,
        pd.DataFrame([{
            "TypeGroup": "NoDiscount",
            "SeasonType": "NoDiscount",
            "Year": min(years),
            "PromotionName": "No Discount",
            "PromotionDescription": "No Discount",
            "DiscountPct": 0.0,
            "PromotionType": PROMO_TYPES["NoDiscount"],
            "PromotionCategory": "No Discount",
            "StartDate": year_windows[min(years)][0],
            "EndDate": year_windows[max(years)][1],
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
            "EndDate",
        ]
    ]


# ---------------------------------------------------------
#  PIPELINE ENTRYPOINT (UNCHANGED)
# ---------------------------------------------------------

def run_promotions(cfg, parquet_folder: Path):
    out_path = parquet_folder / "promotions.parquet"

    promo_cfg = cfg["promotions"]
    defaults_dates = cfg.get("defaults", {}).get("dates") or cfg.get("_defaults", {}).get("dates")

    version_cfg = {**promo_cfg, "global_dates": defaults_dates}

    if not should_regenerate("promotions", version_cfg, out_path):
        skip("Promotions up-to-date; skipping.")
        return

    override = promo_cfg.get("override", {}).get("dates")
    if override and override.get("start") and override.get("end"):
        start = pd.to_datetime(override["start"])
        end = pd.to_datetime(override["end"])
    else:
        start = pd.to_datetime(defaults_dates["start"])
        end = pd.to_datetime(defaults_dates["end"])

    years = list(range(start.year, end.year + 1))

    windows = {}
    for y in years:
        ys = max(pd.Timestamp(f"{y}-01-01"), start)
        ye = min(pd.Timestamp(f"{y}-12-31"), end)
        windows[y] = (ys, ye)

    with stage("Generating Promotions"):
        df = generate_promotions_catalog(
            years=years,
            year_windows=windows,
            num_seasonal=promo_cfg.get("num_seasonal", 20),
            num_clearance=promo_cfg.get("num_clearance", 8),
            num_limited=promo_cfg.get("num_limited", 12),
            seed=promo_cfg.get("override", {}).get("seed", 42),
        )
        df.to_parquet(out_path, index=False)

    save_version("promotions", version_cfg, out_path)
    info(f"Promotions dimension written: {out_path}")
