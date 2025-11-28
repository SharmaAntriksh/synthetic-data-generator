import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_promotions_catalog(
    years=range(2021, 2026),
    num_seasonal=20,
    num_clearance=8,
    num_limited=12,
    seed=42
):
    """
    Generate holiday, seasonal, clearance, limited-time,
    and a final No-Discount promotion.

    FULLY IDENTICAL logic to original — ONLY cleaned.
    """
    np.random.seed(seed)

    years = list(years)
    start_year, end_year = min(years), max(years)

    # --------------------------------------------------------
    # Holiday templates
    # --------------------------------------------------------
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

    PROMOTION_TYPES = {
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

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------
    def mmdd_to_date(mmdd, year):
        m, d = map(int, mmdd.split("-"))
        return datetime(year, m, d)

    def clamp(dt):
        """Ensure start/end stay within the selected year range."""
        if dt.year < start_year:
            return datetime(start_year, dt.month, dt.day)
        if dt.year > end_year:
            return datetime(end_year, dt.month, dt.day)
        return dt

    def overlaps(start, end):
        """Ensure date is inside the generation window."""
        return not (end.year < start_year or start.year > end_year)

    # ========================================================
    # 1. HOLIDAY PROMOTIONS
    # ========================================================
    holiday_promos = []
    for y in years:
        for name, s_mmdd, e_mmdd, min_d, max_d, dmin, dmax in HOLIDAYS:

            s = mmdd_to_date(s_mmdd, y)
            e_month = int(e_mmdd.split("-")[0])
            e_year = y + 1 if e_month < int(s_mmdd.split("-")[0]) else y
            e = mmdd_to_date(e_mmdd, e_year)

            span = (e - s).days
            if span <= min_d:
                p_start = s
            else:
                # Choose a random start inside allowed window
                p_start = s + timedelta(days=np.random.randint(0, span - min_d + 1))

            # Random length between min_d and max_d
            p_end = p_start + timedelta(days=np.random.randint(min_d, max_d + 1))

            p_start = clamp(p_start)
            p_end = clamp(p_end)

            if not overlaps(p_start, p_end):
                continue

            holiday_promos.append({
                "TypeGroup": "Holiday",
                "SeasonType": name,
                "Year": y,
                "PromotionName": f"{name} {y} Promotion",
                "PromotionDescription": f"{name} {y} Holiday Discount",
                "DiscountPct": round(np.random.uniform(dmin, dmax), 2),
                "PromotionType": PROMOTION_TYPES["Holiday"],
                "PromotionCategory": np.random.choice(CATEGORIES),
                "StartDate": pd.Timestamp(p_start),
                "EndDate": pd.Timestamp(p_end),
            })

    # ========================================================
    # 2. SEASONAL PROMOTIONS (unsorted)
    # ========================================================
    seasonal_promos = []
    for _ in range(num_seasonal):
        y = np.random.choice(years)
        stype = np.random.choice(SEASONAL_NAMES)

        start = datetime(y, np.random.randint(1, 13), np.random.randint(1, 28))
        end = start + timedelta(days=np.random.randint(10, 61))

        start, end = clamp(start), clamp(end)
        if overlaps(start, end):
            seasonal_promos.append({
                "TypeGroup": "Seasonal",
                "SeasonType": stype,
                "Year": y,
                "StartDate": pd.Timestamp(start),
                "EndDate": pd.Timestamp(end),
                "DiscountPct": round(np.random.uniform(0.05, 0.30), 2),
                "PromotionType": PROMOTION_TYPES["Seasonal"],
                "PromotionCategory": np.random.choice(CATEGORIES),
            })

    # ========================================================
    # 3. CLEARANCE PROMOTIONS (unsorted)
    # ========================================================
    clearance_promos = []
    for _ in range(num_clearance):
        y = np.random.choice(years)

        start = datetime(y, np.random.randint(1, 13), np.random.randint(1, 28))
        end = start + timedelta(days=np.random.randint(3, 25))

        start, end = clamp(start), clamp(end)
        if overlaps(start, end):
            clearance_promos.append({
                "TypeGroup": "Clearance",
                "SeasonType": "Clearance",
                "Year": y,
                "StartDate": pd.Timestamp(start),
                "EndDate": pd.Timestamp(end),
                "DiscountPct": round(np.random.uniform(0.30, 0.70), 2),
                "PromotionType": PROMOTION_TYPES["Clearance"],
                "PromotionCategory": np.random.choice(CATEGORIES),
            })

    # ========================================================
    # 4. LIMITED-TIME PROMOTIONS (unsorted)
    # ========================================================
    limited_promos = []
    for _ in range(num_limited):
        y = np.random.choice(years)

        start = datetime(y, np.random.randint(1, 13), np.random.randint(1, 28))
        end = start + timedelta(days=np.random.randint(1, 15))

        start, end = clamp(start), clamp(end)
        if overlaps(start, end):
            limited_promos.append({
                "TypeGroup": "Limited",
                "SeasonType": "Limited Time",
                "Year": y,
                "StartDate": pd.Timestamp(start),
                "EndDate": pd.Timestamp(end),
                "DiscountPct": round(np.random.uniform(0.05, 0.35), 2),
                "PromotionType": PROMOTION_TYPES["Limited"],
                "PromotionCategory": np.random.choice(CATEGORIES),
            })

    # ========================================================
    # 5. Combine and number Seasonal/Clearance/Limited
    # ========================================================
    df = pd.DataFrame(
        holiday_promos + seasonal_promos + clearance_promos + limited_promos
    )

    numbered_parts = []
    non_holiday = df[df["TypeGroup"] != "Holiday"]

    for (y, tgroup, stype), group in non_holiday.groupby(["Year", "TypeGroup", "SeasonType"]):
        group = group.sort_values("StartDate").copy()
        group["LocalIndex"] = range(1, len(group) + 1)

        group["PromotionName"] = (
            group["SeasonType"]
            + " "
            + group["Year"].astype(str)
            + " #"
            + group["LocalIndex"].astype(str)
        )

        group["PromotionDescription"] = (
            group["SeasonType"] + " for " + group["Year"].astype(str)
        )

        numbered_parts.append(group)

    # Holidays (no numbering)
    holiday_df = df[df["TypeGroup"] == "Holiday"].copy()
    holiday_df["LocalIndex"] = None

    final = pd.concat([holiday_df] + numbered_parts, ignore_index=True)

    # ========================================================
    # 6. Add NO-DISCOUNT promotion
    # ========================================================
    final = pd.concat(
        [
            final,
            pd.DataFrame(
                [{
                    "TypeGroup": "NoDiscount",
                    "SeasonType": "NoDiscount",
                    "Year": start_year,
                    "PromotionName": "No Discount",
                    "PromotionDescription": "No Discount",
                    "DiscountPct": 0.00,
                    "PromotionType": PROMOTION_TYPES["NoDiscount"],
                    "PromotionCategory": "No Discount",
                    "StartDate": pd.Timestamp(start_year, 1, 1),
                    "EndDate": pd.Timestamp(end_year, 12, 31),
                    "LocalIndex": None,
                }]
            )
        ],
        ignore_index=True
    )

    # ========================================================
    # 7. Final sorting + assign keys
    # ========================================================
    final = final.sort_values("StartDate").reset_index(drop=True)
    final["PromotionKey"] = final.index + 1
    final["PromotionLabel"] = final["PromotionKey"]

    # Final column order (unchanged from original)
    final = final[
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

    return final
