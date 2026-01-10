from datetime import date
import re
from collections import defaultdict

import hashlib
import random


_SALES_RE = re.compile(r"Sales\s+([\d\.]+[KMB]?)", re.IGNORECASE)

START = date(2021, 1, 1)
END = date(2025, 12, 31)

def _stable_rng(key: str) -> random.Random:
    """
    Create a deterministic RNG based on preset name.
    """
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    seed = int(h[:16], 16)
    return random.Random(seed)


def _jitter(value: int, pct: float, rng: random.Random) -> int:
    """
    Apply ±pct jitter to a value.
    Example: pct=0.03 → ±3%
    """
    delta = value * pct
    return max(1, int(value + rng.uniform(-delta, delta)))


# ------------------------------------------------------------------
# Presets are explicit and authoritative.
# If a preset name mentions Customers X, Products Y, or Sales Z,
# the preset MUST actually apply those values.
# ------------------------------------------------------------------

PRESETS = {
    # --------------------------------------------------------------
    # 100K Sales
    # --------------------------------------------------------------
    "Customers 5K | Products 2.2K | Sales 100K": {
        "start": START, "end": END,
        "sales_rows": 100_000,
        "customers": 5_000,
        "products": 2_157,
    },
    "Customers 10K | Products 2.2K | Sales 100K": {
        "start": START, "end": END,
        "sales_rows": 100_000,
        "customers": 10_000,
        "products": 2_157,
    },
    "Customers 10K | Products 5K | Sales 100K": {
        "start": START, "end": END,
        "sales_rows": 100_000,
        "customers": 10_000,
        "products": 5_000,
    },

    # --------------------------------------------------------------
    # 1M Sales
    # --------------------------------------------------------------
    "Customers 20K | Products 3K | Sales 1M": {
        "start": START, "end": END,
        "sales_rows": 1_000_000,
        "customers": 20_000,
        "products": 3_000,
    },
    "Customers 50K | Products 5K | Sales 1M": {
        "start": START, "end": END,
        "sales_rows": 1_000_000,
        "customers": 50_000,
        "products": 5_000,
    },
    "Customers 50K | Products 7K | Sales 1M": {
        "start": START, "end": END,
        "sales_rows": 1_000_000,
        "customers": 50_000,
        "products": 7_000,
    },

    # --------------------------------------------------------------
    # 2M Sales
    # --------------------------------------------------------------
    "Customers 50K | Products 5K | Sales 2M": {
        "start": START, "end": END,
        "sales_rows": 2_000_000,
        "customers": 50_000,
        "products": 5_000,
    },
    "Customers 100K | Products 7K | Sales 2M": {
        "start": START, "end": END,
        "sales_rows": 2_000_000,
        "customers": 100_000,
        "products": 7_000,
    },
    "Customers 100K | Products 10K | Sales 2M": {
        "start": START, "end": END,
        "sales_rows": 2_000_000,
        "customers": 100_000,
        "products": 10_000,
    },

    # --------------------------------------------------------------
    # 5M Sales
    # --------------------------------------------------------------
    "Customers 100K | Products 8K | Sales 5M": {
        "start": START, "end": END,
        "sales_rows": 5_000_000,
        "customers": 100_000,
        "products": 8_000,
    },
    "Customers 250K | Products 12K | Sales 5M": {
        "start": START, "end": END,
        "sales_rows": 5_000_000,
        "customers": 250_000,
        "products": 12_000,
    },
    "Customers 250K | Products 15K | Sales 5M": {
        "start": START, "end": END,
        "sales_rows": 5_000_000,
        "customers": 250_000,
        "products": 15_000,
    },

    # --------------------------------------------------------------
    # 10M Sales
    # --------------------------------------------------------------
    "Customers 250K | Products 10K | Sales 10M": {
        "start": START, "end": END,
        "sales_rows": 10_000_000,
        "customers": 250_000,
        "products": 10_000,
    },
    "Customers 500K | Products 15K | Sales 10M": {
        "start": START, "end": END,
        "sales_rows": 10_000_000,
        "customers": 500_000,
        "products": 15_000,
    },
    "Customers 500K | Products 25K | Sales 10M": {
        "start": START, "end": END,
        "sales_rows": 10_000_000,
        "customers": 500_000,
        "products": 25_000,
    },

    # --------------------------------------------------------------
    # 20M Sales
    # --------------------------------------------------------------
    "Customers 500K | Products 15K | Sales 20M": {
        "start": START, "end": END,
        "sales_rows": 20_000_000,
        "customers": 500_000,
        "products": 15_000,
    },
    "Customers 1M | Products 25K | Sales 20M": {
        "start": START, "end": END,
        "sales_rows": 20_000_000,
        "customers": 1_000_000,
        "products": 25_000,
    },
    "Customers 1M | Products 40K | Sales 20M": {
        "start": START, "end": END,
        "sales_rows": 20_000_000,
        "customers": 1_000_000,
        "products": 40_000,
    },

    # --------------------------------------------------------------
    # 50M Sales
    # --------------------------------------------------------------
    "Customers 1M | Products 25K | Sales 50M": {
        "start": START, "end": END,
        "sales_rows": 50_000_000,
        "customers": 1_000_000,
        "products": 25_000,
    },
    "Customers 2M | Products 50K | Sales 50M": {
        "start": START, "end": END,
        "sales_rows": 50_000_000,
        "customers": 2_000_000,
        "products": 50_000,
    },
    "Customers 2M | Products 75K | Sales 50M": {
        "start": START, "end": END,
        "sales_rows": 50_000_000,
        "customers": 2_000_000,
        "products": 75_000,
    },

    # --------------------------------------------------------------
    # 100M Sales
    # --------------------------------------------------------------
    "Customers 2M | Products 40K | Sales 100M": {
        "start": START, "end": END,
        "sales_rows": 100_000_000,
        "customers": 2_000_000,
        "products": 40_000,
    },
    "Customers 5M | Products 75K | Sales 100M": {
        "start": START, "end": END,
        "sales_rows": 100_000_000,
        "customers": 5_000_000,
        "products": 75_000,
    },
    "Customers 5M | Products 100K | Sales 100M": {
        "start": START, "end": END,
        "sales_rows": 100_000_000,
        "customers": 5_000_000,
        "products": 100_000,
    },

    # --------------------------------------------------------------
    # 150M Sales
    # --------------------------------------------------------------
    "Customers 5M | Products 75K | Sales 150M": {
        "start": START, "end": END,
        "sales_rows": 150_000_000,
        "customers": 5_000_000,
        "products": 75_000,
    },
    "Customers 8M | Products 100K | Sales 150M": {
        "start": START, "end": END,
        "sales_rows": 150_000_000,
        "customers": 8_000_000,
        "products": 100_000,
    },
    "Customers 8M | Products 125K | Sales 150M": {
        "start": START, "end": END,
        "sales_rows": 150_000_000,
        "customers": 8_000_000,
        "products": 125_000,
    },

    # --------------------------------------------------------------
    # 200M Sales
    # --------------------------------------------------------------
    "Customers 8M | Products 100K | Sales 200M": {
        "start": START, "end": END,
        "sales_rows": 200_000_000,
        "customers": 8_000_000,
        "products": 100_000,
    },
    "Customers 10M | Products 125K | Sales 200M": {
        "start": START, "end": END,
        "sales_rows": 200_000_000,
        "customers": 10_000_000,
        "products": 125_000,
    },
    "Customers 10M | Products 150K | Sales 200M": {
        "start": START, "end": END,
        "sales_rows": 200_000_000,
        "customers": 10_000_000,
        "products": 150_000,
    },

    # --------------------------------------------------------------
    # 500M Sales
    # --------------------------------------------------------------
    "Customers 15M | Products 150K | Sales 500M": {
        "start": START, "end": END,
        "sales_rows": 500_000_000,
        "customers": 15_000_000,
        "products": 150_000,
    },
    "Customers 20M | Products 200K | Sales 500M": {
        "start": START, "end": END,
        "sales_rows": 500_000_000,
        "customers": 20_000_000,
        "products": 200_000,
    },
    "Customers 25M | Products 250K | Sales 500M": {
        "start": START, "end": END,
        "sales_rows": 500_000_000,
        "customers": 25_000_000,
        "products": 250_000,
    },
}


def apply_preset(cfg, base_loader, preset_name: str):
    preset = PRESETS[preset_name]

    cfg.clear()
    cfg.update(base_loader())

    rng = _stable_rng(preset_name)

    # Dates stay exact
    cfg["defaults"]["dates"]["start"] = preset["start"]
    cfg["defaults"]["dates"]["end"] = preset["end"]

    # Apply jittered sales rows (±2%)
    cfg["sales"]["total_rows"] = _jitter(
        preset["sales_rows"],
        pct=0.02,
        rng=rng,
    )

    # Apply jittered customers (±3%)
    cfg["customers"]["total_customers"] = _jitter(
        preset["customers"],
        pct=0.03,
        rng=rng,
    )

    # Apply jittered products (±4%), if present
    if "products" in preset:
        cfg["products"]["num_products"] = _jitter(
            preset["products"],
            pct=0.04,
            rng=rng,
        )


def _extract_sales_bucket(name: str) -> str:
    match = _SALES_RE.search(name)
    if not match:
        return "Other"
    return f"{match.group(1)} Sales"


def build_presets_by_sales():
    grouped = defaultdict(dict)

    for preset_name in PRESETS:
        bucket = _extract_sales_bucket(preset_name)
        grouped[bucket][preset_name] = preset_name

    def _sales_key(label):
        num = label.split()[0]
        return float(num[:-1]) * {"K": 1e3, "M": 1e6, "B": 1e9}.get(num[-1], 1)

    return dict(sorted(grouped.items(), key=lambda x: _sales_key(x[0])))
