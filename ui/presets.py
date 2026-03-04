# ui/presets.py
import re
import hashlib
import random
from collections import defaultdict
from typing import Dict, Any


_SALES_RE = re.compile(r"Sales\s+([\d\.]+[KMB]?)", re.IGNORECASE)

START = "2021-01-01"
END = "2025-12-31"


def _stable_rng(key: str) -> random.Random:
    h = hashlib.sha256(key.encode("utf-8")).hexdigest()
    seed = int(h[:16], 16)
    return random.Random(seed)


def _jitter(value: int, pct: float, rng: random.Random) -> int:
    delta = value * pct
    return max(1, int(value + rng.uniform(-delta, delta)))


# ------------------------------------------------------------------
# Compact preset definitions.
#
# Each tuple is (customers, products, sales_rows).
# The label is derived automatically.  All presets share the same
# date window (START / END) and the same jitter behaviour.
# ------------------------------------------------------------------

def _K(n: int) -> str:
    if n >= 1_000_000:
        v = n / 1_000_000
        return f"{v:g}M"
    if n >= 1_000:
        v = n / 1_000
        return f"{v:g}K"
    return str(n)


def _build_label(cust: int, prod: int, sales: int) -> str:
    return f"Customers {_K(cust)} | Products {_K(prod)} | Sales {_K(sales)}"


# (customers, products, sales_rows)
_PRESET_TABLE: list[tuple[int, int, int]] = [
    # 100K Sales
    (5_000,       2_157,    100_000),
    (10_000,      2_157,    100_000),
    (10_000,      5_000,    100_000),
    # 1M Sales
    (20_000,      3_000,  1_000_000),
    (50_000,      5_000,  1_000_000),
    (50_000,      7_000,  1_000_000),
    # 2M Sales
    (50_000,      5_000,  2_000_000),
    (100_000,     7_000,  2_000_000),
    (100_000,    10_000,  2_000_000),
    # 5M Sales
    (100_000,     8_000,  5_000_000),
    (250_000,    12_000,  5_000_000),
    (250_000,    15_000,  5_000_000),
    # 10M Sales
    (250_000,    10_000, 10_000_000),
    (500_000,    15_000, 10_000_000),
    (500_000,    25_000, 10_000_000),
    # 20M Sales
    (500_000,    15_000, 20_000_000),
    (1_000_000,  25_000, 20_000_000),
    (1_000_000,  40_000, 20_000_000),
    # 50M Sales
    (1_000_000,  25_000, 50_000_000),
    (2_000_000,  50_000, 50_000_000),
    (2_000_000,  75_000, 50_000_000),
    # 100M Sales
    (2_000_000,  40_000, 100_000_000),
    (5_000_000,  75_000, 100_000_000),
    (5_000_000, 100_000, 100_000_000),
    # 150M Sales
    (5_000_000,  75_000, 150_000_000),
    (8_000_000, 100_000, 150_000_000),
    (8_000_000, 125_000, 150_000_000),
    # 200M Sales
    (8_000_000,  100_000, 200_000_000),
    (10_000_000, 125_000, 200_000_000),
    (10_000_000, 150_000, 200_000_000),
    # 500M Sales
    (15_000_000, 150_000, 500_000_000),
    (20_000_000, 200_000, 500_000_000),
    (25_000_000, 250_000, 500_000_000),
]


PRESETS: Dict[str, Dict[str, Any]] = {
    _build_label(c, p, s): {
        "start": START,
        "end": END,
        "sales_rows": s,
        "customers": c,
        "products": p,
    }
    for c, p, s in _PRESET_TABLE
}


def apply_preset(cfg, base_loader, preset_name: str) -> None:
    preset = PRESETS[preset_name]

    cfg.clear()
    cfg.update(base_loader())

    cfg.setdefault("defaults", {}).setdefault("dates", {})
    cfg.setdefault("sales", {})
    cfg.setdefault("customers", {})
    cfg.setdefault("products", {})

    rng = _stable_rng(preset_name)

    cfg["defaults"]["dates"]["start"] = preset["start"]
    cfg["defaults"]["dates"]["end"] = preset["end"]

    cfg["sales"]["total_rows"] = _jitter(preset["sales_rows"], pct=0.02, rng=rng)
    cfg["customers"]["total_customers"] = _jitter(preset["customers"], pct=0.03, rng=rng)

    if "products" in preset:
        cfg["products"]["num_products"] = _jitter(preset["products"], pct=0.04, rng=rng)


def _extract_sales_bucket(name: str) -> str:
    match = _SALES_RE.search(name)
    if not match:
        return "Other"
    return f"{match.group(1)} Sales"


def _sales_key(label: str) -> float:
    if label == "Other":
        return float("inf")

    token = label.split()[0]
    if not token:
        return float("inf")

    suffix = token[-1]
    number_part = token[:-1]

    try:
        n = float(number_part) if number_part else float(token)
    except ValueError:
        return float("inf")

    mult = {"K": 1e3, "M": 1e6, "B": 1e9}.get(suffix, 1.0)
    return n * mult


def build_presets_by_sales():
    grouped = defaultdict(dict)

    for preset_name in PRESETS:
        bucket = _extract_sales_bucket(preset_name)
        grouped[bucket][preset_name] = preset_name

    return dict(sorted(grouped.items(), key=lambda x: _sales_key(x[0])))
