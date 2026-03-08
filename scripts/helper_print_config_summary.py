"""Print a JSON config summary for run_generator.ps1.

Usage:  python _print_config_summary.py config.yaml
Exit 2 if PyYAML is not installed (caller silently skips).
"""

import json
import sys

try:
    import yaml
except Exception:
    sys.exit(2)

if len(sys.argv) < 2:
    print("Usage: python helper_print_config_summary.py <config.yaml>", file=sys.stderr)
    sys.exit(1)

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f) or {}


def g(dotpath, default=None):
    cur = cfg
    for part in dotpath.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur if cur is not None else default


defaults_start = g("defaults.dates.start")
defaults_end = g("defaults.dates.end")
sales_start_ov = g("sales.override.dates.start")
sales_end_ov = g("sales.override.dates.end")

eff_start = sales_start_ov or defaults_start
eff_end = sales_end_ov or defaults_end

p_seasonal = g("promotions.num_seasonal", 0) or 0
p_clearance = g("promotions.num_clearance", 0) or 0
p_limited = g("promotions.num_limited", 0) or 0

summary = {
    "dates": {"start": eff_start, "end": eff_end},
    "sales": {
        "file_format": g("sales.file_format"),
        "total_rows": g("sales.total_rows"),
        "chunk_size": g("sales.chunk_size"),
        "row_group_size": g("sales.row_group_size"),
        "compression": g("sales.compression"),
        "out_folder": g("sales.out_folder"),
        "parquet_folder": g("sales.parquet_folder"),
        "delta_output_folder": g("sales.delta_output_folder"),
    },
    "dimensions": {
        "customers": g("customers.total_customers"),
        "stores": g("stores.num_stores"),
        "products": g("products.num_products"),
    },
    "promotions_total": sum(int(v) for v in (p_seasonal, p_clearance, p_limited) if str(v).strip().lstrip('-').isdigit()),
}

print(json.dumps(summary))
