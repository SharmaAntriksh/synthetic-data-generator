from datetime import date

# ------------------------------------------------------------------
# Presets are explicit and authoritative.
# If a preset name mentions Customers X or Sales Y,
# the preset MUST actually apply those values.
# ------------------------------------------------------------------

PRESETS = {
    # --------------------------------------------------------------
    # Demo / quick validation
    # --------------------------------------------------------------
    "Customers 10K | Sales 100K": {
        "start": date(2023, 1, 1),
        "end": date(2023, 3, 31),
        "sales_rows": 100_000,
        "customers": 10_000,
    },

    "Customers 20K | Sales 1M": {
        "start": date(2023, 1, 1),
        "end": date(2023, 3, 31),
        "sales_rows": 1_000_000,
        "customers": 20_000,
    },

    # --------------------------------------------------------------
    # Training / Power BI
    # --------------------------------------------------------------
    "Customers 100K | Sales 2M": {
        "start": date(2022, 1, 1),
        "end": date(2023, 12, 31),
        "sales_rows": 2_000_000,
        "customers": 100_000,
    },

    "Customers 100K | Sales 10M": {
        "start": date(2022, 1, 1),
        "end": date(2023, 12, 31),
        "sales_rows": 10_000_000,
        "customers": 100_000,
    },

    # --------------------------------------------------------------
    # Analytics workloads
    # --------------------------------------------------------------
    "Customers 500K | Sales 10M": {
        "start": date(2020, 1, 1),
        "end": date(2024, 12, 31),
        "sales_rows": 10_000_000,
        "customers": 500_000,
    },

    "Customers 500K | Sales 20M": {
        "start": date(2020, 1, 1),
        "end": date(2024, 12, 31),
        "sales_rows": 20_000_000,
        "customers": 500_000,
    },

    # --------------------------------------------------------------
    # Large / performance testing
    # --------------------------------------------------------------
    "Customers 1M | Sales 20M": {
        "start": date(2019, 1, 1),
        "end": date(2024, 12, 31),
        "sales_rows": 20_000_000,
        "customers": 1_000_000,
    },

    # --------------------------------------------------------------
    # Extreme / stress testing
    # --------------------------------------------------------------
    "Customers 2M | Sales 50M": {
        "start": date(2018, 1, 1),
        "end": date(2024, 12, 31),
        "sales_rows": 50_000_000,
        "customers": 2_000_000,
    },

    "Customers 10K | Products 500 | Sales 100K": {
        "start": date(2023, 1, 1),
        "end": date(2023, 3, 31),
        "sales_rows": 100_000,
        "customers": 10_000,
        "products": 500,
    },

    "Customers 100K | Products 2.5K | Sales 2M": {
        "start": date(2022, 1, 1),
        "end": date(2023, 12, 31),
        "sales_rows": 2_000_000,
        "customers": 100_000,
        "products": 2_500,
    },

    "Customers 500K | Products 10K | Sales 20M": {
        "start": date(2020, 1, 1),
        "end": date(2024, 12, 31),
        "sales_rows": 20_000_000,
        "customers": 500_000,
        "products": 10_000,
    },
}


def apply_preset(cfg, base_loader, preset_name: str):
    """
    Reset config using a named preset.

    Presets may modify:
    - defaults.dates
    - sales.total_rows
    - customers.total_customers
    - products.num_products (optional)

    Other dimensions (stores, promotions)
    remain user-controllable in the UI.
    """
    preset = PRESETS[preset_name]

    # Reset to base config
    cfg.clear()
    cfg.update(base_loader())

    # Apply global dates
    cfg["defaults"]["dates"]["start"] = preset["start"]
    cfg["defaults"]["dates"]["end"] = preset["end"]

    # Apply sales volume
    cfg["sales"]["total_rows"] = preset["sales_rows"]

    # Apply customer scale
    cfg["customers"]["total_customers"] = preset["customers"]

    # Apply product scale (optional per preset)
    if "products" in preset:
        cfg["products"]["num_products"] = preset["products"]
