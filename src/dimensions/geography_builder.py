import os
import pandas as pd

# ============================================================
# COUNTRY → CURRENCY ISO CODE MAPPING
# ============================================================

COUNTRY_TO_ISO = {
    # Core markets
    "United States": "USD",
    "USA": "USD",
    "US": "USD",

    "India": "INR",

    "United Kingdom": "GBP",
    "UK": "GBP",

    "Australia": "AUD",
    "Canada": "CAD",

    # Common EU / European (all EUR)
    "Germany": "EUR",
    "France": "EUR",
    "Spain": "EUR",
    "Italy": "EUR",
    "Netherlands": "EUR",
    "Belgium": "EUR",
    "Portugal": "EUR",
    "Austria": "EUR",
    "Ireland": "EUR",
    "Finland": "EUR",
    "Greece": "EUR",
    "Luxembourg": "EUR",

    # Other currencies (will be filtered unless added to ALLOWED_ISO)
    "Switzerland": "CHF",
    "Sweden": "SEK",
    "Norway": "NOK",
    "Denmark": "DKK",
    "Japan": "JPY",
    "China": "CNY",
    "New Zealand": "NZD",
    "Singapore": "SGD",
    "South Africa": "ZAR",
}

DEFAULT_ISO = "USD"

# ============================================================
# Allowed Currencies (Option B)
# Only rows with these ISO codes will be kept.
# ============================================================
ALLOWED_ISO = {"USD", "EUR", "INR", "GBP", "AUD", "CAD"}


# ============================================================
# BUILD DIM-GEOGRAPHY
# ============================================================

def build_dim_geography(
    source_path: str,
    output_path: str,
    max_rows: int = None
):
    """
    Reads a raw geography parquet, enriches with currency ISOCode,
    filters unsupported ISO codes (Option B),
    rebuilds GeographyKey, and writes final parquet.
    """

    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Source geography not found: {source_path}")

    print(f"Loading geography source: {source_path}")
    df = pd.read_parquet(source_path)

    # Optional row-limiting
    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows)

    # Required columns check
    for col in ["City", "Country", "Continent"]:
        if col not in df.columns:
            raise ValueError(f"Source geography must contain column '{col}'")

    # ============================================================
    # Add ISOCode
    # ============================================================
    df["ISOCode"] = df["Country"].map(COUNTRY_TO_ISO).fillna(DEFAULT_ISO)

    # ============================================================
    # Filter to allowed ISO codes (Option B)
    # ============================================================
    df = df[df["ISOCode"].isin(ALLOWED_ISO)].reset_index(drop=True)

    # ============================================================
    # Rebuild GeographyKey ALWAYS
    # Ensures consistent indexing and removes old/outdated keys
    # ============================================================
    df["GeographyKey"] = df.index + 1

    # ============================================================
    # Save final parquet
    # ============================================================
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Writing dim geography: {output_path}")
    df.to_parquet(output_path, index=False)

    print("DimGeography generated successfully.")
    return df


# ============================================================
# Direct CLI execution
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build DimGeography table")
    parser.add_argument("--src", required=True, help="Source geography parquet path")
    parser.add_argument("--out", required=True, help="Output DimGeography parquet path")
    parser.add_argument("--max", type=int, default=None, help="Max geography rows")

    args = parser.parse_args()

    build_dim_geography(args.src, args.out, args.max)
