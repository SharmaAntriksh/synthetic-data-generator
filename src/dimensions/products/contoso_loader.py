import pandas as pd
from pathlib import Path


def load_contoso_products(output_folder: Path):
    """
    Load Contoso products and normalize only required key columns:
      - ProductSubcategoryKey -> SubcategoryKey
      - ProductCategoryKey    -> CategoryKey (if present)

    IMPORTANT:
      - This function DOES NOT write products.parquet.
      - product_loader.py is the single writer after pricing + active flag are applied.

    Args:
      output_folder: kept for backward compatibility with callers; not used.

    Returns:
      DataFrame with normalized keys and original Contoso columns preserved.
    """
    _ = output_folder  # backward-compatible unused arg

    source_file = Path("data/contoso_products/products.parquet")
    df = pd.read_parquet(source_file)

    # Normalize only required keys (leave everything else intact)
    rename_map = {}
    if "ProductSubcategoryKey" in df.columns:
        rename_map["ProductSubcategoryKey"] = "SubcategoryKey"
    if "ProductCategoryKey" in df.columns:
        rename_map["ProductCategoryKey"] = "CategoryKey"

    if rename_map:
        df = df.rename(columns=rename_map)

    # Validate required columns (SubcategoryKey is required downstream)
    required = ["ProductKey", "SubcategoryKey", "UnitPrice", "UnitCost"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Contoso products file missing required fields: {missing}")

    # Normalize dtypes (be conservative; don't over-normalize other columns)
    df["ProductKey"] = pd.to_numeric(df["ProductKey"], errors="raise").astype("int64")
    df["SubcategoryKey"] = pd.to_numeric(df["SubcategoryKey"], errors="raise").astype("int64")

    # CategoryKey is optional for your downstream Sales requirements, but normalize if present
    if "CategoryKey" in df.columns:
        df["CategoryKey"] = pd.to_numeric(df["CategoryKey"], errors="coerce").astype("Int64")

    df["UnitPrice"] = pd.to_numeric(df["UnitPrice"], errors="coerce").astype("float64")
    df["UnitCost"] = pd.to_numeric(df["UnitCost"], errors="coerce").astype("float64")

    # Basic sanity
    df["UnitPrice"] = df["UnitPrice"].clip(lower=0.0)
    df["UnitCost"] = df["UnitCost"].clip(lower=0.0)
    df["UnitCost"] = df[["UnitCost", "UnitPrice"]].min(axis=1)

    return df
