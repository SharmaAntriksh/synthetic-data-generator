import numpy as np
import pandas as pd
from pathlib import Path

from src.exceptions import DimensionError
from src.utils.logging_utils import info, warn
from .contoso_expander import _stratified_trim_indices

_SOURCE_DIR = Path("data/contoso_products")

# House brand trim parameters (only applied in "all" mode to prevent
# the Contoso house brand from dominating — 710 products vs Fabrikam's 267)
_MAX_HOUSE_BRAND = 490
_HOUSE_BRAND_NAME = "Contoso"


def _normalize_products(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names, dtypes, and prices for a products DataFrame."""
    rename_map = {}
    if "ProductSubcategoryKey" in df.columns:
        rename_map["ProductSubcategoryKey"] = "SubcategoryKey"
    if "ProductCategoryKey" in df.columns:
        rename_map["ProductCategoryKey"] = "CategoryKey"
    if "UnitPrice" in df.columns:
        rename_map["UnitPrice"] = "ListPrice"

    if rename_map:
        df = df.rename(columns=rename_map)

    required = ["ProductKey", "SubcategoryKey", "ListPrice", "UnitCost"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DimensionError(f"Products file missing required fields: {missing}")

    df["ProductKey"] = pd.to_numeric(df["ProductKey"], errors="raise").astype("int64")
    df["SubcategoryKey"] = pd.to_numeric(df["SubcategoryKey"], errors="raise").astype("int64")

    if "CategoryKey" in df.columns:
        df["CategoryKey"] = pd.to_numeric(df["CategoryKey"], errors="coerce").astype("Int64")

    df["ListPrice"] = pd.to_numeric(df["ListPrice"], errors="coerce").astype("float64")
    df["UnitCost"] = pd.to_numeric(df["UnitCost"], errors="coerce").astype("float64")

    nan_lp = df["ListPrice"].isna().sum()
    nan_uc = df["UnitCost"].isna().sum()
    if nan_lp or nan_uc:
        warn(
            f"Products: coerced {nan_lp} ListPrice and {nan_uc} UnitCost "
            f"values to NaN; these will be clipped to 0.0"
        )

    df["ListPrice"] = df["ListPrice"].fillna(0.0).clip(lower=0.0)
    df["UnitCost"] = df["UnitCost"].fillna(0.0).clip(lower=0.0)
    df["UnitCost"] = df[["UnitCost", "ListPrice"]].min(axis=1)

    return df


def _trim_house_brand(df: pd.DataFrame) -> pd.DataFrame:
    """Stratified trim of the house brand to prevent single-brand dominance."""
    if "Brand" not in df.columns:
        return df
    mask = df["Brand"] == _HOUSE_BRAND_NAME
    n_house = int(mask.sum())
    if n_house <= _MAX_HOUSE_BRAND:
        return df

    house_df = df[mask].reset_index(drop=False)
    subcat_vals = house_df["SubcategoryKey"].to_numpy()
    sel = _stratified_trim_indices(subcat_vals, _MAX_HOUSE_BRAND, seed=42)
    keep_orig_idx = house_df.iloc[sel]["index"].to_numpy()

    drop_idx = df.index[mask].difference(keep_orig_idx)
    info(f"Trimmed {_HOUSE_BRAND_NAME} house brand: {n_house} -> {n_house - len(drop_idx)}")
    return df.drop(drop_idx).reset_index(drop=True)


def load_contoso_products(output_folder: Path, catalog: str = "all"):
    """
    Load base product catalog based on catalog selection.

    Args:
      output_folder: kept for backward compatibility with callers; not used.
      catalog: "contoso" (original 2517), "synthetic" (retail + electronics),
               or "all" (combined, with house brand trim).

    Returns:
      DataFrame with normalized keys and columns preserved.
    """
    _ = output_folder

    catalog = str(catalog).strip().lower()

    _combined = _SOURCE_DIR / "products.parquet"
    _source_files = {
        "contoso": ("contoso_products.parquet", "Contoso"),
        "synthetic": ("synthetic_products.parquet", "Synthetic"),
    }

    if catalog in _source_files:
        filename, source_label = _source_files[catalog]
        path = _SOURCE_DIR / filename
        if path.exists():
            df = pd.read_parquet(path)
        else:
            df = pd.read_parquet(_combined)
            df = df[df["Source"] == source_label].copy()
    else:  # "all"
        df = pd.read_parquet(_combined)

    df = _normalize_products(df)

    # House brand trim only when combining both sources
    if catalog == "all":
        df = _trim_house_brand(df)

    return df
