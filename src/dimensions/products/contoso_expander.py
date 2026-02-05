import numpy as np
import pandas as pd


def expand_contoso_products(
    base_products: pd.DataFrame,
    num_products: int,
    seed: int = 42,
    price_jitter_pct: float = 0.0,
) -> pd.DataFrame:
    """
    Expand a base Contoso products table to num_products rows while preserving
    schema and hierarchy.

    IMPORTANT:
      - This expander ONLY modifies product identity fields:
          ProductKey, ProductCode, ProductName (suffix), and adds lineage fields.
      - It DOES NOT modify UnitPrice/UnitCost.
        All pricing changes must happen in apply_product_pricing().

    Args:
      base_products: base products dataframe (Contoso)
      num_products: requested output row count
      seed: deterministic expansion ordering
      price_jitter_pct: retained for backward compatibility; ignored

    Returns:
      Expanded dataframe with:
        - BaseProductKey (lineage)
        - VariantIndex (per base product)
        - ProductKey (new surrogate)
        - ProductCode (business-friendly)
        - ProductName suffix for uniqueness (if ProductName exists)
    """
    # Backward compatible arg; deliberately ignored now
    _ = float(price_jitter_pct)

    base_products = base_products.reset_index(drop=True)
    base_count = len(base_products)

    if num_products <= base_count:
        out = base_products.copy()

        # Ensure lineage columns exist
        if "BaseProductKey" not in out.columns:
            out["BaseProductKey"] = out["ProductKey"]
        if "VariantIndex" not in out.columns:
            out["VariantIndex"] = 0
        if "ProductCode" not in out.columns:
            out["ProductCode"] = out["ProductKey"].astype(str).str.zfill(7)

        return out

    rng = np.random.default_rng(seed)

    # Repeat base rows up to requested size
    repeat_factor = int(np.ceil(num_products / base_count))
    expanded = (
        pd.concat([base_products] * repeat_factor, ignore_index=True)
        .iloc[:num_products]
        .copy()
    )

    # Preserve lineage
    expanded["BaseProductKey"] = expanded["ProductKey"]

    # New surrogate key
    expanded["ProductKey"] = np.arange(1, num_products + 1, dtype="int64")

    # Variant index per base product
    expanded["VariantIndex"] = expanded.groupby("BaseProductKey").cumcount().astype("int64")

    # ProductCode (business-friendly)
    expanded["ProductCode"] = expanded["ProductKey"].astype(str).str.zfill(7)

    # Name suffix for uniqueness (only if ProductName exists)
    if "ProductName" in expanded.columns:
        expanded["ProductName"] = (
            expanded["ProductName"].astype(str)
            + " - V"
            + expanded["VariantIndex"].astype(str).str.zfill(3)
        )

    # Optional: light shuffle of rows (identity-only) while keeping determinism
    # Comment out if you want strict original order replication.
    # expanded = expanded.sample(frac=1.0, random_state=int(seed)).reset_index(drop=True)

    return expanded
