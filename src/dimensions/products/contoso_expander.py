from pathlib import Path
import numpy as np
import pandas as pd


def expand_contoso_products(
    base_products: pd.DataFrame,
    num_products: int,
    seed: int = 42,
    price_jitter_pct: float = 0.0,
) -> pd.DataFrame:
    """
    Expand Contoso products to a larger row count while
    preserving schema, hierarchy, and semantics.

    - Categories & Subcategories are untouched
    - Only Product identity fields are modified
    """

    base_products = base_products.reset_index(drop=True)
    base_count = len(base_products)

    if num_products <= base_count:
        return base_products.copy()

    rng = np.random.default_rng(seed)

    # Repeat base rows
    repeat_factor = int(np.ceil(num_products / base_count))
    expanded = (
        pd.concat([base_products] * repeat_factor, ignore_index=True)
        .iloc[:num_products]
        .copy()
    )

    # Preserve lineage
    expanded["BaseProductKey"] = expanded["ProductKey"]

    # New surrogate ProductKey
    expanded["ProductKey"] = np.arange(
        1, num_products + 1, dtype="int64"
    )

    # Variant index per base product
    expanded["VariantIndex"] = (
        expanded.groupby("BaseProductKey").cumcount()
    )

    # ProductCode (business friendly)
    expanded["ProductCode"] = (
        expanded["ProductKey"].astype(str).str.zfill(7)
    )

    # Name suffix for uniqueness
    expanded["ProductName"] = (
        expanded["ProductName"]
        + " - V"
        + expanded["VariantIndex"].astype(str).str.zfill(3)
    )

    # Optional deterministic price jitter
    if price_jitter_pct > 0:
        price_jitter = rng.uniform(
            1 - price_jitter_pct,
            1 + price_jitter_pct,
            size=len(expanded),
        )

        cost_jitter = rng.uniform(
            1 - price_jitter_pct,
            1.0,
            size=len(expanded),
        )

        expanded["UnitPrice"] = (
            expanded["UnitPrice"] * price_jitter
        ).round(2)

        expanded["UnitCost"] = (
            expanded["UnitCost"] * price_jitter * cost_jitter
        ).round(2)

        # Hard safety
        expanded["UnitCost"] = np.minimum(
            expanded["UnitCost"],
            expanded["UnitPrice"],
        )

    return expanded
