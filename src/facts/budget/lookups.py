"""Budget dimension lookups (worker-side).

Built once per worker process during init_sales_worker.
Stored on State so micro_aggregate() can resolve keys without disk I/O.

Memory footprint: ~few hundred KB (dense arrays sized to max key).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def build_budget_lookups(parquet_dims: Path) -> dict:
    """
    Build dense lookup arrays for budget aggregation.

    Returns dict to be bound onto State:
        budget_store_to_country:  int64[max_store_key+1]  -> country_id
        budget_product_to_cat:    int64[max_product_key+1] -> category_id
        budget_country_labels:    str[num_countries]
        budget_category_labels:   str[num_categories]
        budget_channel_keys:      int16[] (distinct SalesChannelKey values)
    """
    # ---- Store -> Geography -> Country ----
    stores = pd.read_parquet(parquet_dims / "stores.parquet",
                             columns=["StoreKey", "GeographyKey"])
    geo = pd.read_parquet(parquet_dims / "geography.parquet",
                          columns=["GeographyKey", "Country", "ISOCode"])

    store_geo = stores.merge(geo, on="GeographyKey", how="left")

    # Encode countries as dense integer ids
    countries = store_geo["Country"].fillna("Unknown").unique()
    country_to_id = {c: i for i, c in enumerate(countries)}

    max_store = int(store_geo["StoreKey"].max())
    store_to_country = np.full(max_store + 1, -1, dtype=np.int32)
    for _, row in store_geo.iterrows():
        sk = int(row["StoreKey"])
        store_to_country[sk] = country_to_id.get(row["Country"], -1)

    # ---- Product -> Subcategory -> Category ----
    products = pd.read_parquet(parquet_dims / "products.parquet",
                               columns=["ProductKey", "SubcategoryKey"])
    subcat = pd.read_parquet(parquet_dims / "product_subcategory.parquet",
                             columns=["SubcategoryKey", "CategoryKey"])
    cat = pd.read_parquet(parquet_dims / "product_category.parquet",
                          columns=["CategoryKey", "Category"])

    prod_cat = (products
                .merge(subcat, on="SubcategoryKey", how="left")
                .merge(cat, on="CategoryKey", how="left"))

    categories = prod_cat["Category"].fillna("Unknown").unique()
    category_to_id = {c: i for i, c in enumerate(categories)}

    max_prod = int(prod_cat["ProductKey"].max())
    product_to_cat = np.full(max_prod + 1, -1, dtype=np.int32)
    for _, row in prod_cat.iterrows():
        pk = int(row["ProductKey"])
        product_to_cat[pk] = category_to_id.get(row["Category"], -1)

    # ---- Country -> Currency (for FX layer) ----
    country_to_currency = {}
    for _, row in geo.drop_duplicates("Country").iterrows():
        cid = country_to_id.get(row["Country"])
        if cid is not None:
            country_to_currency[cid] = row.get("ISOCode", "USD")

    # ---- Sales channels ----
    sc_path = parquet_dims / "sales_channels.parquet"
    if sc_path.exists():
        sc = pd.read_parquet(sc_path)
        channel_keys = sc["SalesChannelKey"].to_numpy(dtype=np.int16)
        is_digital = sc.get("IsDigital", pd.Series(dtype="int8")).to_numpy()
        is_physical = sc.get("IsPhysical", pd.Series(dtype="int8")).to_numpy()
    else:
        channel_keys = np.arange(1, 6, dtype=np.int16)
        is_digital = np.zeros(5, dtype=np.int8)
        is_physical = np.zeros(5, dtype=np.int8)

    return {
        "budget_store_to_country": store_to_country,
        "budget_product_to_cat": product_to_cat,
        "budget_country_labels": np.array(countries, dtype=object),
        "budget_category_labels": np.array(categories, dtype=object),
        "budget_country_to_currency": country_to_currency,
        "budget_channel_keys": channel_keys,
        "budget_channel_is_digital": is_digital,
        "budget_channel_is_physical": is_physical,
    }


# TODO: vectorize the for-loops above using numpy advanced indexing
# for production — the iterrows() is fine for ~1K stores / ~2K products
# but could be replaced with:
#   store_to_country[store_geo["StoreKey"].to_numpy()] = encoded_country_ids
