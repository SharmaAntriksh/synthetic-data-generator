"""Product dimension multiprocessing workers.

Top-level functions for Windows spawn pickling.  Each worker enriches
a chunk of products, skipping rank-dependent columns (BrandTier and
dependents) which are computed after merge in the main process.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd


def product_enrich_chunk_worker(args: Tuple) -> Dict[str, Any]:
    """Enrich a chunk of products with hash-seeded attributes.

    Reads a pre-split chunk parquet, calls ``enrich_products_attributes``
    with ``_skip_post_merge=True``, writes the enriched result back.
    Returns minimal metadata dict.
    """
    (
        chunk_idx, seed,
        input_path_str, output_path_str,
        cfg_dump, output_folder_str,
    ) = args

    from src.engine.config.config_schema import AppConfig

    cfg = AppConfig.from_raw_dict(dict(cfg_dump))
    output_folder = Path(output_folder_str)

    chunk_df = pd.read_parquet(input_path_str)

    from src.dimensions.products.product_profile import enrich_products_attributes

    enriched = enrich_products_attributes(
        chunk_df, cfg, seed=seed, output_folder=output_folder,
        _skip_post_merge=True,
    )

    enriched.to_parquet(output_path_str, index=False)

    return {
        "chunk_idx": chunk_idx,
        "n_rows": len(enriched),
    }
