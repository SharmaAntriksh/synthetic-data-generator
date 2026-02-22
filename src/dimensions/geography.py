from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from src.utils.logging_utils import info, skip, stage
from src.versioning.version_store import should_regenerate, save_version


# =============================================================
# CURATED GEOGRAPHY ROWS  (unchanged)
# =============================================================

CURATED_ROWS: List[Tuple[str, str, str, str, str]] = [
    ("New York", "NY", "United States", "North America", "USD"),
    ("Los Angeles", "CA", "United States", "North America", "USD"),
    ("Chicago", "IL", "United States", "North America", "USD"),
    ("Houston", "TX", "United States", "North America", "USD"),
    ("Miami", "FL", "United States", "North America", "USD"),

    ("Toronto", "ON", "Canada", "North America", "CAD"),
    ("Vancouver", "BC", "Canada", "North America", "CAD"),
    ("Montreal", "QC", "Canada", "North America", "CAD"),

    ("London", "London", "United Kingdom", "Europe", "GBP"),
    ("Manchester", "Manchester", "United Kingdom", "Europe", "GBP"),

    ("Berlin", "Berlin", "Germany", "Europe", "EUR"),
    ("Munich", "Bavaria", "Germany", "Europe", "EUR"),

    ("Paris", "Île-de-France", "France", "Europe", "EUR"),
    ("Lyon", "Auvergne-Rhône-Alpes", "France", "Europe", "EUR"),

    ("Mumbai", "MH", "India", "Asia", "INR"),
    ("Delhi", "DL", "India", "Asia", "INR"),
    ("Bengaluru", "KA", "India", "Asia", "INR"),

    ("Sydney", "NSW", "Australia", "Oceania", "AUD"),
    ("Melbourne", "VIC", "Australia", "Oceania", "AUD"),
]


OUTPUT_COLS = ["GeographyKey", "City", "State", "Country", "Continent", "ISOCode"]


# =============================================================
# INTERNALS
# =============================================================

def _validate_cfg(cfg: Dict) -> Dict:
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dict")

    if "geography" not in cfg or not isinstance(cfg["geography"], dict):
        raise KeyError("Missing required config section: 'geography'")

    if "exchange_rates" not in cfg or not isinstance(cfg["exchange_rates"], dict):
        raise KeyError("Missing required config section: 'exchange_rates'")

    currencies = cfg["exchange_rates"].get("currencies")
    if not isinstance(currencies, (list, tuple)) or not currencies:
        raise ValueError("exchange_rates.currencies must be a non-empty list of currency ISO codes")

    return cfg["geography"]


def _curated_signature() -> Dict:
    """
    Small signature to ensure changes to CURATED_ROWS trigger regeneration,
    without storing the whole list in version cfg.
    """
    # stable-ish signature: count + a lightweight checksum-like value
    # (sum of string lengths is cheap and deterministic)
    total_len = sum(len(x) for row in CURATED_ROWS for x in row)
    return {"rows": len(CURATED_ROWS), "total_len": total_len}


def _normalize_country_weights(country_weights: Dict) -> Dict[str, float]:
    """
    Returns a normalized copy of weights (sums to 1 when total > 0).
    Allows a "Rest" key as fallback.
    """
    if not country_weights:
        return {}

    if not isinstance(country_weights, dict):
        raise TypeError("geography.country_weights must be a dict")

    cw: Dict[str, float] = {}
    for k, v in country_weights.items():
        try:
            fv = float(v)
        except Exception as e:
            raise ValueError(f"Invalid country weight for '{k}': {v!r}") from e
        if fv < 0:
            raise ValueError(f"Country weight cannot be negative: {k}={fv}")
        cw[str(k)] = fv

    total = float(sum(cw.values()))
    if total > 0:
        cw = {k: (v / total) for k, v in cw.items()}  # normalize to sum=1
    return cw


def _row_weights(df: pd.DataFrame, country_weights_norm: Dict[str, float]) -> np.ndarray:
    """
    Build per-row weights based on Country mapping (vectorized).
    Unmapped countries use 'Rest' if present, else 0.
    """
    if not country_weights_norm:
        # If no weights provided, uniform distribution
        return np.ones(len(df), dtype=np.float64)

    rest = float(country_weights_norm.get("Rest", 0.0))
    w = df["Country"].map(country_weights_norm).astype("float64")
    w = w.fillna(rest).to_numpy(dtype=np.float64)

    s = float(w.sum())
    if s <= 0:
        raise ValueError("All country weights resolved to zero. Check geography.country_weights (including 'Rest').")

    return w / s


# =============================================================
# GENERATOR
# =============================================================

def build_dim_geography(cfg: Dict) -> pd.DataFrame:
    """
    Build curated + weighted geography dimension.
    Filters rows based on allowed currencies from exchange_rates.currencies.
    """
    geo_cfg = _validate_cfg(cfg)

    allowed_iso = set(map(str, cfg["exchange_rates"]["currencies"]))
    target_rows = int(geo_cfg.get("target_rows", 200))
    if target_rows <= 0:
        raise ValueError(f"geography.target_rows must be > 0, got {target_rows}")

    seed = int(geo_cfg.get("seed", 42))

    sampling_cfg = geo_cfg.get("sampling", {}) or {}
    replace_cfg = sampling_cfg.get("replace", True)
    replace = bool(replace_cfg)

    # Base curated DF
    df = pd.DataFrame(
        CURATED_ROWS,
        columns=["City", "State", "Country", "Continent", "ISOCode"],
    )

    # Filter by allowed currency codes
    df = df[df["ISOCode"].isin(allowed_iso)].reset_index(drop=True)
    if df.empty:
        raise ValueError(
            f"No geography rows remain after filtering by allowed currencies: {sorted(allowed_iso)}"
        )

    # If replace is disabled but we need more rows than available, force replace
    if not replace and target_rows > len(df):
        replace = True

    country_weights_norm = _normalize_country_weights(geo_cfg.get("country_weights", {}) or {})
    w = _row_weights(df, country_weights_norm)

    # Weighted sampling via numpy (fast + deterministic)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=target_rows, replace=replace, p=w)

    out = df.iloc[idx].reset_index(drop=True)
    out.insert(0, "GeographyKey", np.arange(1, target_rows + 1, dtype=np.int64))
    return out


# =============================================================
# PIPELINE WRAPPER
# =============================================================

def run_geography(cfg: Dict, parquet_folder: Path) -> None:
    """
    Pipeline wrapper for geography dimension.
    Handles:
    - version check
    - logging
    - writing parquet
    - saving version
    """
    geo_cfg = _validate_cfg(cfg)
    out_path = parquet_folder / "geography.parquet"

    force = bool(geo_cfg.get("_force_regenerate", False))

    # IMPORTANT: include exchange_rates currencies + curated signature in version cfg
    # so changing currencies or curated list triggers regeneration.
    version_cfg = {
        **geo_cfg,
        "exchange_rates": {"currencies": list(map(str, cfg["exchange_rates"]["currencies"]))},
        "_curated_sig": _curated_signature(),
    }

    if not force and not should_regenerate("geography", version_cfg, out_path):
        skip("Geography up-to-date; skipping.")
        return

    with stage("Generating Geography"):
        df = build_dim_geography(cfg)
        df = df[OUTPUT_COLS]  # ensure stable output schema/order
        df.to_parquet(out_path, index=False)

    save_version("geography", version_cfg, out_path)
    info(f"Geography dimension written: {out_path}")
