from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.logging_utils import info, skip, stage, warn
from src.versioning.version_store import should_regenerate, save_version


# =============================================================
# CURATED GEOGRAPHY ROWS
# =============================================================
# Each tuple: (City, State, Country, Continent, ISOCode)
#
# ISOCode is an ISO-4217 *currency* code, not an ISO-3166 country
# code.  The name is retained for backward compatibility with
# downstream consumers (stores, employees, sales, schemas).
#
# To add a new market, append a row here AND list its currency in
# exchange_rates.currencies in config.yaml.  If you add a country
# that already appears in geography.country_weights it will be
# picked up automatically; otherwise it falls under the "Rest"
# weight bucket.
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

def _curated_countries() -> set[str]:
    """Return the set of distinct country names present in the curated pool."""
    return {row[2] for row in CURATED_ROWS}


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


def _curated_signature() -> str:
    """Content-aware hash of CURATED_ROWS so any edit triggers regeneration."""
    raw = repr(CURATED_ROWS).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _resolve_seed(cfg: Dict, geo_cfg: Dict) -> int:
    """Resolve seed using the same override chain as other dimensions:
    geography.override.seed → defaults.seed → fallback 42."""
    override_seed = (geo_cfg.get("override") or {}).get("seed")
    if override_seed is not None:
        return int(override_seed)

    default_seed = cfg.get("defaults", {}).get("seed", 42)
    return int(default_seed)


def _normalize_country_weights(country_weights: Dict) -> Dict[str, float]:
    """Return a normalized copy of weights (sums to 1 when total > 0).

    Raises early if every weight (including Rest) is zero or negative,
    rather than letting the error surface later during sampling.
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
    if total <= 0:
        raise ValueError(
            "All geography.country_weights are zero (including 'Rest'). "
            "At least one weight must be positive."
        )

    return {k: (v / total) for k, v in cw.items()}


def _warn_orphaned_weights(
    country_weights: Dict[str, float],
    available_countries: set[str],
) -> None:
    """Log a warning for country_weights keys that have positive weight
    but no matching rows in the curated pool (and aren't 'Rest')."""
    for name, weight in country_weights.items():
        if name == "Rest":
            continue
        if weight > 0 and name not in available_countries:
            warn(
                f"geography.country_weights lists '{name}' with weight "
                f"{weight}, but no curated rows exist for that country. "
                f"The weight will have no effect."
            )


def _row_weights(df: pd.DataFrame, country_weights_norm: Dict[str, float]) -> np.ndarray:
    """Build per-row weights based on Country mapping (vectorized).

    Unmapped countries use 'Rest' if present, else 0.
    """
    if not country_weights_norm:
        return np.ones(len(df), dtype=np.float64)

    rest = float(country_weights_norm.get("Rest", 0.0))
    w = df["Country"].map(country_weights_norm).astype("float64")
    w = w.fillna(rest).to_numpy(dtype=np.float64)

    s = float(w.sum())
    if s <= 0:
        raise ValueError(
            "All country weights resolved to zero after mapping to available "
            "rows. Check geography.country_weights (including 'Rest')."
        )

    return w / s


# =============================================================
# GENERATOR
# =============================================================

def build_dim_geography(cfg: Dict, *, _geo_cfg: Dict | None = None) -> pd.DataFrame:
    """Build curated + weighted geography dimension.

    Filters rows based on allowed currencies from exchange_rates.currencies,
    then samples ``target_rows`` rows with replacement (by default) using
    per-country weights.

    The output intentionally contains duplicate city/country combinations
    distinguished only by GeographyKey.  This models the common star-schema
    pattern where each key represents a distinct "location instance" (e.g.
    a store footprint or delivery zone) rather than a unique city.
    Downstream tables (stores, sales) join on GeographyKey to spread
    transactions across these weighted location slots.
    """
    geo_cfg = _geo_cfg if _geo_cfg is not None else _validate_cfg(cfg)

    allowed_iso = set(map(str, cfg["exchange_rates"]["currencies"]))
    target_rows = int(geo_cfg.get("target_rows", 200))
    if target_rows <= 0:
        raise ValueError(f"geography.target_rows must be > 0, got {target_rows}")

    seed = _resolve_seed(cfg, geo_cfg)

    sampling_cfg = geo_cfg.get("sampling", {}) or {}
    replace = bool(sampling_cfg.get("replace", True))

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

    if not replace and target_rows > len(df):
        replace = True

    raw_weights = geo_cfg.get("country_weights", {}) or {}
    country_weights_norm = _normalize_country_weights(raw_weights)
    _warn_orphaned_weights(raw_weights, set(df["Country"].unique()))
    w = _row_weights(df, country_weights_norm)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=target_rows, replace=replace, p=w)

    out = df.iloc[idx].reset_index(drop=True)
    out.insert(0, "GeographyKey", np.arange(1, target_rows + 1, dtype=np.int64))
    return out


# =============================================================
# CONFIG NORMALIZER  (registered in engine/config/config.py)
# =============================================================

def normalize_geography_config(geo_cfg: Dict) -> Dict:
    """Validate and coerce geography config at load time.

    Registered as a section normalizer so bad config (wrong types,
    missing keys) is caught early rather than at generation time.
    """
    geo_cfg = dict(geo_cfg)

    # target_rows
    raw_rows = geo_cfg.get("target_rows", 200)
    try:
        target_rows = int(raw_rows)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"geography.target_rows must be an integer, got {raw_rows!r}"
        ) from e
    if target_rows <= 0:
        raise ValueError(f"geography.target_rows must be > 0, got {target_rows}")
    geo_cfg["target_rows"] = target_rows

    # country_weights
    cw = geo_cfg.get("country_weights")
    if cw is not None:
        if not isinstance(cw, dict):
            raise TypeError("geography.country_weights must be a mapping")
        coerced: Dict[str, float] = {}
        for k, v in cw.items():
            try:
                coerced[str(k)] = float(v)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Invalid country weight for '{k}': {v!r}"
                ) from e
        geo_cfg["country_weights"] = coerced

    # sampling sub-block
    sampling = geo_cfg.get("sampling")
    if sampling is not None:
        if not isinstance(sampling, dict):
            raise TypeError("geography.sampling must be a mapping")
        if "replace" in sampling:
            sampling["replace"] = bool(sampling["replace"])
        geo_cfg["sampling"] = sampling

    # override sub-block
    override = geo_cfg.get("override") or {}
    if not isinstance(override, dict):
        raise TypeError("geography.override must be a mapping")
    override.setdefault("seed", None)
    override.setdefault("dates", {})
    override.setdefault("paths", {})
    if override["seed"] is not None:
        override["seed"] = int(override["seed"])
    if not isinstance(override["dates"], dict):
        raise TypeError("geography.override.dates must be a mapping")
    if not isinstance(override["paths"], dict):
        raise TypeError("geography.override.paths must be a mapping")
    geo_cfg["override"] = override

    return geo_cfg


# =============================================================
# PIPELINE WRAPPER
# =============================================================

def run_geography(cfg: Dict, parquet_folder: Path) -> None:
    """Pipeline wrapper: version check → build → write parquet → save version."""
    geo_cfg = _validate_cfg(cfg)
    out_path = parquet_folder / "geography.parquet"

    force = bool(geo_cfg.get("_force_regenerate", False))

    version_cfg = {
        **geo_cfg,
        "exchange_rates": {"currencies": list(map(str, cfg["exchange_rates"]["currencies"]))},
        "_curated_sig": _curated_signature(),
    }

    if not force and not should_regenerate("geography", version_cfg, out_path):
        skip("Geography up-to-date; skipping.")
        return

    with stage("Generating Geography"):
        df = build_dim_geography(cfg, _geo_cfg=geo_cfg)
        df = df[OUTPUT_COLS]
        df.to_parquet(out_path, index=False)

    save_version("geography", version_cfg, out_path)
    info(f"Geography dimension written: {out_path}")
