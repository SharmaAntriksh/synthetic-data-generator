from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.utils.logging_utils import info, skip, stage, warn
from src.versioning.version_store import should_regenerate, save_version


# =============================================================
# STATIC GEOGRAPHY DIMENSION
# =============================================================
# Each tuple: (City, State, Country, Continent, ISOCode)
#
# ISOCode is an ISO-4217 *currency* code, not an ISO-3166 country
# code.  The name is retained for backward compatibility with
# downstream consumers (stores, employees, sales, schemas).
#
# To add a new market, append a row here AND list its currency in
# exchange_rates.currencies in config.yaml.
#
# GeographyKey is assigned deterministically as row index (1-based).
# Every row is a unique City+State+Country combination — no
# sampling, no duplicates, no config knobs.
# =============================================================

CURATED_ROWS: List[Tuple[str, str, str, str, str]] = [
    # North America — United States (12)
    ("New York", "New York", "United States", "North America", "USD"),
    ("Los Angeles", "California", "United States", "North America", "USD"),
    ("Chicago", "Illinois", "United States", "North America", "USD"),
    ("Houston", "Texas", "United States", "North America", "USD"),
    ("Miami", "Florida", "United States", "North America", "USD"),
    ("Seattle", "Washington", "United States", "North America", "USD"),
    ("Dallas", "Texas", "United States", "North America", "USD"),
    ("Atlanta", "Georgia", "United States", "North America", "USD"),
    ("Denver", "Colorado", "United States", "North America", "USD"),
    ("Phoenix", "Arizona", "United States", "North America", "USD"),
    ("Boston", "Massachusetts", "United States", "North America", "USD"),
    ("San Francisco", "California", "United States", "North America", "USD"),

    # North America — Canada (5)
    ("Toronto", "Ontario", "Canada", "North America", "CAD"),
    ("Vancouver", "British Columbia", "Canada", "North America", "CAD"),
    ("Montreal", "Quebec", "Canada", "North America", "CAD"),
    ("Calgary", "Alberta", "Canada", "North America", "CAD"),
    ("Ottawa", "Ontario", "Canada", "North America", "CAD"),

    # Europe — United Kingdom (5)
    ("London", "London", "United Kingdom", "Europe", "GBP"),
    ("Manchester", "Manchester", "United Kingdom", "Europe", "GBP"),
    ("Birmingham", "West Midlands", "United Kingdom", "Europe", "GBP"),
    ("Edinburgh", "Scotland", "United Kingdom", "Europe", "GBP"),
    ("Leeds", "West Yorkshire", "United Kingdom", "Europe", "GBP"),

    # Europe — Germany (5)
    ("Berlin", "Berlin", "Germany", "Europe", "EUR"),
    ("Munich", "Bavaria", "Germany", "Europe", "EUR"),
    ("Hamburg", "Hamburg", "Germany", "Europe", "EUR"),
    ("Frankfurt", "Hesse", "Germany", "Europe", "EUR"),
    ("Cologne", "North Rhine-Westphalia", "Germany", "Europe", "EUR"),

    # Europe — France (5)
    ("Paris", "Île-de-France", "France", "Europe", "EUR"),
    ("Lyon", "Auvergne-Rhône-Alpes", "France", "Europe", "EUR"),
    ("Marseille", "Provence-Alpes-Côte d'Azur", "France", "Europe", "EUR"),
    ("Toulouse", "Occitanie", "France", "Europe", "EUR"),
    ("Nice", "Provence-Alpes-Côte d'Azur", "France", "Europe", "EUR"),

    # Europe — Spain (3)
    ("Madrid", "Madrid", "Spain", "Europe", "EUR"),
    ("Barcelona", "Catalonia", "Spain", "Europe", "EUR"),
    ("Valencia", "Valencia", "Spain", "Europe", "EUR"),

    # Europe — Italy (3)
    ("Rome", "Lazio", "Italy", "Europe", "EUR"),
    ("Milan", "Lombardy", "Italy", "Europe", "EUR"),
    ("Naples", "Campania", "Italy", "Europe", "EUR"),

    # Asia — India (6)
    ("Mumbai", "Maharashtra", "India", "Asia", "INR"),
    ("Delhi", "Delhi", "India", "Asia", "INR"),
    ("Bengaluru", "Karnataka", "India", "Asia", "INR"),
    ("Hyderabad", "Telangana", "India", "Asia", "INR"),
    ("Chennai", "Tamil Nadu", "India", "Asia", "INR"),
    ("Pune", "Maharashtra", "India", "Asia", "INR"),

    # Asia — China (5)
    ("Shanghai", "Shanghai", "China", "Asia", "CNY"),
    ("Beijing", "Beijing", "China", "Asia", "CNY"),
    ("Shenzhen", "Guangdong", "China", "Asia", "CNY"),
    ("Guangzhou", "Guangdong", "China", "Asia", "CNY"),
    ("Chengdu", "Sichuan", "China", "Asia", "CNY"),

    # Asia — Japan (3)
    ("Tokyo", "Tokyo", "Japan", "Asia", "JPY"),
    ("Osaka", "Osaka", "Japan", "Asia", "JPY"),
    ("Yokohama", "Kanagawa", "Japan", "Asia", "JPY"),

    # Asia — South Korea (2)
    ("Seoul", "Seoul", "South Korea", "Asia", "KRW"),
    ("Busan", "Busan", "South Korea", "Asia", "KRW"),

    # Asia — Singapore (1)
    ("Singapore", "Singapore", "Singapore", "Asia", "SGD"),

    # Middle East — UAE (2)
    ("Dubai", "Dubai", "UAE", "Middle East", "AED"),
    ("Abu Dhabi", "Abu Dhabi", "UAE", "Middle East", "AED"),

    # Africa — South Africa (3)
    ("Johannesburg", "Gauteng", "South Africa", "Africa", "ZAR"),
    ("Cape Town", "Western Cape", "South Africa", "Africa", "ZAR"),
    ("Durban", "KwaZulu-Natal", "South Africa", "Africa", "ZAR"),

    # Oceania — Australia (5)
    ("Sydney", "New South Wales", "Australia", "Oceania", "AUD"),
    ("Melbourne", "Victoria", "Australia", "Oceania", "AUD"),
    ("Brisbane", "Queensland", "Australia", "Oceania", "AUD"),
    ("Perth", "Western Australia", "Australia", "Oceania", "AUD"),
    ("Adelaide", "South Australia", "Australia", "Oceania", "AUD"),

    # South America — Brazil (3)
    ("São Paulo", "São Paulo", "Brazil", "South America", "BRL"),
    ("Rio de Janeiro", "Rio de Janeiro", "Brazil", "South America", "BRL"),
    ("Brasília", "Federal District", "Brazil", "South America", "BRL"),

    # South America — Mexico (3)
    ("Mexico City", "Mexico City", "Mexico", "South America", "MXN"),
    ("Guadalajara", "Jalisco", "Mexico", "South America", "MXN"),
    ("Monterrey", "Nuevo León", "Mexico", "South America", "MXN"),
]

OUTPUT_COLS = ["GeographyKey", "City", "State", "Country", "Continent", "ISOCode"]


# =============================================================
# INTERNALS
# =============================================================

def _curated_signature() -> str:
    """Content-aware hash of CURATED_ROWS so any edit triggers regeneration."""
    raw = repr(CURATED_ROWS).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


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


# =============================================================
# GENERATOR
# =============================================================

def build_dim_geography(cfg: Dict, *, _geo_cfg: Dict | None = None) -> pd.DataFrame:
    """Build the static geography dimension.

    Emits one row per unique City+State+Country combination from
    CURATED_ROWS, filtered to currencies listed in
    exchange_rates.currencies.  GeographyKey is assigned as a
    sequential 1-based integer.

    No sampling, no weighting, no duplicates.
    """
    _geo_cfg if _geo_cfg is not None else _validate_cfg(cfg)

    allowed_iso = set(map(str, cfg["exchange_rates"]["currencies"]))

    df = pd.DataFrame(
        CURATED_ROWS,
        columns=["City", "State", "Country", "Continent", "ISOCode"],
    )

    df = df[df["ISOCode"].isin(allowed_iso)].reset_index(drop=True)
    if df.empty:
        raise ValueError(
            f"No geography rows remain after filtering by allowed currencies: {sorted(allowed_iso)}. "
            f"Ensure exchange_rates.currencies includes at least one of: "
            f"{sorted({r[4] for r in CURATED_ROWS})}"
        )

    # Warn about currencies configured but not covered by any geography row
    covered_iso = set(df["ISOCode"].unique())
    uncovered = sorted(allowed_iso - covered_iso)
    if uncovered:
        warn(
            f"exchange_rates.currencies includes {uncovered} but no geography "
            f"rows use those currencies. Add cities to CURATED_ROWS or remove "
            f"the currencies from exchange_rates.currencies."
        )

    df.insert(0, "GeographyKey", np.arange(1, len(df) + 1, dtype=np.int32))
    return df


# =============================================================
# CONFIG NORMALIZER  (registered in engine/config/config.py)
# =============================================================

def normalize_geography_config(geo_cfg: Dict) -> Dict:
    """Validate and coerce geography config at load time."""
    geo_cfg = dict(geo_cfg)

    # Warn about legacy keys that no longer apply
    for legacy_key in ("target_rows", "sampling", "country_weights"):
        if legacy_key in geo_cfg:
            warn(
                f"geography.{legacy_key} is no longer used. "
                f"Geography is now a static dimension derived from CURATED_ROWS. "
                f"Remove this key from config.yaml to silence this warning."
            )

    # override sub-block (kept for seed/dates/paths compatibility)
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

    version_cfg = {
        "exchange_rates": {"currencies": list(map(str, cfg["exchange_rates"]["currencies"]))},
        "_curated_sig": _curated_signature(),
    }

    if not should_regenerate("geography", version_cfg, out_path):
        skip("Geography up-to-date")
        return

    with stage("Generating Geography"):
        df = build_dim_geography(cfg, _geo_cfg=geo_cfg)
        df = df[OUTPUT_COLS]
        df.to_parquet(out_path, index=False)

    save_version("geography", version_cfg, out_path)
    info(f"Geography dimension written: {out_path}  ({len(df)} rows)")
