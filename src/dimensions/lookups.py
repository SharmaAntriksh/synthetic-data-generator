from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.utils.logging_utils import info, skip, stage
from src.versioning.version_store import should_regenerate, save_version


# =========================================================
# Internals
# =========================================================

def _as_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _int_or(x: Any, default: int) -> int:
    try:
        if x is None or x == "":
            return int(default)
        return int(x)
    except (TypeError, ValueError):
        return int(default)


def _strip_force(cfg_section: Dict[str, Any]) -> Dict[str, Any]:
    """Do not let runtime-only flags contaminate version signatures."""
    out = dict(cfg_section)
    out.pop("_force_regenerate", None)
    return out


def _write_parquet(df: pd.DataFrame, out_path: Path, compression: str = "snappy") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Engine default is fine; project uses parquet widely already.
    df.to_parquet(out_path, index=False, compression=compression)


def _build_version_cfg(cfg: Dict[str, Any], dim_key: str, df: pd.DataFrame, schema_version: int = 1) -> Dict[str, Any]:
    dim_cfg = _as_dict(cfg.get(dim_key))
    base = _strip_force(dim_cfg)

    # Keep version cfg small + stable; include row count and schema version.
    return {
        **base,
        "schema_version": int(schema_version),
        "_rows": int(len(df)),
        "_cols": list(df.columns),
    }


def _maybe_override_rows(dim_cfg: Dict[str, Any], required_cols: Sequence[str]) -> Optional[pd.DataFrame]:
    """
    Optional override:
      <dim_key>:
        rows:
          - {<col>: <val>, ...}
          - ...
    """
    rows = dim_cfg.get("rows")
    if not isinstance(rows, list) or not rows:
        return None
    df = pd.DataFrame(rows)

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Override rows missing required columns: {missing}")
    return df[required_cols].copy()


def _run_lookup_dim(
    *,
    cfg: Dict[str, Any],
    dim_key: str,
    out_name: str,
    build_df: Callable[[Dict[str, Any]], pd.DataFrame],
    parquet_folder: Path,
) -> None:
    dim_cfg = _as_dict(cfg.get(dim_key))
    force = bool(dim_cfg.get("_force_regenerate", False))

    parquet_folder = Path(parquet_folder)
    parquet_folder.mkdir(parents=True, exist_ok=True)
    out_path = parquet_folder / out_name

    compression = str(dim_cfg.get("parquet_compression", "snappy"))

    with stage(f"Generating {dim_key}"):
        df = build_df(dim_cfg)

    version_cfg = _build_version_cfg(cfg, dim_key, df)

    if not force and not should_regenerate(dim_key, version_cfg, out_path):
        skip(f"{dim_key} up-to-date; skipping.")
        return

    _write_parquet(df, out_path, compression=compression)
    save_version(dim_key, version_cfg, out_path)
    info(f"{dim_key} written: {out_path}")


# =========================================================
# Default row sets
# =========================================================

def _df_sales_channels(dim_cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Sales channel lookup.

    Backward-compatible override:
      sales_channels:
        rows:
          - {SalesChannelKey: 1, SalesChannel: Store, ChannelGroup: Physical}
          - ...

    If override rows are provided, only the 3 base columns are required; all new columns are derived/fill-defaulted.
    """

    base_required = ["SalesChannelKey", "SalesChannel", "ChannelGroup"]

    # Expanded schema (stable output columns order)
    cols = base_required + [
        "SalesChannelCode",
        "SortOrder",
        "IsDigital",
        "IsPhysical",
        "IsThirdParty",
        "IsB2B",
        "IsAssisted",
        "IsOwnedChannel",
        "TimeProfile",
        "Is24x7",
        "OpenMinute",
        "CloseMinute",
    ]

    def _normalize_code(name: str) -> str:
        s = str(name).strip().upper()
        s = "".join(ch if ch.isalnum() else "_" for ch in s)
        while "__" in s:
            s = s.replace("__", "_")
        return s.strip("_") or "UNKNOWN"

    def _derive_flags(df: pd.DataFrame) -> pd.DataFrame:
        # Ensure base types
        df["SalesChannelKey"] = df["SalesChannelKey"].astype(np.int16)

        # Defaults
        if "SalesChannelCode" not in df.columns:
            df["SalesChannelCode"] = df["SalesChannel"].map(_normalize_code)

        if "SortOrder" not in df.columns:
            df["SortOrder"] = df["SalesChannelKey"].astype(np.int16)

        # Derive based on ChannelGroup + name heuristics
        grp = df["ChannelGroup"].astype(str).str.strip().str.lower()
        nm = df["SalesChannel"].astype(str).str.strip().str.lower()

        if "IsDigital" not in df.columns:
            df["IsDigital"] = (grp == "digital")
        if "IsPhysical" not in df.columns:
            df["IsPhysical"] = (grp == "physical")
        if "IsB2B" not in df.columns:
            df["IsB2B"] = (grp == "business") | nm.str.contains("b2b", na=False)
        if "IsAssisted" not in df.columns:
            df["IsAssisted"] = (grp == "assisted") | nm.str.contains("call|phone|agent", na=False)

        if "IsThirdParty" not in df.columns:
            df["IsThirdParty"] = nm.str.contains("market|reseller|partner", na=False)

        if "IsOwnedChannel" not in df.columns:
            df["IsOwnedChannel"] = ~df["IsThirdParty"].astype(bool)

        if "TimeProfile" not in df.columns:
            # canonical mapping by group
            mp = {
                "physical": "Retail",
                "digital": "Digital",
                "business": "Business",
                "assisted": "Assisted",
                "na": "NA",
            }
            df["TimeProfile"] = grp.map(mp).fillna("Digital")

        if "Is24x7" not in df.columns:
            # Digital defaults to 24x7; others not
            df["Is24x7"] = df["IsDigital"].astype(bool)

        # Open/Close minutes: -1 means N/A / 24x7 / unknown
        if "OpenMinute" not in df.columns:
            df["OpenMinute"] = np.int16(-1)
        if "CloseMinute" not in df.columns:
            df["CloseMinute"] = np.int16(-1)

        # Hard-cast
        df["SortOrder"] = df["SortOrder"].astype(np.int16)
        for c in ["IsDigital", "IsPhysical", "IsThirdParty", "IsB2B", "IsAssisted", "IsOwnedChannel", "Is24x7"]:
            df[c] = df[c].astype(bool)
        df["OpenMinute"] = df["OpenMinute"].astype(np.int16)
        df["CloseMinute"] = df["CloseMinute"].astype(np.int16)

        return df

    # ---- Override support (base_required only) ----
    override_rows = dim_cfg.get("rows")
    if isinstance(override_rows, list) and override_rows:
        df = pd.DataFrame(override_rows)
        missing = [c for c in base_required if c not in df.columns]
        if missing:
            raise ValueError(f"Override rows missing required columns: {missing}")

        df = _derive_flags(df)

        # Ensure all expected cols exist (fill if override omitted)
        for c in cols:
            if c not in df.columns:
                df[c] = None

        return df[cols].copy()

    include_unknown = bool(dim_cfg.get("include_unknown", True))
    include_extended = bool(dim_cfg.get("include_extended", True))

    rows = []
    if include_unknown:
        rows.append(
            dict(
                SalesChannelKey=0,
                SalesChannel="Unknown",
                ChannelGroup="NA",
                SalesChannelCode="UNKNOWN",
                SortOrder=0,
                IsDigital=False,
                IsPhysical=False,
                IsThirdParty=False,
                IsB2B=False,
                IsAssisted=False,
                IsOwnedChannel=False,
                TimeProfile="NA",
                Is24x7=False,
                OpenMinute=-1,
                CloseMinute=-1,
            )
        )

    # Core (keep your existing 1..5 stable) :contentReference[oaicite:2]{index=2}
    rows += [
        dict(SalesChannelKey=1, SalesChannel="Store",       ChannelGroup="Physical", SalesChannelCode="STORE",
             SortOrder=1, IsDigital=False, IsPhysical=True,  IsThirdParty=False, IsB2B=False, IsAssisted=False,
             IsOwnedChannel=True, TimeProfile="Retail",   Is24x7=False, OpenMinute=9*60,  CloseMinute=21*60),
        dict(SalesChannelKey=2, SalesChannel="Online",      ChannelGroup="Digital",  SalesChannelCode="ONLINE",
             SortOrder=2, IsDigital=True,  IsPhysical=False, IsThirdParty=False, IsB2B=False, IsAssisted=False,
             IsOwnedChannel=True, TimeProfile="Digital",  Is24x7=True,  OpenMinute=-1,   CloseMinute=-1),
        dict(SalesChannelKey=3, SalesChannel="Marketplace", ChannelGroup="Digital",  SalesChannelCode="MARKETPLACE",
             SortOrder=3, IsDigital=True,  IsPhysical=False, IsThirdParty=True,  IsB2B=False, IsAssisted=False,
             IsOwnedChannel=False, TimeProfile="Digital", Is24x7=True,  OpenMinute=-1,   CloseMinute=-1),
        dict(SalesChannelKey=4, SalesChannel="B2B",         ChannelGroup="Business", SalesChannelCode="B2B",
             SortOrder=4, IsDigital=False, IsPhysical=False, IsThirdParty=False, IsB2B=True,  IsAssisted=False,
             IsOwnedChannel=True, TimeProfile="Business", Is24x7=False, OpenMinute=8*60,  CloseMinute=18*60),
        dict(SalesChannelKey=5, SalesChannel="CallCenter",  ChannelGroup="Assisted", SalesChannelCode="CALLCENTER",
             SortOrder=5, IsDigital=False, IsPhysical=False, IsThirdParty=False, IsB2B=False, IsAssisted=True,
             IsOwnedChannel=True, TimeProfile="Assisted", Is24x7=False, OpenMinute=8*60,  CloseMinute=20*60),
    ]

    if include_extended:
        rows += [
            dict(SalesChannelKey=6, SalesChannel="Web",          ChannelGroup="Digital",  SalesChannelCode="WEB",
                 SortOrder=6, IsDigital=True,  IsPhysical=False, IsThirdParty=False, IsB2B=False, IsAssisted=False,
                 IsOwnedChannel=True, TimeProfile="Digital",  Is24x7=True,  OpenMinute=-1, CloseMinute=-1),
            dict(SalesChannelKey=7, SalesChannel="MobileApp",    ChannelGroup="Digital",  SalesChannelCode="APP",
                 SortOrder=7, IsDigital=True,  IsPhysical=False, IsThirdParty=False, IsB2B=False, IsAssisted=False,
                 IsOwnedChannel=True, TimeProfile="Digital",  Is24x7=True,  OpenMinute=-1, CloseMinute=-1),
            dict(SalesChannelKey=8, SalesChannel="SocialCommerce", ChannelGroup="Digital", SalesChannelCode="SOCIAL",
                 SortOrder=8, IsDigital=True,  IsPhysical=False, IsThirdParty=True,  IsB2B=False, IsAssisted=False,
                 IsOwnedChannel=False, TimeProfile="Digital", Is24x7=True,  OpenMinute=-1, CloseMinute=-1),
            dict(SalesChannelKey=9, SalesChannel="PartnerReseller", ChannelGroup="Business", SalesChannelCode="PARTNER",
                 SortOrder=9, IsDigital=False, IsPhysical=False, IsThirdParty=True,  IsB2B=True,  IsAssisted=False,
                 IsOwnedChannel=False, TimeProfile="Business", Is24x7=False, OpenMinute=8*60, CloseMinute=18*60),
            dict(SalesChannelKey=10, SalesChannel="Kiosk",       ChannelGroup="Physical", SalesChannelCode="KIOSK",
                 SortOrder=10, IsDigital=False, IsPhysical=True, IsThirdParty=False, IsB2B=False, IsAssisted=False,
                 IsOwnedChannel=True, TimeProfile="Retail",   Is24x7=False, OpenMinute=10*60, CloseMinute=20*60),
        ]

    df = pd.DataFrame(rows, columns=cols)
    return _derive_flags(df)[cols].copy()


def _df_loyalty_tiers(dim_cfg: Dict[str, Any]) -> pd.DataFrame:
    required = ["LoyaltyTierKey", "LoyaltyTier", "TierRank", "PointsMultiplier"]
    override = _maybe_override_rows(dim_cfg, required_cols=required)
    if override is not None:
        override["LoyaltyTierKey"] = override["LoyaltyTierKey"].astype(np.int16)
        override["TierRank"] = override["TierRank"].astype(np.int16)
        override["PointsMultiplier"] = override["PointsMultiplier"].astype(float)
        return override

    rows = [
        (0, "None", 0, 1.00),
        (1, "Silver", 1, 1.05),
        (2, "Gold", 2, 1.10),
        (3, "Platinum", 3, 1.20),
    ]
    df = pd.DataFrame(rows, columns=required)
    df["LoyaltyTierKey"] = df["LoyaltyTierKey"].astype(np.int16)
    df["TierRank"] = df["TierRank"].astype(np.int16)
    df["PointsMultiplier"] = df["PointsMultiplier"].astype(float)
    return df


def _df_customer_acquisition_channels(dim_cfg: Dict[str, Any]) -> pd.DataFrame:
    required = ["CustomerAcquisitionChannelKey", "AcquisitionChannel", "ChannelGroup"]
    override = _maybe_override_rows(dim_cfg, required_cols=required)
    if override is not None:
        override["CustomerAcquisitionChannelKey"] = override["CustomerAcquisitionChannelKey"].astype(np.int16)
        return override

    rows = [
        (1, "Organic", "Owned/Earned"),
        (2, "PaidSearch", "Paid"),
        (3, "Social", "Paid"),
        (4, "Referral", "Owned/Earned"),
        (5, "Email", "Owned/Earned"),
        (6, "Marketplace", "Partner"),
        (7, "Affiliates", "Partner"),
    ]
    df = pd.DataFrame(rows, columns=required)
    df["CustomerAcquisitionChannelKey"] = df["CustomerAcquisitionChannelKey"].astype(np.int16)
    return df



def _df_delivery_performances(dim_cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Delivery outcome / performance lookup (line-level).
    Use DeliveryPerformanceKey on Sales lines; derive labels + IsDelayed via relationship.

    Config:
      delivery_performances:
        include_exception: true|false
        rows: [ ... ]  # optional override (must match required cols)
    """
    required = [
        "DeliveryPerformanceKey",
        "DeliveryPerformance",
        "PerformanceGroup",
        "IsDelayed",
        "SortOrder",
        "LateBucket",
    ]

    override = _maybe_override_rows(dim_cfg, required_cols=required)
    if override is not None:
        override["DeliveryPerformanceKey"] = override["DeliveryPerformanceKey"].astype(np.int16)
        override["SortOrder"] = override["SortOrder"].astype(np.int16)
        override["IsDelayed"] = override["IsDelayed"].astype(bool)
        return override

    include_exception = bool(dim_cfg.get("include_exception", True))

    rows = [
        (0, "NotApplicable", "NA", False, 0, "NA"),
        (1, "Early", "Early", False, 1, "Early"),
        (2, "OnTime", "OnTime", False, 2, "OnTime"),
        (3, "Late (1-2 days)", "Late", True, 3, "Late1_2"),
        (4, "Late (3-5 days)", "Late", True, 4, "Late3_5"),
        (5, "Late (6+ days)", "Late", True, 5, "Late6plus"),
    ]
    if include_exception:
        rows.append((6, "Delivery Exception", "Exception", True, 6, "Exception"))

    df = pd.DataFrame(rows, columns=required)
    df["DeliveryPerformanceKey"] = df["DeliveryPerformanceKey"].astype(np.int16)
    df["SortOrder"] = df["SortOrder"].astype(np.int16)
    df["IsDelayed"] = df["IsDelayed"].astype(bool)
    return df

# =========================================================
# Pipeline entrypoints (run_* wrappers)
# =========================================================

def run_sales_channels(cfg: Dict[str, Any], parquet_folder: Path) -> None:
    _run_lookup_dim(cfg=cfg, dim_key="sales_channels", out_name="sales_channels.parquet",
                    build_df=_df_sales_channels, parquet_folder=parquet_folder)



def run_loyalty_tiers(cfg: Dict[str, Any], parquet_folder: Path) -> None:
    _run_lookup_dim(cfg=cfg, dim_key="loyalty_tiers", out_name="loyalty_tiers.parquet",
                    build_df=_df_loyalty_tiers, parquet_folder=parquet_folder)


def run_customer_acquisition_channels(cfg: Dict[str, Any], parquet_folder: Path) -> None:
    _run_lookup_dim(cfg=cfg, dim_key="customer_acquisition_channels", out_name="customer_acquisition_channels.parquet",
                    build_df=_df_customer_acquisition_channels, parquet_folder=parquet_folder)



def run_lookups(cfg: Dict[str, Any], parquet_folder: Path) -> None:
    """
    Convenience function if you prefer a single call from dimensions_runner.
    You can also call individual run_* functions if you want per-dim force control.
    """
    run_sales_channels(cfg, parquet_folder)
    run_loyalty_tiers(cfg, parquet_folder)
    run_customer_acquisition_channels(cfg, parquet_folder)

