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
    required = ["SalesChannelKey", "SalesChannel", "ChannelGroup"]
    override = _maybe_override_rows(dim_cfg, required_cols=required)
    if override is not None:
        return override

    rows = [
        (1, "Store", "Physical"),
        (2, "Online", "Digital"),
        (3, "Marketplace", "Digital"),
        (4, "B2B", "Business"),
        (5, "CallCenter", "Assisted"),
    ]
    df = pd.DataFrame(rows, columns=required)
    df["SalesChannelKey"] = df["SalesChannelKey"].astype(np.int16)
    return df


def _df_payment_methods(dim_cfg: Dict[str, Any]) -> pd.DataFrame:
    required = ["PaymentMethodKey", "PaymentMethod", "PaymentType", "IsInstant"]
    override = _maybe_override_rows(dim_cfg, required_cols=required)
    if override is not None:
        override["PaymentMethodKey"] = override["PaymentMethodKey"].astype(np.int16)
        override["IsInstant"] = override["IsInstant"].astype(bool)
        return override

    rows = [
        (1, "Cash", "Cash", True),
        (2, "Card", "Card", True),
        (3, "UPI/Wallet", "Digital", True),
        (4, "NetBanking", "Digital", False),
        (5, "CashOnDelivery", "Cash", False),
        (6, "GiftCard", "StoredValue", True),
        (7, "BNPL", "Credit", False),
    ]
    df = pd.DataFrame(rows, columns=required)
    df["PaymentMethodKey"] = df["PaymentMethodKey"].astype(np.int16)
    df["IsInstant"] = df["IsInstant"].astype(bool)
    return df


def _df_fulfillment_methods(dim_cfg: Dict[str, Any]) -> pd.DataFrame:
    required = ["FulfillmentMethodKey", "FulfillmentMethod", "RequiresShipping"]
    override = _maybe_override_rows(dim_cfg, required_cols=required)
    if override is not None:
        override["FulfillmentMethodKey"] = override["FulfillmentMethodKey"].astype(np.int16)
        override["RequiresShipping"] = override["RequiresShipping"].astype(bool)
        return override

    rows = [
        (1, "Takeaway", False),
        (2, "StorePickup", False),
        (3, "ShipFromStore", True),
        (4, "ShipFromWarehouse", True),
        (5, "DigitalDelivery", False),
    ]
    df = pd.DataFrame(rows, columns=required)
    df["FulfillmentMethodKey"] = df["FulfillmentMethodKey"].astype(np.int16)
    df["RequiresShipping"] = df["RequiresShipping"].astype(bool)
    return df


def _df_order_statuses(dim_cfg: Dict[str, Any]) -> pd.DataFrame:
    required = ["OrderStatusKey", "OrderStatus", "IsFinal", "StatusGroup"]
    override = _maybe_override_rows(dim_cfg, required_cols=required)
    if override is not None:
        override["OrderStatusKey"] = override["OrderStatusKey"].astype(np.int16)
        override["IsFinal"] = override["IsFinal"].astype(bool)
        return override

    rows = [
        (1, "Placed", False, "Placed"),
        (2, "Packed", False, "Fulfillment"),
        (3, "Shipped", False, "Fulfillment"),
        (4, "Delivered", True, "Terminal"),
        (5, "Cancelled", True, "Terminal"),
        (6, "Returned", True, "Terminal"),
        (7, "Refunded", True, "Terminal"),
    ]
    df = pd.DataFrame(rows, columns=required)
    df["OrderStatusKey"] = df["OrderStatusKey"].astype(np.int16)
    df["IsFinal"] = df["IsFinal"].astype(bool)
    return df


def _df_payment_statuses(dim_cfg: Dict[str, Any]) -> pd.DataFrame:
    required = ["PaymentStatusKey", "PaymentStatus", "IsFinal"]
    override = _maybe_override_rows(dim_cfg, required_cols=required)
    if override is not None:
        override["PaymentStatusKey"] = override["PaymentStatusKey"].astype(np.int16)
        override["IsFinal"] = override["IsFinal"].astype(bool)
        return override

    rows = [
        (1, "Authorized", False),
        (2, "Captured", True),
        (3, "Failed", True),
        (4, "RefundInitiated", False),
        (5, "Refunded", True),
        (6, "Chargeback", True),
    ]
    df = pd.DataFrame(rows, columns=required)
    df["PaymentStatusKey"] = df["PaymentStatusKey"].astype(np.int16)
    df["IsFinal"] = df["IsFinal"].astype(bool)
    return df


def _df_delivery_service_levels(dim_cfg: Dict[str, Any]) -> pd.DataFrame:
    required = ["DeliveryServiceLevelKey", "DeliveryServiceLevel", "SortOrder"]
    override = _maybe_override_rows(dim_cfg, required_cols=required)
    if override is not None:
        override["DeliveryServiceLevelKey"] = override["DeliveryServiceLevelKey"].astype(np.int16)
        override["SortOrder"] = override["SortOrder"].astype(np.int16)
        return override

    rows = [
        (1, "Standard", 1),
        (2, "Express", 2),
        (3, "SameDay", 3),
        (4, "Scheduled", 4),
    ]
    df = pd.DataFrame(rows, columns=required)
    df["DeliveryServiceLevelKey"] = df["DeliveryServiceLevelKey"].astype(np.int16)
    df["SortOrder"] = df["SortOrder"].astype(np.int16)
    return df


def _df_shipping_carriers(dim_cfg: Dict[str, Any]) -> pd.DataFrame:
    required = ["ShippingCarrierKey", "CarrierName", "CarrierType"]
    override = _maybe_override_rows(dim_cfg, required_cols=required)
    if override is not None:
        override["ShippingCarrierKey"] = override["ShippingCarrierKey"].astype(np.int16)
        return override

    # Generic carriers (intentionally not country-specific).
    rows = [
        (1, "FastShip", "Courier"),
        (2, "BlueDart", "Courier"),
        (3, "DHL", "Courier"),
        (4, "FedEx", "Courier"),
        (5, "UPS", "Courier"),
        (6, "PostalService", "Postal"),
        (7, "LocalCourier", "Courier"),
        (8, "SameDayRunners", "SameDay"),
    ]
    df = pd.DataFrame(rows, columns=required)
    df["ShippingCarrierKey"] = df["ShippingCarrierKey"].astype(np.int16)
    return df


def _df_discount_types(dim_cfg: Dict[str, Any]) -> pd.DataFrame:
    required = ["DiscountTypeKey", "DiscountType", "DiscountGroup"]
    override = _maybe_override_rows(dim_cfg, required_cols=required)
    if override is not None:
        override["DiscountTypeKey"] = override["DiscountTypeKey"].astype(np.int16)
        return override

    rows = [
        (1, "None", "None"),
        (2, "Promo", "Promotion"),
        (3, "Coupon", "Promotion"),
        (4, "Markdown", "Pricing"),
        (5, "Bundle", "Pricing"),
        (6, "Employee", "Internal"),
        (7, "PriceMatch", "Pricing"),
    ]
    df = pd.DataFrame(rows, columns=required)
    df["DiscountTypeKey"] = df["DiscountTypeKey"].astype(np.int16)
    return df


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


def _df_time_buckets(dim_cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Modes:
      time_buckets:
        mode: "4" | "24"
    """
    mode = str(dim_cfg.get("mode", "4")).strip()

    if mode == "24":
        required = ["TimeBucketKey", "TimeBucket", "StartHour", "EndHour"]
        rows = []
        for h in range(24):
            rows.append((h, f"{h:02d}:00-{(h + 1) % 24:02d}:00", h, (h + 1) % 24))
        df = pd.DataFrame(rows, columns=required)
        df["TimeBucketKey"] = df["TimeBucketKey"].astype(np.int16)
        df["StartHour"] = df["StartHour"].astype(np.int16)
        df["EndHour"] = df["EndHour"].astype(np.int16)
        return df

    # Default: 4 buckets
    required = ["TimeBucketKey", "TimeBucket", "StartHour", "EndHour"]
    override = _maybe_override_rows(dim_cfg, required_cols=required)
    if override is not None:
        override["TimeBucketKey"] = override["TimeBucketKey"].astype(np.int16)
        override["StartHour"] = override["StartHour"].astype(np.int16)
        override["EndHour"] = override["EndHour"].astype(np.int16)
        return override

    rows = [
        (1, "Morning", 6, 12),
        (2, "Afternoon", 12, 17),
        (3, "Evening", 17, 22),
        (4, "Night", 22, 6),
    ]
    df = pd.DataFrame(rows, columns=required)
    df["TimeBucketKey"] = df["TimeBucketKey"].astype(np.int16)
    df["StartHour"] = df["StartHour"].astype(np.int16)
    df["EndHour"] = df["EndHour"].astype(np.int16)
    return df


# =========================================================
# Pipeline entrypoints (run_* wrappers)
# =========================================================

def run_sales_channels(cfg: Dict[str, Any], parquet_folder: Path) -> None:
    _run_lookup_dim(cfg=cfg, dim_key="sales_channels", out_name="sales_channels.parquet",
                    build_df=_df_sales_channels, parquet_folder=parquet_folder)


def run_payment_methods(cfg: Dict[str, Any], parquet_folder: Path) -> None:
    _run_lookup_dim(cfg=cfg, dim_key="payment_methods", out_name="payment_methods.parquet",
                    build_df=_df_payment_methods, parquet_folder=parquet_folder)


def run_fulfillment_methods(cfg: Dict[str, Any], parquet_folder: Path) -> None:
    _run_lookup_dim(cfg=cfg, dim_key="fulfillment_methods", out_name="fulfillment_methods.parquet",
                    build_df=_df_fulfillment_methods, parquet_folder=parquet_folder)


def run_order_statuses(cfg: Dict[str, Any], parquet_folder: Path) -> None:
    _run_lookup_dim(cfg=cfg, dim_key="order_statuses", out_name="order_statuses.parquet",
                    build_df=_df_order_statuses, parquet_folder=parquet_folder)


def run_payment_statuses(cfg: Dict[str, Any], parquet_folder: Path) -> None:
    _run_lookup_dim(cfg=cfg, dim_key="payment_statuses", out_name="payment_statuses.parquet",
                    build_df=_df_payment_statuses, parquet_folder=parquet_folder)


def run_shipping_carriers(cfg: Dict[str, Any], parquet_folder: Path) -> None:
    _run_lookup_dim(cfg=cfg, dim_key="shipping_carriers", out_name="shipping_carriers.parquet",
                    build_df=_df_shipping_carriers, parquet_folder=parquet_folder)


def run_delivery_service_levels(cfg: Dict[str, Any], parquet_folder: Path) -> None:
    _run_lookup_dim(cfg=cfg, dim_key="delivery_service_levels", out_name="delivery_service_levels.parquet",
                    build_df=_df_delivery_service_levels, parquet_folder=parquet_folder)


def run_discount_types(cfg: Dict[str, Any], parquet_folder: Path) -> None:
    _run_lookup_dim(cfg=cfg, dim_key="discount_types", out_name="discount_types.parquet",
                    build_df=_df_discount_types, parquet_folder=parquet_folder)


def run_loyalty_tiers(cfg: Dict[str, Any], parquet_folder: Path) -> None:
    _run_lookup_dim(cfg=cfg, dim_key="loyalty_tiers", out_name="loyalty_tiers.parquet",
                    build_df=_df_loyalty_tiers, parquet_folder=parquet_folder)


def run_customer_acquisition_channels(cfg: Dict[str, Any], parquet_folder: Path) -> None:
    _run_lookup_dim(cfg=cfg, dim_key="customer_acquisition_channels", out_name="customer_acquisition_channels.parquet",
                    build_df=_df_customer_acquisition_channels, parquet_folder=parquet_folder)


def run_time_buckets(cfg: Dict[str, Any], parquet_folder: Path) -> None:
    _run_lookup_dim(cfg=cfg, dim_key="time_buckets", out_name="time_buckets.parquet",
                    build_df=_df_time_buckets, parquet_folder=parquet_folder)


def run_lookups(cfg: Dict[str, Any], parquet_folder: Path) -> None:
    """
    Convenience function if you prefer a single call from dimensions_runner.
    You can also call individual run_* functions if you want per-dim force control.
    """
    run_sales_channels(cfg, parquet_folder)
    run_payment_methods(cfg, parquet_folder)
    run_fulfillment_methods(cfg, parquet_folder)
    run_order_statuses(cfg, parquet_folder)
    run_payment_statuses(cfg, parquet_folder)
    run_shipping_carriers(cfg, parquet_folder)
    run_delivery_service_levels(cfg, parquet_folder)
    run_discount_types(cfg, parquet_folder)
    run_loyalty_tiers(cfg, parquet_folder)
    run_customer_acquisition_channels(cfg, parquet_folder)
    run_time_buckets(cfg, parquet_folder)
