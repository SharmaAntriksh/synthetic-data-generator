"""Complaints pipeline runner — generates complaints.parquet using
accumulated (CustomerKey, SalesOrderNumber, SalesOrderLineNumber) triples
from the sales pipeline.

Runs AFTER sales generation.  A configurable fraction of customers file
complaints, with most complaints linked to specific order lines and the
remainder being general service complaints.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.facts.complaints.accumulator import ComplaintsAccumulator
from src.utils.logging_utils import info, skip


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NS_PER_DAY: int = 86_400_000_000_000

_COMPLAINT_TYPES_ORDER = [
    "Product Defect",
    "Wrong Item",
    "Late Delivery",
    "Damaged In Transit",
    "Billing Error",
    "Missing Parts",
]

_COMPLAINT_TYPES_GENERAL = [
    "Service",
    "Billing Error",
]

_COMPLAINT_DETAILS: Dict[str, list] = {
    "Product Defect": [
        "Item stopped working after a few uses",
        "Missing components in the package",
        "Product does not match the description",
        "Visible damage on arrival, packaging intact",
        "Color/size differs from what was ordered",
        "Battery drains significantly faster than advertised",
        "Strong chemical smell from the product",
    ],
    "Wrong Item": [
        "Received a completely different product",
        "Correct product but wrong variant/color",
        "Order contained someone else's items",
        "Received duplicate items instead of separate products",
    ],
    "Late Delivery": [
        "Package arrived well past the estimated date",
        "Tracking showed delivered but not received for days",
        "Delivery was rescheduled without notification",
        "Only partial order delivered, rest still pending",
    ],
    "Damaged In Transit": [
        "Box was crushed, item broken inside",
        "Water damage to product and packaging",
        "Item arrived with scratches and dents",
        "Screen cracked during shipping",
    ],
    "Billing Error": [
        "Charged twice for the same order",
        "Discount code was not applied at checkout",
        "Final charge higher than the displayed price",
        "Refund from previous return still not received",
        "Charged for items that were cancelled",
    ],
    "Service": [
        "Staff was unhelpful and dismissive",
        "Long wait time with no resolution",
        "Received conflicting information from support",
        "Store would not honor the posted return policy",
        "No follow-up after filing a complaint",
    ],
    "Missing Parts": [
        "Assembly hardware not included",
        "Accessories listed on the box were missing",
        "Manual/documentation not in the package",
    ],
}

_SEVERITY_VALUES = np.array(["Low", "Medium", "High", "Critical"], dtype=object)
_SEVERITY_WEIGHTS = np.array([0.25, 0.40, 0.25, 0.10])

_CHANNEL_VALUES = np.array(["Email", "Phone", "In-Store", "Website", "Chat"], dtype=object)
_CHANNEL_WEIGHTS = np.array([0.30, 0.25, 0.15, 0.15, 0.15])

_STATUS_VALUES = np.array(["Resolved", "Closed", "Open", "Escalated"], dtype=object)

_RESOLUTION_TYPES = np.array(
    ["Replacement", "Refund", "Discount", "Apology", "Store Credit"], dtype=object
)
_RESOLUTION_WEIGHTS = np.array([0.25, 0.30, 0.20, 0.10, 0.15])

# Fraction of complaints that are order-linked vs general
_ORDER_LINKED_RATE = 0.75


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ComplaintsCfg:
    enabled: bool = False
    complaint_rate: float = 0.03
    repeat_complaint_rate: float = 0.15
    max_complaints: int = 5
    resolution_rate: float = 0.85
    escalation_rate: float = 0.10
    avg_response_days: int = 5
    max_response_days: int = 30
    seed: int = 600
    write_chunk_rows: int = 250_000


def _read_cfg(cfg: Any) -> _ComplaintsCfg:
    cc = getattr(cfg, "complaints", None)
    if cc is None:
        return _ComplaintsCfg()
    return _ComplaintsCfg(
        enabled=bool(getattr(cc, "enabled", False)),
        complaint_rate=float(getattr(cc, "complaint_rate", 0.03)),
        repeat_complaint_rate=float(getattr(cc, "repeat_complaint_rate", 0.15)),
        max_complaints=int(getattr(cc, "max_complaints", 5)),
        resolution_rate=float(getattr(cc, "resolution_rate", 0.85)),
        escalation_rate=float(getattr(cc, "escalation_rate", 0.10)),
        avg_response_days=int(getattr(cc, "avg_response_days", 5)),
        max_response_days=int(getattr(cc, "max_response_days", 30)),
        seed=int(getattr(cc, "seed", None) or 600),
        write_chunk_rows=int(getattr(cc, "write_chunk_rows", 250_000)),
    )


def _parse_global_dates(cfg: Any) -> Tuple[pd.Timestamp, pd.Timestamp]:
    defaults = getattr(cfg, "defaults", None)
    if defaults is None:
        defaults = getattr(cfg, "_defaults", None)
    gd = getattr(defaults, "dates", None) if defaults else None
    if gd is None:
        raise ValueError("Cannot resolve global dates for complaints.")
    start_raw = gd.get("start", None) if isinstance(gd, dict) else getattr(gd, "start", None)
    end_raw = gd.get("end", None) if isinstance(gd, dict) else getattr(gd, "end", None)
    if start_raw is None or end_raw is None:
        raise ValueError("Global dates must have both 'start' and 'end'.")
    return pd.Timestamp(start_raw), pd.Timestamp(end_raw)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def _complaints_schema() -> pa.Schema:
    return pa.schema([
        pa.field("ComplaintKey", pa.int64()),
        pa.field("CustomerKey", pa.int64()),
        pa.field("SalesOrderNumber", pa.int64(), nullable=True),
        pa.field("LineNumber", pa.int64(), nullable=True),
        pa.field("ComplaintDate", pa.date32()),
        pa.field("ResolutionDate", pa.date32(), nullable=True),
        pa.field("ComplaintType", pa.string()),
        pa.field("ComplaintDetail", pa.string()),
        pa.field("Severity", pa.string()),
        pa.field("Channel", pa.string()),
        pa.field("Status", pa.string()),
        pa.field("ResolutionType", pa.string(), nullable=True),
        pa.field("ResponseDays", pa.int32(), nullable=True),
    ])


# ---------------------------------------------------------------------------
# Complaint generation
# ---------------------------------------------------------------------------

def _generate_complaints(
    order_data: pd.DataFrame,
    c: _ComplaintsCfg,
    g_start: pd.Timestamp,
    g_end: pd.Timestamp,
) -> pa.Table:
    rng = np.random.default_rng(c.seed)

    g_start_ns = np.int64(g_start.value)
    g_end_ns = np.int64(g_end.value)

    # Get unique customers from order data
    unique_customers = order_data["CustomerKey"].unique()
    n_customers = len(unique_customers)

    schema = _complaints_schema()
    if n_customers == 0:
        return pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema)

    # Select complaining customers
    n_complainers = max(1, int(round(n_customers * c.complaint_rate)))
    complainer_keys = rng.choice(unique_customers, size=n_complainers, replace=False)

    # Determine number of complaints per customer
    complaints_per = np.ones(n_complainers, dtype=np.int32)
    repeat_mask = rng.random(n_complainers) < c.repeat_complaint_rate
    n_repeaters = repeat_mask.sum()
    if n_repeaters > 0:
        # Repeat complainers get 2-max_complaints
        complaints_per[repeat_mask] = rng.integers(
            2, c.max_complaints + 1, size=n_repeaters
        ).astype(np.int32)

    total_rows = int(complaints_per.sum())

    # Build per-customer order lookup
    cust_orders: Dict[int, pd.DataFrame] = {}
    for ck in complainer_keys:
        cust_orders[int(ck)] = order_data[order_data["CustomerKey"] == ck]

    # Pre-allocate output arrays
    out_ckey = np.empty(total_rows, dtype=np.int64)
    out_so = np.full(total_rows, -1, dtype=np.int64)  # -1 = NULL
    out_ln = np.full(total_rows, -1, dtype=np.int64)
    out_date_ns = np.empty(total_rows, dtype=np.int64)
    out_res_date_ns = np.full(total_rows, -1, dtype=np.int64)  # -1 = NULL
    out_type = np.empty(total_rows, dtype=object)
    out_detail = np.empty(total_rows, dtype=object)
    out_severity = np.empty(total_rows, dtype=object)
    out_channel = np.empty(total_rows, dtype=object)
    out_status = np.empty(total_rows, dtype=object)
    out_res_type = np.empty(total_rows, dtype=object)
    out_resp_days = np.full(total_rows, -1, dtype=np.int32)  # -1 = NULL

    # Flatten all complaint type+detail pairs for order-linked
    all_order_types = []
    all_order_details = []
    for ct in _COMPLAINT_TYPES_ORDER:
        for detail in _COMPLAINT_DETAILS[ct]:
            all_order_types.append(ct)
            all_order_details.append(detail)
    all_order_types = np.array(all_order_types, dtype=object)
    all_order_details = np.array(all_order_details, dtype=object)

    # Flatten for general complaints
    all_general_types = []
    all_general_details = []
    for ct in _COMPLAINT_TYPES_GENERAL:
        for detail in _COMPLAINT_DETAILS[ct]:
            all_general_types.append(ct)
            all_general_details.append(detail)
    all_general_types = np.array(all_general_types, dtype=object)
    all_general_details = np.array(all_general_details, dtype=object)

    row = 0
    for i in range(n_complainers):
        ck = int(complainer_keys[i])
        n_complaints = int(complaints_per[i])
        orders = cust_orders[ck]

        for _ in range(n_complaints):
            out_ckey[row] = ck

            is_order_linked = rng.random() < _ORDER_LINKED_RATE and len(orders) > 0

            if is_order_linked:
                # Pick a random order line
                order_idx = rng.integers(0, len(orders))
                order_row = orders.iloc[order_idx]
                out_so[row] = int(order_row["SalesOrderNumber"])
                out_ln[row] = int(order_row["SalesOrderLineNumber"])

                # Pick type+detail from order-linked pool
                td_idx = rng.integers(0, len(all_order_types))
                out_type[row] = all_order_types[td_idx]
                out_detail[row] = all_order_details[td_idx]
            else:
                # General complaint — SalesOrderNumber and LineNumber stay NULL (-1)
                td_idx = rng.integers(0, len(all_general_types))
                out_type[row] = all_general_types[td_idx]
                out_detail[row] = all_general_details[td_idx]

            # Complaint date: random within global date range
            span = g_end_ns - g_start_ns
            out_date_ns[row] = g_start_ns + rng.integers(0, max(1, span))

            # Severity and channel
            out_severity[row] = rng.choice(_SEVERITY_VALUES, p=_SEVERITY_WEIGHTS)
            out_channel[row] = rng.choice(_CHANNEL_VALUES, p=_CHANNEL_WEIGHTS)

            # Status and resolution
            if rng.random() < c.resolution_rate:
                status = rng.choice(["Resolved", "Closed"])
                out_status[row] = status
                out_res_type[row] = rng.choice(_RESOLUTION_TYPES, p=_RESOLUTION_WEIGHTS)
                resp_days = int(rng.exponential(c.avg_response_days))
                resp_days = min(resp_days, c.max_response_days)
                resp_days = max(resp_days, 0)
                out_resp_days[row] = resp_days
                out_res_date_ns[row] = out_date_ns[row] + np.int64(resp_days) * _NS_PER_DAY
                # Clamp resolution date to global end
                if out_res_date_ns[row] > g_end_ns:
                    out_res_date_ns[row] = g_end_ns
                    out_resp_days[row] = max(
                        0,
                        int((g_end_ns - out_date_ns[row]) // _NS_PER_DAY),
                    )
            else:
                # Unresolved
                if rng.random() < c.escalation_rate:
                    out_status[row] = "Escalated"
                else:
                    out_status[row] = "Open"
                # ResolutionDate, ResolutionType, ResponseDays stay NULL

            row += 1

    # Build complaint keys
    out_ckey_final = out_ckey[:row]
    complaint_keys = np.arange(1, row + 1, dtype=np.int64)

    # Convert dates
    complaint_dates = out_date_ns[:row].view("datetime64[ns]").astype("datetime64[ms]")

    # Handle nullable SalesOrderNumber / LineNumber
    so_arr = out_so[:row]
    ln_arr = out_ln[:row]
    so_mask = so_arr == -1
    ln_mask = ln_arr == -1

    # Handle nullable ResolutionDate
    res_date_arr = out_res_date_ns[:row]
    res_date_mask = res_date_arr == -1
    res_dates_dt = res_date_arr.copy()
    res_dates_dt[res_date_mask] = 0  # placeholder for view
    res_dates_dt = res_dates_dt.view("datetime64[ns]").astype("datetime64[ms]")

    # Handle nullable ResponseDays
    resp_days_arr = out_resp_days[:row]
    resp_days_mask = resp_days_arr == -1

    # Build full table
    so_pa = pa.array(so_arr.tolist(), type=pa.int64(), mask=so_mask)
    ln_pa = pa.array(ln_arr.tolist(), type=pa.int64(), mask=ln_mask)
    res_date_pa = pa.array(res_dates_dt.tolist(), type=pa.date32(), mask=res_date_mask)
    resp_days_pa = pa.array(resp_days_arr.tolist(), type=pa.int32(), mask=resp_days_mask)

    table = pa.table(
        [
            pa.array(complaint_keys, type=pa.int64()),
            pa.array(out_ckey_final, type=pa.int64()),
            so_pa,
            ln_pa,
            pa.array(complaint_dates, type=pa.date32()),
            res_date_pa,
            pa.array(out_type[:row], type=pa.string()),
            pa.array(out_detail[:row], type=pa.string()),
            pa.array(out_severity[:row], type=pa.string()),
            pa.array(out_channel[:row], type=pa.string()),
            pa.array(out_status[:row], type=pa.string()),
            pa.array(out_res_type[:row].tolist(), type=pa.string()),
            resp_days_pa,
        ],
        schema=schema,
    )
    return table


# ---------------------------------------------------------------------------
# Format-aware writer
# ---------------------------------------------------------------------------

_COMPLAINTS_CSV_COLUMNS = [
    "ComplaintKey", "CustomerKey", "SalesOrderNumber", "LineNumber",
    "ComplaintDate", "ResolutionDate", "ComplaintType", "ComplaintDetail",
    "Severity", "Channel", "Status", "ResolutionType", "ResponseDays",
]

_COMPLAINTS_CSV_INT_COLS = ("ComplaintKey", "CustomerKey", "SalesOrderNumber", "LineNumber", "ResponseDays")


def _prepare_complaints_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Select columns and cast integers for clean CSV output."""
    out = df.copy()
    for col in _COMPLAINTS_CSV_COLUMNS:
        if col not in out.columns:
            out[col] = None
    out = out[_COMPLAINTS_CSV_COLUMNS]

    for col in _COMPLAINTS_CSV_INT_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    return out


def _write_complaints(table: pa.Table, complaints_dir: Path, file_format: str) -> None:
    """Write complaints table in the requested format (parquet, csv, or delta)."""
    name = "complaints"

    if file_format == "deltaparquet":
        # Delta table lives at complaints_dir itself (the directory IS the delta table)
        try:
            from deltalake import write_deltalake
        except ImportError:
            from deltalake.writer import write_deltalake
        write_deltalake(str(complaints_dir), table, mode="overwrite")
        info(f"Wrote {name} delta ({table.num_rows:,} rows)")
        return

    # Parquet (always written for parquet and csv formats)
    parquet_path = complaints_dir / f"{name}.parquet"
    pq.write_table(
        table, str(parquet_path),
        compression="snappy",
        row_group_size=500_000,
        use_dictionary=True,
    )

    if file_format == "csv":
        csv_path = complaints_dir / f"{name}.csv"
        df = table.to_pandas()
        csv_df = _prepare_complaints_csv(df)
        csv_df.to_csv(str(csv_path), index=False)
        info(f"Wrote {csv_path.name} ({table.num_rows:,} rows)")
    else:
        info(f"Wrote {parquet_path.name} ({table.num_rows:,} rows)")


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_complaints_pipeline(
    *,
    accumulator: ComplaintsAccumulator,
    parquet_dims: Path,
    fact_out: Path,
    cfg: Any,
    file_format: str = "parquet",
) -> Optional[Dict[str, Any]]:
    """Generate complaints output using accumulated sales data."""
    c = _read_cfg(cfg)
    if not c.enabled:
        return None

    if not accumulator.has_data:
        skip("Complaints: no sales data accumulated; skipping.")
        return None

    g_start, g_end = _parse_global_dates(cfg)
    order_data = accumulator.finalize()

    complaints_dir = Path(fact_out) / "complaints"
    complaints_dir.mkdir(parents=True, exist_ok=True)

    table = _generate_complaints(
        order_data=order_data,
        c=c,
        g_start=g_start,
        g_end=g_end,
    )

    n_rows = table.num_rows
    if n_rows == 0:
        skip("Complaints: generated 0 rows; skipping write.")
        return None

    _write_complaints(table, complaints_dir, file_format)

    # For deltaparquet the delta table is written directly at complaints_dir;
    # the directory structure is the delta table itself — no cleanup needed.

    return {
        "complaints": str(complaints_dir),
        "complaints_rows": n_rows,
    }
