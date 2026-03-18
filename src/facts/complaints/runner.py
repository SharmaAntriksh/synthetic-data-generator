"""Complaints pipeline runner — generates complaints.parquet using
accumulated (CustomerKey, SalesOrderNumber, SalesOrderLineNumber) triples
from the sales pipeline.

Runs AFTER sales generation.  A configurable fraction of customers file
complaints, with most complaints linked to specific order lines and the
remainder being general service complaints.
"""
from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa

from src.facts.complaints.accumulator import ComplaintsAccumulator
from src.facts.shared.writers import write_fact_table
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

# Below this complainer count, use the serial path (spawning overhead not worth it)
_PARALLEL_THRESHOLD = 50_000


# ---------------------------------------------------------------------------
# Worker function — must be top-level for Windows spawn pickling
# ---------------------------------------------------------------------------

def _complaints_worker_task(args: Tuple) -> Dict[str, np.ndarray]:
    """Generate complaint rows for a chunk of customers.

    Args (tuple):
        chunk_idx:             int — chunk sequence number (for seeding)
        seed:                  int — base seed from _ComplaintsCfg
        n_chunks:              int — total number of chunks (for SeedSequence.spawn)
        complainer_keys_chunk: np.ndarray[int64] — CustomerKeys for this chunk
        complaints_per_chunk:  np.ndarray[int32] — complaint counts per customer
        order_arrays_chunk:    dict of numpy arrays — order data for this chunk's customers
            Keys: "CustomerKey", "SalesOrderNumber", "SalesOrderLineNumber"
        config_scalars:        dict — complaint config scalars (resolution_rate, etc.)
        date_range:            tuple[int, int] — (g_start_ns, g_end_ns)

    Returns:
        dict of numpy arrays for the chunk's complaint rows
    """
    (
        chunk_idx,
        seed,
        n_chunks,
        complainer_keys_chunk,
        complaints_per_chunk,
        order_arrays_chunk,
        config_scalars,
        date_range,
    ) = args

    # Each worker gets its own independent RNG stream
    child_rng = np.random.default_rng(
        np.random.SeedSequence(seed).spawn(n_chunks)[chunk_idx]
    )

    g_start_ns, g_end_ns = date_range
    resolution_rate = config_scalars["resolution_rate"]
    escalation_rate = config_scalars["escalation_rate"]
    avg_response_days = config_scalars["avg_response_days"]
    max_response_days = config_scalars["max_response_days"]

    # Reconstruct per-customer order lookup from the filtered arrays
    ck_arr = order_arrays_chunk["CustomerKey"]
    so_arr = order_arrays_chunk["SalesOrderNumber"]
    ln_arr = order_arrays_chunk["SalesOrderLineNumber"]

    # Build dict: CustomerKey -> (so_array, ln_array)
    cust_orders: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    if len(ck_arr) > 0:
        # Sort by CustomerKey for groupby-style splitting
        sort_idx = np.argsort(ck_arr, kind="stable")
        sorted_ck = ck_arr[sort_idx]
        sorted_so = so_arr[sort_idx]
        sorted_ln = ln_arr[sort_idx]
        boundaries = np.where(np.diff(sorted_ck))[0] + 1
        ck_groups = np.split(sorted_ck, boundaries)
        so_groups = np.split(sorted_so, boundaries)
        ln_groups = np.split(sorted_ln, boundaries)
        for g_ck, g_so, g_ln in zip(ck_groups, so_groups, ln_groups):
            cust_orders[int(g_ck[0])] = (g_so, g_ln)

    # Flatten all complaint type+detail pairs for order-linked
    all_order_types: List[str] = []
    all_order_details: List[str] = []
    for ct in _COMPLAINT_TYPES_ORDER:
        for detail in _COMPLAINT_DETAILS[ct]:
            all_order_types.append(ct)
            all_order_details.append(detail)
    aot = np.array(all_order_types, dtype=object)
    aod = np.array(all_order_details, dtype=object)

    # Flatten for general complaints
    all_general_types: List[str] = []
    all_general_details: List[str] = []
    for ct in _COMPLAINT_TYPES_GENERAL:
        for detail in _COMPLAINT_DETAILS[ct]:
            all_general_types.append(ct)
            all_general_details.append(detail)
    agt = np.array(all_general_types, dtype=object)
    agd = np.array(all_general_details, dtype=object)

    n_complainers = len(complainer_keys_chunk)
    total_rows = int(complaints_per_chunk.sum())

    # Pre-allocate output arrays
    out_ckey = np.empty(total_rows, dtype=np.int64)
    out_so = np.full(total_rows, -1, dtype=np.int64)
    out_ln = np.full(total_rows, -1, dtype=np.int64)
    out_date_ns = np.empty(total_rows, dtype=np.int64)
    out_res_date_ns = np.full(total_rows, -1, dtype=np.int64)
    out_type = np.empty(total_rows, dtype=object)
    out_detail = np.empty(total_rows, dtype=object)
    out_severity = np.empty(total_rows, dtype=object)
    out_channel = np.empty(total_rows, dtype=object)
    out_status = np.empty(total_rows, dtype=object)
    out_res_type = np.empty(total_rows, dtype=object)
    out_resp_days = np.full(total_rows, -1, dtype=np.int32)

    span = g_end_ns - g_start_ns

    row = 0
    for i in range(n_complainers):
        ck = int(complainer_keys_chunk[i])
        n_complaints = int(complaints_per_chunk[i])
        orders = cust_orders.get(ck)
        n_orders = len(orders[0]) if orders is not None else 0

        for _ in range(n_complaints):
            out_ckey[row] = ck

            is_order_linked = child_rng.random() < _ORDER_LINKED_RATE and n_orders > 0

            if is_order_linked:
                order_idx = child_rng.integers(0, n_orders)
                out_so[row] = int(orders[0][order_idx])
                out_ln[row] = int(orders[1][order_idx])
                td_idx = child_rng.integers(0, len(aot))
                out_type[row] = aot[td_idx]
                out_detail[row] = aod[td_idx]
            else:
                td_idx = child_rng.integers(0, len(agt))
                out_type[row] = agt[td_idx]
                out_detail[row] = agd[td_idx]

            out_date_ns[row] = g_start_ns + child_rng.integers(0, max(1, span))

            out_severity[row] = child_rng.choice(_SEVERITY_VALUES, p=_SEVERITY_WEIGHTS)
            out_channel[row] = child_rng.choice(_CHANNEL_VALUES, p=_CHANNEL_WEIGHTS)

            if child_rng.random() < resolution_rate:
                status = child_rng.choice(["Resolved", "Closed"])
                out_status[row] = status
                out_res_type[row] = child_rng.choice(_RESOLUTION_TYPES, p=_RESOLUTION_WEIGHTS)
                resp_days = int(child_rng.exponential(avg_response_days))
                resp_days = min(resp_days, max_response_days)
                resp_days = max(resp_days, 0)
                out_resp_days[row] = resp_days
                out_res_date_ns[row] = out_date_ns[row] + np.int64(resp_days) * _NS_PER_DAY
                if out_res_date_ns[row] > g_end_ns:
                    out_res_date_ns[row] = g_end_ns
                    out_resp_days[row] = max(
                        0,
                        int((g_end_ns - out_date_ns[row]) // _NS_PER_DAY),
                    )
            else:
                if child_rng.random() < escalation_rate:
                    out_status[row] = "Escalated"
                else:
                    out_status[row] = "Open"
                out_res_type[row] = None

            row += 1

    return {
        "ckey": out_ckey[:row],
        "so": out_so[:row],
        "ln": out_ln[:row],
        "date_ns": out_date_ns[:row],
        "res_date_ns": out_res_date_ns[:row],
        "type": out_type[:row],
        "detail": out_detail[:row],
        "severity": out_severity[:row],
        "channel": out_channel[:row],
        "status": out_status[:row],
        "res_type": out_res_type[:row],
        "resp_days": out_resp_days[:row],
    }


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
# Helpers: assemble a PyArrow table from raw column arrays
# ---------------------------------------------------------------------------

def _arrays_to_table(
    out_ckey: np.ndarray,
    out_so: np.ndarray,
    out_ln: np.ndarray,
    out_date_ns: np.ndarray,
    out_res_date_ns: np.ndarray,
    out_type: np.ndarray,
    out_detail: np.ndarray,
    out_severity: np.ndarray,
    out_channel: np.ndarray,
    out_status: np.ndarray,
    out_res_type: np.ndarray,
    out_resp_days: np.ndarray,
    key_offset: int,
) -> pa.Table:
    """Convert pre-allocated numpy arrays into a PyArrow table segment."""
    schema = _complaints_schema()
    n_rows = len(out_ckey)

    complaint_keys = np.arange(key_offset + 1, key_offset + n_rows + 1, dtype=np.int64)

    complaint_dates = out_date_ns.view("datetime64[ns]").astype("datetime64[ms]")

    so_mask = out_so == -1
    ln_mask = out_ln == -1

    res_date_mask = out_res_date_ns == -1
    res_dates_copy = out_res_date_ns.copy()
    res_dates_copy[res_date_mask] = 0
    res_dates_dt = res_dates_copy.view("datetime64[ns]").astype("datetime64[ms]")

    resp_days_mask = out_resp_days == -1

    so_pa = pa.array(out_so.tolist(), type=pa.int64(), mask=so_mask)
    ln_pa = pa.array(out_ln.tolist(), type=pa.int64(), mask=ln_mask)
    res_date_pa = pa.array(res_dates_dt.tolist(), type=pa.date32(), mask=res_date_mask)
    resp_days_pa = pa.array(out_resp_days.tolist(), type=pa.int32(), mask=resp_days_mask)

    return pa.table(
        [
            pa.array(complaint_keys, type=pa.int64()),
            pa.array(out_ckey, type=pa.int64()),
            so_pa,
            ln_pa,
            pa.array(complaint_dates, type=pa.date32()),
            res_date_pa,
            pa.array(out_type, type=pa.string()),
            pa.array(out_detail, type=pa.string()),
            pa.array(out_severity, type=pa.string()),
            pa.array(out_channel, type=pa.string()),
            pa.array(out_status, type=pa.string()),
            pa.array(out_res_type.tolist(), type=pa.string()),
            resp_days_pa,
        ],
        schema=schema,
    )


# ---------------------------------------------------------------------------
# Complaint generation — serial path (small datasets / fallback)
# ---------------------------------------------------------------------------

def _generate_complaints_serial(
    order_data: pd.DataFrame,
    c: _ComplaintsCfg,
    g_start: pd.Timestamp,
    g_end: pd.Timestamp,
) -> pa.Table:
    rng = np.random.default_rng(c.seed)

    g_start_ns = np.int64(g_start.value)
    g_end_ns = np.int64(g_end.value)

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
        complaints_per[repeat_mask] = rng.integers(
            2, c.max_complaints + 1, size=n_repeaters
        ).astype(np.int32)

    total_rows = int(complaints_per.sum())

    # Build per-customer order lookup (single groupby instead of per-key filter)
    _complainer_set = set(int(k) for k in complainer_keys)
    _complainer_mask = order_data["CustomerKey"].isin(_complainer_set)
    cust_orders: Dict[int, pd.DataFrame] = {
        int(ck): grp for ck, grp in order_data[_complainer_mask].groupby("CustomerKey")
    }

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
                # Explicitly set nullable fields to None (not uninitialized)
                out_res_type[row] = None

            row += 1

    return _arrays_to_table(
        out_ckey[:row],
        out_so[:row],
        out_ln[:row],
        out_date_ns[:row],
        out_res_date_ns[:row],
        out_type[:row],
        out_detail[:row],
        out_severity[:row],
        out_channel[:row],
        out_status[:row],
        out_res_type[:row],
        out_resp_days[:row],
        key_offset=0,
    )


# ---------------------------------------------------------------------------
# Complaint generation — parallel path (large datasets)
# ---------------------------------------------------------------------------

def _generate_complaints_parallel(
    order_data: pd.DataFrame,
    c: _ComplaintsCfg,
    g_start: pd.Timestamp,
    g_end: pd.Timestamp,
    n_workers: int,
) -> pa.Table:
    """Partition complainers across workers and merge results."""
    from src.facts.sales.sales_worker.pool import PoolRunSpec, iter_imap_unordered

    g_start_ns = np.int64(g_start.value)
    g_end_ns = np.int64(g_end.value)

    unique_customers = order_data["CustomerKey"].unique()
    n_customers = len(unique_customers)

    schema = _complaints_schema()
    if n_customers == 0:
        return pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema)

    # --- Setup phase (main process) ---
    # Use a seeded RNG for the setup (complainer selection + complaint counts)
    # so results are deterministic regardless of worker count.
    setup_rng = np.random.default_rng(c.seed)

    n_complainers = max(1, int(round(n_customers * c.complaint_rate)))
    complainer_keys = setup_rng.choice(unique_customers, size=n_complainers, replace=False)

    complaints_per = np.ones(n_complainers, dtype=np.int32)
    repeat_mask = setup_rng.random(n_complainers) < c.repeat_complaint_rate
    n_repeaters = int(repeat_mask.sum())
    if n_repeaters > 0:
        complaints_per[repeat_mask] = setup_rng.integers(
            2, c.max_complaints + 1, size=n_repeaters
        ).astype(np.int32)

    # Build per-customer order lookup filtered to complainers only
    _complainer_set = set(int(k) for k in complainer_keys)
    _complainer_mask = order_data["CustomerKey"].isin(_complainer_set)
    filtered = order_data[_complainer_mask]

    # Convert to numpy arrays for pickle-efficient IPC
    orders_ck = filtered["CustomerKey"].to_numpy(dtype=np.int64, copy=True)
    orders_so = filtered["SalesOrderNumber"].to_numpy(dtype=np.int64, copy=True)
    orders_ln = filtered["SalesOrderLineNumber"].to_numpy(dtype=np.int64, copy=True)

    # --- Partition complainers into chunks ---
    n_chunks = min(n_complainers, n_workers * 2)
    n_chunks = max(2, n_chunks)
    # Actual workers bounded by chunks
    actual_workers = min(n_chunks, n_workers)

    chunk_indices = np.array_split(np.arange(n_complainers), n_chunks)

    config_scalars = {
        "resolution_rate": c.resolution_rate,
        "escalation_rate": c.escalation_rate,
        "avg_response_days": c.avg_response_days,
        "max_response_days": c.max_response_days,
    }
    date_range = (int(g_start_ns), int(g_end_ns))

    tasks = []
    for chunk_idx, indices in enumerate(chunk_indices):
        if len(indices) == 0:
            continue
        chunk_keys = complainer_keys[indices].astype(np.int64)
        chunk_counts = complaints_per[indices]

        # Filter order arrays to only this chunk's customers (minimize pickle size)
        chunk_key_set = set(int(k) for k in chunk_keys)
        mask = np.isin(orders_ck, list(chunk_key_set))
        order_arrays_chunk = {
            "CustomerKey": orders_ck[mask],
            "SalesOrderNumber": orders_so[mask],
            "SalesOrderLineNumber": orders_ln[mask],
        }

        tasks.append((
            chunk_idx,
            c.seed,
            n_chunks,
            chunk_keys,
            chunk_counts,
            order_arrays_chunk,
            config_scalars,
            date_range,
        ))

    info(
        f"Complaints parallel: {len(tasks)} chunks across {actual_workers} workers "
        f"({n_complainers:,} complainers)"
    )

    pool_spec = PoolRunSpec(
        processes=actual_workers,
        chunksize=1,
        label="complaints",
    )

    # Collect results from workers (arrival order — keys are renumbered globally)
    all_arrays: Dict[str, List[np.ndarray]] = {
        k: [] for k in [
            "ckey", "so", "ln", "date_ns", "res_date_ns",
            "type", "detail", "severity", "channel", "status",
            "res_type", "resp_days",
        ]
    }

    for result in iter_imap_unordered(
        tasks=tasks,
        task_fn=_complaints_worker_task,
        spec=pool_spec,
    ):
        for col in all_arrays:
            all_arrays[col].append(result[col])

    # --- Merge phase ---
    # ComplaintKeys are assigned globally after concatenation.

    if not all_arrays["ckey"]:
        return pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema)

    merged: Dict[str, np.ndarray] = {
        col: np.concatenate(arrs) for col, arrs in all_arrays.items()
    }

    return _arrays_to_table(
        merged["ckey"],
        merged["so"],
        merged["ln"],
        merged["date_ns"],
        merged["res_date_ns"],
        merged["type"],
        merged["detail"],
        merged["severity"],
        merged["channel"],
        merged["status"],
        merged["res_type"],
        merged["resp_days"],
        key_offset=0,
    )


# ---------------------------------------------------------------------------
# Unified dispatcher
# ---------------------------------------------------------------------------

def _generate_complaints(
    order_data: pd.DataFrame,
    c: _ComplaintsCfg,
    g_start: pd.Timestamp,
    g_end: pd.Timestamp,
    workers: Optional[int] = None,
) -> pa.Table:
    """Generate the complaints table; dispatches to parallel or serial path."""
    unique_customers = order_data["CustomerKey"].unique()
    n_customers = len(unique_customers)
    n_complainers = max(1, int(round(n_customers * c.complaint_rate))) if n_customers > 0 else 0

    n_cpus = max(1, cpu_count() - 1)
    if workers is not None and workers >= 1:
        n_cpus = min(n_cpus, workers)

    use_parallel = n_complainers >= _PARALLEL_THRESHOLD and n_cpus >= 2

    if use_parallel:
        return _generate_complaints_parallel(
            order_data=order_data,
            c=c,
            g_start=g_start,
            g_end=g_end,
            n_workers=n_cpus,
        )
    else:
        return _generate_complaints_serial(
            order_data=order_data,
            c=c,
            g_start=g_start,
            g_end=g_end,
        )


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


def _write_complaints(table: pa.Table, complaints_dir: Path, file_format: str,
                      csv_chunk_size: Optional[int] = None) -> None:
    """Write complaints table in the requested format (parquet, csv, or delta)."""
    write_fact_table(table, complaints_dir, "complaints", file_format,
                     csv_prep_fn=_prepare_complaints_csv,
                     csv_chunk_size=csv_chunk_size)


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

    # Resolve worker count from config if available
    sales_cfg = getattr(cfg, "sales", None)
    workers: Optional[int] = int(getattr(sales_cfg, "workers", 0) or 0) or None

    table = _generate_complaints(
        order_data=order_data,
        c=c,
        g_start=g_start,
        g_end=g_end,
        workers=workers,
    )

    n_rows = table.num_rows
    if n_rows == 0:
        skip("Complaints: generated 0 rows; skipping write.")
        return None

    _csv_chunk = int(getattr(sales_cfg, "chunk_size", 0) or 0) if sales_cfg else 0
    _write_complaints(table, complaints_dir, file_format, csv_chunk_size=_csv_chunk)

    # For deltaparquet the delta table is written directly at complaints_dir;
    # the directory structure is the delta table itself — no cleanup needed.

    return {
        "complaints": str(complaints_dir),
        "complaints_rows": n_rows,
    }
