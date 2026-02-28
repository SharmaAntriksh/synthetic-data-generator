from __future__ import annotations

import os
from typing import Any, List, Optional

import numpy as np

from ..output_paths import (
    OutputPaths,
    TABLE_SALES,
    TABLE_SALES_ORDER_DETAIL,
    TABLE_SALES_ORDER_HEADER,
)
from ..sales_logic import bind_globals
from .schemas import build_worker_schemas
from ..output_paths import TABLE_SALES_RETURN


EMPLOYEE_KEY_MIN_NON_MANAGER = 40_000_000

# ---------------------------------------------------------------------
# Small helpers (kept close to monolith behavior for compatibility)
# ---------------------------------------------------------------------


def build_buckets_from_key(
    key: Any,
    *,
    max_key: Optional[int] = None,
    max_buckets: int = 1_000_000,
) -> list[np.ndarray]:
    """
    Build index buckets for non-negative integer keys.

    Returns: buckets[k] = row indices where key == k.

    Safety:
      - By default, refuses to allocate more than `max_buckets` buckets to avoid OOM
        if called with raw, high-magnitude surrogate keys (e.g., EmployeeKey ~ 40,000,000+).
      - Override via `max_key=` or `max_buckets=` if you *intend* to allocate a large table.
    """
    key_np = np.asarray(key, dtype=np.int64)
    if key_np.size == 0:
        return []
    if key_np.min() < 0:
        raise RuntimeError("Key values must be non-negative integers")

    max_k = int(key_np.max()) if max_key is None else int(max_key)
    K = max_k + 1
    if K > int(max_buckets):
        raise RuntimeError(
            f"Refusing to allocate {K:,} buckets (max_key={max_k:,}); "
            f"set max_key/max_buckets explicitly if this is intended."
        )

    order = np.argsort(key_np, kind="mergesort")
    k_sorted = key_np[order]
    starts = np.flatnonzero(np.r_[True, k_sorted[1:] != k_sorted[:-1]])
    ends = np.r_[starts[1:], k_sorted.size]

    buckets: list[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(K)]
    for s, e in zip(starts, ends):
        k = int(k_sorted[int(s)])
        buckets[k] = order[int(s) : int(e)].astype(np.int64, copy=False)
    return buckets


def int_or(v: Any, default: int) -> int:
    try:
        if v is None or v == "":
            return int(default)
        return int(v)
    except Exception:
        return int(default)


def float_or(v: Any, default: float) -> float:
    try:
        if v is None or v == "":
            return float(default)
        return float(v)
    except Exception:
        return float(default)


def str_or(v: Any, default: str) -> str:
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def as_int64(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.int64)


def as_f64(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


def dense_map(mapping: Optional[dict]) -> Optional[np.ndarray]:
    if not mapping:
        return None
    keys = np.fromiter((int(k) for k in mapping.keys()), dtype=np.int64)
    vals = np.fromiter((int(v) for v in mapping.values()), dtype=np.int64)
    if keys.size == 0:
        return None
    max_key = int(keys.max())
    if max_key < 0:
        return None
    arr = np.full(max_key + 1, -1, dtype=np.int64)
    arr[keys] = vals
    return arr


def infer_T_from_date_pool(date_pool: Any) -> int:
    dp = np.asarray(date_pool, dtype="datetime64[D]")
    return int(np.unique(dp.astype("datetime64[M]")).size)


# back-compat aliases (keep older imports stable)
_build_buckets_from_brand_key = lambda brand_key: build_buckets_from_key(brand_key, max_key=int(np.asarray(brand_key, dtype=np.int64).max()) if np.asarray(brand_key).size else None)

# Prefer direct names in new code.
_int_or = int_or
_float_or = float_or
_str_or = str_or
_as_int64 = as_int64
_as_f64 = as_f64
_dense_map = dense_map
_infer_T_from_date_pool = infer_T_from_date_pool


# ---------------------------------------------------------------------
# Employee salesperson assignment

# Canonical: day-accurate effective-dated bridge index
# ---------------------------------------------------------------------


def _normalize_assignment_arrays(
    *,
    store_keys: np.ndarray,
    assign_store: Any,
    assign_emp: Any,
    assign_start: Any,
    assign_end: Any,
    assign_fte: Any = None,
    assign_is_primary: Any = None,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]]:
    """
    Normalize and validate employee assignment arrays.

    Returns:
      (assign_store_i64, assign_emp_i64, start_D, end_D, fte_f64, is_primary_bool, max_store_key)

    Notes:
      - NaT start -> FAR_PAST
      - NaT end   -> FAR_FUTURE
      - Filters invalid store keys outside [0, max_store_key]
      - Filters invalid windows where start > end
      - Keeps fte/is_primary aligned with filtered rows
    """
    if assign_store is None or assign_emp is None or assign_start is None or assign_end is None:
        return None

    store_keys = np.asarray(store_keys, dtype=np.int64)
    if store_keys.size == 0:
        return None
    max_store_key = int(store_keys.max())

    a_store = np.asarray(assign_store, dtype=np.int64)
    a_emp = np.asarray(assign_emp, dtype=np.int64)
    if a_store.size == 0 or a_emp.size == 0:
        return None
    if a_store.shape[0] != a_emp.shape[0]:
        raise RuntimeError("employee assignment arrays must align (StoreKey vs EmployeeKey)")

    start_raw = np.asarray(assign_start, dtype="datetime64[D]")
    end_raw = np.asarray(assign_end, dtype="datetime64[D]")
    if start_raw.shape[0] != a_store.shape[0] or end_raw.shape[0] != a_store.shape[0]:
        raise RuntimeError("employee assignment arrays must align (dates vs keys)")

    # Optional aligned arrays
    fte = np.ones(a_store.shape[0], dtype=np.float64) if assign_fte is None else np.asarray(assign_fte, dtype=np.float64)
    if fte.shape[0] != a_store.shape[0]:
        raise RuntimeError("employee_assign_fte must align with employee assignments")

    is_primary = (
        np.zeros(a_store.shape[0], dtype=bool)
        if assign_is_primary is None
        else np.asarray(assign_is_primary, dtype=bool)
    )
    if is_primary.shape[0] != a_store.shape[0]:
        raise RuntimeError("employee_assign_is_primary must align with employee assignments")

    FAR_FUTURE = np.datetime64("2262-04-11", "D")
    FAR_PAST = np.datetime64("1900-01-01", "D")

    start_fixed = start_raw.copy()
    end_fixed = end_raw.copy()

    nat_s = np.isnat(start_fixed)
    if nat_s.any():
        start_fixed[nat_s] = FAR_PAST

    nat_e = np.isnat(end_fixed)
    if nat_e.any():
        end_fixed[nat_e] = FAR_FUTURE

    # Filter store key validity
    valid_store = (a_store >= 0) & (a_store <= max_store_key)
    if not valid_store.all():
        a_store = a_store[valid_store]
        a_emp = a_emp[valid_store]
        start_fixed = start_fixed[valid_store]
        end_fixed = end_fixed[valid_store]
        fte = fte[valid_store]
        is_primary = is_primary[valid_store]
        if a_store.size == 0:
            return None

    # Filter window validity
    ok_window = start_fixed <= end_fixed
    if not ok_window.all():
        a_store = a_store[ok_window]
        a_emp = a_emp[ok_window]
        start_fixed = start_fixed[ok_window]
        end_fixed = end_fixed[ok_window]
        fte = fte[ok_window]
        is_primary = is_primary[ok_window]
        if a_store.size == 0:
            return None

    return a_store, a_emp, start_fixed, end_fixed, fte, is_primary, max_store_key


def _build_salesperson_effective_by_store(
    *,
    store_keys: np.ndarray,
    assign_store: Any,
    assign_emp: Any,
    assign_start: Any,
    assign_end: Any,
    assign_fte: Any = None,
    assign_is_primary: Any = None,
    primary_boost: float = 2.0,
) -> Optional[dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]]:
    """
    Canonical structure for Sales employee assignment.

    Returns:
      { store_key: (emp_keys, start_dates[D], end_dates[D], weights[f64]) }

    - Enforces StartDate/EndDate at DAY granularity.
    - Treats missing EndDate as open-ended.
    - Treats missing StartDate as FAR_PAST (always eligible until EndDate).
    """
    norm = _normalize_assignment_arrays(
        store_keys=store_keys,
        assign_store=assign_store,
        assign_emp=assign_emp,
        assign_start=assign_start,
        assign_end=assign_end,
        assign_fte=assign_fte,
        assign_is_primary=assign_is_primary,
    )
    if norm is None:
        return None

    a_store, a_emp, start_fixed, end_fixed, fte, is_primary, max_store_key = norm

    weights = fte * np.where(is_primary, float(primary_boost), 1.0)

    order = np.argsort(a_store, kind="mergesort")
    s_sorted = a_store[order]
    starts = np.flatnonzero(np.r_[True, s_sorted[1:] != s_sorted[:-1]])
    ends = np.r_[starts[1:], s_sorted.size]

    out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for s, e in zip(starts, ends):
        idx = order[int(s) : int(e)]
        store = int(s_sorted[int(s)])
        if store < 0 or store > max_store_key:
            continue
        out[store] = (
            a_emp[idx].astype(np.int64, copy=False),
            start_fixed[idx].astype("datetime64[D]", copy=False),
            end_fixed[idx].astype("datetime64[D]", copy=False),
            weights[idx].astype(np.float64, copy=False),
        )

    return out


# ---------------------------------------------------------------------
# Optional legacy: month-rounded single salesperson per store-month
# (kept only for backward-compat; OFF by default)
# ---------------------------------------------------------------------


def _build_salesperson_by_store_month(
    *,
    store_keys: np.ndarray,
    date_pool: Any,
    assign_store: Any,
    assign_emp: Any,
    assign_start: Any,
    assign_end: Any,
    assign_fte: Any = None,
    assign_is_primary: Any = None,
    primary_boost: float = 2.0,
    seed: int = 12345,
) -> Optional[np.ndarray]:
    """
    Legacy: choose a single salesperson per store-month.

    This is a *month-rounded* view used only for backward compatibility.
    Uses an event-sweep per store to avoid O(assignments × months) nested loops.
    """
    store_keys = np.asarray(store_keys, dtype=np.int64)
    if store_keys.size == 0:
        return None

    dp = np.asarray(date_pool, dtype="datetime64[D]")
    if dp.size == 0:
        return None

    months_int = dp.astype("datetime64[M]").astype(np.int64)
    month0_int = int(months_int.min())
    T = int(np.unique(months_int).size)
    if T <= 0:
        return None

    norm = _normalize_assignment_arrays(
        store_keys=store_keys,
        assign_store=assign_store,
        assign_emp=assign_emp,
        assign_start=assign_start,
        assign_end=assign_end,
        assign_fte=assign_fte,
        assign_is_primary=assign_is_primary,
    )
    if norm is None:
        return None

    a_store, a_emp, start_fixed, end_fixed, fte, is_primary, max_store_key = norm

    weights = fte * np.where(is_primary, float(primary_boost), 1.0)

    start_off = start_fixed.astype("datetime64[M]").astype(np.int64) - month0_int
    end_off = end_fixed.astype("datetime64[M]").astype(np.int64) - month0_int
    start_off = np.clip(start_off, 0, T - 1).astype(np.int64, copy=False)
    end_off = np.clip(end_off, 0, T - 1).astype(np.int64, copy=False)

    out = np.full((max_store_key + 1, T), -1, dtype=np.int64)

    order = np.argsort(a_store, kind="mergesort")
    store_sorted = a_store[order]
    starts = np.flatnonzero(np.r_[True, store_sorted[1:] != store_sorted[:-1]])
    ends = np.r_[starts[1:], store_sorted.size]

    rng = np.random.default_rng(int(seed))
    for s, e in zip(starts, ends):
        store = int(store_sorted[int(s)])
        if store < 0 or store > max_store_key:
            continue

        idxs = order[int(s) : int(e)]
        if idxs.size == 0:
            continue

        # month -> list of assignment row indices starting/ending
        starts_ev: list[list[int]] = [[] for _ in range(T)]
        ends_ev: list[list[int]] = [[] for _ in range(T + 1)]  # end+1

        for ii in idxs:
            so = int(start_off[int(ii)])
            eo = int(end_off[int(ii)])
            if eo < so:
                continue
            starts_ev[so].append(int(ii))
            ends_ev[eo + 1].append(int(ii))

        active: set[int] = set()
        for m in range(T):
            for ii in starts_ev[m]:
                active.add(ii)
            for ii in ends_ev[m]:
                active.discard(ii)

            if not active:
                continue

            cand = np.fromiter(active, dtype=np.int64)
            w = weights[cand]
            sw = float(w.sum())
            pick = 0 if sw <= 1e-12 else int(rng.choice(cand.size, p=(w / sw)))
            out[store, m] = int(a_emp[int(cand[pick])])

    return out


# ---------------------------------------------------------------------
# Brand popularity model helper (unchanged)
# ---------------------------------------------------------------------


def _build_brand_prob_by_month_rotate_winner(
    rng: np.random.Generator,
    *,
    T: int,
    B: int,
    winner_boost: float = 2.5,
    noise_sd: float = 0.15,
    min_share: float = 0.02,
    year_len_months: int = 12,
) -> np.ndarray:
    if T <= 0 or B <= 0:
        raise RuntimeError(f"Invalid T/B for brand_prob_by_month: T={T}, B={B}")

    year_len = max(1, int(year_len_months))
    base = np.ones(B, dtype=np.float64) / float(B)
    out = np.empty((T, B), dtype=np.float64)

    for t in range(T):
        winner = (t // year_len) % B
        v = base.copy()
        v[winner] *= float(winner_boost)
        if noise_sd > 0:
            v *= np.exp(rng.normal(loc=0.0, scale=float(noise_sd), size=B))
        if min_share > 0:
            v = np.maximum(v, float(min_share))
        s = float(v.sum())
        out[t] = (v / s) if s > 0 else base
    return out


# ---------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------


def init_sales_worker(worker_cfg: dict) -> None:
    try:
        product_np = worker_cfg["product_np"]
        product_brand_key = worker_cfg.get("product_brand_key")
        store_keys = worker_cfg["store_keys"]

        store_to_geo = worker_cfg.get("store_to_geo")
        geo_to_currency = worker_cfg.get("geo_to_currency")

        promo_keys_all = worker_cfg["promo_keys_all"]
        promo_pct_all = worker_cfg["promo_pct_all"]
        promo_start_all = worker_cfg["promo_start_all"]
        promo_end_all = worker_cfg["promo_end_all"]

        customer_keys = worker_cfg["customer_keys"]
        customer_is_active_in_sales = worker_cfg.get("customer_is_active_in_sales")
        customer_start_month = worker_cfg.get("customer_start_month")
        customer_end_month = worker_cfg.get("customer_end_month")
        customer_base_weight = worker_cfg.get("customer_base_weight")

        date_pool = worker_cfg["date_pool"]
        date_prob = worker_cfg["date_prob"]

        # employee store assignments (bridge)
        employee_assign_store_key = worker_cfg.get("employee_assign_store_key")
        employee_assign_employee_key = worker_cfg.get("employee_assign_employee_key")
        employee_assign_start_date = worker_cfg.get("employee_assign_start_date")
        employee_assign_end_date = worker_cfg.get("employee_assign_end_date")
        employee_assign_fte = worker_cfg.get("employee_assign_fte")
        employee_assign_is_primary = worker_cfg.get("employee_assign_is_primary")
        employee_primary_boost = float(worker_cfg.get("employee_primary_boost", 2.0))
        employee_seed = int(worker_cfg.get("employee_salesperson_seed", worker_cfg.get("seed_master", 12345)))
        employee_assign_role = worker_cfg.get("employee_assign_role")
        salesperson_roles = worker_cfg.get("salesperson_roles", ["Sales Associate"])

        op = worker_cfg["output_paths"]
        if not isinstance(op, dict):
            raise RuntimeError("output_paths must be a dict")

        file_format = worker_cfg.get("file_format") or op.get("file_format")
        out_folder = worker_cfg.get("out_folder") or op.get("out_folder")
        if not file_format:
            raise RuntimeError("file_format is required (worker_cfg.file_format or output_paths.file_format)")
        if not out_folder:
            raise RuntimeError("out_folder is required (worker_cfg.out_folder or output_paths.out_folder)")

        row_group_size = int(worker_cfg.get("row_group_size", 2_000_000))
        compression = str(worker_cfg.get("compression", "snappy"))

        # ------------------------------------------------------------
        # CRITICAL: SalesOrderNumber uniqueness depends on a CONSTANT
        # per-run stride for chunk order-id ranges (NOT per-task batch size).
        # Prefer 'order_id_stride_orders'; fall back to 'chunk_size'.
        # ------------------------------------------------------------
        stride_raw = worker_cfg.get("order_id_stride_orders", None)
        if stride_raw is None:
            stride_raw = worker_cfg.get("chunk_size", None)

        chunk_size = int(stride_raw or 0)
        if chunk_size <= 0:
            raise RuntimeError(
                "Missing/invalid order-id stride. Set worker_cfg.order_id_stride_orders (preferred) "
                "or worker_cfg.chunk_size to a positive int. This value partitions SalesOrderNumber "
                "space across chunks and must be constant across the run."
            )

        # ------------------------------------------------------------
        # Per-run id for SalesOrderNumber (0..999).
        # If caller doesn't provide it, derive deterministically from seed_master.
        # ------------------------------------------------------------
        run_id_raw = worker_cfg.get("order_id_run_id", None)
        if run_id_raw is None:
            seed_master = int(worker_cfg.get("seed_master", 0) or 0)
            order_id_run_id = int(seed_master % 1000)
        else:
            order_id_run_id = int(run_id_raw)

        if order_id_run_id < 0 or order_id_run_id > 999:
            raise RuntimeError(f"order_id_run_id must be in [0,999], got {order_id_run_id}")

        no_discount_key = worker_cfg["no_discount_key"]
        delta_output_folder = worker_cfg.get("delta_output_folder") or op.get("delta_output_folder")
        merged_file = worker_cfg.get("merged_file") or op.get("merged_file")
        write_delta = worker_cfg.get("write_delta", False)

        skip_order_cols = worker_cfg["skip_order_cols"]
        sales_output = str_or(worker_cfg.get("sales_output"), "sales").lower()
        if sales_output not in {"sales", "sales_order", "both"}:
            raise RuntimeError(f"Invalid sales_output: {sales_output}")

        skip_order_cols_requested = bool(worker_cfg.get("skip_order_cols_requested", skip_order_cols))

        returns_enabled = bool(worker_cfg.get("returns_enabled", False))
        returns_rate = float(worker_cfg.get("returns_rate", 0.0))
        returns_min_lag_days = int(worker_cfg.get("returns_min_lag_days", 0))
        returns_max_lag_days = int(worker_cfg.get("returns_max_lag_days", 60))
        if returns_min_lag_days < 0:
            returns_min_lag_days = 0
        if returns_max_lag_days < 0:
            returns_max_lag_days = 0
        if returns_min_lag_days > returns_max_lag_days:
            # Clamp defensively to avoid crashing the pool on bad configs.
            returns_min_lag_days = returns_max_lag_days
        returns_reason_keys = worker_cfg.get("returns_reason_keys")
        returns_reason_probs = worker_cfg.get("returns_reason_probs")

        if sales_output in {"sales_order", "both"}:
            skip_order_cols = False

        partition_enabled = bool(worker_cfg.get("partition_enabled", False))
        partition_cols = worker_cfg.get("partition_cols") or []
        models_cfg = worker_cfg.get("models_cfg")

        parquet_dict_exclude = worker_cfg.get("parquet_dict_exclude")
        write_pyarrow = worker_cfg.get("write_pyarrow", True)

        # NEW: configurable cap for SalesOrderLineNumber per SalesOrderNumber
        max_lines_per_order = int_or(worker_cfg.get("max_lines_per_order"), 5)
        if max_lines_per_order < 1:
            raise RuntimeError(f"max_lines_per_order must be >= 1, got {max_lines_per_order}")

        legacy_salesperson_by_store_month = bool(worker_cfg.get("legacy_salesperson_by_store_month", False))

    except KeyError as e:
        raise RuntimeError(f"Missing worker_cfg key: {e}") from e

    product_np = np.asarray(product_np)

    brand_to_row_idx = None
    if product_brand_key is not None:
        product_brand_key = as_int64(product_brand_key)
        if product_brand_key.shape[0] != product_np.shape[0]:
            raise RuntimeError("product_brand_key must align with product_np row count")
        brand_to_row_idx = _build_buckets_from_brand_key(product_brand_key)

    store_keys = as_int64(store_keys)

    # ------------------------------------------------------------
    # Filter employee assignment rows to sales-eligible roles
    # ------------------------------------------------------------
    if employee_assign_employee_key is not None and employee_assign_store_key is not None:
        emp_key = np.asarray(employee_assign_employee_key, dtype=np.int64)

        if employee_assign_role is not None:
            role_arr = np.asarray(employee_assign_role).astype(str)
            allowed = np.asarray(list(salesperson_roles), dtype=str)
            mask = np.isin(role_arr, allowed)
        else:
            # fallback if role not provided: at least exclude Store Manager key range (30M..40M)
            mask = emp_key >= EMPLOYEE_KEY_MIN_NON_MANAGER

        employee_assign_store_key = np.asarray(employee_assign_store_key, dtype=np.int64)[mask]
        employee_assign_employee_key = emp_key[mask]
        employee_assign_start_date = np.asarray(employee_assign_start_date, dtype="datetime64[D]")[mask]
        employee_assign_end_date = np.asarray(employee_assign_end_date, dtype="datetime64[D]")[mask]

        if employee_assign_fte is not None:
            employee_assign_fte = np.asarray(employee_assign_fte, dtype=np.float64)[mask]
        if employee_assign_is_primary is not None:
            employee_assign_is_primary = np.asarray(employee_assign_is_primary, dtype=bool)[mask]

        salesperson_global_pool = np.unique(employee_assign_employee_key).astype(np.int64, copy=False)
    else:
        salesperson_global_pool = None

    salesperson_effective_by_store = _build_salesperson_effective_by_store(
        store_keys=store_keys,
        assign_store=employee_assign_store_key,
        assign_emp=employee_assign_employee_key,
        assign_start=employee_assign_start_date,
        assign_end=employee_assign_end_date,
        assign_fte=employee_assign_fte,
        assign_is_primary=employee_assign_is_primary,
        primary_boost=employee_primary_boost,
    )

    salesperson_by_store_month = None
    if legacy_salesperson_by_store_month:
        salesperson_by_store_month = _build_salesperson_by_store_month(
            store_keys=store_keys,
            date_pool=date_pool,
            assign_store=employee_assign_store_key,
            assign_emp=employee_assign_employee_key,
            assign_start=employee_assign_start_date,
            assign_end=employee_assign_end_date,
            assign_fte=employee_assign_fte,
            assign_is_primary=employee_assign_is_primary,
            primary_boost=employee_primary_boost,
            seed=employee_seed,
        )

    promo_keys_all = as_int64(promo_keys_all)
    promo_pct_all = as_f64(promo_pct_all)
    promo_start_all = np.asarray(promo_start_all, dtype="datetime64[D]")
    promo_end_all = np.asarray(promo_end_all, dtype="datetime64[D]")

    customer_keys = as_int64(customer_keys)

    if customer_is_active_in_sales is not None:
        customer_is_active_in_sales = as_int64(customer_is_active_in_sales)
        if customer_is_active_in_sales.shape[0] != customer_keys.shape[0]:
            raise RuntimeError("customer_is_active_in_sales must align with customer_keys length")

    if customer_start_month is not None:
        customer_start_month = as_int64(customer_start_month)
        if customer_start_month.shape[0] != customer_keys.shape[0]:
            raise RuntimeError("customer_start_month must align with customer_keys length")

    if customer_end_month is not None:
        customer_end_month = as_int64(customer_end_month)
        if customer_end_month.shape[0] != customer_keys.shape[0]:
            raise RuntimeError("customer_end_month must align with customer_keys length")

    if customer_base_weight is not None:
        customer_base_weight = np.asarray(customer_base_weight, dtype=np.float64)
        if customer_base_weight.shape[0] != customer_keys.shape[0]:
            raise RuntimeError("customer_base_weight must align with customer_keys length")

    brand_prob_by_month = None
    if isinstance(models_cfg, dict):
        models_root = models_cfg.get("models") if isinstance(models_cfg.get("models"), dict) else models_cfg
        brand_cfg = models_root.get("brand_popularity") if isinstance(models_root, dict) else None
        if brand_cfg:
            T = infer_T_from_date_pool(date_pool)
            B = int(product_brand_key.max()) + 1 if product_brand_key is not None and product_brand_key.size else 0
            rng_bp = np.random.default_rng(int(int_or(brand_cfg.get("seed"), 1234)))
            brand_prob_by_month = _build_brand_prob_by_month_rotate_winner(
                rng_bp,
                T=T,
                B=B,
                winner_boost=float_or(brand_cfg.get("winner_boost"), 2.5),
                noise_sd=float_or(brand_cfg.get("noise_sd"), 0.15),
                min_share=float_or(brand_cfg.get("min_share"), 0.02),
                year_len_months=int_or(brand_cfg.get("year_len_months"), 12),
            )

    store_to_geo_arr = dense_map(store_to_geo) if isinstance(store_to_geo, dict) else None
    geo_to_currency_arr = dense_map(geo_to_currency) if isinstance(geo_to_currency, dict) else None

    output_paths = OutputPaths(
        out_folder=out_folder,
        delta_output_folder=delta_output_folder,
        file_format=file_format,
        merged_file=merged_file,
    )

    tables: list[str] = []
    if sales_output in {"sales", "both"}:
        tables.append(TABLE_SALES)
    if sales_output in {"sales_order", "both"}:
        tables += [TABLE_SALES_ORDER_DETAIL, TABLE_SALES_ORDER_HEADER]
    if returns_enabled:
        tables.append(TABLE_SALES_RETURN)
    for t in tables:
        output_paths.ensure_dirs(t)

    bundle = build_worker_schemas(
        file_format=file_format,
        skip_order_cols=bool(skip_order_cols),
        skip_order_cols_requested=bool(skip_order_cols_requested),
        returns_enabled=bool(returns_enabled),
        parquet_dict_exclude=set(parquet_dict_exclude) if parquet_dict_exclude else None,
        models_cfg=models_cfg if isinstance(models_cfg, dict) else None,
    )

    bind_globals(
        {
            "product_np": product_np,
            "brand_to_row_idx": brand_to_row_idx,
            "product_brand_key": product_brand_key,
            "brand_prob_by_month": brand_prob_by_month,
            "store_keys": store_keys,
            "promo_keys_all": promo_keys_all,
            "promo_pct_all": promo_pct_all,
            "promo_start_all": promo_start_all,
            "promo_end_all": promo_end_all,
            "customer_keys": customer_keys,
            "customer_is_active_in_sales": customer_is_active_in_sales,
            "customer_start_month": customer_start_month,
            "customer_end_month": customer_end_month,
            "customer_base_weight": customer_base_weight,
            "store_to_geo_arr": store_to_geo_arr,
            "geo_to_currency_arr": geo_to_currency_arr,
            "date_pool": date_pool,
            "date_prob": date_prob,
            "file_format": file_format,
            "out_folder": out_folder,

            # CRITICAL: constant per-run stride used to partition SalesOrderNumber ranges
            "chunk_size": int(max(1, chunk_size)),
            "order_id_stride_orders": int(max(1, chunk_size)),
            "order_id_run_id": int(order_id_run_id),
            "max_lines_per_order": int(max_lines_per_order),
            
            "row_group_size": int(max(1, row_group_size)),
            "compression": compression,
            "output_paths": output_paths,
            "sales_output": sales_output,
            "skip_order_cols_requested": bool(skip_order_cols_requested),
            "skip_order_cols": bool(skip_order_cols),
            "delta_output_folder": os.path.normpath(delta_output_folder) if delta_output_folder else None,
            "write_delta": bool(write_delta),
            "no_discount_key": no_discount_key,
            "partition_enabled": bool(partition_enabled),
            "partition_cols": list(partition_cols),
            "models_cfg": models_cfg,
            "write_pyarrow": bool(write_pyarrow),
            "date_cols_by_table": bundle.date_cols_by_table,
            "schema_no_order": bundle.schema_no_order,
            "schema_with_order": bundle.schema_with_order,
            "schema_no_order_delta": bundle.schema_no_order_delta,
            "schema_with_order_delta": bundle.schema_with_order_delta,
            "sales_schema": bundle.sales_schema_gen,
            "sales_schema_out": bundle.sales_schema_out,
            "schema_by_table": bundle.schema_by_table,
            "parquet_dict_cols_by_table": bundle.parquet_dict_cols_by_table,
            "parquet_dict_exclude": bundle.parquet_dict_exclude,
            "parquet_dict_cols": bundle.parquet_dict_cols,
            "returns_enabled": bool(returns_enabled),
            "returns_rate": float(returns_rate),
            "returns_min_lag_days": int(max(0, returns_min_lag_days)),
            "returns_max_lag_days": int(returns_max_lag_days),
            "returns_reason_keys": returns_reason_keys,
            "returns_reason_probs": returns_reason_probs,
            # EMPLOYEE assignment (canonical + optional legacy)
            "salesperson_effective_by_store": salesperson_effective_by_store,
            "salesperson_by_store_month": salesperson_by_store_month,
            "salesperson_global_pool": salesperson_global_pool,
        }
    )