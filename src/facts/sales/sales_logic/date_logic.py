import numpy as np


def _stable_row_hash(order_dates: np.ndarray, product_keys: np.ndarray) -> np.ndarray:
    """
    Deterministic, vectorized hash for row-level mode (skip_order_cols=True).
    Avoids consuming RNG so results are stable even if pipeline call order changes.
    """
    # Convert dates to int days since epoch
    d = order_dates.astype("datetime64[D]").astype("int64", copy=False)
    p = product_keys.astype("int64", copy=False)

    # A simple mix (xorshift-like) to spread bits
    x = d * 0x9E3779B97F4A7C15
    x ^= (p + 0xBF58476D1CE4E5B9)
    x ^= (x >> 30)
    x *= 0x94D049BB133111EB
    x ^= (x >> 31)

    return (x & 0x7FFFFFFFFFFFFFFF).astype(np.int64, copy=False)


def compute_dates(rng, n, product_keys, order_ids_int, order_dates):
    """
    Compute due dates, delivery dates, delivery status, and order delay flag.

    Supports:
    - order_ids_int present  → order-level coherent behavior
    - order_ids_int is None → row-level fallback (skip_order_cols=True)

    Returns dict of numpy arrays:
      due_date: datetime64[D]
      delivery_date: datetime64[D]
      delivery_status: fixed-width unicode
      is_order_delayed: int8
    """
    if n <= 0:
        return {
            "due_date": np.empty(0, dtype="datetime64[D]"),
            "delivery_date": np.empty(0, dtype="datetime64[D]"),
            "delivery_status": np.empty(0, dtype="U15"),
            "is_order_delayed": np.empty(0, dtype=np.int8),
        }

    product_keys = np.asarray(product_keys)
    order_dates = np.asarray(order_dates).astype("datetime64[D]", copy=False)

    has_orders = order_ids_int is not None

    if has_orders:
        order_ids_int = np.asarray(order_ids_int, dtype=np.int64)

        # Map rows → order index
        unique_orders, inv_idx = np.unique(order_ids_int, return_inverse=True)
        order_hash = unique_orders.astype(np.int64, copy=False)

        # Expand order-level hashes to rows
        hash_vals = order_hash[inv_idx]
    else:
        # Deterministic per-row hash without consuming RNG
        hash_vals = _stable_row_hash(order_dates, product_keys)

    # ------------------------------------------------------------
    # Due dates: 3..7 days after order date
    # ------------------------------------------------------------
    due_offset = (hash_vals % 5) + 3
    due_date = order_dates + due_offset.astype("timedelta64[D]")

    # ------------------------------------------------------------
    # Seeds (vectorized)
    # ------------------------------------------------------------
    order_seed = hash_vals % 100
    product_seed = (hash_vals + product_keys.astype(np.int64, copy=False)) % 100
    line_seed = (product_keys.astype(np.int64, copy=False) + (hash_vals % 100)) % 100

    # ------------------------------------------------------------
    # Base delivery offset (relative to due date)
    # ------------------------------------------------------------
    delivery_offset = np.zeros(n, dtype=np.int64)

    # Condition C: small delay (1..4)
    mask_c = (order_seed >= 60) & (order_seed < 85) & (product_seed >= 60)
    delivery_offset[mask_c] = (line_seed[mask_c] % 4) + 1

    # Condition D: larger delay (2..6)
    mask_d = order_seed >= 85
    delivery_offset[mask_d] = (product_seed[mask_d] % 5) + 2

    # ------------------------------------------------------------
    # Early deliveries
    #   - Order-level when we have order ids
    #   - Row-level otherwise
    # ------------------------------------------------------------
    if has_orders:
        # One early flag per order (10%)
        early_order = rng.random(len(unique_orders)) < 0.10
        # Early days per order: 1..2
        early_days_per_order = rng.integers(1, 3, size=len(unique_orders), dtype=np.int64)
        early_days_rows = early_days_per_order[inv_idx]

        early_mask = early_order[inv_idx]
        if early_mask.any():
            # Early overrides delay (consistent with original behavior)
            delivery_offset[early_mask] = -early_days_rows[early_mask]
    else:
        early_mask = rng.random(n) < 0.10
        if early_mask.any():
            early_days = rng.integers(1, 3, size=n, dtype=np.int64)
            delivery_offset[early_mask] = -early_days[early_mask]

    # Final delivery date
    delivery_date = due_date + delivery_offset.astype("timedelta64[D]")

    # ------------------------------------------------------------
    # Delivery status
    # ------------------------------------------------------------
    delivery_status = np.full(n, "On Time", dtype="U15")
    delivery_status[delivery_date < due_date] = "Early Delivery"
    delivery_status[delivery_date > due_date] = "Delayed"

    # ------------------------------------------------------------
    # Order delayed flag
    # ------------------------------------------------------------
    if has_orders:
        delayed_line = (delivery_status == "Delayed")

        # Any delayed line → order delayed (order-level coherence)
        delayed_any = np.bincount(inv_idx, weights=delayed_line.astype(np.int8)).astype(bool)
        is_order_delayed = delayed_any[inv_idx].astype(np.int8)
    else:
        is_order_delayed = (delivery_status == "Delayed").astype(np.int8)

    return {
        "due_date": due_date,
        "delivery_date": delivery_date,
        "delivery_status": delivery_status,
        "is_order_delayed": is_order_delayed,
    }
