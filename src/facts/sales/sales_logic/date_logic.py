import numpy as np

def compute_dates(rng, n, product_keys, order_ids_int, order_dates):
    hash_vals = order_ids_int

    # Due dates
    due_offset = (hash_vals % 5).astype(np.int64) + 3
    due_date = order_dates + due_offset.astype("timedelta64[D]")

    # Seeds
    line_seed = (product_keys + (hash_vals % 100)) % 100
    order_seed = hash_vals % 100
    product_seed = (hash_vals + product_keys) % 100

    base_offset = np.zeros(n, dtype=np.int64)

    mask_c = (60 <= order_seed) & (order_seed < 85) & (product_seed >= 60)
    base_offset[mask_c] = (line_seed[mask_c] % 4) + 1

    mask_d = order_seed >= 85
    base_offset[mask_d] = (product_seed[mask_d] % 5) + 2

    early_mask = rng.random(n) < 0.10
    early_days = rng.integers(1, 3, n)

    delivery_offset = base_offset.copy()
    delivery_offset[early_mask] = -early_days[early_mask]

    delivery_date = due_date + delivery_offset.astype("timedelta64[D]")

    # Status
    delivery_status = np.full(n, "On Time", dtype="U15")
    delivery_status[delivery_date < due_date] = "Early Delivery"
    delivery_status[delivery_date > due_date] = "Delayed"

    # Order delay: any line in order delayed?
    delayed_line = (delivery_status == "Delayed")
    _, inv_idx = np.unique(order_ids_int, return_inverse=True)
    delayed_any = np.bincount(inv_idx, weights=delayed_line).astype(bool)
    is_order_delayed = delayed_any[inv_idx].astype(np.int8)

    return {
        "due_date": due_date,
        "delivery_date": delivery_date,
        "delivery_status": delivery_status,
        "is_order_delayed": is_order_delayed
    }
