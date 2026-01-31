import numpy as np


def apply_promotions(
    rng, n, order_dates,
    promo_keys_all, promo_pct_all,
    promo_start_all, promo_end_all,
    no_discount_key=1,
    promo_intensity=None,
):
    promo_keys = np.full(n, no_discount_key, dtype=np.int64)
    promo_pct = np.zeros(n, dtype=np.float64)

    if promo_keys_all is None or promo_keys_all.size == 0:
        return promo_keys, promo_pct

    # ------------------------------------------------------------
    # Active promotion mask (rows × promos)
    # ------------------------------------------------------------
    od = order_dates[:, None]
    active_mask = (od >= promo_start_all) & (od <= promo_end_all)

    # Exclude no-discount promos
    valid_mask = active_mask & (promo_keys_all != no_discount_key)

    row_has_promo = valid_mask.any(axis=1)
    rows = np.nonzero(row_has_promo)[0]

    if rows.size == 0:
        return promo_keys, promo_pct

    # ------------------------------------------------------------
    # Vectorized per-row random choice
    # ------------------------------------------------------------
    # Generate random noise only where promos are valid
    noise = rng.random(valid_mask.shape)

    if promo_intensity is not None:
        promo_intensity = np.asarray(promo_intensity, dtype=np.float64)
        promo_intensity = np.clip(promo_intensity, 0.0, 1.0)

        # Higher intensity → higher chance to win
        # Shape: (rows, 1) broadcast across promos
        noise = noise * promo_intensity[:, None]

    noise[~valid_mask] = -1.0

    # Pick the promo with max noise per row
    chosen_idx = noise.argmax(axis=1)

    # Assign only rows that had at least one promo
    promo_keys[rows] = promo_keys_all[chosen_idx[rows]]
    
    if promo_intensity is not None:
        promo_pct[rows] = (
            promo_pct_all[chosen_idx[rows]] * promo_intensity[rows]
        )
    else:
        promo_pct[rows] = promo_pct_all[chosen_idx[rows]]

    return promo_keys, promo_pct
