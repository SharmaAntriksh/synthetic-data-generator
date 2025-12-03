import numpy as np

def apply_promotions(
    rng, n, order_dates,
    promo_keys_all, promo_pct_all,
    promo_start_all, promo_end_all,
    no_discount_key=1
):
    promo_keys = np.full(n, no_discount_key, dtype=np.int64)
    promo_pct = np.zeros(n, dtype=np.float64)

    if promo_keys_all is None or promo_keys_all.size == 0:
        return promo_keys, promo_pct

    od_exp = order_dates[:, None]
    start_ok = od_exp >= promo_start_all
    end_ok = od_exp <= promo_end_all
    mask_all = start_ok & end_ok

    for i in range(n):
        active = np.where(mask_all[i])[0]
        if active.size == 0:
            continue

        actual_promos = [j for j in active if promo_keys_all[j] != no_discount_key]

        if actual_promos:
            idx = rng.choice(actual_promos)
            promo_keys[i] = promo_keys_all[idx]
            promo_pct[i] = promo_pct_all[idx]

    return promo_keys, promo_pct
