import numpy as np


def _as_date32(x):
    """
    Normalize incoming date arrays to datetime64[D] for consistent comparisons.
    """
    if x is None:
        return None
    a = np.asarray(x)
    if a.size == 0:
        return a
    # ensure day resolution
    return a.astype("datetime64[D]", copy=False)


def _safe_clip_pct(p):
    p = np.asarray(p, dtype=np.float64)
    p = np.where(np.isfinite(p), p, 0.0)
    return np.clip(p, 0.0, 1.0)


def apply_promotions(
    rng,
    n,
    order_dates,
    promo_keys_all,
    promo_pct_all,
    promo_start_all,
    promo_end_all,
    no_discount_key=1,
    promo_weight_all=None,
):
    """
    Assign at most one promotion per row.

    Default behavior (matches previous):
      - Uniform random choice among active promotions for that row.

    Optional:
      - promo_weight_all: 1D array aligned with promo_keys_all (non-negative weights)
        enables weighted choice among active promos (still random per row).

    Returns:
      promo_keys: int64 array (len n)
      promo_pct:  float64 array (len n), clipped to [0, 1]
    """
    promo_keys = np.full(int(n), int(no_discount_key), dtype=np.int64)
    promo_pct = np.zeros(int(n), dtype=np.float64)

    if n <= 0:
        return promo_keys, promo_pct

    if promo_keys_all is None:
        return promo_keys, promo_pct

    promo_keys_all = np.asarray(promo_keys_all, dtype=np.int64)
    if promo_keys_all.size == 0:
        return promo_keys, promo_pct

    promo_pct_all = _safe_clip_pct(promo_pct_all)

    promo_start_all = _as_date32(promo_start_all)
    promo_end_all = _as_date32(promo_end_all)
    order_dates = _as_date32(order_dates)

    if promo_start_all is None or promo_end_all is None or order_dates is None:
        return promo_keys, promo_pct

    # ------------------------------------------------------------
    # Active promotion mask (rows × promos)
    # ------------------------------------------------------------
    # Small/medium promos: full mask is fine and fast.
    # Very large promo sets: avoid allocating huge matrices.
    P = promo_keys_all.size

    # Exclude no-discount promos up front
    promo_valid_glob = (promo_keys_all != int(no_discount_key))
    if not promo_valid_glob.any():
        return promo_keys, promo_pct

    # If weights given, normalize them and force invalid promos to weight 0
    weights = None
    if promo_weight_all is not None:
        w = np.asarray(promo_weight_all, dtype=np.float64)
        if w.shape[0] != P:
            raise ValueError("promo_weight_all must align with promo_keys_all length")
        w = np.where(np.isfinite(w), w, 0.0)
        w = np.maximum(w, 0.0)
        w[~promo_valid_glob] = 0.0
        if w.sum() > 0:
            weights = w

    # Heuristic threshold to protect memory; tune as needed
    # rows×promos noise allocation is the main risk.
    USE_MATRIX_PATH = P <= 512

    if USE_MATRIX_PATH:
        od = order_dates[:, None]  # (n,1)
        active_mask = (od >= promo_start_all) & (od <= promo_end_all)  # (n,P)
        valid_mask = active_mask & promo_valid_glob  # broadcast promo_valid_glob (P,)

        row_has_promo = valid_mask.any(axis=1)
        rows = np.nonzero(row_has_promo)[0]
        if rows.size == 0:
            return promo_keys, promo_pct

        # --------------------------------------------------------
        # Choose promo per row
        # --------------------------------------------------------
        if weights is None:
            # uniform among valid promos: "max random noise" trick
            noise = rng.random(valid_mask.shape)
            noise[~valid_mask] = -1.0
            chosen_idx = noise.argmax(axis=1)
        else:
            # weighted among valid promos:
            # sample by argmax(log(u)/w) equivalent is complicated; use weighted noise:
            # score = u ** (1 / w) favors larger w; set invalid promos to -inf score
            u = rng.random(valid_mask.shape)
            score = np.full(valid_mask.shape, -np.inf, dtype=np.float64)
            w = weights[None, :]  # (1,P)
            # only compute where valid and w>0
            ok = valid_mask & (w > 0)
            score[ok] = np.log(u[ok]) / w[ok]
            chosen_idx = score.argmax(axis=1)

        promo_keys[rows] = promo_keys_all[chosen_idx[rows]]
        promo_pct[rows] = promo_pct_all[chosen_idx[rows]]
        return promo_keys, promo_pct

    # ------------------------------------------------------------
    # Memory-safe path for large P:
    # Iterate rows but keep promo checks vectorized per row.
    # This avoids creating n×P arrays.
    # ------------------------------------------------------------
    # (Month-sliced generation typically means n is per-month and P is modest,
    #  but this keeps you safe if promotions scale up.)
    for i in range(int(n)):
        od = order_dates[i]
        active = promo_valid_glob & (od >= promo_start_all) & (od <= promo_end_all)
        idx = np.nonzero(active)[0]
        if idx.size == 0:
            continue

        if weights is None:
            j = rng.choice(idx)
        else:
            w = weights[idx]
            s = w.sum()
            if s <= 0:
                j = rng.choice(idx)
            else:
                p = w / s
                j = rng.choice(idx, p=p)

        promo_keys[i] = promo_keys_all[j]
        promo_pct[i] = promo_pct_all[j]

    return promo_keys, promo_pct
