"""Promotion assignment: date-windowed, optionally weighted promo selection.

Assigns a PromotionKey to each sales row based on which promotions are
active on that row's order date.  Discount amounts are never computed
here — they live in the Promotions dimension and are joined at query time
via PromotionKey.
"""

from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _as_date32(x):
    """Normalize incoming date arrays to datetime64[D] for consistent comparisons."""
    if x is None:
        return None
    a = np.asarray(x)
    if a.size == 0:
        return a
    return a.astype("datetime64[D]", copy=False)


def _sanitize_weights(promo_weight_all, promo_valid_glob):
    """
    Returns:
      weights: float64 array aligned with promo_keys_all, or None if unusable
    """
    if promo_weight_all is None:
        return None

    w = np.asarray(promo_weight_all, dtype=np.float64)
    w = np.where(np.isfinite(w), w, 0.0)
    w = np.maximum(w, 0.0)
    w[~promo_valid_glob] = 0.0

    return w if w.sum() > 0.0 else None


# ----------------------------------------------------------------
# Main promotion assignment
# ----------------------------------------------------------------

def apply_promotions(
    rng,
    n,
    order_dates,
    promo_keys_all,
    promo_start_all,
    promo_end_all,
    no_discount_key=1,
    promo_weight_all=None,
    *,
    channel_keys=None,
    promo_channel_group=None,
):
    """
    Assign at most one PromotionKey per row.

    For each row, picks uniformly (or weighted) among promotions whose
    [StartDate, EndDate] window covers that row's order date.

    CORRELATION #5: When channel_keys and promo_channel_group are provided,
    promotions are filtered by channel affinity:
      - promo_channel_group=0 → any channel
      - promo_channel_group=1 → physical channels only (1=Store, 5=CallCenter, 10=Kiosk)
      - promo_channel_group=2 → digital channels only (2,3,6,7,8)

    Rows with no active promotion receive ``no_discount_key``.

    Returns:
      promo_keys: int64 array (len n)
    """
    n = int(n)
    promo_keys = np.full(n, int(no_discount_key), dtype=np.int32)

    if n <= 0:
        return promo_keys
    if promo_keys_all is None:
        return promo_keys

    promo_keys_all = np.asarray(promo_keys_all, dtype=np.int32)
    P = int(promo_keys_all.size)
    if P == 0:
        return promo_keys

    promo_start_all = _as_date32(promo_start_all)
    promo_end_all = _as_date32(promo_end_all)
    order_dates = _as_date32(order_dates)

    if promo_start_all is None or promo_end_all is None or order_dates is None:
        return promo_keys
    if order_dates.shape[0] != n:
        raise ValueError("order_dates length must match n")
    if promo_start_all.shape[0] != P or promo_end_all.shape[0] != P:
        raise ValueError("promo_start_all/promo_end_all must align with promo_keys_all length")

    promo_valid_glob = (promo_keys_all != int(no_discount_key))
    if not promo_valid_glob.any():
        return promo_keys

    weights = _sanitize_weights(promo_weight_all, promo_valid_glob)
    if promo_weight_all is not None and weights is None:
        weights = None

    # ------------------------------------------------------------
    # Group rows by unique date (fast path for month-sliced generation)
    # ------------------------------------------------------------
    unique_dates, inv = np.unique(order_dates, return_inverse=True)
    U = unique_dates.size

    # Build channel-group masks for promo filtering
    # Physical: 1=Store, 5=CallCenter, 10=Kiosk
    # Digital:  2=Online, 3=Marketplace, 6=Web, 7=MobileApp, 8=SocialCommerce
    # Business: 4=B2B, 9=PartnerReseller (accept only universal promos)
    _PHYSICAL_CH = frozenset({1, 5, 10})
    _DIGITAL_CH = frozenset({2, 3, 6, 7, 8})
    _BUSINESS_CH = frozenset({4, 9})
    _has_ch_filter = (channel_keys is not None and promo_channel_group is not None
                      and len(promo_channel_group) == P)

    # U is typically <= 31 when month-sliced; building lists is fast
    # and avoids a U×P boolean matrix
    active_indices_per_date = []
    for d in unique_dates:
        active = promo_valid_glob & (d >= promo_start_all) & (d <= promo_end_all)
        active_indices_per_date.append(np.nonzero(active)[0])

    group_counts = np.bincount(inv, minlength=U).astype(np.int64)
    group_order = np.argsort(inv, kind="mergesort")
    group_starts = np.zeros(U + 1, dtype=np.int64)
    np.cumsum(group_counts, out=group_starts[1:])

    for code in range(U):
        idx = active_indices_per_date[code]
        if idx.size == 0:
            continue

        count = int(group_counts[code])
        gs = int(group_starts[code])
        ge = int(group_starts[code + 1])
        rows = group_order[gs:ge]

        # CORRELATION #5: filter promos by channel affinity.
        # Batch by channel type (physical/digital/business/other) — only
        # 4 groups, so we filter promos once per group and vectorized-
        # sample all rows in that group at once.  O(groups × promos)
        # instead of O(rows × promos).
        if _has_ch_filter:
            row_channels = np.asarray(channel_keys, dtype=np.int16)[rows]
            _pcg = np.asarray(promo_channel_group, dtype=np.int8)
            _pcg_idx = _pcg[idx]  # promo channel groups for active promos

            # Classify each row into a channel type code: 0=other, 1=physical, 2=digital, 3=business
            _ch_type = np.zeros(count, dtype=np.int8)
            for _phk in _PHYSICAL_CH:
                _ch_type[row_channels == _phk] = 1
            for _dgk in _DIGITAL_CH:
                _ch_type[row_channels == _dgk] = 2
            for _bk in _BUSINESS_CH:
                _ch_type[row_channels == _bk] = 3

            for _ct in np.unique(_ch_type):
                _ct_mask = _ch_type == _ct
                _ct_count = int(_ct_mask.sum())
                if _ct_count == 0:
                    continue
                _ct_rows = rows[_ct_mask]

                # Filter promos: group=0 (any) always accepted
                _ok = _pcg_idx == 0
                if _ct == 1:    # physical
                    _ok |= _pcg_idx == 1
                elif _ct == 2:  # digital
                    _ok |= _pcg_idx == 2
                elif _ct == 3:  # business — only universal promos
                    pass
                else:           # unknown — accept all
                    _ok[:] = True

                _filtered = idx[_ok]
                if _filtered.size == 0:
                    continue  # keep no_discount_key

                # Vectorized batch sample for all rows in this channel type
                if weights is None:
                    _chosen = _filtered[rng.integers(0, _filtered.size, size=_ct_count)]
                else:
                    _w = weights[_filtered]
                    _s = float(_w.sum())
                    if _s <= 0.0:
                        _chosen = _filtered[rng.integers(0, _filtered.size, size=_ct_count)]
                    else:
                        _cdf = np.cumsum(_w, dtype=np.float64)
                        _cdf /= _cdf[-1]
                        _cdf[-1] = 1.0
                        _u = rng.random(_ct_count)
                        _j = np.searchsorted(_cdf, _u, side="right")
                        _chosen = _filtered[np.minimum(_j, _filtered.size - 1)]

                promo_keys[_ct_rows] = promo_keys_all[_chosen]
        else:
            if weights is None:
                chosen = idx[rng.integers(0, idx.size, size=count)]
            else:
                w = weights[idx]
                s = float(w.sum())
                if s <= 0.0:
                    chosen = idx[rng.integers(0, idx.size, size=count)]
                else:
                    cdf = np.cumsum(w, dtype=np.float64)
                    cdf /= cdf[-1]
                    cdf[-1] = 1.0  # prevent fp rounding from leaving cdf[-1] < 1.0
                    u = rng.random(count)
                    j = np.searchsorted(cdf, u, side="right")
                    chosen = idx[np.minimum(j, idx.size - 1)]

            promo_keys[rows] = promo_keys_all[chosen]

    return promo_keys
