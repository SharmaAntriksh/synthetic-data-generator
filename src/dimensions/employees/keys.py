"""EmployeeKey encoding/decoding — the single source of truth for the
EmployeeKey numbering scheme shared by the employee dimension generator and the
employee-store-assignment bridge.

Encoding scheme (by org tier)::

    CEO            = 1                                              (constant)
    VP Operations  = 2                                              (constant)
    Region manager = REGION_KEY_BASE   (10_000) + RegionId
    District mgr   = DISTRICT_KEY_BASE  (20_000) + DistrictId
    Store manager  = STORE_MGR_KEY_BASE (30_000_000) + StoreKey
    Staff          = STAFF_KEY_BASE (40_000_000)
                       + StoreKey * STAFF_KEY_STORE_MULT (1_000)
                       + within_store_idx  (1-based, < STAFF_KEY_STORE_MULT)
    Online rep     = ONLINE_EMP_KEY_BASE (50_000_000) + StoreKey

The staff band reserves ``STAFF_KEY_STORE_MULT`` slots per store; a store with
that many staff would spill into the next store's slot range, so ``encode_staff``
guards with :data:`MAX_STAFF_PER_STORE`.

Region- and district-manager keys sit below ``STORE_MGR_KEY_BASE``; for the
"does this key carry a home store" question they are :attr:`KeyBand.CORPORATE`
(they have no store) and :func:`decode_home_store_key` returns NA for them.

Dtype note (byte-identity): the array encoders replicate the historical inline
arithmetic exactly — ``BASE + int32_array`` stays int32 under numpy promotion,
then ``.astype(np.int64)`` widens. No overflow occurs (physical StoreKey
< 10_000 and within-store idx < 1_000, so the largest staff key is ~4.001e7,
far below 2**31).
"""
from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

from src.defaults import (
    ONLINE_EMP_KEY_BASE,
    STORE_MGR_KEY_BASE,
    STAFF_KEY_BASE,
    STAFF_KEY_STORE_MULT,
)
from src.exceptions import DimensionError

# Corporate constant keys and region/district additive offsets.
CEO_KEY: int = 1
VP_OPS_KEY: int = 2
REGION_KEY_BASE: int = 10_000
DISTRICT_KEY_BASE: int = 20_000

# A store may hold at most this many staff before within-store indices would
# spill into the next store's slot range in the staff band. Equal by
# construction to STAFF_KEY_STORE_MULT (the per-store slot stride).
MAX_STAFF_PER_STORE: int = STAFF_KEY_STORE_MULT


class KeyBand(Enum):
    """The store-relevance tiers the EmployeeKey scheme partitions into.

    CORPORATE covers CEO/VP and region/district managers — none carry a home
    store. The three store-level bands each map back to a StoreKey.
    """

    CORPORATE = "corporate"
    STORE_MANAGER = "store_manager"
    STAFF = "staff"
    ONLINE_REP = "online_rep"

    @classmethod
    def of(cls, employee_key) -> "KeyBand":
        """Classify a single EmployeeKey into its band."""
        ek = int(employee_key)
        if ek >= ONLINE_EMP_KEY_BASE:
            return cls.ONLINE_REP
        if ek >= STAFF_KEY_BASE:
            return cls.STAFF
        if ek >= STORE_MGR_KEY_BASE:
            return cls.STORE_MANAGER
        return cls.CORPORATE


# ---------------------------------------------------------------------------
# Encoders — polymorphic over scalars and numpy int arrays.
# ---------------------------------------------------------------------------
def encode_region(rid) -> np.int32:
    """Region-manager key: ``REGION_KEY_BASE + RegionId`` (np.int32 scalar)."""
    return np.int32(REGION_KEY_BASE + int(rid))


def encode_district(did) -> np.int32:
    """District-manager key: ``DISTRICT_KEY_BASE + DistrictId`` (np.int32 scalar)."""
    return np.int32(DISTRICT_KEY_BASE + int(did))


def encode_store_manager(store_keys):
    """Store-manager keys: ``STORE_MGR_KEY_BASE + StoreKey`` as int64."""
    return (STORE_MGR_KEY_BASE + np.asarray(store_keys)).astype(np.int64)


def encode_staff(store_keys, within_store_idx):
    """Staff keys: ``STAFF_KEY_BASE + StoreKey*STAFF_KEY_STORE_MULT + idx`` as int64.

    Raises :class:`DimensionError` if any within-store index reaches
    :data:`MAX_STAFF_PER_STORE` — the key would spill into the next store's
    slot range and produce a duplicate / wrong-store-decoding key.
    """
    idx = np.asarray(within_store_idx)
    if np.any(idx >= MAX_STAFF_PER_STORE):
        raise DimensionError(
            f"within_store_idx {int(np.max(idx))} >= MAX_STAFF_PER_STORE "
            f"({MAX_STAFF_PER_STORE}); staff keys would collide with the next "
            "store's slot range. Lower the per-store staff count."
        )
    return (
        STAFF_KEY_BASE + np.asarray(store_keys) * STAFF_KEY_STORE_MULT + idx
    ).astype(np.int64)


def encode_online_rep(store_keys):
    """Online-rep keys: ``ONLINE_EMP_KEY_BASE + StoreKey`` as int64."""
    return (ONLINE_EMP_KEY_BASE + np.asarray(store_keys)).astype(np.int64)


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------
def decode_home_store_key(employee_keys):
    """Derive the home StoreKey from EmployeeKey(s).

      * online rep  (ek >= ONLINE_EMP_KEY_BASE)         -> ek - 50M
      * store mgr   (STORE_MGR_KEY_BASE <= ek < 40M)    -> ek - 30M
      * staff       (ek >= 40M and not online)          -> (ek - 40M) // 1000
      * corporate/region/district (else)                -> NA

    Scalar in -> scalar out (int or ``pd.NA``); Series/array in -> nullable
    ``Int32`` Series aligned to the input. The Series path reproduces the
    historical ``_infer_home_store_key`` arithmetic byte-for-byte.
    """
    if np.ndim(employee_keys) == 0:
        ek = int(employee_keys)
        if ek >= ONLINE_EMP_KEY_BASE:
            return ek - ONLINE_EMP_KEY_BASE
        if STORE_MGR_KEY_BASE <= ek < STAFF_KEY_BASE:
            return ek - STORE_MGR_KEY_BASE
        if ek >= STAFF_KEY_BASE:
            return (ek - STAFF_KEY_BASE) // STAFF_KEY_STORE_MULT
        return pd.NA

    keys = employee_keys if isinstance(employee_keys, pd.Series) else pd.Series(employee_keys)
    ek = keys.astype(np.int64)
    out = pd.Series([pd.NA] * len(keys), dtype="Int32")

    online_mask = ek >= ONLINE_EMP_KEY_BASE
    if online_mask.any():
        out.loc[online_mask] = (ek.loc[online_mask] - ONLINE_EMP_KEY_BASE).astype("Int32")

    mgr_mask = (ek >= STORE_MGR_KEY_BASE) & (ek < STAFF_KEY_BASE)
    if mgr_mask.any():
        out.loc[mgr_mask] = (ek.loc[mgr_mask] - STORE_MGR_KEY_BASE).astype("Int32")

    staff_mask = (ek >= STAFF_KEY_BASE) & ~online_mask
    if staff_mask.any():
        out.loc[staff_mask] = (
            (ek.loc[staff_mask] - STAFF_KEY_BASE) // STAFF_KEY_STORE_MULT
        ).astype("Int32")

    return out


class EmployeeKeyCodec:
    """Grouped, stateless access to the encode/decode functions.

    Exists so callers can spell the scheme as ``EmployeeKeyCodec.encode_*`` /
    ``.decode_home_store_key``; the implementations are the module-level
    functions above.
    """

    STORE_MGR_KEY_BASE = STORE_MGR_KEY_BASE
    STAFF_KEY_BASE = STAFF_KEY_BASE
    STAFF_KEY_STORE_MULT = STAFF_KEY_STORE_MULT
    ONLINE_EMP_KEY_BASE = ONLINE_EMP_KEY_BASE
    MAX_STAFF_PER_STORE = MAX_STAFF_PER_STORE
    CEO_KEY = CEO_KEY
    VP_OPS_KEY = VP_OPS_KEY

    encode_region = staticmethod(encode_region)
    encode_district = staticmethod(encode_district)
    encode_store_manager = staticmethod(encode_store_manager)
    encode_staff = staticmethod(encode_staff)
    encode_online_rep = staticmethod(encode_online_rep)
    decode_home_store_key = staticmethod(decode_home_store_key)
