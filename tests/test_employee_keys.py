"""Unit tests for the online/physical boundary predicate and the
``EmployeeKeyCodec`` (Phase 2 of the store-cluster refactor).

Two concerns are pinned here:

1. **Boundary predicate** (``src.defaults.is_online_store_key`` /
   ``is_physical_store_key``): the single canonical spelling of the
   online/physical split.  The rule is ``online ⇔ StoreKey > ONLINE_STORE_KEY_BASE``
   — ``BASE`` itself is physical (it never occurs as a real StoreKey, but the
   classification must still be total and unambiguous).  Every call site that
   used to spell this four different ways (``> BASE`` / ``>= BASE`` / ``< BASE``
   / ``<= BASE``) now routes through these two functions, so their behavior at
   the boundary is load-bearing.

2. **EmployeeKey codec** (``src.dimensions.employees.keys``): the encode/decode
   round-trip for the three store-level bands (store manager, staff, online rep)
   plus the ``MAX_STAFF_PER_STORE`` guard that keeps a store's staff keys from
   spilling into the next store's slot range.  Corporate (org-level) keys carry
   no store and must decode to NA.

Key encoding (mirrors ``src/dimensions/employees/generator.py``):
    store manager = STORE_MGR_KEY_BASE (30M) + StoreKey
    staff         = STAFF_KEY_BASE (40M) + StoreKey * STAFF_KEY_STORE_MULT (1000) + within_store_idx
    online rep    = ONLINE_EMP_KEY_BASE (50M) + StoreKey
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.defaults import (
    ONLINE_STORE_KEY_BASE,
    STORE_MGR_KEY_BASE,
    STAFF_KEY_BASE,
    STAFF_KEY_STORE_MULT,
    ONLINE_EMP_KEY_BASE,
    is_online_store_key,
    is_physical_store_key,
)
from src.dimensions.employees.keys import (
    KeyBand,
    MAX_STAFF_PER_STORE,
    encode_store_manager,
    encode_staff,
    encode_online_rep,
    decode_home_store_key,
)


# ---------------------------------------------------------------------------
# Boundary predicate
# ---------------------------------------------------------------------------
class TestBoundaryPredicate:
    """``is_online_store_key`` / ``is_physical_store_key`` are total, mutually
    exclusive, and put ``BASE`` on the physical side."""

    @pytest.mark.parametrize(
        "sk, expect_online",
        [
            (ONLINE_STORE_KEY_BASE - 1, False),   # 9_999  — physical
            (ONLINE_STORE_KEY_BASE, False),        # 10_000 — the boundary, physical
            (ONLINE_STORE_KEY_BASE + 1, True),     # 10_001 — first online key
            (1, False),                            # smallest physical key
            (ONLINE_STORE_KEY_BASE + 500_000, True),  # a large online key
        ],
    )
    def test_classification_at_boundary(self, sk, expect_online):
        assert is_online_store_key(sk) is expect_online
        assert is_physical_store_key(sk) is (not expect_online)

    def test_scalar_int_returns_plain_bool(self):
        # A scalar Python int must yield a plain ``bool`` (not np.bool_), so the
        # ``is`` / ``is not`` identity checks downstream are meaningful.
        for sk in (1, ONLINE_STORE_KEY_BASE, ONLINE_STORE_KEY_BASE + 1):
            assert type(is_online_store_key(sk)) is bool
            assert type(is_physical_store_key(sk)) is bool

    def test_predicates_are_complementary_at_base(self):
        b = ONLINE_STORE_KEY_BASE
        # Exactly one classification; BASE is physical.
        assert is_online_store_key(b) is not is_physical_store_key(b)
        assert not is_online_store_key(b)
        assert is_physical_store_key(b)

    def test_polymorphic_over_numpy_array(self):
        arr = np.array(
            [1, ONLINE_STORE_KEY_BASE, ONLINE_STORE_KEY_BASE + 1], dtype=np.int64
        )
        online = is_online_store_key(arr)
        physical = is_physical_store_key(arr)
        assert online.tolist() == [False, False, True]
        assert physical.tolist() == [True, True, False]
        # Total and mutually exclusive elementwise.
        assert bool((online ^ physical).all())

    def test_polymorphic_over_pandas_series(self):
        s = pd.Series([1, ONLINE_STORE_KEY_BASE, ONLINE_STORE_KEY_BASE + 1])
        online = is_online_store_key(s)
        assert online.tolist() == [False, False, True]


# ---------------------------------------------------------------------------
# EmployeeKey codec — encode/decode round-trip
# ---------------------------------------------------------------------------
class TestEmployeeKeyCodecEncode:
    """Encoders reproduce the exact arithmetic used by the generators."""

    def test_encode_store_manager(self):
        assert encode_store_manager(1) == STORE_MGR_KEY_BASE + 1
        assert encode_store_manager(42) == STORE_MGR_KEY_BASE + 42

    def test_encode_staff(self):
        assert encode_staff(1, 0) == STAFF_KEY_BASE + 1 * STAFF_KEY_STORE_MULT + 0
        assert encode_staff(7, 3) == STAFF_KEY_BASE + 7 * STAFF_KEY_STORE_MULT + 3

    def test_encode_online_rep(self):
        sk = ONLINE_STORE_KEY_BASE + 1
        assert encode_online_rep(sk) == ONLINE_EMP_KEY_BASE + sk


class TestEmployeeKeyCodecRoundTrip:
    """encode → decode returns the original StoreKey for every store-level band."""

    @pytest.mark.parametrize("sk", [1, 2, 50, 9_999])
    def test_store_manager_round_trip(self, sk):
        ek = encode_store_manager(sk)
        assert int(decode_home_store_key(ek)) == sk

    @pytest.mark.parametrize("sk", [1, 2, 50, 9_999])
    @pytest.mark.parametrize("idx", [0, 1, 999])
    def test_staff_round_trip(self, sk, idx):
        ek = encode_staff(sk, idx)
        # Staff decode strips the within-store index → recovers the StoreKey.
        assert int(decode_home_store_key(ek)) == sk

    @pytest.mark.parametrize("sk", [ONLINE_STORE_KEY_BASE + 1, ONLINE_STORE_KEY_BASE + 2])
    def test_online_rep_round_trip(self, sk):
        ek = encode_online_rep(sk)
        assert int(decode_home_store_key(ek)) == sk

    def test_corporate_keys_decode_to_na(self):
        # CEO=1, VP=2, region mgr = 10_000+RegionId, district mgr = 20_000+DistrictId.
        # None carry a store → all decode to NA.
        for corporate_ek in (1, 2, ONLINE_STORE_KEY_BASE + 5, 20_000 + 3):
            decoded = decode_home_store_key(corporate_ek)
            assert decoded is pd.NA or (
                hasattr(decoded, "__len__") is False and pd.isna(decoded)
            )

    def test_decode_vectorized(self):
        # Mixed band vector: manager, staff, online rep, corporate.
        ek = np.array(
            [
                encode_store_manager(5),
                encode_staff(5, 2),
                encode_online_rep(ONLINE_STORE_KEY_BASE + 1),
                1,  # CEO — corporate, no store
            ],
            dtype=np.int64,
        )
        decoded = decode_home_store_key(ek)
        vals = pd.array(decoded, dtype="Int32")
        assert int(vals[0]) == 5
        assert int(vals[1]) == 5
        assert int(vals[2]) == ONLINE_STORE_KEY_BASE + 1
        assert pd.isna(vals[3])


# ---------------------------------------------------------------------------
# KeyBand classification
# ---------------------------------------------------------------------------
class TestKeyBand:
    """Each key maps to exactly one band."""

    def test_band_of_each_kind(self):
        assert KeyBand.of(encode_store_manager(1)) is KeyBand.STORE_MANAGER
        assert KeyBand.of(encode_staff(1, 0)) is KeyBand.STAFF
        assert KeyBand.of(encode_online_rep(ONLINE_STORE_KEY_BASE + 1)) is KeyBand.ONLINE_REP

    def test_corporate_band(self):
        # Below STORE_MGR_KEY_BASE ⇒ corporate hierarchy (CEO/VP/region/district).
        assert KeyBand.of(1) is KeyBand.CORPORATE
        assert KeyBand.of(20_000 + 7) is KeyBand.CORPORATE

    def test_online_rep_not_confused_with_staff(self):
        # Online rep band (50M+) is above the staff band (40M+); the classifier
        # must not misread an online rep as staff.
        ek = encode_online_rep(ONLINE_STORE_KEY_BASE + 1)
        assert KeyBand.of(ek) is KeyBand.ONLINE_REP
        assert KeyBand.of(ek) is not KeyBand.STAFF


# ---------------------------------------------------------------------------
# MAX_STAFF_PER_STORE guard
# ---------------------------------------------------------------------------
class TestMaxStaffGuard:
    """The staff band reserves ``STAFF_KEY_STORE_MULT`` slots per store; an index
    at or beyond that would collide into the next store's range."""

    def test_max_equals_store_mult(self):
        assert MAX_STAFF_PER_STORE == STAFF_KEY_STORE_MULT

    def test_encode_staff_accepts_last_valid_index(self):
        # idx == MAX_STAFF_PER_STORE - 1 is the last non-colliding slot.
        ek = encode_staff(1, MAX_STAFF_PER_STORE - 1)
        assert int(decode_home_store_key(ek)) == 1

    def test_encode_staff_rejects_overflow_index(self):
        from src.exceptions import DimensionError
        with pytest.raises(DimensionError):
            encode_staff(1, MAX_STAFF_PER_STORE)

    def test_encode_staff_rejects_index_beyond_overflow(self):
        from src.exceptions import DimensionError
        with pytest.raises(DimensionError):
            encode_staff(1, MAX_STAFF_PER_STORE + 5)
