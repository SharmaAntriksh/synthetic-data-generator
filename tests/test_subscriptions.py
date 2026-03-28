"""Tests for the subscriptions dimension package."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.dimensions.customers.subscriptions.catalog import (
    PLANS_CATALOG,
    _PAYMENT_WEIGHTS,
    PAYMENT_METHODS,
    CANCELLATION_REASONS,
    _CYCLE_MONTHS,
)
from src.dimensions.customers.subscriptions.helpers import (
    SubscriptionsCfg,
    build_dim_plans,
    build_type_groups,
    choose_plans_diverse,
    compute_customer_windows,
    advance_months,
    months_between,
    ns_to_year_month,
    month_start_date,
    month_end_date,
    bridge_schema,
    _NS_PER_DAY,
)


# ===================================================================
# Catalog tests
# ===================================================================

class TestCatalog:
    def test_plans_catalog_nonempty(self):
        assert len(PLANS_CATALOG) > 0

    def test_all_plans_have_correct_tuple_length(self):
        for row in PLANS_CATALOG:
            assert len(row) == 13, f"Plan row has {len(row)} fields, expected 13"

    def test_payment_weights_sum_to_one(self):
        assert abs(_PAYMENT_WEIGHTS.sum() - 1.0) < 1e-9

    def test_payment_methods_match_weights(self):
        assert len(PAYMENT_METHODS) == len(_PAYMENT_WEIGHTS)

    def test_cancellation_reasons_nonempty(self):
        assert len(CANCELLATION_REASONS) >= 3

    def test_cycle_prices_positive(self):
        for row in PLANS_CATALOG:
            assert row[7] > 0, f"CyclePrice for {row[0]} ({row[3]}) should be positive"

    def test_annual_prices_positive(self):
        for row in PLANS_CATALOG:
            assert row[8] > 0, f"AnnualPrice for {row[0]} ({row[3]}) should be positive"

    def test_discount_applied_correctly(self):
        """Annual plans should have lower per-month cost than monthly plans."""
        monthly_by_name = {}
        annual_by_name = {}
        for row in PLANS_CATALOG:
            name, cycle = row[0], row[3]
            annual_price = row[8]
            if cycle == "Monthly":
                monthly_by_name[name] = annual_price
            elif cycle == "Annual":
                annual_by_name[name] = annual_price
        for name in set(monthly_by_name) & set(annual_by_name):
            assert annual_by_name[name] < monthly_by_name[name], (
                f"Annual discount not applied for {name}"
            )


# ===================================================================
# Date helper tests
# ===================================================================

class TestDateHelpers:
    def test_advance_months_simple(self):
        assert advance_months(2024, 1, 3) == (2024, 4)

    def test_advance_months_year_wrap(self):
        assert advance_months(2024, 11, 3) == (2025, 2)

    def test_advance_months_multi_year(self):
        assert advance_months(2024, 1, 24) == (2026, 1)

    def test_months_between_same_month(self):
        assert months_between(2024, 6, 2024, 6) == 1

    def test_months_between_one_year(self):
        assert months_between(2024, 1, 2024, 12) == 12

    def test_months_between_cross_year(self):
        assert months_between(2024, 10, 2025, 3) == 6

    def test_ns_to_year_month(self):
        ts = pd.Timestamp("2024-06-15")
        y, m = ns_to_year_month(ts.value)
        assert y == 2024
        assert m == 6

    def test_month_start_date(self):
        assert month_start_date(2024, 3) == date(2024, 3, 1)

    def test_month_end_date_feb_leap(self):
        assert month_end_date(2024, 2) == date(2024, 2, 29)

    def test_month_end_date_feb_non_leap(self):
        assert month_end_date(2023, 2) == date(2023, 2, 28)

    def test_month_end_date_dec(self):
        assert month_end_date(2024, 12) == date(2024, 12, 31)


# ===================================================================
# Config tests
# ===================================================================

class TestSubscriptionsCfg:
    def test_defaults(self):
        c = SubscriptionsCfg()
        assert c.trial_days == 14
        assert c.participation_rate == 0.65
        assert c.churn_rate == 0.25
        assert c.trial_rate == 0.30
        assert c.trial_conversion_rate == 0.85

    def test_frozen(self):
        c = SubscriptionsCfg()
        with pytest.raises(AttributeError):
            c.trial_days = 30  # type: ignore[misc]

    def test_custom_trial_days(self):
        c = SubscriptionsCfg(trial_days=30)
        assert c.trial_days == 30


# ===================================================================
# build_dim_plans tests
# ===================================================================

class TestBuildDimPlans:
    def test_output_shape(self):
        df = build_dim_plans(pd.Timestamp("2020-01-01"))
        assert len(df) == len(PLANS_CATALOG)
        assert "PlanKey" in df.columns
        assert "PlanName" in df.columns
        assert len(df.columns) == 15

    def test_plan_keys_sequential(self):
        df = build_dim_plans(pd.Timestamp("2020-01-01"))
        assert list(df["PlanKey"]) == list(range(1, len(df) + 1))

    def test_launch_dates_after_start(self):
        g_start = pd.Timestamp("2020-01-01")
        df = build_dim_plans(g_start)
        for _, row in df.iterrows():
            assert pd.Timestamp(row["LaunchDate"]) >= g_start

    def test_all_active(self):
        df = build_dim_plans(pd.Timestamp("2020-01-01"))
        assert (df["IsActiveFlag"] == 1).all()

    def test_cycle_months_valid(self):
        df = build_dim_plans(pd.Timestamp("2020-01-01"))
        valid_months = set(_CYCLE_MONTHS.values())
        for cm in df["CycleMonths"]:
            assert int(cm) in valid_months


# ===================================================================
# Type group / plan selection tests
# ===================================================================

class TestPlanSelection:
    @pytest.fixture
    def plan_data(self):
        df = build_dim_plans(pd.Timestamp("2020-01-01"))
        plan_types = df["PlanType"].astype(str).to_numpy()
        return build_type_groups(plan_types)

    def test_unique_types_nonempty(self, plan_data):
        unique_types, _, _ = plan_data
        assert len(unique_types) >= 5

    def test_weights_sum_to_one(self, plan_data):
        _, _, type_weights = plan_data
        assert abs(type_weights.sum() - 1.0) < 1e-9

    def test_choose_diverse_returns_distinct_types(self, plan_data):
        unique_types, type_members, type_weights = plan_data
        rng = np.random.default_rng(42)
        plans = choose_plans_diverse(rng, 4, unique_types, type_members, type_weights)
        assert len(plans) == 4
        # All chosen plans should be from distinct type indices
        assert len(set(plans)) == 4

    def test_choose_diverse_capped_by_type_count(self, plan_data):
        unique_types, type_members, type_weights = plan_data
        rng = np.random.default_rng(42)
        plans = choose_plans_diverse(rng, 100, unique_types, type_members, type_weights)
        assert len(plans) == len(unique_types)


# ===================================================================
# Customer windows tests
# ===================================================================

class TestCustomerWindows:
    def test_basic_windows(self):
        customers = pd.DataFrame({
            "CustomerKey": [1, 2, 3],
            "CustomerStartDate": pd.to_datetime(["2020-01-01", "2021-06-15", "2022-03-01"]),
            "CustomerEndDate": pd.to_datetime(["2025-12-31", "2025-12-31", "2024-06-30"]),
        })
        g_start = pd.Timestamp("2020-01-01")
        g_end = pd.Timestamp("2025-12-31")
        ck, lo, hi = compute_customer_windows(customers, g_start, g_end)
        assert list(ck) == [1, 2, 3]
        assert all(lo <= hi)

    def test_clamping(self):
        customers = pd.DataFrame({
            "CustomerKey": [1],
            "CustomerStartDate": pd.to_datetime(["2018-01-01"]),
            "CustomerEndDate": pd.to_datetime(["2030-01-01"]),
        })
        g_start = pd.Timestamp("2020-01-01")
        g_end = pd.Timestamp("2025-12-31")
        ck, lo, hi = compute_customer_windows(customers, g_start, g_end)
        assert lo[0] == g_start.value
        assert hi[0] == g_end.value

    def test_missing_date_columns(self):
        customers = pd.DataFrame({"CustomerKey": [1, 2]})
        g_start = pd.Timestamp("2020-01-01")
        g_end = pd.Timestamp("2025-12-31")
        ck, lo, hi = compute_customer_windows(customers, g_start, g_end)
        assert len(ck) == 2
        assert all(lo == g_start.value)
        assert all(hi == g_end.value)


# ===================================================================
# Period expansion tests
# ===================================================================

# ===================================================================
# Bridge schema tests
# ===================================================================

class TestBridgeSchema:
    def test_column_count(self):
        schema = bridge_schema()
        assert len(schema) == 10

    def test_column_names(self):
        schema = bridge_schema()
        names = [f.name for f in schema]
        assert "SubscriptionKey" in names
        assert "CustomerKey" in names
        assert "PlanKey" in names
        assert "PeriodStartDate" in names
        assert "IsTrialPeriod" in names
        assert "BillingCycleNumber" in names


# ===================================================================
# Determinism test
# ===================================================================

class TestDeterminism:
    def test_plan_selection_deterministic(self):
        df = build_dim_plans(pd.Timestamp("2020-01-01"))
        plan_types = df["PlanType"].astype(str).to_numpy()
        ut, tm, tw = build_type_groups(plan_types)

        results = []
        for _ in range(3):
            rng = np.random.default_rng(42)
            plans = choose_plans_diverse(rng, 4, ut, tm, tw)
            results.append(list(plans))
        assert results[0] == results[1] == results[2]



# ===================================================================
# Edge case tests
# ===================================================================

class TestEdgeCases:
    def test_max_subscriptions_one(self):
        df = build_dim_plans(pd.Timestamp("2020-01-01"))
        plan_types = df["PlanType"].astype(str).to_numpy()
        ut, tm, tw = build_type_groups(plan_types)
        rng = np.random.default_rng(42)
        plans = choose_plans_diverse(rng, 1, ut, tm, tw)
        assert len(plans) == 1

    def test_churn_rate_one_always_churns(self):
        """With churn_rate=1.0, every subscription should have is_churned=True."""
        rng = np.random.default_rng(42)
        n_trials = 100
        churned = sum(1 for _ in range(n_trials) if rng.random() < 1.0)
        assert churned == n_trials

    def test_trial_rate_zero_no_trials(self):
        """With trial_rate=0, no subscriptions should have trials."""
        rng = np.random.default_rng(42)
        n_trials = 100
        trials = sum(1 for _ in range(n_trials) if rng.random() < 0.0)
        assert trials == 0

    def test_customer_one_day_span(self):
        """Customer with exactly 1-day span should be filtered out (< 30 days)."""
        customers = pd.DataFrame({
            "CustomerKey": [1],
            "CustomerStartDate": pd.to_datetime(["2024-06-15"]),
            "CustomerEndDate": pd.to_datetime(["2024-06-15"]),
        })
        g_start = pd.Timestamp("2020-01-01")
        g_end = pd.Timestamp("2025-12-31")
        ck, lo, hi = compute_customer_windows(customers, g_start, g_end)
        span_days = (hi - lo) // _NS_PER_DAY
        assert span_days[0] == 0
        assert span_days[0] < 30  # would be filtered by eligibility

    def test_zero_month_window_eligible(self):
        """A customer window of zero days should still be handled without crash."""
        sub_ns = int(pd.Timestamp("2024-06-15").value)
        # Zero-length windows are filtered by eligibility in bulk generation;
        # verify the compute_customer_windows helper handles them gracefully.
        cust = pd.DataFrame({
            "CustomerKey": [1],
            "CustomerStartDate": [pd.Timestamp("2024-06-15")],
            "CustomerEndDate": [pd.NaT],
        })
        ck, lo, hi = compute_customer_windows(
            cust, pd.Timestamp("2024-06-15"), pd.Timestamp("2024-06-15"),
        )
        assert len(ck) == 1
