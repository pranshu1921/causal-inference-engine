import pytest
import numpy as np
import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st
import hypothesis.extra.numpy as npst

from src.stats_tests import StatisticalTests


@pytest.fixture
def tester():
    return StatisticalTests(alpha=0.05)


@pytest.fixture
def two_groups():
    rng = np.random.default_rng(7)
    group_a = rng.normal(55, 10, 150)
    group_b = rng.normal(50, 10, 150)
    return group_a, group_b


def test_ttest_returns_expected_keys(tester, two_groups):
    a, b = two_groups
    result = tester.two_sample_ttest(a, b)
    assert {"test", "statistic", "p_value", "mean_diff", "significant"}.issubset(result)


def test_ttest_detects_true_difference(tester, two_groups):
    a, b = two_groups
    result = tester.two_sample_ttest(a, b)
    # Groups differ by 5 — should be significant with 150 samples each
    assert result["significant"] is True


def test_ttest_no_difference(tester):
    rng = np.random.default_rng(99)
    a = rng.normal(50, 10, 200)
    b = rng.normal(50, 10, 200)
    result = tester.two_sample_ttest(a, b)
    # p-value should be large (not significant) — we check it's above 0.01 at minimum
    assert result["p_value"] > 0.01


def test_mann_whitney_keys(tester, two_groups):
    a, b = two_groups
    result = tester.mann_whitney(a, b)
    assert {"statistic", "p_value", "significant"}.issubset(result)


def test_bootstrap_ci_contains_true_diff(tester):
    rng = np.random.default_rng(3)
    a = rng.normal(55, 8, 200)
    b = rng.normal(50, 8, 200)
    result = tester.bootstrap_ci(a, b, n_bootstrap=500)
    # True diff is 5 — CI should contain it
    assert result["ci_lower"] < 5 < result["ci_upper"]


def test_sample_size_calculator_positive(tester):
    result = tester.sample_size_calculator(
        baseline_mean=50, mde=2.0, std_dev=10.0, power=0.8
    )
    assert result["required_n_per_group"] > 0
    assert result["total_required_n"] == result["required_n_per_group"] * 2


def test_normality_test_on_normal_data(tester):
    rng = np.random.default_rng(1)
    data = rng.normal(0, 1, 300)
    result = tester.normality_test(data)
    assert result["is_normal"] is True


def test_normality_test_on_skewed_data(tester):
    rng = np.random.default_rng(2)
    data = rng.exponential(1, 300)
    result = tester.normality_test(data)
    assert result["is_normal"] is False


# ── Hypothesis property-based tests ───────────────────────────────────────────

@given(
    a=npst.arrays(dtype=np.float64, shape=st.integers(30, 100),
                  elements=st.floats(0, 100, allow_nan=False, allow_infinity=False)),
    b=npst.arrays(dtype=np.float64, shape=st.integers(30, 100),
                  elements=st.floats(0, 100, allow_nan=False, allow_infinity=False)),
)
@settings(max_examples=50, deadline=2000)
def test_ttest_p_value_always_between_0_and_1(a, b):
    tester = StatisticalTests()
    result = tester.two_sample_ttest(a, b)
    assert 0.0 <= result["p_value"] <= 1.0


@given(
    baseline=st.floats(10, 100, allow_nan=False),
    mde=st.floats(0.5, 20, allow_nan=False),
    std_dev=st.floats(1, 50, allow_nan=False),
)
@settings(max_examples=50, deadline=2000)
def test_sample_size_always_positive(baseline, mde, std_dev):
    tester = StatisticalTests()
    result = tester.sample_size_calculator(baseline, mde, std_dev)
<<<<<<< HEAD
    assert result["required_n_per_group"] > 0
=======
    assert result["required_n_per_group"] > 0
>>>>>>> d97457968c3131a46966c05252068361d9400900
