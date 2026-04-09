import pytest
import pandas as pd
import numpy as np
from src.psm import PropensityScoreMatcher, PSMError


@pytest.fixture
def sample_df():
    """A simple synthetic dataset for testing PSM."""
    rng = np.random.default_rng(42)
    n = 300
    age = rng.integers(22, 55, size=n)
    treatment = (age > 38).astype(int)
    outcome = 50 + 5 * treatment + rng.normal(0, 5, size=n)
    return pd.DataFrame({
        "treatment": treatment,
        "outcome": outcome,
        "age": age,
        "score": rng.normal(60, 10, size=n),
    })


def test_psm_runs_without_error(sample_df):
    matcher = PropensityScoreMatcher(caliper=0.1)
    matcher.fit(sample_df, "treatment", "outcome", ["age", "score"])
    assert matcher.ate_ is not None


def test_psm_matched_df_is_balanced(sample_df):
    matcher = PropensityScoreMatcher(caliper=0.1)
    matcher.fit(sample_df, "treatment", "outcome", ["age", "score"])
    balance = matcher.get_balance_stats(sample_df)
    # After matching, at least half the covariates should be balanced
    n_balanced = balance["balanced"].sum()
    assert n_balanced >= len(balance) // 2


def test_psm_summary_keys(sample_df):
    matcher = PropensityScoreMatcher()
    matcher.fit(sample_df, "treatment", "outcome", ["age", "score"])
    summary = matcher.summary()
    required_keys = {"ate", "ate_std", "ate_ci_lower", "ate_ci_upper", "matched_pairs"}
    assert required_keys.issubset(summary.keys())


def test_psm_propensity_scores_between_0_and_1(sample_df):
    matcher = PropensityScoreMatcher()
    matcher.fit(sample_df, "treatment", "outcome", ["age", "score"])
    scores = matcher.propensity_scores_
    assert np.all(scores >= 0) and np.all(scores <= 1)


def test_psm_raises_on_missing_column(sample_df):
    matcher = PropensityScoreMatcher()
    with pytest.raises(Exception):
        matcher.fit(sample_df, "treatment", "outcome", ["age", "nonexistent_col"])


def test_psm_raises_on_empty_group():
    df = pd.DataFrame({
        "treatment": [1] * 50,
        "outcome": np.random.normal(0, 1, 50),
        "age": np.random.randint(20, 50, 50),
    })
    matcher = PropensityScoreMatcher()
    with pytest.raises(PSMError):
        matcher.fit(df, "treatment", "outcome", ["age"])