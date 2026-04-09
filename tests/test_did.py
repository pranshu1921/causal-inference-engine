import pytest
import pandas as pd
import numpy as np
from src.did import DifferenceInDifferences, DIDError


@pytest.fixture
def did_df():
    """Synthetic pre/post panel dataset for DiD testing."""
    rng = np.random.default_rng(0)
    n = 200
    treatment = rng.integers(0, 2, size=n)
    time = rng.integers(0, 2, size=n)
    # True DiD effect is 3.0
    outcome = 50 + 3 * treatment * time + 2 * treatment + 1.5 * time + rng.normal(0, 4, n)
    return pd.DataFrame({
        "treatment": treatment,
        "outcome": outcome,
        "time": time,
        "age": rng.integers(22, 55, size=n),
    })


def test_did_runs_without_error(did_df):
    did = DifferenceInDifferences()
    did.fit(did_df, "treatment", "outcome", "time")
    assert did.did_estimate_ is not None


def test_did_estimate_is_close_to_truth(did_df):
    # True effect is 3.0 — estimate should be within ±1.5
    did = DifferenceInDifferences()
    did.fit(did_df, "treatment", "outcome", "time")
    assert abs(did.did_estimate_ - 3.0) < 1.5


def test_did_summary_keys(did_df):
    did = DifferenceInDifferences()
    did.fit(did_df, "treatment", "outcome", "time")
    summary = did.summary()
    required_keys = {"did_estimate", "p_value", "ci_lower", "ci_upper", "significant"}
    assert required_keys.issubset(summary.keys())


def test_did_group_means_shape(did_df):
    did = DifferenceInDifferences()
    did.fit(did_df, "treatment", "outcome", "time")
    means = did.get_group_means(did_df)
    # Should have 4 rows: 2 groups × 2 periods
    assert means.shape[0] == 4


def test_did_raises_on_missing_column(did_df):
    did = DifferenceInDifferences()
    with pytest.raises(DIDError):
        did.fit(did_df, "treatment", "outcome", "nonexistent_time")


def test_did_raises_on_non_binary_time(did_df):
    did_df = did_df.copy()
    did_df["time"] = did_df["time"] * 5  # make it non-binary
    did = DifferenceInDifferences()
    with pytest.raises(DIDError):
        did.fit(did_df, "treatment", "outcome", "time")
