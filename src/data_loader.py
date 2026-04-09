import pandas as pd
import numpy as np


class DataValidationError(Exception):
    """Raised when the uploaded dataset fails validation checks."""
    pass


class DataLoader:
    """
    Handles CSV ingestion and basic validation for causal inference analysis.

    Expected CSV format:
        - A binary treatment column (0 or 1)
        - A numeric outcome column
        - One or more numeric covariate columns
    """

    def __init__(self, treatment_col: str, outcome_col: str, covariate_cols: list[str]):
        self.treatment_col = treatment_col
        self.outcome_col = outcome_col
        self.covariate_cols = covariate_cols

    def load(self, filepath: str) -> pd.DataFrame:
        """Load and validate a CSV file. Returns a cleaned DataFrame."""
        try:
            df = pd.read_csv(filepath)
        except FileNotFoundError:
            raise DataValidationError(f"File not found: {filepath}")
        except Exception as e:
            raise DataValidationError(f"Could not read file: {e}")

        df = self._validate(df)
        df = self._clean(df)
        return df

    def load_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean a DataFrame passed directly (e.g. from Streamlit uploader)."""
        df = self._validate(df.copy())
        df = self._clean(df)
        return df

    def _validate(self, df: pd.DataFrame) -> pd.DataFrame:
        required_cols = [self.treatment_col, self.outcome_col] + self.covariate_cols
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise DataValidationError(f"Missing required columns: {missing}")

        treatment_values = df[self.treatment_col].dropna().unique()
        if not set(treatment_values).issubset({0, 1}):
            raise DataValidationError(
                f"Treatment column '{self.treatment_col}' must be binary (0 or 1). "
                f"Found values: {treatment_values}"
            )

        if not pd.api.types.is_numeric_dtype(df[self.outcome_col]):
            raise DataValidationError(
                f"Outcome column '{self.outcome_col}' must be numeric."
            )

        min_rows = 50
        if len(df) < min_rows:
            raise DataValidationError(
                f"Dataset too small. Need at least {min_rows} rows, got {len(df)}."
            )

        return df

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_keep = [self.treatment_col, self.outcome_col] + self.covariate_cols
        df = df[cols_to_keep].copy()

        before = len(df)
        df = df.dropna()
        dropped = before - len(df)
        if dropped > 0:
            print(f"[DataLoader] Dropped {dropped} rows with missing values.")

        return df.reset_index(drop=True)

    def summary(self, df: pd.DataFrame) -> dict:
        """Returns a quick summary dict for display in the dashboard."""
        return {
            "total_rows": len(df),
            "treated": int(df[self.treatment_col].sum()),
            "control": int((df[self.treatment_col] == 0).sum()),
            "outcome_mean_treated": round(
                df[df[self.treatment_col] == 1][self.outcome_col].mean(), 4
            ),
            "outcome_mean_control": round(
                df[df[self.treatment_col] == 0][self.outcome_col].mean(), 4
            ),
            "covariates": self.covariate_cols,
        }


def generate_sample_dataset(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic HR-style dataset for demo purposes.

    Simulates a training program intervention where:
        - treatment = 1 means employee received training
        - outcome = performance score (0-100)
        - covariates = age, tenure, department_encoded, prior_score
    """
    rng = np.random.default_rng(seed)

    age = rng.integers(22, 60, size=n)
    tenure = rng.integers(0, 20, size=n)
    prior_score = rng.normal(60, 10, size=n).clip(20, 95)
    department = rng.integers(0, 4, size=n)

    # Treatment assignment is not random — older, longer-tenured employees
    # are more likely to receive training (selection bias built in)
    propensity = 1 / (1 + np.exp(-(0.03 * age + 0.05 * tenure - 2.5)))
    treatment = rng.binomial(1, propensity)

    # Outcome: training improves performance by ~5 points on average
    noise = rng.normal(0, 8, size=n)
    outcome = prior_score + 5 * treatment + 0.2 * tenure + noise
    outcome = outcome.clip(0, 100)

    df = pd.DataFrame({
        "treatment": treatment,
        "performance_score": outcome.round(2),
        "age": age,
        "tenure": tenure,
        "department": department,
        "prior_score": prior_score.round(2),
    })

    return df
