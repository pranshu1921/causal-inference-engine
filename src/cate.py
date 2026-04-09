import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


class CATEError(Exception):
    """Raised when CATE estimation fails."""
    pass


class CATEEstimator:
    """
    Conditional Average Treatment Effect (CATE) estimator using the T-Learner approach.

    T-Learner trains two separate models:
        - mu_1: outcome model for treated units (treatment == 1)
        - mu_0: outcome model for control units (treatment == 0)

    The individual treatment effect for any unit is estimated as:
        ITE(x) = mu_1(x) - mu_0(x)

    CATE for a subgroup is the mean of ITE across that subgroup.

    This approach is straightforward and works well for moderate-sized datasets.
    It does not require EconML or DoWhy.
    """

    def __init__(self, n_estimators: int = 100, max_depth: int = 3, random_state: int = 42):
        """
        Args:
            n_estimators: Number of boosting stages.
            max_depth: Maximum depth of each tree. Keep low to avoid overfitting.
            random_state: Seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state

        self.mu_1 = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        self.mu_0 = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
        self.scaler = StandardScaler()
        self.ite_scores_ = None
        self._feature_cols = None
        self._is_fitted = False

    def fit(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        feature_cols: list[str],
    ) -> "CATEEstimator":
        """
        Fit the T-Learner.

        Args:
            df: Input DataFrame.
            treatment_col: Binary treatment indicator column.
            outcome_col: Numeric outcome column.
            feature_cols: Covariates used for CATE estimation.

        Returns:
            self
        """
        self._treatment_col = treatment_col
        self._outcome_col = outcome_col
        self._feature_cols = feature_cols

        treated = df[df[treatment_col] == 1]
        control = df[df[treatment_col] == 0]

        if len(treated) < 10 or len(control) < 10:
            raise CATEError("Need at least 10 samples in each group to fit CATE.")

        X_treated = treated[feature_cols].values
        y_treated = treated[outcome_col].values

        X_control = control[feature_cols].values
        y_control = control[outcome_col].values

        # Scale features using the full dataset
        X_all = df[feature_cols].values
        self.scaler.fit(X_all)

        self.mu_1.fit(self.scaler.transform(X_treated), y_treated)
        self.mu_0.fit(self.scaler.transform(X_control), y_control)

        # Predict potential outcomes for all units
        X_all_scaled = self.scaler.transform(X_all)
        po_treated = self.mu_1.predict(X_all_scaled)
        po_control = self.mu_0.predict(X_all_scaled)

        self.ite_scores_ = po_treated - po_control
        self._is_fitted = True

        return self

    def get_ite_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns the original DataFrame with individual treatment effects appended.
        """
        if not self._is_fitted:
            raise CATEError("Run fit() before calling get_ite_dataframe().")

        result = df.copy()
        result["ite"] = self.ite_scores_
        return result

    def cate_by_subgroup(self, df: pd.DataFrame, subgroup_col: str) -> pd.DataFrame:
        """
        Computes mean CATE for each value of a categorical subgroup column.

        Args:
            df: Original DataFrame (before ITE was appended).
            subgroup_col: Column to group by.

        Returns:
            DataFrame with columns: subgroup_value, mean_cate, std_cate, count
        """
        if not self._is_fitted:
            raise CATEError("Run fit() first.")

        ite_df = self.get_ite_dataframe(df)
        grouped = (
            ite_df.groupby(subgroup_col)["ite"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        grouped.columns = [subgroup_col, "mean_cate", "std_cate", "count"]
        grouped["mean_cate"] = grouped["mean_cate"].round(4)
        grouped["std_cate"] = grouped["std_cate"].round(4)
        return grouped.sort_values("mean_cate", ascending=False)

    def ate(self) -> float:
        """Returns the overall ATE — the mean of all individual treatment effects."""
        if not self._is_fitted:
            raise CATEError("Run fit() first.")
        return round(float(np.mean(self.ite_scores_)), 4)

    def feature_importance(self) -> pd.DataFrame:
        """
        Returns feature importances from the treated outcome model (mu_1).
        Useful for understanding which features drive heterogeneity.
        """
        if not self._is_fitted:
            raise CATEError("Run fit() first.")

        importance_df = pd.DataFrame({
            "feature": self._feature_cols,
            "importance": self.mu_1.feature_importances_.round(4),
        }).sort_values("importance", ascending=False)

        return importance_df