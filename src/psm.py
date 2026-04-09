import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class PSMError(Exception):
    """Raised when PSM analysis encounters an unrecoverable issue."""
    pass


class PropensityScoreMatcher:
    """
    Propensity Score Matching (PSM) for reducing selection bias
    in observational studies.

    Steps:
        1. Fit a logistic regression to estimate propensity scores
        2. Match each treated unit to its nearest control unit
        3. Compute ATE on the matched sample
        4. Return balance diagnostics before and after matching
    """

    def __init__(self, caliper: float = 0.05, random_state: int = 42):
        """
        Args:
            caliper: Maximum allowable difference in propensity scores for a match.
                     Set to None to disable caliper matching.
            random_state: Seed for reproducibility.
        """
        self.caliper = caliper
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.propensity_model = LogisticRegression(
            max_iter=1000, random_state=random_state
        )
        self.propensity_scores_ = None
        self.matched_df_ = None
        self.ate_ = None
        self.ate_std_ = None

    def fit(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        covariate_cols: list[str],
    ) -> "PropensityScoreMatcher":
        """
        Fit the propensity score model and run matching.

        Args:
            df: Input DataFrame (already cleaned).
            treatment_col: Name of the binary treatment column.
            outcome_col: Name of the numeric outcome column.
            covariate_cols: List of covariate column names for matching.

        Returns:
            self, for method chaining.
        """
        self._treatment_col = treatment_col
        self._outcome_col = outcome_col
        self._covariate_cols = covariate_cols

        X = df[covariate_cols].values
        t = df[treatment_col].values
        y = df[outcome_col].values

        X_scaled = self.scaler.fit_transform(X)
        self.propensity_model.fit(X_scaled, t)
        scores = self.propensity_model.predict_proba(X_scaled)[:, 1]
        self.propensity_scores_ = scores

        df = df.copy()
        df["_propensity_score"] = scores

        self.matched_df_ = self._match(df, treatment_col, outcome_col)
        self._compute_ate()

        return self

    def _match(
        self, df: pd.DataFrame, treatment_col: str, outcome_col: str
    ) -> pd.DataFrame:
        treated = df[df[treatment_col] == 1].copy()
        control = df[df[treatment_col] == 0].copy()

        if len(treated) == 0 or len(control) == 0:
            raise PSMError("Need both treated and control units to run PSM.")

        control_scores = control[["_propensity_score"]].values
        nn = NearestNeighbors(n_neighbors=1, algorithm="ball_tree")
        nn.fit(control_scores)

        treated_scores = treated[["_propensity_score"]].values
        distances, indices = nn.kneighbors(treated_scores)

        matched_rows = []
        for i, (dist, idx) in enumerate(zip(distances.flatten(), indices.flatten())):
            if self.caliper is not None and dist > self.caliper:
                continue
            treated_row = treated.iloc[i].to_dict()
            control_row = control.iloc[idx].to_dict()
            treated_row["_match_id"] = i
            treated_row["_distance"] = round(dist, 6)
            control_row["_match_id"] = i
            control_row["_distance"] = round(dist, 6)
            matched_rows.extend([treated_row, control_row])

        if len(matched_rows) == 0:
            raise PSMError(
                "No matches found within caliper. Try increasing the caliper value."
            )

        matched_df = pd.DataFrame(matched_rows).reset_index(drop=True)
        return matched_df

    def _compute_ate(self):
        treated = self.matched_df_[
            self.matched_df_[self._treatment_col] == 1
        ][self._outcome_col].values

        control = self.matched_df_[
            self.matched_df_[self._treatment_col] == 0
        ][self._outcome_col].values

        diffs = treated - control
        self.ate_ = float(np.mean(diffs))
        self.ate_std_ = float(np.std(diffs) / np.sqrt(len(diffs)))

    def get_balance_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes standardized mean differences (SMD) before and after matching.
        SMD < 0.1 is generally considered good balance.
        """
        rows = []
        for col in self._covariate_cols:
            t_vals = df[df[self._treatment_col] == 1][col]
            c_vals = df[df[self._treatment_col] == 0][col]
            pooled_std = np.sqrt((t_vals.std() ** 2 + c_vals.std() ** 2) / 2)
            smd_before = (t_vals.mean() - c_vals.mean()) / (pooled_std + 1e-9)

            t_matched = self.matched_df_[
                self.matched_df_[self._treatment_col] == 1
            ][col]
            c_matched = self.matched_df_[
                self.matched_df_[self._treatment_col] == 0
            ][col]
            pooled_std_m = np.sqrt((t_matched.std() ** 2 + c_matched.std() ** 2) / 2)
            smd_after = (t_matched.mean() - c_matched.mean()) / (pooled_std_m + 1e-9)

            rows.append({
                "covariate": col,
                "smd_before": round(smd_before, 4),
                "smd_after": round(smd_after, 4),
                "balanced": abs(smd_after) < 0.1,
            })

        return pd.DataFrame(rows)

    def summary(self) -> dict:
        """Returns a summary of PSM results for display."""
        if self.ate_ is None:
            raise PSMError("Run fit() before calling summary().")

        n_treated_original = int(
            (self.matched_df_[self._treatment_col] == 1).sum()
        )

        return {
            "ate": round(self.ate_, 4),
            "ate_std": round(self.ate_std_, 4),
            "ate_ci_lower": round(self.ate_ - 1.96 * self.ate_std_, 4),
            "ate_ci_upper": round(self.ate_ + 1.96 * self.ate_std_, 4),
            "matched_pairs": n_treated_original,
            "caliper_used": self.caliper,
        }