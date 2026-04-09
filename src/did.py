import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


class DIDError(Exception):
    """Raised when DiD analysis encounters an issue."""
    pass


class DifferenceInDifferences:
    """
    Difference-in-Differences (DiD) estimator using OLS regression.

    Model:
        outcome ~ treatment + post + treatment:post + covariates

    The coefficient on treatment:post is the DiD estimate — the
    Average Treatment Effect on the Treated (ATT).

    Requires:
        - A binary treatment column (0/1)
        - A binary time column indicating pre (0) vs post (1) period
        - A numeric outcome column
    """

    def __init__(self):
        self.result_ = None
        self.did_estimate_ = None
        self.p_value_ = None
        self._treatment_col = None
        self._outcome_col = None
        self._time_col = None

    def fit(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        time_col: str,
        covariate_cols: list[str] | None = None,
    ) -> "DifferenceInDifferences":
        """
        Fit the DiD model.

        Args:
            df: Input DataFrame.
            treatment_col: Binary column — 1 = treatment group, 0 = control.
            outcome_col: Numeric outcome variable.
            time_col: Binary column — 1 = post period, 0 = pre period.
            covariate_cols: Optional additional controls to include in the regression.

        Returns:
            self
        """
        self._validate_inputs(df, treatment_col, outcome_col, time_col)

        self._treatment_col = treatment_col
        self._outcome_col = outcome_col
        self._time_col = time_col

        df = df.copy()
        df = df.rename(columns={
            treatment_col: "_treatment",
            outcome_col: "_outcome",
            time_col: "_time",
        })

        formula = "_outcome ~ _treatment + _time + _treatment:_time"

        if covariate_cols:
            safe_covariates = []
            for col in covariate_cols:
                safe_name = col.replace(" ", "_").replace("-", "_")
                df[safe_name] = df[col] if col in df.columns else df[safe_name]
                safe_covariates.append(safe_name)
            formula += " + " + " + ".join(safe_covariates)

        model = smf.ols(formula=formula, data=df)
        self.result_ = model.fit()

        self.did_estimate_ = float(self.result_.params["_treatment:_time"])
        self.p_value_ = float(self.result_.pvalues["_treatment:_time"])
        self.conf_int_ = self.result_.conf_int().loc["_treatment:_time"].tolist()

        return self

    def _validate_inputs(
        self,
        df: pd.DataFrame,
        treatment_col: str,
        outcome_col: str,
        time_col: str,
    ):
        for col in [treatment_col, outcome_col, time_col]:
            if col not in df.columns:
                raise DIDError(f"Column '{col}' not found in dataset.")

        for col in [treatment_col, time_col]:
            unique_vals = df[col].dropna().unique()
            if not set(unique_vals).issubset({0, 1}):
                raise DIDError(
                    f"Column '{col}' must be binary (0 or 1). Found: {unique_vals}"
                )

        if not pd.api.types.is_numeric_dtype(df[outcome_col]):
            raise DIDError(f"Outcome column '{outcome_col}' must be numeric.")

    def get_group_means(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a table of pre/post means by group.
        Useful for the parallel trends assumption visual check.
        """
        df = df.copy()
        grouped = (
            df.groupby([self._treatment_col, self._time_col])[self._outcome_col]
            .mean()
            .reset_index()
        )
        grouped.columns = ["treatment", "time", "mean_outcome"]
        grouped["group"] = grouped["treatment"].map({1: "Treated", 0: "Control"})
        grouped["period"] = grouped["time"].map({0: "Pre", 1: "Post"})
        return grouped[["group", "period", "mean_outcome"]].round(4)

    def summary(self) -> dict:
        """Returns DiD results as a plain dictionary."""
        if self.did_estimate_ is None:
            raise DIDError("Run fit() before calling summary().")

        return {
            "did_estimate": round(self.did_estimate_, 4),
            "p_value": round(self.p_value_, 4),
            "ci_lower": round(self.conf_int_[0], 4),
            "ci_upper": round(self.conf_int_[1], 4),
            "significant": self.p_value_ < 0.05,
            "r_squared": round(self.result_.rsquared, 4),
            "n_obs": int(self.result_.nobs),
        }

    def regression_table(self) -> pd.DataFrame:
        """Returns a formatted regression summary table."""
        params = self.result_.params
        pvalues = self.result_.pvalues
        conf = self.result_.conf_int()

        table = pd.DataFrame({
            "coefficient": params.round(4),
            "p_value": pvalues.round(4),
            "ci_lower": conf[0].round(4),
            "ci_upper": conf[1].round(4),
        })
        table["significant"] = table["p_value"] < 0.05
<<<<<<< HEAD
        return table
=======
        return table
>>>>>>> d97457968c3131a46966c05252068361d9400900
