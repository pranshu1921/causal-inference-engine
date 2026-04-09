import numpy as np
import pandas as pd
from scipy import stats


class StatisticalTests:
    """
    Collection of statistical tests for A/B experiment analysis.

    All methods return a result dictionary with consistent keys:
        - statistic: the test statistic
        - p_value: two-sided p-value
        - significant: bool, whether p < alpha
        - interpretation: plain English summary
    """

    def __init__(self, alpha: float = 0.05):
        """
        Args:
            alpha: Significance level. Default 0.05.
        """
        self.alpha = alpha

    def two_sample_ttest(
        self,
        group_a: np.ndarray | pd.Series,
        group_b: np.ndarray | pd.Series,
        equal_var: bool = False,
    ) -> dict:
        """
        Welch's t-test (default) or Student's t-test for comparing two group means.

        Args:
            group_a: Outcome values for group A (e.g. treatment).
            group_b: Outcome values for group B (e.g. control).
            equal_var: Set True to use Student's t-test (assumes equal variance).
        """
        stat, p = stats.ttest_ind(group_a, group_b, equal_var=equal_var)
        mean_diff = np.mean(group_a) - np.mean(group_b)

        return {
            "test": "Welch's t-test" if not equal_var else "Student's t-test",
            "statistic": round(float(stat), 4),
            "p_value": round(float(p), 4),
            "mean_diff": round(float(mean_diff), 4),
            "significant": float(p) < self.alpha,
            "interpretation": self._interpret(float(p), "mean difference"),
        }

    def mann_whitney(
        self,
        group_a: np.ndarray | pd.Series,
        group_b: np.ndarray | pd.Series,
    ) -> dict:
        """
        Mann-Whitney U test — non-parametric alternative to the t-test.
        Use when the outcome is not normally distributed.
        """
        stat, p = stats.mannwhitneyu(group_a, group_b, alternative="two-sided")

        return {
            "test": "Mann-Whitney U",
            "statistic": round(float(stat), 4),
            "p_value": round(float(p), 4),
            "significant": float(p) < self.alpha,
            "interpretation": self._interpret(float(p), "distribution difference"),
        }

    def chi_square(
        self,
        df: pd.DataFrame,
        col_a: str,
        col_b: str,
    ) -> dict:
        """
        Chi-square test of independence between two categorical columns.

        Args:
            df: DataFrame containing the columns.
            col_a: First categorical column name.
            col_b: Second categorical column name.
        """
        contingency = pd.crosstab(df[col_a], df[col_b])
        stat, p, dof, _ = stats.chi2_contingency(contingency)

        return {
            "test": "Chi-square",
            "statistic": round(float(stat), 4),
            "p_value": round(float(p), 4),
            "degrees_of_freedom": int(dof),
            "significant": float(p) < self.alpha,
            "interpretation": self._interpret(float(p), "association"),
        }

    def bootstrap_ci(
        self,
        group_a: np.ndarray | pd.Series,
        group_b: np.ndarray | pd.Series,
        n_bootstrap: int = 1000,
        ci: float = 0.95,
        seed: int = 42,
    ) -> dict:
        """
        Bootstrap confidence interval for the difference in means.

        Args:
            group_a: Treatment group outcomes.
            group_b: Control group outcomes.
            n_bootstrap: Number of bootstrap samples.
            ci: Confidence interval level (default 0.95).
            seed: Random seed for reproducibility.
        """
        rng = np.random.default_rng(seed)
        group_a = np.array(group_a)
        group_b = np.array(group_b)

        boot_diffs = []
        for _ in range(n_bootstrap):
            sample_a = rng.choice(group_a, size=len(group_a), replace=True)
            sample_b = rng.choice(group_b, size=len(group_b), replace=True)
            boot_diffs.append(np.mean(sample_a) - np.mean(sample_b))

        lower = np.percentile(boot_diffs, (1 - ci) / 2 * 100)
        upper = np.percentile(boot_diffs, (1 + ci) / 2 * 100)
        observed_diff = np.mean(group_a) - np.mean(group_b)

        return {
            "observed_diff": round(float(observed_diff), 4),
            "ci_lower": round(float(lower), 4),
            "ci_upper": round(float(upper), 4),
            "ci_level": ci,
            "n_bootstrap": n_bootstrap,
            "significant": not (lower <= 0 <= upper),
        }

    def normality_test(self, values: np.ndarray | pd.Series) -> dict:
        """
        Shapiro-Wilk normality test.
        For n > 5000 falls back to D'Agostino-Pearson (Shapiro-Wilk is slow on large samples).
        """
        values = np.array(values)

        if len(values) > 5000:
            stat, p = stats.normaltest(values)
            test_name = "D'Agostino-Pearson"
        else:
            stat, p = stats.shapiro(values)
            test_name = "Shapiro-Wilk"

        return {
            "test": test_name,
            "statistic": round(float(stat), 4),
            "p_value": round(float(p), 4),
            "is_normal": float(p) >= self.alpha,
            "interpretation": (
                "Data appears normally distributed (p >= alpha)."
                if float(p) >= self.alpha
                else "Data does not appear normally distributed (p < alpha). "
                     "Consider Mann-Whitney U instead of t-test."
            ),
        }

    def sample_size_calculator(
        self,
        baseline_mean: float,
        mde: float,
        std_dev: float,
        power: float = 0.8,
    ) -> dict:
        """
        Calculates required sample size per group for a two-sided t-test.

        Args:
            baseline_mean: Expected mean in the control group.
            mde: Minimum detectable effect (absolute difference).
            std_dev: Expected standard deviation of the outcome.
            power: Desired statistical power. Default 0.8.
        """
        effect_size = mde / std_dev
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        z_beta = stats.norm.ppf(power)
        n = ((z_alpha + z_beta) / effect_size) ** 2

        return {
            "required_n_per_group": int(np.ceil(n)),
            "total_required_n": int(np.ceil(n)) * 2,
            "effect_size_cohens_d": round(effect_size, 4),
            "alpha": self.alpha,
            "power": power,
            "baseline_mean": baseline_mean,
            "mde": mde,
        }

    def _interpret(self, p_value: float, effect_label: str) -> str:
        if p_value < self.alpha:
            return (
                f"Statistically significant {effect_label} detected "
                f"(p={p_value:.4f} < {self.alpha})."
            )
        return (
            f"No statistically significant {effect_label} detected "
            f"(p={p_value:.4f} >= {self.alpha})."
        )