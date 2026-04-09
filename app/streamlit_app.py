import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
import numpy as np

from src.data_loader import DataLoader, DataValidationError, generate_sample_dataset
from src.psm import PropensityScoreMatcher, PSMError
from src.did import DifferenceInDifferences, DIDError
from src.stats_tests import StatisticalTests
from src.cate import CATEEstimator, CATEError
from src import visualizations as viz


st.set_page_config(
    page_title="Causal Inference Engine",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Causal Inference A/B Testing Engine")
st.caption(
    "Run PSM, DiD, and CATE analysis on your A/B test or observational data."
)

# ── Sidebar: data upload and column config ─────────────────────────────────────

with st.sidebar:
    st.header("Dataset")

    use_sample = st.checkbox("Use built-in sample dataset", value=True)

    if use_sample:
        df_raw = generate_sample_dataset(n=1000)
        st.success("Sample HR dataset loaded (1,000 rows).")
    else:
        uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
        if uploaded is None:
            st.info("Upload a CSV to get started.")
            st.stop()
        df_raw = pd.read_csv(uploaded)

    st.divider()
    st.header("Column configuration")

    all_cols = list(df_raw.columns)

    treatment_col = st.selectbox(
        "Treatment column (binary 0/1)",
        options=all_cols,
        index=all_cols.index("treatment") if "treatment" in all_cols else 0,
    )
    outcome_col = st.selectbox(
        "Outcome column (numeric)",
        options=[c for c in all_cols if c != treatment_col],
        index=0,
    )
    remaining_cols = [c for c in all_cols if c not in [treatment_col, outcome_col]]
    covariate_cols = st.multiselect(
        "Covariate columns (for matching)",
        options=remaining_cols,
        default=remaining_cols[:4] if len(remaining_cols) >= 4 else remaining_cols,
    )

    time_col = st.selectbox(
        "Time column for DiD (binary pre=0/post=1) — optional",
        options=["None"] + all_cols,
        index=0,
    )
    time_col = None if time_col == "None" else time_col

    if not covariate_cols:
        st.warning("Select at least one covariate to run analysis.")
        st.stop()

    st.divider()
    st.header("PSM settings")
    caliper = st.slider("Caliper (max score distance)", 0.01, 0.3, 0.05, 0.01)

    st.header("Stats settings")
    alpha = st.slider("Significance level (alpha)", 0.01, 0.10, 0.05, 0.01)


# ── Load + validate data ───────────────────────────────────────────────────────

try:
    loader = DataLoader(treatment_col, outcome_col, covariate_cols)
    df = loader.load_from_dataframe(df_raw)
except DataValidationError as e:
    st.error(f"Data validation error: {e}")
    st.stop()

summary = loader.summary(df)

# ── Dataset overview ───────────────────────────────────────────────────────────

st.subheader("Dataset overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total rows", summary["total_rows"])
col2.metric("Treated", summary["treated"])
col3.metric("Control", summary["control"])
col4.metric(
    "Raw mean difference",
    round(summary["outcome_mean_treated"] - summary["outcome_mean_control"], 3),
)

with st.expander("Preview data"):
    st.dataframe(df.head(50), use_container_width=True)

st.divider()

# ── Tabs for each analysis ─────────────────────────────────────────────────────

tab_psm, tab_did, tab_cate, tab_stats = st.tabs([
    "Propensity Score Matching",
    "Difference-in-Differences",
    "CATE Estimation",
    "Statistical Tests",
])


# ── Tab 1: PSM ─────────────────────────────────────────────────────────────────

with tab_psm:
    st.subheader("Propensity Score Matching")
    st.write(
        "Matches treated and control units on their estimated probability of treatment "
        "to reduce selection bias before comparing outcomes."
    )

    run_psm = st.button("Run PSM", key="btn_psm")
    if run_psm:
        with st.spinner("Running propensity score matching..."):
            try:
                matcher = PropensityScoreMatcher(caliper=caliper)
                matcher.fit(df, treatment_col, outcome_col, covariate_cols)
                psm_summary = matcher.summary()
                balance_df = matcher.get_balance_stats(df)

                c1, c2, c3 = st.columns(3)
                c1.metric("ATE (matched)", psm_summary["ate"])
                c2.metric(
                    "95% CI",
                    f"[{psm_summary['ate_ci_lower']}, {psm_summary['ate_ci_upper']}]"
                )
                c3.metric("Matched pairs", psm_summary["matched_pairs"])

                col_a, col_b = st.columns(2)
                with col_a:
                    st.plotly_chart(
                        viz.plot_propensity_score_distribution(
                            df, matcher.propensity_scores_, treatment_col
                        ),
                        use_container_width=True,
                    )
                with col_b:
                    st.plotly_chart(
                        viz.plot_covariate_balance(balance_df),
                        use_container_width=True,
                    )

                st.plotly_chart(
                    viz.plot_ate_with_ci(psm_summary, "PSM"),
                    use_container_width=True,
                )

                st.subheader("Balance diagnostics")
                st.dataframe(balance_df, use_container_width=True)

                unbalanced = balance_df[~balance_df["balanced"]]
                if len(unbalanced) > 0:
                    st.warning(
                        f"{len(unbalanced)} covariate(s) remain imbalanced after matching: "
                        f"{unbalanced['covariate'].tolist()}. "
                        "Try adjusting the caliper or adding more covariates."
                    )
                else:
                    st.success("All covariates are balanced after matching (SMD < 0.1).")

            except PSMError as e:
                st.error(f"PSM failed: {e}")


# ── Tab 2: DiD ─────────────────────────────────────────────────────────────────

with tab_did:
    st.subheader("Difference-in-Differences")

    if time_col is None:
        st.info(
            "DiD requires a time column (pre/post indicator). "
            "Select one in the sidebar under 'Time column for DiD'."
        )
    else:
        st.write(
            "Estimates the treatment effect by comparing the change in outcomes over "
            "time between the treatment and control groups."
        )

        run_did = st.button("Run DiD", key="btn_did")
        if run_did:
            with st.spinner("Fitting DiD model..."):
                try:
                    did = DifferenceInDifferences()
                    did.fit(df, treatment_col, outcome_col, time_col, covariate_cols)
                    did_summary = did.summary()
                    group_means = did.get_group_means(df)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("DiD Estimate", did_summary["did_estimate"])
                    c2.metric("p-value", did_summary["p_value"])
                    c3.metric(
                        "Significant",
                        "Yes ✓" if did_summary["significant"] else "No ✗"
                    )

                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.plotly_chart(
                            viz.plot_did_parallel_trends(group_means),
                            use_container_width=True,
                        )
                    with col_b:
                        st.plotly_chart(
                            viz.plot_ate_with_ci(did_summary, "DiD"),
                            use_container_width=True,
                        )

                    with st.expander("Full regression table"):
                        st.dataframe(did.regression_table(), use_container_width=True)

                except DIDError as e:
                    st.error(f"DiD failed: {e}")


# ── Tab 3: CATE ────────────────────────────────────────────────────────────────

with tab_cate:
    st.subheader("CATE — Conditional Average Treatment Effect")
    st.write(
        "Estimates how the treatment effect varies across different subgroups using "
        "a T-Learner with GradientBoosting."
    )

    subgroup_col = st.selectbox(
        "Subgroup column for CATE breakdown",
        options=covariate_cols,
        key="cate_subgroup",
    )

    run_cate = st.button("Run CATE", key="btn_cate")
    if run_cate:
        with st.spinner("Fitting CATE model..."):
            try:
                cate_model = CATEEstimator()
                cate_model.fit(df, treatment_col, outcome_col, covariate_cols)

                overall_ate = cate_model.ate()
                cate_by_group = cate_model.cate_by_subgroup(df, subgroup_col)
                importance_df = cate_model.feature_importance()
                ite_df = cate_model.get_ite_dataframe(df)

                st.metric("Overall ATE (CATE model)", overall_ate)

                col_a, col_b = st.columns(2)
                with col_a:
                    st.plotly_chart(
                        viz.plot_ite_distribution(cate_model.ite_scores_),
                        use_container_width=True,
                    )
                with col_b:
                    st.plotly_chart(
                        viz.plot_cate_by_subgroup(cate_by_group, subgroup_col),
                        use_container_width=True,
                    )

                st.plotly_chart(
                    viz.plot_feature_importance(importance_df),
                    use_container_width=True,
                )

                with st.expander("CATE by subgroup — full table"):
                    st.dataframe(cate_by_group, use_container_width=True)

            except CATEError as e:
                st.error(f"CATE failed: {e}")


# ── Tab 4: Statistical tests ───────────────────────────────────────────────────

with tab_stats:
    st.subheader("Statistical Tests")

    tester = StatisticalTests(alpha=alpha)
    treated_outcomes = df[df[treatment_col] == 1][outcome_col]
    control_outcomes = df[df[treatment_col] == 0][outcome_col]

    st.plotly_chart(
        viz.plot_outcome_distributions(df, treatment_col, outcome_col),
        use_container_width=True,
    )

    st.subheader("Normality check")
    norm_result = tester.normality_test(df[outcome_col].values)
    c1, c2 = st.columns(2)
    c1.metric("Test", norm_result["test"])
    c2.metric("p-value", norm_result["p_value"])
    st.info(norm_result["interpretation"])

    st.subheader("Group comparison")
    col_t, col_mw, col_boot = st.columns(3)

    ttest_result = tester.two_sample_ttest(treated_outcomes, control_outcomes)
    mw_result = tester.mann_whitney(treated_outcomes, control_outcomes)
    boot_result = tester.bootstrap_ci(treated_outcomes.values, control_outcomes.values)

    with col_t:
        st.markdown("**Welch's t-test**")
        st.metric("Mean difference", ttest_result["mean_diff"])
        st.metric("p-value", ttest_result["p_value"])
        st.caption(ttest_result["interpretation"])

    with col_mw:
        st.markdown("**Mann-Whitney U**")
        st.metric("Statistic", mw_result["statistic"])
        st.metric("p-value", mw_result["p_value"])
        st.caption(mw_result["interpretation"])

    with col_boot:
        st.markdown("**Bootstrap 95% CI**")
        st.metric("Observed diff", boot_result["observed_diff"])
        st.metric(
            "95% CI",
            f"[{boot_result['ci_lower']}, {boot_result['ci_upper']}]"
        )
        st.caption(
            "Significant" if boot_result["significant"] else "Not significant"
        )

    st.divider()
    st.subheader("Sample size calculator")

    sc1, sc2, sc3, sc4 = st.columns(4)
    baseline = sc1.number_input("Baseline mean", value=float(control_outcomes.mean().round(2)))
    mde = sc2.number_input("Min detectable effect", value=1.0, step=0.1)
    std_dev = sc3.number_input("Std deviation", value=float(df[outcome_col].std().round(2)))
    power = sc4.slider("Power", 0.7, 0.95, 0.8, 0.05)

    ss_result = tester.sample_size_calculator(baseline, mde, std_dev, power)
    s1, s2, s3 = st.columns(3)
    s1.metric("Required per group", ss_result["required_n_per_group"])
    s2.metric("Total required", ss_result["total_required_n"])
    s3.metric("Cohen's d", ss_result["effect_size_cohens_d"])