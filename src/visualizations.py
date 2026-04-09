import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


COLORS = {
    "treated": "#5A8FF5",
    "control": "#F5875A",
    "positive": "#4CAF7D",
    "negative": "#E05C5C",
    "neutral": "#888888",
}


def plot_propensity_score_distribution(
    df: pd.DataFrame,
    propensity_scores: np.ndarray,
    treatment_col: str,
) -> go.Figure:
    """Overlapping histogram of propensity scores for treated vs control."""
    df = df.copy()
    df["_propensity"] = propensity_scores

    treated_scores = df[df[treatment_col] == 1]["_propensity"]
    control_scores = df[df[treatment_col] == 0]["_propensity"]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=treated_scores,
        name="Treated",
        opacity=0.6,
        marker_color=COLORS["treated"],
        nbinsx=30,
    ))
    fig.add_trace(go.Histogram(
        x=control_scores,
        name="Control",
        opacity=0.6,
        marker_color=COLORS["control"],
        nbinsx=30,
    ))
    fig.update_layout(
        barmode="overlay",
        title="Propensity Score Distribution",
        xaxis_title="Propensity Score",
        yaxis_title="Count",
        legend=dict(orientation="h", y=1.1),
        height=380,
    )
    return fig


def plot_covariate_balance(balance_df: pd.DataFrame) -> go.Figure:
    """
    Love plot — standardized mean differences before and after matching.
    The vertical dashed line at 0.1 is the commonly accepted balance threshold.
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=balance_df["smd_before"].abs(),
        y=balance_df["covariate"],
        mode="markers",
        name="Before matching",
        marker=dict(color=COLORS["control"], size=10, symbol="circle"),
    ))
    fig.add_trace(go.Scatter(
        x=balance_df["smd_after"].abs(),
        y=balance_df["covariate"],
        mode="markers",
        name="After matching",
        marker=dict(color=COLORS["treated"], size=10, symbol="diamond"),
    ))

    fig.add_vline(
        x=0.1,
        line_dash="dash",
        line_color="gray",
        annotation_text="Balance threshold (0.1)",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Covariate Balance (Love Plot)",
        xaxis_title="Absolute Standardized Mean Difference",
        yaxis_title="Covariate",
        legend=dict(orientation="h", y=1.1),
        height=max(300, len(balance_df) * 50),
    )
    return fig


def plot_ate_with_ci(results: dict, method_name: str) -> go.Figure:
    """Bar chart showing ATE with 95% confidence interval."""
    ate = results.get("ate") or results.get("did_estimate")
    ci_lower = results.get("ate_ci_lower") or results.get("ci_lower")
    ci_upper = results.get("ate_ci_upper") or results.get("ci_upper")

    color = COLORS["positive"] if ate >= 0 else COLORS["negative"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[method_name],
        y=[ate],
        error_y=dict(
            type="data",
            symmetric=False,
            array=[ci_upper - ate],
            arrayminus=[ate - ci_lower],
            color="black",
            thickness=2,
        ),
        marker_color=color,
        width=0.35,
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=f"Average Treatment Effect — {method_name}",
        yaxis_title="Estimated ATE",
        height=350,
        showlegend=False,
    )
    return fig


def plot_did_parallel_trends(group_means_df: pd.DataFrame) -> go.Figure:
    """
    Line chart showing pre/post means for treated and control groups.
    Used to visually inspect the parallel trends assumption.
    """
    period_order = {"Pre": 0, "Post": 1}
    group_means_df = group_means_df.copy()
    group_means_df["period_order"] = group_means_df["period"].map(period_order)
    group_means_df = group_means_df.sort_values("period_order")

    fig = go.Figure()
    for group, color in [("Treated", COLORS["treated"]), ("Control", COLORS["control"])]:
        subset = group_means_df[group_means_df["group"] == group]
        fig.add_trace(go.Scatter(
            x=subset["period"],
            y=subset["mean_outcome"],
            mode="lines+markers",
            name=group,
            line=dict(color=color, width=2),
            marker=dict(size=9),
        ))

    fig.update_layout(
        title="Parallel Trends Check",
        xaxis_title="Period",
        yaxis_title="Mean Outcome",
        legend=dict(orientation="h", y=1.1),
        height=370,
    )
    return fig


def plot_ite_distribution(ite_scores: np.ndarray) -> go.Figure:
    """Histogram of individual treatment effects from CATE estimation."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=ite_scores,
        nbinsx=40,
        marker_color=COLORS["treated"],
        opacity=0.8,
    ))
    fig.add_vline(
        x=float(np.mean(ite_scores)),
        line_dash="dash",
        line_color="black",
        annotation_text=f"Mean ITE: {np.mean(ite_scores):.3f}",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Distribution of Individual Treatment Effects (CATE)",
        xaxis_title="Estimated ITE",
        yaxis_title="Count",
        height=370,
    )
    return fig


def plot_cate_by_subgroup(cate_df: pd.DataFrame, subgroup_col: str) -> go.Figure:
    """
    Horizontal bar chart of mean CATE per subgroup value.
    Bars are colored green (positive effect) or red (negative effect).
    """
    cate_df = cate_df.copy().sort_values("mean_cate")
    colors = [
        COLORS["positive"] if v >= 0 else COLORS["negative"]
        for v in cate_df["mean_cate"]
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=cate_df["mean_cate"],
        y=cate_df[subgroup_col].astype(str),
        orientation="h",
        marker_color=colors,
        error_x=dict(
            type="data",
            array=cate_df["std_cate"].tolist(),
            color="black",
            thickness=1.5,
        ),
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title=f"CATE by {subgroup_col}",
        xaxis_title="Mean Treatment Effect",
        yaxis_title=subgroup_col,
        height=max(300, len(cate_df) * 45),
    )
    return fig


def plot_feature_importance(importance_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of feature importances from the CATE model."""
    importance_df = importance_df.sort_values("importance")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=importance_df["importance"],
        y=importance_df["feature"],
        orientation="h",
        marker_color=COLORS["treated"],
    ))
    fig.update_layout(
        title="Feature Importance (CATE Model)",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=max(300, len(importance_df) * 40),
    )
    return fig


def plot_outcome_distributions(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
) -> go.Figure:
    """Side-by-side box plots of outcome distribution for each group."""
    treated = df[df[treatment_col] == 1][outcome_col]
    control = df[df[treatment_col] == 0][outcome_col]

    fig = go.Figure()
    fig.add_trace(go.Box(
        y=treated,
        name="Treated",
        marker_color=COLORS["treated"],
        boxmean=True,
    ))
    fig.add_trace(go.Box(
        y=control,
        name="Control",
        marker_color=COLORS["control"],
        boxmean=True,
    ))
    fig.update_layout(
        title=f"Outcome Distribution by Group — {outcome_col}",
        yaxis_title=outcome_col,
        height=370,
    )
    return fig
