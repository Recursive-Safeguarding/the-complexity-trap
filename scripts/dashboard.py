#!/usr/bin/env python3
"""
Complexity Trap Analysis Dashboard (MVP)

Single-file Streamlit dashboard for analyzing context management experiments.
Implements: Pareto scatter, Exit status bar, Instance explorer.

Usage:
    streamlit run scripts/dashboard.py
    # Or with custom project/entity:
    DASHBOARD_PROJECT=my-project DASHBOARD_ENTITY=my-entity streamlit run scripts/dashboard.py
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Paper baselines from arXiv:2508.21433 - Table 1 (all 5 model configurations)
# NOTE: Paper reports solve_rate (bugs actually fixed)
# Our WandB data now has solve_rate from SWE-bench evaluation (--evaluate flag)
# solve_rate = n_resolved / n_instances (directly comparable to paper)
#
# Each entry includes: solve_rate, avg_cost, confidence intervals (ci), and relative change vs raw (delta)
PAPER_BASELINES = {
    "qwen3-32b": {
        "raw": {"solve_rate": 0.170, "avg_cost": 1.12, "rate_ci": 0.033, "cost_ci": 0.18},
        "observation_masking": {"solve_rate": 0.150, "avg_cost": 0.55, "rate_ci": 0.031, "cost_ci": 0.09,
                                "rate_delta": -0.118, "cost_delta": -0.509},
        "llm_summary": {"solve_rate": 0.160, "avg_cost": 0.50, "rate_ci": 0.033, "cost_ci": 0.07,
                        "rate_delta": -0.059, "cost_delta": -0.554},
    },
    "qwen3-32b-thinking": {
        "raw": {"solve_rate": 0.230, "avg_cost": 0.51, "rate_ci": 0.037, "cost_ci": 0.07},
        "observation_masking": {"solve_rate": 0.246, "avg_cost": 0.46, "rate_ci": 0.038, "cost_ci": 0.05,
                                "rate_delta": 0.070, "cost_delta": -0.098},
        "llm_summary": {"solve_rate": 0.248, "avg_cost": 0.51, "rate_ci": 0.039, "cost_ci": 0.06,
                        "rate_delta": 0.073, "cost_delta": 0.0},
    },
    "qwen3-coder-480b": {
        "raw": {"solve_rate": 0.534, "avg_cost": 1.29, "rate_ci": 0.043, "cost_ci": 0.26},
        "observation_masking": {"solve_rate": 0.548, "avg_cost": 0.61, "rate_ci": 0.044, "cost_ci": 0.06,
                                "rate_delta": 0.026, "cost_delta": -0.527},
        "llm_summary": {"solve_rate": 0.538, "avg_cost": 0.64, "rate_ci": 0.042, "cost_ci": 0.06,
                        "rate_delta": 0.007, "cost_delta": -0.504},
        # Hybrid from Section 5.3: N=43, M=W=10 on SWE-bench Verified-50
        "hybrid": {"solve_rate": 0.540, "avg_cost": 0.50, "rate_ci": 0.044, "cost_ci": 0.05,
                   "rate_delta": 0.011, "cost_delta": -0.612},
    },
    "gemini-2.5-flash": {
        "raw": {"solve_rate": 0.328, "avg_cost": 0.41, "rate_ci": 0.041, "cost_ci": 0.08},
        "observation_masking": {"solve_rate": 0.356, "avg_cost": 0.18, "rate_ci": 0.042, "cost_ci": 0.03,
                                "rate_delta": 0.085, "cost_delta": -0.561},
        "llm_summary": {"solve_rate": 0.360, "avg_cost": 0.24, "rate_ci": 0.041, "cost_ci": 0.04,
                        "rate_delta": 0.098, "cost_delta": -0.415},
    },
    "gemini-2.5-flash-thinking": {
        "raw": {"solve_rate": 0.404, "avg_cost": 0.56, "rate_ci": 0.043, "cost_ci": 0.10},
        "observation_masking": {"solve_rate": 0.364, "avg_cost": 0.24, "rate_ci": 0.042, "cost_ci": 0.04,
                                "rate_delta": -0.099, "cost_delta": -0.571},
        "llm_summary": {"solve_rate": 0.314, "avg_cost": 0.25, "rate_ci": 0.040, "cost_ci": 0.05,
                        "rate_delta": -0.223, "cost_delta": -0.554},
    },
}

# Table 2: LLM-Summary generation costs per model
SUMMARY_COSTS = {
    "qwen3-32b": {"cost": 0.0143, "pct": 2.86},
    "qwen3-32b-thinking": {"cost": 0.0033, "pct": 0.65},
    "qwen3-coder-480b": {"cost": 0.0439, "pct": 7.20},
    "gemini-2.5-flash": {"cost": 0.0161, "pct": 6.71},
    "gemini-2.5-flash-thinking": {"cost": 0.0131, "pct": 5.24},
}

# Hyperparameters from paper
STRATEGY_PARAMS = {
    "observation_masking": {"M": 10, "description": "Keep last M=10 observations"},
    "llm_summary": {"N": 21, "M": 10, "description": "Summarize every N=21 turns, keep M=10 tail"},
    "hybrid": {"N": 43, "M": 10, "W": 10, "description": "N=43 summary trigger, M=W=10 for masking+tail"},
}

# Exit status colors (matches WandB hook taxonomy)
EXIT_COLORS = {
    "exit_submitted": "#22c55e",  # green - success
    "exit_cost": "#f59e0b",  # amber - cost limit
    "exit_context": "#ef4444",  # red - context overflow
    "exit_timeout": "#3b82f6",  # blue - timeout
    "exit_format": "#a855f7",  # purple - format error
    "exit_other": "#6b7280",  # gray - other
}


@st.cache_data(ttl=600)
def fetch_runs(project: str, entity: str | None = None) -> pd.DataFrame:
    """Fetch all runs from WandB, return as DataFrame."""
    import wandb

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project

    try:
        runs = api.runs(path)
    except wandb.errors.CommError as e:
        st.error(f"Failed to connect to WandB: {e}")
        st.info("Check your WANDB_API_KEY environment variable.")
        st.stop()

    records = []
    for r in runs:
        config = r.config or {}
        summary = r.summary._json_dict if r.summary else {}

        record = {
            "run_id": r.id,
            "run_name": r.name,
            "state": r.state,
            "created_at": r.created_at,
            # Config
            "model": config.get("model", "unknown"),
            "strategy": config.get("strategy", "unknown"),
            "summarizer": config.get("summarizer_model", "same"),
            "instances_subset": config.get("instances_subset", "verified"),
            # Core metrics
            "n_instances": summary.get("n_instances", 0),
            "n_submitted": summary.get("n_submitted", 0),
            "n_resolved": summary.get("n_resolved", 0),
            "submission_rate": summary.get("submission_rate", 0),
            "resolved_rate": summary.get("resolved_rate", 0),  # From evaluation
            "solve_rate": summary.get("solve_rate", 0),  # n_resolved / n_instances
            "avg_cost": summary.get("avg_cost") or 0,  # Handle None for Bedrock
            "avg_turns": summary.get("avg_turns", 0),
            "cache_hit_rate": summary.get("cache_hit_rate", 0),
            "total_cost": summary.get("total_cost") or 0,
            # Exit distribution (NOTE: forward slash in WandB names!)
            "exit_submitted": summary.get("exit/submitted", 0),
            "exit_cost": summary.get("exit/exit_cost", 0),
            "exit_context": summary.get("exit/exit_context", 0),
            "exit_timeout": summary.get("exit/exit_timeout", 0),
            "exit_format": summary.get("exit/exit_format", 0),
            "exit_other": summary.get("exit/other", 0),
            # Cost breakdown
            "total_agent_cost": summary.get("total_agent_cost") or 0,
            "total_summary_cost": summary.get("total_summary_cost") or 0,
            "summary_cost_fraction": summary.get("summary_cost_fraction", 0),
        }
        records.append(record)

    return pd.DataFrame(records)


def build_pareto_plot(df: pd.DataFrame, show_baselines: bool = True) -> go.Figure:
    """Build Pareto scatter: cost (log x) vs solve rate (y)."""
    # Filter out runs with missing cost
    df_valid = df[df["avg_cost"] > 0].copy()

    if df_valid.empty:
        fig = go.Figure()
        fig.add_annotation(text="No runs with cost data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_dark")
        return fig

    fig = px.scatter(
        df_valid,
        x="avg_cost",
        y="solve_rate",
        color="strategy",
        symbol="model",
        size="n_instances",
        hover_name="run_name",
        hover_data=["model", "strategy", "n_instances", "avg_turns", "submission_rate"],
        labels={"avg_cost": "Avg Cost ($)", "solve_rate": "Solve Rate"},
    )

    fig.update_xaxes(type="log", title="Avg Cost ($)")
    fig.update_yaxes(title="Solve Rate", tickformat=".0%")
    fig.update_layout(
        template="plotly_dark",
        title="",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        margin=dict(t=20, b=100),
    )

    if show_baselines:
        # Add paper reference points (using solve_rate from paper)
        for strategy, vals in PAPER_BASELINES.get("qwen3-coder-480b", {}).items():
            fig.add_scatter(
                x=[vals["avg_cost"]],
                y=[vals["solve_rate"]],
                mode="markers",
                marker=dict(symbol="x", size=14, color="white", line=dict(width=2)),
                name=f"Paper: {strategy}",
                showlegend=False,
                hovertemplate=f"Paper: {strategy}<br>Cost: ${vals['avg_cost']:.2f}<br>Solve Rate: {vals['solve_rate']:.1%}<extra></extra>",
            )

    return fig


def build_exit_status_bar(df: pd.DataFrame) -> go.Figure:
    """Build stacked bar chart of exit status distribution by strategy."""
    exit_cols = ["exit_submitted", "exit_cost", "exit_context", "exit_timeout", "exit_format", "exit_other"]

    # Sum exit counts per strategy
    exit_df = df.groupby("strategy")[exit_cols].sum().reset_index()

    # Melt for plotting
    exit_melted = exit_df.melt(id_vars="strategy", var_name="exit_type", value_name="count")

    # Pretty labels
    label_map = {
        "exit_submitted": "Submitted",
        "exit_cost": "Cost Limit",
        "exit_context": "Context Overflow",
        "exit_timeout": "Timeout",
        "exit_format": "Format Error",
        "exit_other": "Other",
    }
    exit_melted["exit_label"] = exit_melted["exit_type"].map(label_map)

    fig = px.bar(
        exit_melted,
        x="strategy",
        y="count",
        color="exit_type",
        labels={"count": "Count", "strategy": "Strategy"},
        color_discrete_map=EXIT_COLORS,
        category_orders={"exit_type": exit_cols},
    )

    fig.update_layout(
        template="plotly_dark",
        title="",
        height=500,
        legend=dict(
            title_text="",
            orientation="h",
            yanchor="top",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=11),
        ),
        barmode="stack",
        margin=dict(t=20, b=100),
        xaxis_tickangle=-30,
    )

    # Update legend labels
    for trace in fig.data:
        if trace.name in label_map:
            trace.name = label_map[trace.name]

    return fig


def render_sidebar(df: pd.DataFrame) -> dict[str, Any]:
    """Render sidebar filters and return filter values."""
    with st.sidebar:
        st.header("Filters")

        # Model filter
        all_models = sorted(df["model"].unique())
        models = st.multiselect("Models", all_models, default=all_models)

        # Strategy filter
        all_strategies = sorted(df["strategy"].unique())
        strategies = st.multiselect("Strategies", all_strategies, default=all_strategies)

        # Exit status filter
        exit_options = ["All", "Submitted Only", "Failed Only"]
        exit_filter = st.selectbox("Exit Status", exit_options)

        # Min instances filter
        min_instances = st.slider("Min Instances", 0, 100, 0)

        # Show baselines checkbox
        show_baselines = st.checkbox("Show Paper Baselines", value=True)

        st.divider()

        # Refresh button
        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.rerun()

        # Project info
        project = os.environ.get("DASHBOARD_PROJECT") or os.environ.get("WANDB_PROJECT", "")
        st.caption(f"Project: {project}")

    return {
        "models": models,
        "strategies": strategies,
        "exit_filter": exit_filter,
        "min_instances": min_instances,
        "show_baselines": show_baselines,
    }


def apply_filters(df: pd.DataFrame, filters: dict[str, Any]) -> pd.DataFrame:
    """Apply sidebar filters to DataFrame."""
    filtered = df.copy()

    # Model filter
    if filters["models"]:
        filtered = filtered[filtered["model"].isin(filters["models"])]

    # Strategy filter
    if filters["strategies"]:
        filtered = filtered[filtered["strategy"].isin(filters["strategies"])]

    # Exit status filter
    if filters["exit_filter"] == "Submitted Only":
        filtered = filtered[filtered["exit_submitted"] > 0]
    elif filters["exit_filter"] == "Failed Only":
        filtered = filtered[filtered["exit_submitted"] == 0]

    # Min instances filter
    filtered = filtered[filtered["n_instances"] >= filters["min_instances"]]

    return filtered


def render_metrics(df: pd.DataFrame):
    """Render top-level metrics row."""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Runs", len(df))

    with col2:
        st.metric("Models", df["model"].nunique())

    with col3:
        st.metric("Strategies", df["strategy"].nunique())

    with col4:
        best_rate = df["solve_rate"].max() if not df.empty else 0
        st.metric("Best Solve Rate", f"{best_rate:.1%}")


def render_instance_explorer(df: pd.DataFrame):
    """Render instance explorer table."""
    st.subheader("Run Explorer")

    if df.empty:
        st.warning("No runs match the current filters.")
        return

    # Prepare display columns
    display_cols = [
        "run_name",
        "model",
        "strategy",
        "n_instances",
        "n_resolved",
        "solve_rate",
        "avg_cost",
        "avg_turns",
        "exit_submitted",
        "exit_cost",
    ]

    display_df = df[display_cols].copy()
    display_df["solve_rate"] = display_df["solve_rate"].apply(lambda x: f"{x:.1%}")
    display_df["avg_cost"] = display_df["avg_cost"].apply(lambda x: f"${x:.3f}" if x > 0 else "N/A")

    # Rename columns for display
    display_df.columns = [
        "Run",
        "Model",
        "Strategy",
        "N",
        "Resolved",
        "Solve Rate",
        "Avg Cost",
        "Avg Turns",
        "Submitted",
        "Cost Exit",
    ]

    st.dataframe(
        display_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Run": st.column_config.TextColumn(width="large"),
            "Model": st.column_config.TextColumn(width="medium"),
            "Strategy": st.column_config.TextColumn(width="medium"),
        },
    )

    # Show reproduction command for selected run
    if not df.empty:
        st.caption("Select a run above to see details. Run IDs can be used for WandB queries.")


def build_pareto_with_all_baselines(df: pd.DataFrame) -> go.Figure:
    """Build Pareto scatter with ALL paper baselines (all 5 models)."""
    df_valid = df[df["avg_cost"] > 0].copy()

    if df_valid.empty:
        fig = go.Figure()
        fig.add_annotation(text="No runs with cost data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_dark")
        return fig

    fig = px.scatter(
        df_valid,
        x="avg_cost",
        y="solve_rate",
        color="strategy",
        symbol="model",
        size="n_instances",
        hover_name="run_name",
        hover_data=["model", "strategy", "n_instances", "avg_turns", "n_resolved"],
        labels={"avg_cost": "Avg Cost ($)", "solve_rate": "Solve Rate"},
    )

    fig.update_xaxes(type="log", title="Avg Cost ($)")
    fig.update_yaxes(title="Solve Rate", tickformat=".0%")

    # Add ALL paper baselines with different colors per model (with error bars)
    model_colors = {
        "qwen3-32b": "#60a5fa",  # blue
        "qwen3-32b-thinking": "#34d399",  # green
        "qwen3-coder-480b": "#f472b6",  # pink
        "gemini-2.5-flash": "#fbbf24",  # yellow
        "gemini-2.5-flash-thinking": "#a78bfa",  # purple
    }

    for model, strategies in PAPER_BASELINES.items():
        color = model_colors.get(model, "white")
        for strategy, vals in strategies.items():
            rate_ci = vals.get("rate_ci", 0)
            cost_ci = vals.get("cost_ci", 0)
            fig.add_scatter(
                x=[vals["avg_cost"]],
                y=[vals["solve_rate"]],
                error_y=dict(type="data", array=[rate_ci], visible=True, color=color, thickness=1.5),
                error_x=dict(type="data", array=[cost_ci], visible=True, color=color, thickness=1.5),
                mode="markers",
                marker=dict(symbol="x", size=12, color=color, line=dict(width=2, color="white")),
                name=f"Paper: {model}",
                showlegend=False,
                hovertemplate=f"Paper: {model}<br>Strategy: {strategy}<br>Solve Rate: {vals['solve_rate']:.1%} ±{rate_ci:.1%}<br>Cost: ${vals['avg_cost']:.2f} ±${cost_ci:.2f}<extra></extra>",
            )

    fig.update_layout(
        template="plotly_dark",
        title="",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
        ),
        margin=dict(t=20, b=120),
    )

    return fig


def build_turn_boxplot(df: pd.DataFrame) -> go.Figure:
    """Figure 4 equivalent: Turn count distribution by strategy."""
    if df.empty or "avg_turns" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="No turn data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_dark")
        return fig

    fig = px.box(
        df,
        x="strategy",
        y="avg_turns",
        color="strategy",
        points="outliers",
        labels={"avg_turns": "Avg Turns per Instance", "strategy": "Strategy"},
    )
    fig.update_layout(
        template="plotly_dark",
        title="",
        height=500,
        showlegend=False,
        xaxis_tickangle=-30,
        margin=dict(t=20, b=100),
    )
    return fig


def build_cost_reduction_bar(df: pd.DataFrame) -> go.Figure:
    """Cost reduction % relative to raw baseline."""
    if df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_dark")
        return fig

    # Get raw baseline cost per model
    raw_df = df[df["strategy"] == "raw"]
    if raw_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No 'raw' baseline runs found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_dark")
        return fig

    raw_costs = raw_df.groupby("model")["avg_cost"].mean()

    # Aggregate by model×strategy first, then calculate reduction
    other_df = df[df["strategy"] != "raw"]
    if other_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No non-raw strategies to compare", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_dark")
        return fig

    agg = other_df.groupby(["model", "strategy"]).agg({
        "avg_cost": "mean",
        "n_instances": "sum",
    }).reset_index()

    reductions = []
    for _, row in agg.iterrows():
        if row["model"] in raw_costs.index:
            raw_cost = raw_costs[row["model"]]
            if raw_cost > 0 and row["avg_cost"] > 0:
                reduction = (1 - row["avg_cost"] / raw_cost) * 100
                reductions.append({
                    "model": row["model"],
                    "strategy": row["strategy"],
                    "cost_reduction_pct": reduction,
                })

    if not reductions:
        fig = go.Figure()
        fig.add_annotation(text="Need both raw and other strategies for comparison", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_dark")
        return fig

    reduction_df = pd.DataFrame(reductions)

    fig = px.bar(
        reduction_df,
        x="strategy",
        y="cost_reduction_pct",
        color="model",
        barmode="group",
        labels={"cost_reduction_pct": "Cost Reduction (%)", "strategy": "Strategy"},
    )
    fig.add_hline(
        y=50,
        line_dash="dash",
        line_color="white",
        annotation_text="Paper target: 50%",
        annotation_position="top right",
    )
    fig.update_layout(
        template="plotly_dark",
        title="",
        height=500,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.15,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=20, b=100),
    )
    return fig


def build_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Build Table 1 equivalent: Model×Strategy comparison with paper baselines."""
    if df.empty:
        return pd.DataFrame()

    # Aggregate our data by model×strategy
    agg = df.groupby(["model", "strategy"]).agg({
        "solve_rate": "mean",
        "avg_cost": "mean",
        "n_instances": "sum",
        "n_resolved": "sum",
    }).reset_index()

    rows = []
    for _, row in agg.iterrows():
        model = row["model"]
        strategy = row["strategy"]

        # Find matching paper baseline (prefer longest match to avoid "qwen3-32b" matching "qwen3-32b-thinking")
        paper_model = None
        best_match_len = 0
        for pm in PAPER_BASELINES:
            if pm in model.lower() or model.lower() in pm:
                if len(pm) > best_match_len:
                    paper_model = pm
                    best_match_len = len(pm)

        paper_vals = PAPER_BASELINES.get(paper_model, {}).get(strategy, {}) if paper_model else {}
        paper_rate = paper_vals.get("solve_rate")
        paper_cost = paper_vals.get("avg_cost")
        paper_rate_ci = paper_vals.get("rate_ci")
        paper_rate_delta = paper_vals.get("rate_delta")
        paper_cost_delta = paper_vals.get("cost_delta")

        # Calculate our delta vs paper
        rate_delta_vs_paper = None
        cost_delta_vs_paper = None
        if paper_rate and row["solve_rate"] > 0:
            rate_delta_vs_paper = row["solve_rate"] - paper_rate  # Absolute difference
        if paper_cost and row["avg_cost"] > 0:
            cost_delta_vs_paper = ((row["avg_cost"] - paper_cost) / paper_cost) * 100

        rows.append({
            "Model": model,
            "Strategy": strategy,
            "Our Rate": f"{row['solve_rate']:.1%}",
            "Paper Rate": f"{paper_rate:.1%} ±{paper_rate_ci:.1%}" if paper_rate and paper_rate_ci else (f"{paper_rate:.1%}" if paper_rate else "—"),
            "Rate Δ": f"{rate_delta_vs_paper:+.1%}" if rate_delta_vs_paper is not None else "—",
            "Our Cost": f"${row['avg_cost']:.2f}" if row["avg_cost"] > 0 else "—",
            "Paper Cost": f"${paper_cost:.2f}" if paper_cost else "—",
            "Cost Δ": f"{cost_delta_vs_paper:+.0f}%" if cost_delta_vs_paper is not None else "—",
            "Resolved": int(row["n_resolved"]),
            "N": int(row["n_instances"]),
        })

    return pd.DataFrame(rows)


def render_paper_comparison(df: pd.DataFrame):
    """Render Paper Comparison page matching arXiv:2508.21433 figures."""
    st.header("Paper Comparison")
    st.caption("Comparing our results to arXiv:2508.21433 - The Complexity Trap")

    # Info about metrics
    st.info(
        "**Metrics from SWE-bench evaluation**\n\n"
        "- **solve_rate** = n_resolved / n_instances (directly comparable to paper)\n"
        "- All runs include `--evaluate` flag for SWE-bench evaluation"
    )

    # Strategy hyperparameters info
    with st.expander("Strategy hyperparameters (from paper)"):
        params_df = pd.DataFrame([
            {"Strategy": k, "Parameters": v["description"]}
            for k, v in STRATEGY_PARAMS.items()
        ])
        st.dataframe(params_df, width="stretch", hide_index=True)

    st.divider()

    # Table 1: Model×Strategy comparison
    st.subheader("Table 1: Model×Strategy comparison")
    comparison_df = build_comparison_table(df)
    if not comparison_df.empty:
        st.dataframe(
            comparison_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Model": st.column_config.TextColumn(width="medium"),
                "Strategy": st.column_config.TextColumn(width="small"),
            },
        )
        st.caption("Δ columns: positive = we're higher than paper, negative = we're lower. Paper shows relative change vs raw baseline.")
    else:
        st.info("No data to compare. Run experiments first.")

    st.divider()

    # Figure 1: Cost vs Rate with ALL baselines (with error bars)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Figure 1: Cost vs solve rate")
        st.caption("Our solve_rate vs paper baselines (X markers with error bars)")
        fig_pareto = build_pareto_with_all_baselines(df)
        st.plotly_chart(fig_pareto, width="stretch", key="paper_pareto")

    with col2:
        st.subheader("Figure 4: Trajectory length")
        st.caption("Paper finding: LLM-Summary → longer trajectories")
        fig_turns = build_turn_boxplot(df)
        st.plotly_chart(fig_turns, width="stretch", key="paper_turns")

    st.divider()

    # Cost reduction and LLM-Summary costs side by side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cost reduction vs raw baseline")
        st.caption("Paper target: ~50% cost reduction for masking/summary")
        fig_reduction = build_cost_reduction_bar(df)
        st.plotly_chart(fig_reduction, width="stretch", key="paper_reduction")

    with col2:
        st.subheader("Table 2: LLM-Summary generation costs")
        st.caption("Overhead of summary generation per instance")
        summary_rows = [
            {"Model": model, "Summary Cost": f"${vals['cost']:.4f}", "% of Total": f"{vals['pct']:.2f}%"}
            for model, vals in SUMMARY_COSTS.items()
        ]
        st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

    st.divider()

    # OpenHands generalization note
    st.subheader("OpenHands generalization (Section 5.1)")
    st.info(
        "The paper shows initial evidence of findings generalizing to **OpenHands** scaffold:\n\n"
        "- Tested on 50-instance SWE-bench Verified slice with Gemini 2.5 Flash\n"
        "- Observation Masking (M=10, M=58) vs LLM-Summary (N=21, M=10)\n"
        "- Results in Figure 5a confirm masking is cost-effective across scaffolds\n\n"
        "*See `background-documents/data/openhands_scatter_plot.png` for the figure.*"
    )

    st.divider()

    # Paper baselines with confidence intervals
    with st.expander("Paper baselines reference (Table 1 with confidence intervals)"):
        baseline_rows = []
        for model, strategies in PAPER_BASELINES.items():
            for strategy, vals in strategies.items():
                rate_delta = vals.get("rate_delta")
                cost_delta = vals.get("cost_delta")
                baseline_rows.append({
                    "Model": model,
                    "Strategy": strategy,
                    "Solve Rate": f"{vals['solve_rate']:.1%} ±{vals.get('rate_ci', 0):.1%}",
                    "Avg Cost": f"${vals['avg_cost']:.2f} ±${vals.get('cost_ci', 0):.2f}",
                    "Rate Δ": f"{rate_delta:+.1%}" if rate_delta is not None else "—",
                    "Cost Δ": f"{cost_delta:+.1%}" if cost_delta is not None else "—",
                })
        st.dataframe(pd.DataFrame(baseline_rows), width="stretch", hide_index=True)


def main():
    # Load .env file
    from dotenv import load_dotenv
    load_dotenv()

    st.set_page_config(
        page_title="Complexity Trap Dashboard",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("The Complexity Trap - Experiment Analysis")
    st.caption("Analyzing context management strategies for SWE-agent")

    # Get project from env (set in .env or pass as env vars)
    project = os.environ.get("DASHBOARD_PROJECT") or os.environ.get("WANDB_PROJECT")
    entity = os.environ.get("DASHBOARD_ENTITY") or os.environ.get("WANDB_ENTITY")

    if not project:
        st.error("Missing WANDB_PROJECT or DASHBOARD_PROJECT environment variable.")
        st.info("Set in .env or run with: DASHBOARD_PROJECT=your-project streamlit run scripts/dashboard.py")
        st.stop()

    # Fetch data with loading spinner
    with st.spinner(f"Fetching runs from WandB project: {project}..."):
        df = fetch_runs(project, entity)

    if df.empty:
        st.warning("No runs found in this project. Check your project name and WandB credentials.")
        st.stop()

    # Render sidebar and get filters
    filters = render_sidebar(df)

    # Apply filters
    filtered_df = apply_filters(df, filters)

    # Tab navigation
    tab1, tab2 = st.tabs(["Overview", "Paper Comparison"])

    with tab1:
        # Metrics row
        render_metrics(filtered_df)

        st.divider()

        # Main content: two columns
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Cost vs solve rate")
            st.caption("Upper-left = Pareto optimal (lower cost, higher solve rate)")
            fig_pareto = build_pareto_plot(filtered_df, show_baselines=filters["show_baselines"])
            st.plotly_chart(fig_pareto, width="stretch", key="overview_pareto")

        with col2:
            st.subheader("Exit status distribution")
            st.caption("Why runs end (submitted = success)")
            fig_exit = build_exit_status_bar(filtered_df)
            st.plotly_chart(fig_exit, width="stretch", key="overview_exit")

        st.divider()

        # Instance explorer
        render_instance_explorer(filtered_df)

    with tab2:
        render_paper_comparison(filtered_df)

    # Footer
    st.divider()
    st.caption("Data source: WandB | Paper: arXiv:2508.21433")


if __name__ == "__main__":
    main()
