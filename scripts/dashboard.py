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

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Import shared constants and data fetching from Streamlit-free module
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from dashboard_shared import (
    PAPER_BASELINES,
    SUMMARY_COSTS,
    STRATEGY_PARAMS,
    EXIT_COLORS,
    fetch_runs as _fetch_runs_uncached,
    dedupe_latest_runs,
    get_project_config,
)


@st.cache_data(ttl=600)
def fetch_runs(project: str, entity: str | None = None) -> pd.DataFrame:
    """Fetch WandB runs."""
    try:
        return _fetch_runs_uncached(project, entity, use_cache=True)
    except RuntimeError as e:
        st.error(str(e))
        st.info("Check your WANDB_API_KEY environment variable.")
        st.stop()


def build_pareto_plot(df: pd.DataFrame, show_baselines: bool = True) -> go.Figure:
    """Build Pareto scatter: cost (log x) vs solve rate (y)."""
    # Map 0→epsilon for log scale
    df_valid = df[df["avg_cost"].notna()].copy()
    if "eval_complete" in df_valid.columns:
        df_valid = df_valid[df_valid["eval_complete"]]
    df_valid = df_valid[df_valid["solve_rate"].notna()]
    df_valid["avg_cost_display"] = df_valid["avg_cost"].apply(lambda x: max(x, 0.001))

    if df_valid.empty:
        fig = go.Figure()
        fig.add_annotation(text="No runs with cost data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_dark")
        return fig

    fig = px.scatter(
        df_valid,
        x="avg_cost_display",
        y="solve_rate",
        color="strategy",
        symbol="model",
        size="n_instances",
        hover_name="run_name",
        hover_data=["model", "strategy", "n_instances", "avg_turns", "submission_rate", "avg_cost"],
        labels={"avg_cost_display": "Avg Cost ($)", "solve_rate": "Solve Rate", "avg_cost": "Actual Cost"},
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

        # Run explorer duplicates toggle
        show_all_runs = st.checkbox("Show all runs in explorer (include duplicates)", value=False)

        st.divider()

        # Refresh button
        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.rerun()

        # Project info
        project, _ = get_project_config()
        st.caption(f"Project: {project or 'Not configured'}")

    return {
        "models": models,
        "strategies": strategies,
        "exit_filter": exit_filter,
        "min_instances": min_instances,
        "show_baselines": show_baselines,
        "show_all_runs": show_all_runs,
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
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    eval_df = df[df["eval_complete"]] if "eval_complete" in df.columns else df

    with col1:
        st.metric("Total Runs", len(df))

    with col2:
        st.metric("Models", df["model"].nunique())

    with col3:
        st.metric("Strategies", df["strategy"].nunique())

    with col4:
        best_rate = eval_df["solve_rate"].max() if not eval_df.empty else np.nan
        st.metric("Best Solve Rate", f"{best_rate:.1%}" if pd.notna(best_rate) else "N/A")

    with col5:
        total_resolved = int(df["n_resolved"].sum()) if "n_resolved" in df.columns else 0
        total_instances = int(df["n_instances"].sum()) if "n_instances" in df.columns else 0
        st.metric("Resolved", f"{total_resolved}/{total_instances}")

    with col6:
        total_cost = df["total_cost"].sum() if "total_cost" in df.columns else np.nan
        st.metric("Total Cost", f"${total_cost:.2f}" if pd.notna(total_cost) and total_cost > 0 else "N/A")

    st.caption("Summaries and comparisons use the latest run per run_name (deduped).")


def render_strategy_summary(df: pd.DataFrame):
    """Render compact strategy comparison cards."""
    if df.empty:
        return

    eval_df = df[df["eval_complete"]] if "eval_complete" in df.columns else df
    if eval_df.empty:
        return

    # Group by strategy
    strategies = eval_df.groupby("strategy").agg({
        "solve_rate": "mean",
        "avg_cost": "mean",
        "n_resolved": "sum",
        "n_instances": "sum",
    }).reset_index()

    if strategies.empty:
        return

    # Sort by solve_rate descending
    strategies = strategies.sort_values("solve_rate", ascending=False)

    # Create columns for each strategy
    cols = st.columns(len(strategies))
    for col, (_, row) in zip(cols, strategies.iterrows()):
        with col:
            rate = row["solve_rate"]
            cost = row["avg_cost"]
            resolved = int(row["n_resolved"])
            total = int(row["n_instances"])

            # Color based on performance
            if rate >= 0.6:
                color = "🟢"
            elif rate >= 0.4:
                color = "🟡"
            else:
                color = "🔴"

            st.markdown(f"**{row['strategy']}** {color}")
            st.caption(f"{rate:.1%} · ${cost:.2f}/inst · {resolved}/{total}")


def render_instance_explorer(df: pd.DataFrame):
    """Render instance explorer table with sorting and formatting."""
    st.subheader("Run Explorer")

    if df.empty:
        st.warning("No runs match the current filters.")
        return

    # filter controls
    col1, col2, col3 = st.columns(3)
    with col1:
        min_instances = st.number_input("Min instances", min_value=0, value=0, step=5)
    with col2:
        eval_only = st.checkbox("Evaluated only", value=False)
    with col3:
        min_solve_pct = st.slider("Min solve rate", 0, 100, 0, 5, format="%d%%")
        min_solve = min_solve_pct / 100.0  # convert to 0-1 range for filter

    # apply filters
    filtered_df = df.copy()
    if min_instances > 0:
        filtered_df = filtered_df[filtered_df["n_instances"] >= min_instances]
    if eval_only and "eval_complete" in filtered_df.columns:
        filtered_df = filtered_df[filtered_df["eval_complete"]]
    if min_solve > 0:
        filtered_df = filtered_df[filtered_df["solve_rate"].fillna(0) >= min_solve]

    if filtered_df.empty:
        st.info("No runs match the current filters.")
        return

    # prepare display dataframe with numeric values for proper sorting
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
    if "eval_complete" in filtered_df.columns:
        display_cols.append("eval_complete")

    display_df = filtered_df[display_cols].copy()

    # use native Streamlit column config for formatting (keeps values sortable)
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "run_name": st.column_config.TextColumn("Run", width="large"),
            "model": st.column_config.TextColumn("Model", width="medium"),
            "strategy": st.column_config.TextColumn("Strategy", width="small"),
            "n_instances": st.column_config.NumberColumn("N", format="%d"),
            "n_resolved": st.column_config.NumberColumn("Resolved", format="%d"),
            "solve_rate": st.column_config.ProgressColumn(
                "Solve Rate",
                format="percent",
                min_value=0,
                max_value=1,
            ),
            "avg_cost": st.column_config.NumberColumn("Avg Cost", format="$%.3f"),
            "avg_turns": st.column_config.NumberColumn("Turns", format="%.1f"),
            "exit_submitted": st.column_config.NumberColumn("✓", format="%d", help="Submitted count"),
            "exit_cost": st.column_config.NumberColumn("$", format="%d", help="Cost exit count"),
            "eval_complete": st.column_config.CheckboxColumn("Eval", help="Evaluation complete"),
        },
    )

    st.caption(f"Showing {len(display_df)} runs. Click column headers to sort.")


def build_pareto_with_all_baselines(df: pd.DataFrame) -> go.Figure:
    """Build Pareto scatter with ALL paper baselines (all 5 models)."""
    df_valid = df[df["avg_cost"].notna()].copy()
    if "eval_complete" in df_valid.columns:
        df_valid = df_valid[df_valid["eval_complete"]]
    df_valid = df_valid[df_valid["solve_rate"].notna()]
    df_valid["avg_cost_display"] = df_valid["avg_cost"].apply(lambda x: max(x, 0.001))

    if df_valid.empty:
        fig = go.Figure()
        fig.add_annotation(text="No runs with cost data", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_dark")
        return fig

    fig = px.scatter(
        df_valid,
        x="avg_cost_display",
        y="solve_rate",
        color="strategy",
        symbol="model",
        size="n_instances",
        hover_name="run_name",
        hover_data=["model", "strategy", "n_instances", "avg_turns", "n_resolved", "avg_cost"],
        labels={"avg_cost_display": "Avg Cost ($)", "solve_rate": "Solve Rate", "avg_cost": "Actual Cost"},
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

    # Filter to eval_complete runs only for accurate cost comparison
    eval_df = df[df["eval_complete"]] if "eval_complete" in df.columns else df

    # Get raw baseline cost per model (use median for robustness to outliers)
    raw_df = eval_df[eval_df["strategy"] == "raw"]
    if raw_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No 'raw' baseline runs found", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_dark")
        return fig

    raw_costs = raw_df.groupby("model")["avg_cost"].median()

    # Aggregate by model×strategy first, then calculate reduction
    other_df = eval_df[eval_df["strategy"] != "raw"]
    if other_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No non-raw strategies to compare", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(template="plotly_dark")
        return fig

    agg = other_df.groupby(["model", "strategy"]).agg({
        "avg_cost": "median",  # median is robust to outliers
        "n_instances": "sum",
    }).reset_index()

    reductions = []
    for _, row in agg.iterrows():
        if row["model"] in raw_costs.index:
            raw_cost = raw_costs[row["model"]]
            # Allow $0 cost (local models) - this is 100% reduction
            if pd.notna(raw_cost) and raw_cost > 0 and pd.notna(row["avg_cost"]) and row["avg_cost"] >= 0:
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
    # y=0 reference line (break-even point)
    fig.add_hline(
        y=0,
        line_dash="solid",
        line_color="gray",
        annotation_text="Break-even",
        annotation_position="bottom right",
    )
    # y=50 reference line (paper target)
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
    eval_df = df[df["eval_complete"]] if "eval_complete" in df.columns else df
    if eval_df.empty:
        return pd.DataFrame()

    # Weighted aggregation
    def weighted_agg(group):
        n_total = group["n_instances"].sum()
        n_resolved = group["n_resolved"].sum()
        # Filter valid cost entries for weighted average
        valid_costs = group[group["avg_cost"].notna() & (group["n_instances"] > 0)]
        if len(valid_costs) > 0 and n_total > 0:
            weighted_cost = (valid_costs["avg_cost"] * valid_costs["n_instances"]).sum() / valid_costs["n_instances"].sum()
        else:
            weighted_cost = np.nan
        return pd.Series({
            "solve_rate": n_resolved / n_total if n_total > 0 else 0,
            "avg_cost": weighted_cost,
            "n_instances": n_total,
            "n_resolved": n_resolved,
        })

    agg = eval_df.groupby(["model", "strategy"]).apply(weighted_agg).reset_index()

    rows = []
    for _, row in agg.iterrows():
        model = row["model"]
        strategy = row["strategy"]

        # Find matching paper baseline (prefer longest match to avoid "qwen3-32b" matching "qwen3-32b-thinking")
        paper_model = None
        best_match_len = 0
        for pm in PAPER_BASELINES:
            # Only match if baseline key is substring of our model name
            # (not reverse - avoids qwen3-32b matching qwen3-32b-thinking)
            if pm in model.lower():
                if len(pm) > best_match_len:
                    paper_model = pm
                    best_match_len = len(pm)

        paper_vals = PAPER_BASELINES.get(paper_model, {}).get(strategy, {}) if paper_model else {}
        paper_rate = paper_vals.get("solve_rate")
        paper_cost = paper_vals.get("avg_cost")
        paper_rate_ci = paper_vals.get("rate_ci")
        paper_rate_delta = paper_vals.get("rate_delta")
        paper_cost_delta = paper_vals.get("cost_delta")

        # Delta vs paper
        rate_delta_vs_paper = None
        cost_delta_vs_paper = None
        if paper_rate and pd.notna(row["solve_rate"]) and row["solve_rate"] > 0:
            rate_delta_vs_paper = row["solve_rate"] - paper_rate  # Absolute difference
        if paper_cost and pd.notna(row["avg_cost"]) and row["avg_cost"] > 0:
            cost_delta_vs_paper = ((row["avg_cost"] - paper_cost) / paper_cost) * 100

        our_cost = row["avg_cost"]
        rows.append({
            "Model": model,
            "Strategy": strategy,
            "Our Rate": f"{row['solve_rate']:.1%}",
            "Paper Rate": f"{paper_rate:.1%} ±{paper_rate_ci:.1%}" if paper_rate and paper_rate_ci else (f"{paper_rate:.1%}" if paper_rate else "—"),
            "Rate Δ": f"{rate_delta_vs_paper:+.1%}" if rate_delta_vs_paper is not None else "—",
            "_rate_delta_num": rate_delta_vs_paper,  # numeric for styling
            "Our Cost": f"${our_cost:.2f}" if pd.notna(our_cost) and our_cost > 0 else "—",
            "Paper Cost": f"${paper_cost:.2f}" if paper_cost else "—",
            "Cost Δ": f"{cost_delta_vs_paper:+.0f}%" if cost_delta_vs_paper is not None else "—",
            "_cost_delta_num": cost_delta_vs_paper,  # numeric for styling
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
        "- Only runs with completed evaluation (`eval_complete`) are used in solve-rate comparisons"
    )

    # Strategy hyperparameters info
    with st.expander("Strategy hyperparameters (from paper)"):
        params_df = pd.DataFrame([
            {"Strategy": k, "Parameters": v["description"]}
            for k, v in STRATEGY_PARAMS.items()
        ])
        st.dataframe(params_df, use_container_width=True, hide_index=True)

    st.divider()

    # Table 1: Model×Strategy comparison
    st.subheader("Table 1: Model×Strategy comparison")
    comparison_df = build_comparison_table(df)
    if not comparison_df.empty:
        # display columns (hide numeric helper columns)
        display_cols = [c for c in comparison_df.columns if not c.startswith("_")]
        display_df = comparison_df[display_cols].copy()

        # color styling for delta columns using applymap on specific columns
        def style_rate_delta(val, row_idx):
            """Green for positive (we're better), red for negative."""
            num_val = comparison_df.iloc[row_idx]["_rate_delta_num"]
            if pd.isna(num_val) or num_val is None:
                return ""
            if num_val > 0:
                return "background-color: rgba(34, 197, 94, 0.3)"  # green
            elif num_val < 0:
                return "background-color: rgba(239, 68, 68, 0.3)"  # red
            return ""

        def style_cost_delta(val, row_idx):
            """Green for negative (cost savings), red for positive (cost increase)."""
            num_val = comparison_df.iloc[row_idx]["_cost_delta_num"]
            if pd.isna(num_val) or num_val is None:
                return ""
            if num_val < 0:
                return "background-color: rgba(34, 197, 94, 0.3)"  # green = savings
            elif num_val > 0:
                return "background-color: rgba(239, 68, 68, 0.3)"  # red = increase
            return ""

        # build style matrix
        def apply_delta_styles(df):
            styles = pd.DataFrame("", index=df.index, columns=df.columns)
            if "Rate Δ" in df.columns:
                for idx in df.index:
                    styles.loc[idx, "Rate Δ"] = style_rate_delta(df.loc[idx, "Rate Δ"], idx)
            if "Cost Δ" in df.columns:
                for idx in df.index:
                    styles.loc[idx, "Cost Δ"] = style_cost_delta(df.loc[idx, "Cost Δ"], idx)
            return styles

        styled_df = display_df.style.apply(lambda _: apply_delta_styles(display_df), axis=None)

        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Model": st.column_config.TextColumn(width="medium"),
                "Strategy": st.column_config.TextColumn(width="small"),
            },
        )
        st.caption("Δ columns: green = favorable (higher rate or lower cost), red = unfavorable. Paper shows relative change vs raw baseline.")
    else:
        st.info("No data to compare. Run experiments first.")

    st.divider()

    # Figure 1: Cost vs Rate with ALL baselines (with error bars)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Figure 1: Cost vs solve rate")
        st.caption("Our solve_rate vs paper baselines (X markers with error bars)")
        fig_pareto = build_pareto_with_all_baselines(df)
        st.plotly_chart(fig_pareto, use_container_width=True, key="paper_pareto")

    with col2:
        st.subheader("Figure 4: Trajectory length")
        st.caption("Paper finding: LLM-Summary → longer trajectories")
        fig_turns = build_turn_boxplot(df)
        st.plotly_chart(fig_turns, use_container_width=True, key="paper_turns")

    st.divider()

    # Cost reduction and LLM-Summary costs side by side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cost reduction vs raw baseline")
        st.caption("Paper target: ~50% cost reduction for masking/summary")
        fig_reduction = build_cost_reduction_bar(df)
        st.plotly_chart(fig_reduction, use_container_width=True, key="paper_reduction")

    with col2:
        st.subheader("Table 2: LLM-Summary generation costs")
        st.caption("Overhead of summary generation per instance")
        summary_rows = [
            {"Model": model, "Summary Cost": f"${vals['cost']:.4f}", "% of Total": f"{vals['pct']:.2f}%"}
            for model, vals in SUMMARY_COSTS.items()
        ]
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

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
        st.dataframe(pd.DataFrame(baseline_rows), use_container_width=True, hide_index=True)


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
    project, entity = get_project_config()

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
    summary_df = dedupe_latest_runs(filtered_df)
    explorer_df = filtered_df if filters.get("show_all_runs") else summary_df

    # Tab navigation
    tab1, tab2 = st.tabs(["Overview", "Paper Comparison"])

    with tab1:
        # Metrics row
        render_metrics(summary_df)

        # Strategy summary (quick comparison)
        render_strategy_summary(summary_df)

        st.divider()

        # Main content: two columns
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Cost vs solve rate")
            st.caption("Upper-left = Pareto optimal (lower cost, higher solve rate)")
            fig_pareto = build_pareto_plot(summary_df, show_baselines=filters["show_baselines"])
            st.plotly_chart(fig_pareto, use_container_width=True, key="overview_pareto")

        with col2:
            st.subheader("Exit status distribution")
            st.caption("Why runs end (submitted = success)")
            fig_exit = build_exit_status_bar(summary_df)
            st.plotly_chart(fig_exit, use_container_width=True, key="overview_exit")

        st.divider()

        # Instance explorer
        render_instance_explorer(explorer_df)

    with tab2:
        render_paper_comparison(summary_df)

    # Footer
    st.divider()
    st.caption("Data source: WandB | Paper: arXiv:2508.21433")


if __name__ == "__main__":
    import sys

    if "--tui" in sys.argv:
        # Launch TUI version
        from dashboard_tui import main as tui_main
        tui_main()
    elif "--help" in sys.argv or "-h" in sys.argv:
        print("""
Complexity Trap Dashboard

Usage:
    streamlit run scripts/dashboard.py      # Web dashboard (default)
    python scripts/dashboard.py --tui       # Terminal UI dashboard

Environment variables:
    DASHBOARD_PROJECT or WANDB_PROJECT      # WandB project name (required)
    DASHBOARD_ENTITY or WANDB_ENTITY        # WandB entity (optional)
        """)
    else:
        main()
