#!/usr/bin/env python3
"""
Pure analysis functions for WandB experiment queries.

This module contains stateless functions that operate on DataFrames.
No I/O, no side effects - just data transformations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dashboard_shared import (
    dedupe_latest_runs,
    find_paper_baseline,
    calculate_rate_delta,
    calculate_cost_delta,
)


@dataclass
class QueryResult:
    data: pd.DataFrame
    title: str
    insights: list[str]
    columns: list[str]
    sort_by: str | None = None
    sort_ascending: bool = False


def aggregate_by_model_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate runs by model×strategy using instance-weighted averages.

    Returns DataFrame with columns:
    - model, strategy, solve_rate, avg_cost, n_instances, n_resolved
    """
    if df.empty:
        return pd.DataFrame()

    # only use evaluated runs
    eval_df = df[df["eval_complete"]] if "eval_complete" in df.columns else df
    if eval_df.empty:
        return pd.DataFrame()

    # aggregate by model×strategy using explicit column selection (pandas 1.x/2.x compatible)
    rows = []
    for (model, strategy), group in eval_df.groupby(["model", "strategy"]):
        n_total = group["n_instances"].sum()
        n_resolved = group["n_resolved"].sum()

        # weighted cost average
        valid_costs = group[group["avg_cost"].notna() & (group["n_instances"] > 0)]
        if len(valid_costs) > 0:
            weighted_cost = (
                valid_costs["avg_cost"] * valid_costs["n_instances"]
            ).sum() / valid_costs["n_instances"].sum()
        else:
            weighted_cost = np.nan

        # weighted avg compactions (if column exists)
        if "avg_compactions" in group.columns:
            valid_comp = group[group["avg_compactions"].notna() & (group["n_instances"] > 0)]
            if len(valid_comp) > 0:
                avg_compactions = (
                    valid_comp["avg_compactions"] * valid_comp["n_instances"]
                ).sum() / valid_comp["n_instances"].sum()
            else:
                avg_compactions = np.nan
        else:
            avg_compactions = np.nan

        rows.append({
            "model": model,
            "strategy": strategy,
            "solve_rate": n_resolved / n_total if n_total > 0 else np.nan,
            "avg_cost": weighted_cost,
            "avg_compactions": avg_compactions,
            "n_instances": n_total,
            "n_resolved": n_resolved,
        })

    return pd.DataFrame(rows)


def compute_summary(df: pd.DataFrame) -> QueryResult:
    """Compute one-liner summary of best performer."""
    agg = aggregate_by_model_strategy(df)
    if agg.empty:
        return QueryResult(
            data=pd.DataFrame(),
            title="Summary",
            insights=["No evaluated runs found."],
            columns=[],
        )

    # find best by solve rate (ignore unknown denominators)
    valid = agg[agg["solve_rate"].notna()]
    if valid.empty:
        return QueryResult(
            data=pd.DataFrame(),
            title="Summary",
            insights=["No runs with valid instance denominators."],
            columns=[],
        )
    best_idx = valid["solve_rate"].idxmax()
    best = valid.loc[best_idx]

    # paper comparison
    baseline = find_paper_baseline(best["model"], best["strategy"])
    paper_delta = ""
    if baseline:
        rate_delta = calculate_rate_delta(best["solve_rate"], baseline.get("solve_rate"))
        if rate_delta != "—":
            paper_delta = f" — vs paper: {rate_delta}"

    # build summary line
    total_instances = int(agg["n_instances"].sum())
    n_strategies = agg["strategy"].nunique()
    cost_str = f"${best['avg_cost']:.2f}" if pd.notna(best["avg_cost"]) else "N/A"

    summary_line = (
        f"Best: {best['model']} {best['strategy']} @ {best['solve_rate']:.1%} "
        f"({cost_str}){paper_delta} | "
        f"{n_strategies} strategies, {total_instances} instances"
    )

    return QueryResult(
        data=agg,
        title="Summary",
        insights=[summary_line],
        columns=["model", "strategy", "solve_rate", "avg_cost", "n_instances"],
    )


def compute_leaderboard(
    df: pd.DataFrame,
    strategy_filter: str | None = None,
    model_filter: str | None = None,
    min_instances: int = 10,
) -> QueryResult:
    """Compute ranked leaderboard of model×strategy combinations."""
    agg = aggregate_by_model_strategy(df)
    if agg.empty:
        return QueryResult(
            data=pd.DataFrame(),
            title="Leaderboard",
            insights=["No evaluated runs found."],
            columns=[],
        )

    # apply filters
    if strategy_filter:
        agg = agg[agg["strategy"].str.contains(strategy_filter, case=False, na=False)]
    if model_filter:
        agg = agg[agg["model"].str.contains(model_filter, case=False, na=False)]
    if min_instances > 0:
        agg = agg[agg["n_instances"] >= min_instances]

    if agg.empty:
        return QueryResult(
            data=pd.DataFrame(),
            title="Leaderboard",
            insights=["No runs match the filters."],
            columns=[],
        )

    # add paper deltas
    paper_rate_deltas = []
    paper_cost_deltas = []
    for _, row in agg.iterrows():
        baseline = find_paper_baseline(row["model"], row["strategy"])
        if baseline:
            paper_rate_deltas.append(
                calculate_rate_delta(row["solve_rate"], baseline.get("solve_rate"))
            )
            paper_cost_deltas.append(
                calculate_cost_delta(row["avg_cost"], baseline.get("avg_cost"))
            )
        else:
            paper_rate_deltas.append("—")
            paper_cost_deltas.append("—")

    agg["rate_delta"] = paper_rate_deltas
    agg["cost_delta"] = paper_cost_deltas

    # sort by solve rate descending
    agg = agg.sort_values("solve_rate", ascending=False).reset_index(drop=True)
    agg["rank"] = range(1, len(agg) + 1)

    # insights
    insights = []
    if len(agg) > 0:
        best = agg.iloc[0]
        insights.append(
            f"Top performer: {best['model']} {best['strategy']} "
            f"at {best['solve_rate']:.1%} solve rate"
        )

    return QueryResult(
        data=agg,
        title="Leaderboard",
        insights=insights,
        columns=[
            "rank",
            "model",
            "strategy",
            "solve_rate",
            "rate_delta",
            "avg_cost",
            "cost_delta",
            "n_instances",
        ],
        sort_by="solve_rate",
        sort_ascending=False,
    )


def compute_paper_comparison(
    df: pd.DataFrame,
    model_filter: str | None = None,
) -> QueryResult:
    """Compare our results directly to paper baselines."""
    agg = aggregate_by_model_strategy(df)
    if agg.empty:
        return QueryResult(
            data=pd.DataFrame(),
            title="Paper Comparison",
            insights=["No evaluated runs found."],
            columns=[],
        )

    if model_filter:
        agg = agg[agg["model"].str.contains(model_filter, case=False, na=False)]

    if agg.empty:
        return QueryResult(
            data=pd.DataFrame(),
            title="Paper Comparison",
            insights=["No runs match the model filter."],
            columns=[],
        )

    # build comparison rows
    rows = []
    for _, row in agg.iterrows():
        baseline = find_paper_baseline(row["model"], row["strategy"])
        paper_rate = baseline.get("solve_rate") if baseline else None
        paper_cost = baseline.get("avg_cost") if baseline else None

        rows.append(
            {
                "model": row["model"],
                "strategy": row["strategy"],
                "our_rate": row["solve_rate"],
                "paper_rate": paper_rate,
                "rate_delta": calculate_rate_delta(row["solve_rate"], paper_rate),
                "our_cost": row["avg_cost"],
                "paper_cost": paper_cost,
                "cost_delta": calculate_cost_delta(row["avg_cost"], paper_cost),
                "n_instances": row["n_instances"],
            }
        )

    result_df = pd.DataFrame(rows)

    # generate insights
    insights = []
    if len(result_df) > 0:
        # find biggest win vs paper
        rate_wins = [
            (r["model"], r["strategy"], r["our_rate"] - r["paper_rate"])
            for _, r in result_df.iterrows()
            if r["paper_rate"] is not None and not pd.isna(r["our_rate"])
        ]
        if rate_wins:
            best_win = max(rate_wins, key=lambda x: x[2])
            if best_win[2] > 0:
                insights.append(
                    f"Biggest win vs paper: {best_win[0]} {best_win[1]} (+{best_win[2]:.1%})"
                )
            else:
                insights.append("All strategies underperform paper baselines.")

    # include model column only if multiple models
    columns = ["strategy", "our_rate", "paper_rate", "rate_delta", "our_cost", "paper_cost", "cost_delta", "n_instances"]
    if len(result_df) > 0 and result_df["model"].nunique() > 1:
        columns = ["model"] + columns

    return QueryResult(
        data=result_df,
        title="Our Results vs Paper (arXiv:2508.21433)",
        insights=insights,
        columns=columns,
    )


def compute_strategy_comparison(
    df: pd.DataFrame,
    model_filter: str | None = None,
) -> QueryResult:
    """Compare strategies for a specific model (or all models if not filtered)."""
    agg = aggregate_by_model_strategy(df)
    if agg.empty:
        return QueryResult(
            data=pd.DataFrame(),
            title="Strategy Comparison",
            insights=["No evaluated runs found."],
            columns=[],
        )

    if model_filter:
        agg = agg[agg["model"].str.contains(model_filter, case=False, na=False)]

    if agg.empty:
        return QueryResult(
            data=pd.DataFrame(),
            title="Strategy Comparison",
            insights=["No runs match the model filter."],
            columns=[],
        )

    # for each model, compute delta vs raw baseline
    rows = []
    for model in agg["model"].unique():
        model_data = agg[agg["model"] == model]
        raw_row = model_data[model_data["strategy"] == "raw"]

        if len(raw_row) > 0:
            raw_rate = raw_row.iloc[0]["solve_rate"]
            raw_cost = raw_row.iloc[0]["avg_cost"]
        else:
            # use first strategy as baseline
            raw_rate = model_data.iloc[0]["solve_rate"]
            raw_cost = model_data.iloc[0]["avg_cost"]

        for _, row in model_data.iterrows():
            rate_vs_raw = "—"
            if row["strategy"] != "raw" and pd.notna(raw_rate) and pd.notna(row["solve_rate"]):
                rate_delta = row["solve_rate"] - raw_rate
                rate_vs_raw = f"{rate_delta:+.1%}"

            cost_vs_raw = "—"
            if (
                row["strategy"] != "raw"
                and pd.notna(raw_cost)
                and pd.notna(row["avg_cost"])
                and raw_cost > 0
            ):
                cost_delta = ((row["avg_cost"] - raw_cost) / raw_cost) * 100
                cost_vs_raw = f"{cost_delta:+.0f}%"

            rows.append(
                {
                    "model": row["model"],
                    "strategy": row["strategy"],
                    "solve_rate": row["solve_rate"],
                    "rate_vs_raw": rate_vs_raw,
                    "avg_cost": row["avg_cost"],
                    "cost_vs_raw": cost_vs_raw,
                    "avg_compactions": row.get("avg_compactions", np.nan),
                    "n_instances": row["n_instances"],
                }
            )

    result_df = pd.DataFrame(rows)

    # sort: raw first, then by solve rate
    def sort_key(strategy: str) -> tuple:
        if strategy == "raw":
            return (0, "")
        return (1, strategy)

    result_df["_sort"] = result_df["strategy"].apply(sort_key)
    result_df = result_df.sort_values(["model", "_sort"]).drop(columns=["_sort"])

    # insights
    insights = []
    if len(result_df) > 0:
        # check if context management helps or hurts
        non_raw = result_df[result_df["strategy"] != "raw"]
        if len(non_raw) > 0:
            valid_deltas = non_raw[non_raw["rate_vs_raw"] != "—"]["rate_vs_raw"]
            avg_delta = valid_deltas.apply(
                lambda x: float(x.replace("%", "").replace("+", "")) / 100
            ).mean()
            if pd.notna(avg_delta):
                if avg_delta < -0.02:
                    insights.append(
                        "⚠️ Context management strategies HURT performance (opposite of paper)"
                    )
                elif avg_delta > 0.02:
                    insights.append(
                        "✅ Context management strategies HELP performance (matches paper)"
                    )

    return QueryResult(
        data=result_df,
        title="Strategy Comparison",
        insights=insights,
        columns=[
            "model",
            "strategy",
            "solve_rate",
            "rate_vs_raw",
            "avg_cost",
            "cost_vs_raw",
            "avg_compactions",
            "n_instances",
        ],
    )


def compute_failures(
    df: pd.DataFrame,
    model_filter: str | None = None,
    strategy_filter: str | None = None,
) -> QueryResult:
    """Analyze exit status distribution."""
    work_df = dedupe_latest_runs(df)

    if model_filter:
        work_df = work_df[
            work_df["model"].str.contains(model_filter, case=False, na=False)
        ]
    if strategy_filter:
        work_df = work_df[
            work_df["strategy"].str.contains(strategy_filter, case=False, na=False)
        ]

    if work_df.empty:
        return QueryResult(
            data=pd.DataFrame(),
            title="Exit Status Analysis",
            insights=["No runs match the filters."],
            columns=[],
        )

    # aggregate exit counts
    exit_cols = [
        "exit_submitted",
        "exit_cost",
        "exit_context",
        "exit_timeout",
        "exit_format",
        "exit_other",
    ]
    exit_cols = [c for c in exit_cols if c in work_df.columns]

    totals = {col: work_df[col].sum() for col in exit_cols}
    total_all = sum(totals.values())

    rows = []
    for col in exit_cols:
        count = totals[col]
        pct = (count / total_all * 100) if total_all > 0 else 0
        friendly_name = col.replace("exit_", "").replace("_", " ").title()
        rows.append(
            {
                "status": friendly_name,
                "count": int(count),
                "percentage": f"{pct:.1f}%",
            }
        )

    result_df = pd.DataFrame(rows)

    # insights
    insights = []
    if total_all > 0:
        submitted = totals.get("exit_submitted", 0)
        submission_rate = submitted / total_all * 100
        insights.append(f"Submission rate: {submission_rate:.1f}%")

        # flag concerning failure modes
        context_pct = totals.get("exit_context", 0) / total_all * 100
        if context_pct > 5:
            insights.append(f"⚠️ {context_pct:.1f}% hit context limits")

        cost_pct = totals.get("exit_cost", 0) / total_all * 100
        if cost_pct > 10:
            insights.append(f"⚠️ {cost_pct:.1f}% hit cost limits")

    return QueryResult(
        data=result_df,
        title="Exit Status Analysis",
        insights=insights,
        columns=["status", "count", "percentage"],
    )


def compute_runs(
    df: pd.DataFrame,
    model_filter: str | None = None,
    strategy_filter: str | None = None,
    min_instances: int = 0,
    eval_only: bool = False,
) -> QueryResult:
    """List runs with filtering."""
    work_df = df.copy()

    # Backwards-compatibility: older cached dataframes or external callers may
    # not include newer hyperparameter columns.
    required_defaults = {
        "summarizer": "same",
        "instances_subset": "verified",
        "hp_obs_n": np.nan,
        "hp_sum_n": np.nan,
        "hp_sum_keep_m": np.nan,
        "hp_limit_aware": False,
        "hp_limit_fraction": np.nan,
        "hp_limit_min_tokens": np.nan,
        "eval_complete": False,
    }
    for col, default in required_defaults.items():
        if col not in work_df.columns:
            work_df[col] = default

    if model_filter:
        work_df = work_df[
            work_df["model"].str.contains(model_filter, case=False, na=False)
        ]
    if strategy_filter:
        work_df = work_df[
            work_df["strategy"].str.contains(strategy_filter, case=False, na=False)
        ]
    if min_instances > 0:
        work_df = work_df[work_df["n_instances"] >= min_instances]
    if eval_only:
        work_df = work_df[work_df["eval_complete"] == True]  # noqa: E712

    if work_df.empty:
        return QueryResult(
            data=pd.DataFrame(),
            title="Runs",
            insights=["No runs match the filters."],
            columns=[],
        )

    # select and format columns
    result_df = work_df[
        [
            "run_name",
            "model",
            "strategy",
            "summarizer",
            "instances_subset",
            "hp_obs_n",
            "hp_sum_n",
            "hp_sum_keep_m",
            "hp_limit_aware",
            "hp_limit_fraction",
            "hp_limit_min_tokens",
            "n_instances",
            "n_resolved",
            "solve_rate",
            "avg_cost",
            "eval_complete",
        ]
    ].copy()

    result_df = result_df.sort_values("run_name", ascending=False)

    insights = [f"Found {len(result_df)} runs matching filters"]

    return QueryResult(
        data=result_df,
        title="Runs",
        insights=insights,
        columns=[
            "run_name",
            "model",
            "strategy",
            "summarizer",
            "instances_subset",
            "hp_obs_n",
            "hp_sum_n",
            "hp_sum_keep_m",
            "hp_limit_aware",
            "hp_limit_fraction",
            "hp_limit_min_tokens",
            "n_instances",
            "n_resolved",
            "solve_rate",
            "avg_cost",
            "eval_complete",
        ],
    )
