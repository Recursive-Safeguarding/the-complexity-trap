#!/usr/bin/env python3
"""
Complexity Trap TUI Dashboard

Terminal-based dashboard for analyzing context management experiments.
Designed for VPS/SSH workflows where browser access is inconvenient.

Usage:
    python scripts/dashboard_tui.py                  # Full interactive TUI
    python scripts/dashboard_tui.py --view summary   # One-line summary (for scripts/automation)
    python scripts/dashboard_tui.py --view compact   # Compact multi-line summary
    python scripts/dashboard_tui.py --view json      # Full JSON output

    # With custom project:
    DASHBOARD_PROJECT=my-project python scripts/dashboard_tui.py --view summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import from shared module (Streamlit-free)
from dashboard_shared import (
    PAPER_BASELINES,
    fetch_runs,
    dedupe_latest_runs,
    get_project_config,
)

console = Console()

# Color scheme
COLORS = {
    "title": "bold bright_cyan",
    "subtitle": "dim cyan",
    "model": "bright_cyan",
    "strategy": "bright_magenta",
    "good": "bright_green",
    "bad": "bright_red",
    "warning": "bright_yellow",
    "neutral": "white",
    "dim": "dim white",
    "highlight": "bold bright_white on blue",
    "header": "bold bright_white",
    "border": "bright_blue",
}

# Exit status colors
EXIT_COLORS = {
    "exit_submitted": ("Submitted", "bright_green", "✓"),
    "exit_cost": ("Cost Limit", "bright_yellow", "$"),
    "exit_context": ("Context", "bright_red", "⚠"),
    "exit_timeout": ("Timeout", "bright_blue", "⏱"),
    "exit_format": ("Format", "bright_magenta", "✗"),
    "exit_other": ("Other", "dim", "?"),
}


def colorize_rate(rate: float) -> str:
    """Return colored rate string based on value."""
    if pd.isna(rate):
        return "[dim]N/A[/]"
    if rate >= 0.5:
        return f"[bright_green]{rate:.1%}[/]"
    elif rate >= 0.3:
        return f"[bright_yellow]{rate:.1%}[/]"
    elif rate > 0:
        return f"[bright_red]{rate:.1%}[/]"
    else:
        return f"[dim]{rate:.1%}[/]"


def colorize_cost(cost: float, baseline: float | None = None) -> str:
    """Return colored cost string, optionally compared to baseline."""
    if pd.isna(cost) or cost <= 0:
        return "[dim]N/A[/]"

    cost_str = f"${cost:.2f}"

    if baseline and baseline > 0:
        ratio = cost / baseline
        if ratio <= 0.5:
            return f"[bright_green]{cost_str}[/]"  # >50% reduction
        elif ratio <= 0.8:
            return f"[bright_yellow]{cost_str}[/]"  # 20-50% reduction
        elif ratio > 1.2:
            return f"[bright_red]{cost_str}[/]"  # >20% increase

    return f"[white]{cost_str}[/]"


def colorize_delta(delta: float, is_cost: bool = False) -> str:
    """Return colored delta string. For cost, negative is good."""
    if is_cost:
        # For cost: negative (reduction) is good
        if delta <= -30:
            return f"[bright_green]{delta:+.0f}%[/]"
        elif delta <= -10:
            return f"[green]{delta:+.0f}%[/]"
        elif delta >= 10:
            return f"[bright_red]{delta:+.0f}%[/]"
        else:
            return f"[white]{delta:+.0f}%[/]"
    else:
        # For rate: positive is good
        if delta >= 0.05:
            return f"[bright_green]{delta:+.1%}[/]"
        elif delta >= 0:
            return f"[green]{delta:+.1%}[/]"
        elif delta >= -0.05:
            return f"[yellow]{delta:+.1%}[/]"
        else:
            return f"[bright_red]{delta:+.1%}[/]"


def build_comparison_table(df: pd.DataFrame) -> Table:
    """Build Table 1 equivalent: Model×Strategy comparison with paper baselines."""
    table = Table(
        title="[bold bright_cyan]📊 Model×Strategy Comparison with Paper[/]",
        box=box.ROUNDED,
        border_style="bright_blue",
        header_style="bold bright_white",
        title_style="bold bright_cyan",
        show_lines=True,
        padding=(0, 1),
    )
    table.add_column("Model", style="bright_cyan", no_wrap=True)
    table.add_column("Strategy", style="bright_magenta")
    table.add_column("Our Rate", justify="right", style="white")
    table.add_column("Raw", justify="right", style="dim")
    table.add_column("Δ vs Raw", justify="right")
    table.add_column("Our Cost", justify="right", style="white")
    table.add_column("Raw", justify="right", style="dim")
    table.add_column("Δ Cost", justify="right")
    table.add_column("N", justify="right", style="dim")

    if df.empty:
        return table
    eval_df = df[df["eval_complete"]] if "eval_complete" in df.columns else df
    if eval_df.empty:
        return table

    # Validate required columns
    required = {"model", "strategy", "solve_rate", "avg_cost", "n_instances", "n_resolved"}
    if not required.issubset(df.columns):
        return table

    # Aggregate by model×strategy using instance-weighted averages
    def weighted_agg(group):
        n_total = group["n_instances"].sum()
        n_resolved = group["n_resolved"].sum()
        valid_costs = group[group["avg_cost"].notna() & (group["n_instances"] > 0)]
        if len(valid_costs) > 0 and n_total > 0:
            weighted_cost = (valid_costs["avg_cost"] * valid_costs["n_instances"]).sum() / valid_costs["n_instances"].sum()
        else:
            weighted_cost = np.nan
        return pd.Series({
            "solve_rate": n_resolved / n_total if n_total > 0 else 0,
            "avg_cost": weighted_cost,
            "n_instances": n_total,
        })

    agg = eval_df.groupby(["model", "strategy"]).apply(weighted_agg, include_groups=False).reset_index()

    for _, row in agg.iterrows():
        model = row["model"]
        strategy = row["strategy"]

        # Find matching paper baseline
        paper_model = None
        best_match_len = 0
        for pm in PAPER_BASELINES:
            if pm in model.lower() or model.lower() in pm:
                if len(pm) > best_match_len:
                    paper_model = pm
                    best_match_len = len(pm)

        paper_vals = (
            PAPER_BASELINES.get(paper_model, {}).get(strategy, {})
            if paper_model
            else {}
        )
        paper_rate = paper_vals.get("solve_rate")
        paper_cost = paper_vals.get("avg_cost")

        # Get RAW baseline for delta calculation (compare all strategies to raw)
        raw_vals = (
            PAPER_BASELINES.get(paper_model, {}).get("raw", {})
            if paper_model
            else {}
        )
        raw_rate = raw_vals.get("solve_rate")
        raw_cost = raw_vals.get("avg_cost")

        # Calculate deltas vs RAW baseline (not vs same strategy)
        rate_delta_text = "[dim]—[/]"
        cost_delta_text = "[dim]—[/]"
        if raw_rate and row["solve_rate"] > 0:
            delta = row["solve_rate"] - raw_rate
            rate_delta_text = colorize_delta(delta, is_cost=False)
        if raw_cost and row["avg_cost"] > 0:
            delta_pct = ((row["avg_cost"] - raw_cost) / raw_cost) * 100
            cost_delta_text = colorize_delta(delta_pct, is_cost=True)

        table.add_row(
            f"[bright_cyan]{model[:18]}[/]",
            f"[bright_magenta]{strategy}[/]",
            colorize_rate(row["solve_rate"]),
            f"[dim]{raw_rate:.1%}[/]" if raw_rate else "[dim]—[/]",
            rate_delta_text,
            colorize_cost(row["avg_cost"], raw_cost),
            f"[dim]${raw_cost:.2f}[/]" if raw_cost else "[dim]—[/]",
            cost_delta_text,
            f"[dim]{int(row['n_instances'])}[/]",
        )

    return table


def build_runs_table(df: pd.DataFrame) -> Table:
    """Build run explorer table."""
    table = Table(
        title="[bold bright_cyan]📋 All Runs[/]",
        box=box.ROUNDED,
        border_style="bright_blue",
        header_style="bold bright_white",
        show_lines=True,
        padding=(0, 1),
    )
    table.add_column("Run", style="white", max_width=22, no_wrap=True)
    table.add_column("Model", style="bright_cyan", max_width=14)
    table.add_column("Strategy", style="bright_magenta")
    table.add_column("N", justify="right", style="dim")
    table.add_column("Resolved", justify="right")
    table.add_column("Rate", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Turns", justify="right", style="dim")

    if df.empty:
        return table

    # Validate required columns
    required = {"run_name", "model", "strategy", "n_instances", "n_resolved", "solve_rate", "avg_cost", "avg_turns"}
    if not required.issubset(df.columns):
        return table

    for _, row in df.iterrows():
        eval_complete = bool(row.get("eval_complete", True))
        resolved_color = "bright_green" if eval_complete and row["n_resolved"] > 0 else "dim"
        # Handle NaN in avg_turns
        turns = row["avg_turns"]
        turns_str = f"{turns:.0f}" if pd.notna(turns) else "—"
        resolved_str = f"{int(row['n_resolved'])}" if eval_complete and pd.notna(row["n_resolved"]) else "—"
        rate_str = colorize_rate(row["solve_rate"]) if eval_complete else "[dim]—[/]"

        table.add_row(
            row["run_name"][:22],
            f"[bright_cyan]{row['model'][:14]}[/]",
            f"[bright_magenta]{row['strategy']}[/]",
            str(row["n_instances"]),
            f"[{resolved_color}]{resolved_str}[/]",
            rate_str,
            colorize_cost(row["avg_cost"]),
            turns_str,
        )

    return table


def build_baselines_table() -> Table:
    """Build paper baselines reference table."""
    table = Table(
        title="[bold bright_cyan]📚 Paper Baselines (arXiv:2508.21433)[/]",
        box=box.ROUNDED,
        border_style="bright_blue",
        header_style="bold bright_white",
        show_lines=True,
        padding=(0, 1),
    )
    table.add_column("Model", style="bright_cyan")
    table.add_column("Strategy", style="bright_magenta")
    table.add_column("Solve Rate", justify="right")
    table.add_column("Avg Cost", justify="right")
    table.add_column("Rate Δ vs Raw", justify="right")
    table.add_column("Cost Δ vs Raw", justify="right")

    for model, strategies in PAPER_BASELINES.items():
        for strategy, vals in strategies.items():
            rate_delta = vals.get("rate_delta")
            cost_delta = vals.get("cost_delta")

            # Color the rate based on absolute value
            rate_str = colorize_rate(vals["solve_rate"])

            # Cost coloring (lower is better)
            cost = vals["avg_cost"]
            if cost <= 0.3:
                cost_str = f"[bright_green]${cost:.2f}[/]"
            elif cost <= 0.6:
                cost_str = f"[bright_yellow]${cost:.2f}[/]"
            else:
                cost_str = f"[white]${cost:.2f}[/]"

            # Delta coloring
            if rate_delta is not None:
                rate_delta_str = colorize_delta(rate_delta, is_cost=False)
            else:
                rate_delta_str = "[dim]baseline[/]"

            if cost_delta is not None:
                cost_delta_str = colorize_delta(cost_delta * 100, is_cost=True)
            else:
                cost_delta_str = "[dim]baseline[/]"

            table.add_row(
                f"[bright_cyan]{model}[/]",
                f"[bright_magenta]{strategy}[/]",
                rate_str,
                cost_str,
                rate_delta_str,
                cost_delta_str,
            )

    return table


def build_exit_panel(df: pd.DataFrame) -> Panel:
    """Build colored exit status panel with bar chart."""
    if df.empty:
        return Panel("[dim]No data[/]", title="Exit Status")

    exit_cols = list(EXIT_COLORS.keys())
    # Only use columns that exist in DataFrame
    exit_cols = [col for col in exit_cols if col in df.columns]
    if not exit_cols:
        return Panel("[dim]No exit data[/]", title="Exit Status")

    totals = {col: df[col].sum() for col in exit_cols}
    max_val = max(totals.values()) if totals.values() else 1
    total_all = sum(totals.values())

    lines = []
    for col in exit_cols:
        label, color, icon = EXIT_COLORS[col]
        count = totals.get(col, 0)
        pct = (count / total_all * 100) if total_all > 0 else 0
        bar_len = int((count / max_val) * 25) if max_val > 0 else 0
        bar = "█" * bar_len + "░" * (25 - bar_len)
        lines.append(
            f"  [{color}]{icon}[/] {label:12} [{color}]{bar}[/] "
            f"[{color}]{count:4}[/] [dim]({pct:4.1f}%)[/]"
        )

    return Panel(
        "\n".join(lines),
        title="[bold bright_cyan]📈 Exit Status Distribution[/]",
        border_style="bright_blue",
        padding=(1, 2),
    )


def build_metrics_panel(df: pd.DataFrame) -> Panel:
    """Build colorful metrics summary panel."""
    if df.empty:
        return Panel("[dim]No data[/]", title="📊 Summary", border_style="bright_blue")

    df = dedupe_latest_runs(df)

    # Safe access with defaults for missing columns
    total_runs = len(df)
    models = df["model"].nunique() if "model" in df.columns else 0
    strategies = df["strategy"].nunique() if "strategy" in df.columns else 0
    eval_df = df[df["eval_complete"]] if "eval_complete" in df.columns else df
    best_rate = eval_df["solve_rate"].max() if "solve_rate" in eval_df.columns else 0
    total_instances = eval_df["n_instances"].sum() if "n_instances" in eval_df.columns else 0
    total_resolved = eval_df["n_resolved"].sum() if "n_resolved" in eval_df.columns else 0
    avg_cost = df["avg_cost"].mean() if "avg_cost" in df.columns else 0

    # Create individual metric texts
    metrics = [
        f"[bold bright_cyan]Runs:[/] [bright_white]{total_runs}[/]",
        f"[bold bright_magenta]Models:[/] [bright_white]{models}[/]",
        f"[bold bright_yellow]Strategies:[/] [bright_white]{strategies}[/]",
        f"[bold bright_green]Best Rate:[/] {colorize_rate(best_rate)}",
        f"[bold bright_blue]Instances:[/] [bright_white]{total_instances}[/]",
        f"[bold bright_green]Resolved:[/] [bright_white]{total_resolved}[/]",
        f"[bold bright_yellow]Avg Cost:[/] {colorize_cost(avg_cost)}",
    ]

    return Panel(
        "  │  ".join(metrics),
        title="[bold bright_cyan]📊 Summary[/]",
        border_style="bright_blue",
        padding=(0, 1),
    )


def get_summary_stats(df: pd.DataFrame) -> dict:
    """Extract key statistics for non-interactive output."""
    if df.empty:
        return {
            "status": "empty",
            "runs": 0,
            "models": 0,
            "strategies": 0,
            "instances": 0,
            "resolved": 0,
            "best_rate": 0.0,
            "avg_cost": 0.0,
            "by_strategy": {},
        }

    def _safe_float(val, default=0.0):
        """Convert to float, returning default if NaN/None."""
        if pd.isna(val):
            return default
        return float(val)

    def _weighted_cost(subset):
        """Calculate instance-weighted average cost."""
        valid = subset[subset["avg_cost"].notna() & (subset["n_instances"] > 0)]
        if len(valid) == 0:
            return 0.0
        return (valid["avg_cost"] * valid["n_instances"]).sum() / valid["n_instances"].sum()

    df = dedupe_latest_runs(df)
    eval_df = df[df["eval_complete"]] if "eval_complete" in df.columns else df

    # Aggregate by strategy using weighted averages (evaluated runs only)
    by_strategy = {}
    if "strategy" in eval_df.columns:
        for strategy in eval_df["strategy"].unique():
            strat_df = eval_df[eval_df["strategy"] == strategy]
            n_inst = int(strat_df["n_instances"].sum()) if "n_instances" in strat_df else 0
            n_res = int(strat_df["n_resolved"].sum()) if "n_resolved" in strat_df else 0
            by_strategy[strategy] = {
                "runs": len(strat_df),
                "instances": n_inst,
                "resolved": n_res,
                "solve_rate": n_res / n_inst if n_inst > 0 else 0.0,  # Exact rate
                "avg_cost": _safe_float(_weighted_cost(strat_df)),
            }

    # Overall stats with weighted averages
    total_instances = int(eval_df["n_instances"].sum()) if "n_instances" in eval_df.columns else 0
    total_resolved = int(eval_df["n_resolved"].sum()) if "n_resolved" in eval_df.columns else 0
    overall_rate = total_resolved / total_instances if total_instances > 0 else 0.0

    return {
        "status": "ok",
        "runs": len(df),
        "models": int(df["model"].nunique()) if "model" in df.columns else 0,
        "strategies": int(df["strategy"].nunique()) if "strategy" in df.columns else 0,
        "instances": total_instances,
        "resolved": total_resolved,
        "best_rate": _safe_float(eval_df["solve_rate"].max()) if "solve_rate" in eval_df.columns else 0.0,
        "avg_cost": _safe_float(_weighted_cost(df)) if "avg_cost" in df.columns else 0.0,
        "by_strategy": by_strategy,
    }


def _sanitize_for_json(obj):
    """Recursively replace NaN/inf with None and convert numpy types for valid JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (float, np.floating)):
        # Handle both Python float and numpy floats (np.float64, etc.)
        if pd.isna(obj) or np.isinf(obj):
            return None
        return float(obj)  # Convert to Python float for JSON
    elif isinstance(obj, (int, np.integer)):
        return int(obj)  # Convert numpy int to Python int
    elif isinstance(obj, (bool, np.bool_)):
        return bool(obj)  # Convert numpy bool to Python bool
    return obj


def output_json(df: pd.DataFrame, project: str, entity: str | None) -> None:
    """Output full JSON data (valid JSON with null for missing values)."""
    stats = get_summary_stats(df)

    # Add run details
    runs = []
    if not df.empty:
        for _, row in df.iterrows():
            # Get values with explicit NaN handling
            solve_rate = row.get("solve_rate")
            avg_cost = row.get("avg_cost")
            avg_turns = row.get("avg_turns")
            run_data = {
                "name": row.get("run_name", ""),
                "model": row.get("model", ""),
                "strategy": row.get("strategy", ""),
                "n_instances": int(row.get("n_instances", 0)),
                "n_resolved": int(row.get("n_resolved", 0)),
                "solve_rate": float(solve_rate) if pd.notna(solve_rate) else None,
                "avg_cost": float(avg_cost) if pd.notna(avg_cost) else None,
                "avg_turns": float(avg_turns) if pd.notna(avg_turns) else None,
            }
            runs.append(run_data)

    output = {
        "project": project,
        "entity": entity,
        "summary": _sanitize_for_json(stats),  # Sanitize nested NaN values
        "summary_deduped": True,
        "runs": runs,
        "paper_baselines": PAPER_BASELINES,
    }

    print(json.dumps(output, indent=2))


def output_summary(df: pd.DataFrame, project: str) -> None:
    """Output single-line summary for scripts/automation."""
    stats = get_summary_stats(df)

    if stats["status"] == "empty":
        print(f"[{project}] No runs found")
        return

    # Format: [project] runs=N models=M best=X% resolved=R/I avg_cost=$C
    best_pct = stats["best_rate"] * 100
    avg_cost = stats["avg_cost"]
    cost_str = f"${avg_cost:.2f}" if avg_cost > 0 else "N/A"

    # Strategy breakdown
    strat_parts = []
    for strat, data in stats["by_strategy"].items():
        rate_pct = data["solve_rate"] * 100
        strat_parts.append(f"{strat}:{rate_pct:.1f}%")
    strat_str = " ".join(strat_parts) if strat_parts else ""

    print(
        f"[{project}] "
        f"runs={stats['runs']} "
        f"models={stats['models']} "
        f"best={best_pct:.1f}% "
        f"resolved={stats['resolved']}/{stats['instances']} "
        f"avg_cost={cost_str} "
        f"| {strat_str} "
        f"(deduped)"
    )


def output_compact(df: pd.DataFrame, project: str, entity: str | None) -> None:
    """Output compact multi-line summary."""
    stats = get_summary_stats(df)

    print(f"=== {project} {'(' + entity + ')' if entity else ''} (deduped) ===")

    if stats["status"] == "empty":
        print("No runs found.")
        return

    print(f"Runs: {stats['runs']} | Models: {stats['models']} | Strategies: {stats['strategies']}")
    print(f"Instances: {stats['instances']} | Resolved: {stats['resolved']} | Best Rate: {stats['best_rate']:.1%}")
    print(f"Average Cost: ${stats['avg_cost']:.2f}" if stats['avg_cost'] > 0 else "Average Cost: N/A")
    print()

    if stats["by_strategy"]:
        print("By Strategy:")
        for strat, data in sorted(stats["by_strategy"].items()):
            cost_str = f"${data['avg_cost']:.2f}" if data['avg_cost'] > 0 else "N/A"
            print(f"  {strat:20} rate={data['solve_rate']:5.1%} resolved={data['resolved']:3}/{data['instances']:<3} cost={cost_str}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Complexity Trap TUI Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
View modes (--view):
  summary   Single-line summary (for scripts/automation)
  compact   Multi-line compact summary (no colors)
  json      Full JSON output with all data

Examples:
  python scripts/dashboard_tui.py --view summary
  python scripts/dashboard_tui.py --view json -p my-project
        """,
    )

    parser.add_argument(
        "--view", "-v",
        choices=["summary", "compact", "json"],
        help="Non-interactive output mode: summary, compact, or json",
    )
    parser.add_argument(
        "--project", "-p",
        help="WandB project name (overrides DASHBOARD_PROJECT)",
    )
    parser.add_argument(
        "--entity", "-e",
        help="WandB entity (overrides DASHBOARD_ENTITY)",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Show all runs in the explorer table (include duplicates)",
    )

    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    # Get project config, with CLI overrides
    project, entity = get_project_config()
    if args.project:
        project = args.project
    if args.entity:
        entity = args.entity

    # Determine if we're in non-interactive mode
    non_interactive = args.view is not None

    if not project:
        if non_interactive:
            print("ERROR: Missing WANDB_PROJECT or DASHBOARD_PROJECT", file=sys.stderr)
            sys.exit(1)
        console.print(Panel(
            "[bright_red]Missing WANDB_PROJECT or DASHBOARD_PROJECT environment variable.[/]\n\n"
            "Set in .env or run with:\n"
            "[bright_cyan]DASHBOARD_PROJECT=your-project python scripts/dashboard_tui.py[/]",
            title="[bold bright_red]⚠ Configuration Error[/]",
            border_style="bright_red",
        ))
        return

    # Fetch data
    if non_interactive:
        # Silent fetch for non-interactive modes
        try:
            df = fetch_runs(project, entity, use_cache=False)
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)

        # Output in requested format
        if args.view == "json":
            output_json(df, project, entity)
        elif args.view == "summary":
            output_summary(df, project)
        elif args.view == "compact":
            output_compact(df, project, entity)
        return

    # Interactive TUI mode below
    # Header
    console.print()
    console.print(Panel(
        f"[bold bright_white]The Complexity Trap[/] │ "
        f"[bright_cyan]Project:[/] [bright_white]{project}[/]"
        + (f" │ [bright_magenta]Entity:[/] [bright_white]{entity}[/]" if entity else ""),
        style="on dark_blue",
        border_style="bright_cyan",
    ))
    console.print()

    # Fetch data with spinner
    with console.status("[bold bright_green]⏳ Fetching data from WandB...[/]", spinner="dots"):
        try:
            df = fetch_runs(project, entity, use_cache=False)
        except Exception as e:
            console.print(Panel(
                f"[bright_red]{e}[/]",
                title="[bold bright_red]⚠ Error Loading Data[/]",
                border_style="bright_red",
            ))
            return

    if df.empty:
        console.print(Panel(
            "[bright_yellow]No runs found in this project.[/]\n"
            "Run some experiments first!",
            title="[bold bright_yellow]📭 Empty Project[/]",
            border_style="bright_yellow",
        ))
        return

    # Metrics summary (deduped by run_name)
    console.print(build_metrics_panel(df))
    console.print()

    # Exit status chart (deduped by run_name)
    console.print(build_exit_panel(dedupe_latest_runs(df)))
    console.print()

    # Paper comparison table (deduped by run_name)
    console.print(build_comparison_table(dedupe_latest_runs(df)))
    console.print()

    # Run explorer (optional duplicates)
    runs_df = df if args.all_runs else dedupe_latest_runs(df)
    console.print(build_runs_table(runs_df))
    console.print()

    # Paper baselines
    console.print(build_baselines_table())
    console.print()

    # Footer
    console.print(Panel(
        "[dim]Data source: WandB │ Paper: arXiv:2508.21433 │ "
        "Press [bright_cyan]Ctrl+C[/] to exit[/]",
        border_style="dim",
    ))
    console.print()


if __name__ == "__main__":
    main()
