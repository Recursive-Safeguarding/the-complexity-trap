#!/usr/bin/env python3
"""
CLI Query Tool for WandB Experiment Analysis.

Answers common questions about ML experiments from the command line.

Usage:
    python scripts/query.py summary                           # One-liner status
    python scripts/query.py leaderboard                       # Best models ranked
    python scripts/query.py --model glm-4.7 paper-comparison  # Compare to paper
    python scripts/query.py --model glm-4.7 --format markdown compare-strategies

Output Formats:
    --format table     Rich terminal table (default)
    --format markdown  Copy-paste ready for slides/docs
    --format json      For scripts and pipelines
    --format csv       For spreadsheets
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

from dashboard_shared import fetch_runs, get_project_config
from query_formatters import render
from query_metrics import (
    compute_failures,
    compute_leaderboard,
    compute_paper_comparison,
    compute_runs,
    compute_strategy_comparison,
    compute_summary,
)


def cmd_summary(df, args) -> None:
    """Quick one-liner summary."""
    result = compute_summary(df)

    if args.format == "table":
        # for summary, just print the insight line
        if result.insights:
            print(result.insights[0])
        else:
            print("No data available.")
    else:
        render(result, args.format)


def cmd_leaderboard(df, args) -> None:
    """Best models ranked by solve rate."""
    result = compute_leaderboard(
        df,
        strategy_filter=args.strategy,
        model_filter=args.model,
        min_instances=args.min_instances,
    )
    render(result, args.format)


def cmd_paper_comparison(df, args) -> None:
    """Compare results to paper baselines."""
    result = compute_paper_comparison(df, model_filter=args.model)
    render(result, args.format)


def cmd_compare_strategies(df, args) -> None:
    """Compare strategies for a model."""
    result = compute_strategy_comparison(df, model_filter=args.model)
    render(result, args.format)


def cmd_failures(df, args) -> None:
    """Analyze exit status distribution."""
    result = compute_failures(
        df,
        model_filter=args.model,
        strategy_filter=args.strategy,
    )
    render(result, args.format)


def cmd_runs(df, args) -> None:
    """List runs with filtering."""
    result = compute_runs(
        df,
        model_filter=args.model,
        strategy_filter=args.strategy,
        min_instances=args.min_instances,
        eval_only=args.eval_only,
    )
    render(result, args.format)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Query WandB experiment data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s summary                                    # Quick status one-liner
  %(prog)s --format markdown leaderboard              # Markdown for slides
  %(prog)s --model glm-4.7 paper-comparison           # Compare to paper
  %(prog)s --model glm-4.7 compare-strategies         # Strategy breakdown
  %(prog)s --model glm-4.7 failures                   # Exit status analysis
  %(prog)s --eval-only runs                           # List evaluated runs

Output Formats:
  --format table      Rich terminal table with colors (default)
  --format markdown   Copy-paste ready for slides/docs
  --format json       For scripts: | jq '.data[0].solve_rate'
  --format csv        For spreadsheets: > results.csv

Filters:
  --model X           Partial match on model name
  --strategy X        Partial match on strategy
  --min-instances N   Skip runs with fewer than N instances
  --eval-only         Only runs that completed evaluation

Full documentation: docs/usage/query_tool.md
        """,
    )

    # global options
    parser.add_argument(
        "--project",
        "-p",
        help="WandB project (default: $WANDB_PROJECT or $DASHBOARD_PROJECT)",
    )
    parser.add_argument(
        "--entity",
        "-e",
        help="WandB entity (default: $WANDB_ENTITY or $DASHBOARD_ENTITY)",
    )
    parser.add_argument(
        "--format",
        "-f",
        default="table",
        choices=["table", "markdown", "json", "csv"],
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--model",
        "-m",
        help="Filter by model name (partial match)",
    )
    parser.add_argument(
        "--strategy",
        "-s",
        help="Filter by strategy (partial match)",
    )
    parser.add_argument(
        "--min-instances",
        type=int,
        default=10,
        help="Minimum instances per run (default: 10)",
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Only include evaluated runs",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print debug info (matched runs, counts)",
    )

    # subcommands
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser(
        "summary",
        help="Quick one-liner status of best performer",
    )
    subparsers.add_parser(
        "leaderboard",
        help="Best models ranked by solve rate with paper deltas",
    )
    subparsers.add_parser(
        "paper-comparison",
        help="Compare results directly to paper baselines (arXiv:2508.21433)",
    )
    subparsers.add_parser(
        "compare-strategies",
        help="Compare raw/obs_masking/llm_summary/hybrid for a model",
    )
    subparsers.add_parser(
        "failures",
        help="Exit status breakdown (why instances fail)",
    )
    subparsers.add_parser(
        "runs",
        help="List runs with filtering",
    )

    args = parser.parse_args()

    # get project config
    project = args.project
    entity = args.entity
    if not project:
        project, entity_from_env = get_project_config()
        if not entity:
            entity = entity_from_env

    if not project:
        print(
            "ERROR: No project specified. Set WANDB_PROJECT or use --project",
            file=sys.stderr,
        )
        sys.exit(1)

    # fetch data
    if args.debug:
        print(f"[DEBUG] Fetching from project: {entity}/{project}", file=sys.stderr)

    try:
        df = fetch_runs(project, entity)
    except Exception as e:
        print(f"ERROR: Failed to fetch runs: {e}", file=sys.stderr)
        sys.exit(1)

    if args.debug:
        print(f"[DEBUG] Fetched {len(df)} runs", file=sys.stderr)
        if args.model:
            matches = df[df["model"].str.contains(args.model, case=False, na=False)]
            print(f"[DEBUG] {len(matches)} runs match --model {args.model}", file=sys.stderr)

    # dispatch to command handler
    commands = {
        "summary": cmd_summary,
        "leaderboard": cmd_leaderboard,
        "paper-comparison": cmd_paper_comparison,
        "compare-strategies": cmd_compare_strategies,
        "failures": cmd_failures,
        "runs": cmd_runs,
    }

    handler = commands.get(args.command)
    if handler:
        handler(df, args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
