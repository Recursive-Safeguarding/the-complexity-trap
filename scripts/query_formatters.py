#!/usr/bin/env python3
"""
Output formatters for WandB experiment queries.

Supports: table (rich), markdown, json, csv
"""

from __future__ import annotations

import json
import numbers
import sys
from typing import TYPE_CHECKING

import pandas as pd
from rich.console import Console
from rich.table import Table
from rich import box

if TYPE_CHECKING:
    from query_metrics import QueryResult


def render(
    result: "QueryResult",
    format: str = "table",
    file=None,
) -> None:
    """Render a QueryResult in the specified format.

    Args:
        result: QueryResult from a query function
        format: One of 'table', 'markdown', 'json', 'csv'
        file: Output file (default: stdout)
    """
    if file is None:
        file = sys.stdout

    if format == "table":
        _render_table(result, file)
    elif format == "markdown":
        _render_markdown(result, file)
    elif format == "json":
        _render_json(result, file)
    elif format == "csv":
        _render_csv(result, file)
    else:
        raise ValueError(f"Unknown format: {format}")


def _render_table(result: "QueryResult", file) -> None:
    """Render as rich table to terminal."""
    console = Console(file=file)

    # print insights first
    if result.insights:
        for insight in result.insights:
            console.print(f"[dim]{insight}[/]")
        console.print()

    if result.data.empty:
        console.print("[yellow]No data to display.[/]")
        return

    # build table
    table = Table(
        title=f"[bold cyan]{result.title}[/]",
        box=box.ROUNDED,
        border_style="bright_blue",
        header_style="bold white",
        show_lines=False,
    )

    # add columns
    for col in result.columns:
        justify = "right" if col in ("rank", "n_instances", "n_resolved", "count") else "left"
        style = _get_column_style(col)
        table.add_column(col, justify=justify, style=style)

    # add rows
    for _, row in result.data.iterrows():
        values = []
        for col in result.columns:
            val = row.get(col, "")
            values.append(_format_cell(col, val))
        table.add_row(*values)

    console.print(table)


def _render_markdown(result: "QueryResult", file) -> None:
    """Render as markdown table."""
    lines = []

    # title
    lines.append(f"## {result.title}")
    lines.append("")

    # insights
    if result.insights:
        for insight in result.insights:
            lines.append(f"**{insight}**")
        lines.append("")

    if result.data.empty:
        lines.append("*No data to display.*")
        file.write("\n".join(lines))
        return

    # header row
    header = "| " + " | ".join(result.columns) + " |"
    separator = "|" + "|".join(["---"] * len(result.columns)) + "|"
    lines.append(header)
    lines.append(separator)

    # data rows
    for _, row in result.data.iterrows():
        cells = []
        for col in result.columns:
            val = row.get(col, "")
            cells.append(_format_cell_md(col, val))
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    file.write("\n".join(lines))


def _render_json(result: "QueryResult", file) -> None:
    """Render as JSON."""
    output = {
        "title": result.title,
        "insights": result.insights,
        "columns": result.columns,
        "data": result.data.to_dict(orient="records") if not result.data.empty else [],
    }

    # handle NaN values
    def clean_nan(obj):
        if isinstance(obj, dict):
            return {k: clean_nan(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_nan(v) for v in obj]
        elif isinstance(obj, float) and pd.isna(obj):
            return None
        return obj

    output = clean_nan(output)
    json.dump(output, file, indent=2)
    file.write("\n")


def _render_csv(result: "QueryResult", file) -> None:
    """Render as CSV."""
    if result.data.empty:
        file.write("# No data\n")
        return

    # filter to only requested columns
    df = result.data[result.columns] if result.columns else result.data
    df.to_csv(file, index=False)


def _get_column_style(col: str) -> str:
    """Get rich style for a column."""
    styles = {
        "model": "cyan",
        "strategy": "magenta",
        "solve_rate": "green",
        "our_rate": "green",
        "paper_rate": "dim",
        "avg_cost": "yellow",
        "our_cost": "yellow",
        "paper_cost": "dim",
        "rate_delta": "white",
        "cost_delta": "white",
        "rate_vs_raw": "white",
        "cost_vs_raw": "white",
        "rank": "dim",
        "n_instances": "dim",
        "status": "white",
        "percentage": "cyan",
    }
    return styles.get(col, "white")


def _format_cell(col: str, val) -> str:
    """Format a cell value for rich table display."""
    if pd.isna(val):
        return "[dim]—[/]"

    if col in ("solve_rate", "our_rate", "paper_rate"):
        if isinstance(val, numbers.Real):
            rate = float(val)
            if rate >= 0.5:
                return f"[bright_green]{rate:.1%}[/]"
            elif rate >= 0.3:
                return f"[yellow]{rate:.1%}[/]"
            else:
                return f"[red]{rate:.1%}[/]"

    if col in ("avg_cost", "our_cost", "paper_cost"):
        if isinstance(val, numbers.Real):
            return f"${float(val):.2f}"

    if col == "rate_delta":
        if isinstance(val, str) and val.startswith("+"):
            return f"[bright_green]{val}[/]"
        elif isinstance(val, str) and val.startswith("-"):
            return f"[bright_red]{val}[/]"

    if col == "cost_delta":
        if isinstance(val, str) and val.startswith("-"):
            return f"[bright_green]{val}[/]"  # negative cost delta is good
        elif isinstance(val, str) and val.startswith("+"):
            return f"[bright_red]{val}[/]"

    if col == "rate_vs_raw":
        if isinstance(val, str) and val.startswith("+"):
            return f"[bright_green]{val}[/]"
        elif isinstance(val, str) and val.startswith("-"):
            return f"[bright_red]{val}[/]"

    if col == "cost_vs_raw":
        if isinstance(val, str) and val.startswith("-"):
            return f"[bright_green]{val}[/]"
        elif isinstance(val, str) and val.startswith("+"):
            return f"[bright_red]{val}[/]"

    if col == "eval_complete":
        return "[green]✓[/]" if val else "[red]✗[/]"

    return str(val)


def _format_cell_md(col: str, val) -> str:
    """Format a cell value for markdown display (no rich markup)."""
    if pd.isna(val):
        return "—"

    if col in ("solve_rate", "our_rate", "paper_rate"):
        if isinstance(val, numbers.Real):
            return f"{float(val):.1%}"

    if col in ("avg_cost", "our_cost", "paper_cost"):
        if isinstance(val, numbers.Real):
            return f"${float(val):.2f}"

    if col == "eval_complete":
        return "✓" if val else "✗"

    return str(val)
