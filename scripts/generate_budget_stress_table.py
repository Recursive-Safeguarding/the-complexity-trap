#!/usr/bin/env python3
"""Generate LaTeX table for budget-stress results (L=40k).

Compaction trigger numbers are derived from `scripts/compaction_trigger_stats.py`
(`summarize_run`) so reporting uses a single canonical source.

Usage:
  python scripts/generate_budget_stress_table.py
  python scripts/generate_budget_stress_table.py --output budget_stress_table.tex
  python scripts/generate_budget_stress_table.py --trajectories-root trajectories
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from compaction_trigger_stats import summarize_run
from dashboard_shared import dedupe_latest_runs, fetch_runs


def _warn(msg: str, quiet_missing: bool) -> None:
    if not quiet_missing:
        print(f"WARNING: {msg}", file=sys.stderr)


def _rate_to_pct(rate: float | None) -> str:
    if rate is None:
        return "N/A"
    return f"{rate:.1%}"


def _norm_summarizer(value: object) -> str:
    if value is None:
        return "same"
    s = str(value).strip()
    if not s or s.lower() in ("none", "nan"):
        return "same"
    if s == "reuse-agent-model":
        return "same"
    return s


def _boollike(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return False
        return bool(value)
    if value is None:
        return False
    s = str(value).strip().lower()
    return s in ("1", "true", "yes", "y", "t")


def _safe_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _safe_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not math.isfinite(parsed):
            return None
        return int(parsed)


def resolve_trajectory_run_dir(
    trajectories_root: Path,
    run_name: str,
) -> tuple[Path | None, str | None]:
    """Resolve run_name to a concrete trajectory directory.

    Search order:
    1) trajectories/<run_name>
    2) trajectories/<owner>/<run_name> (one level deep)
    """
    if not run_name:
        return None, "run_name is empty"
    if not trajectories_root.exists():
        return None, f"trajectories root does not exist: {trajectories_root}"
    if not trajectories_root.is_dir():
        return None, f"trajectories root is not a directory: {trajectories_root}"

    candidates: list[Path] = []

    direct = trajectories_root / run_name
    if direct.is_dir():
        candidates.append(direct)

    for owner_dir in sorted(p for p in trajectories_root.iterdir() if p.is_dir()):
        candidate = owner_dir / run_name
        if candidate.is_dir():
            candidates.append(candidate)

    # Deduplicate paths while preserving discovery order.
    deduped: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    if not deduped:
        return None, f"trajectory directory not found for run_name={run_name}"
    if len(deduped) == 1:
        return deduped[0], None

    # Deterministic pick: newest mtime, then lexical path as tie-break.
    chosen = sorted(deduped, key=lambda p: (p.stat().st_mtime, str(p)))[-1]
    options = ", ".join(str(p) for p in deduped)
    warning = (
        f"multiple trajectory directories for run_name={run_name}; "
        f"using {chosen} among [{options}]"
    )
    return chosen, warning


def trigger_rate_from_summary(run_row: pd.Series, run_summary: dict[str, Any]) -> str:
    """Select an appropriate trigger-like percentage for the table row.

    - Limit-aware runs: use debug-log trigger rate (instances with any trigger).
    - Summary periodic runs: use traj summary-call rate (instances with summaries).
    - Raw baseline: not applicable.
    - Periodic masking: not measurable via current artifacts -> N/A.
    """
    strategy = str(run_row.get("strategy", ""))
    limit_aware = _boollike(run_row.get("hp_limit_aware", False))

    if strategy == "raw":
        return "--"

    log_rate = run_summary["log"].get("trigger_rate")
    traj_rate = run_summary["traj"].get("summary_rate")

    if limit_aware and log_rate is not None:
        return _rate_to_pct(log_rate)

    if strategy in ("llm_summary", "on_demand", "hybrid") and traj_rate is not None:
        return _rate_to_pct(traj_rate)

    return "N/A"


def _latest_row(df: pd.DataFrame) -> pd.Series | None:
    if df.empty:
        return None
    work = df.copy()
    if "created_at" in work.columns:
        work["_created_at_ts"] = pd.to_datetime(work["created_at"], errors="coerce")
        work = work.sort_values("_created_at_ts", ascending=True)
    return work.iloc[-1]


def build_budget_stress_rows(
    df: pd.DataFrame,
    trajectories_root: Path,
    quiet_missing: bool,
) -> tuple[list[dict[str, str]], float | None]:
    """Build table rows and return (rows, raw_solve_rate)."""
    work = df.copy()

    for col, default in (
        ("instances_subset", ""),
        ("hp_limit_min_tokens", 0),
        ("hp_limit_aware", False),
        ("eval_complete", False),
        ("run_name", ""),
        ("strategy", ""),
        ("model", ""),
        ("summarizer", "same"),
    ):
        if col not in work.columns:
            work[col] = default

    subset_norm = work["instances_subset"].astype(str).str.lower().str.replace("_", "-", regex=False)
    work["instances_subset_norm"] = subset_norm
    work["summarizer"] = work["summarizer"].apply(_norm_summarizer)
    work["hp_limit_min_tokens_norm"] = work["hp_limit_min_tokens"].apply(_safe_int)

    # Keep only latest run per run_name after filtering to evaluated runs.
    work = work[work["eval_complete"].apply(_boollike)].copy()
    work = dedupe_latest_runs(work)

    mask_budget = (
        (work["model"] == "glm-4.7")
        & (work["instances_subset_norm"].isin(["verified-mini", "verifiedmini", "mini"]))
        & (work["hp_limit_min_tokens_norm"] == 40000)
    )
    budget_df = work[mask_budget].copy()

    if budget_df.empty:
        return [], None

    raw_mask = (
        (work["model"] == "glm-4.7")
        & (work["strategy"] == "raw")
        & (work["instances_subset_norm"].isin(["verified-mini", "verifiedmini", "mini"]))
    )
    raw_row = _latest_row(work[raw_mask])
    raw_solve_rate = _safe_float(raw_row.get("solve_rate")) if raw_row is not None else None
    if raw_solve_rate is None:
        _warn(
            "no evaluated raw baseline found for glm-4.7 verified-mini; "
            "delta vs raw will be reported as N/A",
            quiet_missing=quiet_missing,
        )

    rows: list[dict[str, str]] = []
    for _, run in budget_df.sort_values(["strategy", "run_name"]).iterrows():
        run_name = str(run.get("run_name", "") or "")
        run_dir, warning = resolve_trajectory_run_dir(trajectories_root, run_name)
        if warning:
            _warn(warning, quiet_missing=quiet_missing)

        if run_dir is None:
            trigger_rate = "N/A"
        else:
            try:
                run_summary = summarize_run(run_dir, source="auto")
                trigger_rate = trigger_rate_from_summary(run, run_summary)
            except Exception as exc:  # defensive for malformed files/dirs
                _warn(f"failed to summarize run_name={run_name} at {run_dir}: {exc}", quiet_missing=quiet_missing)
                trigger_rate = "N/A"

        strategy = str(run.get("strategy", ""))
        summarizer = _norm_summarizer(run.get("summarizer"))
        if strategy == "on_demand":
            if summarizer in ("same", "glm-4.7"):
                config_name = "On-demand summary (self)"
            elif summarizer == "minimax-m2.1":
                config_name = "On-demand summary (minimax)"
            else:
                config_name = f"On-demand summary ({summarizer})"
        elif strategy == "observation_masking":
            config_name = "Limit-aware masking" if _boollike(run.get("hp_limit_aware", False)) else "Periodic masking"
        elif strategy == "llm_summary":
            if summarizer in ("same", "glm-4.7"):
                config_name = "Periodic summary (self)"
            else:
                config_name = f"Periodic summary ({summarizer})"
        elif strategy == "hybrid":
            config_name = "Hybrid"
        else:
            config_name = strategy

        solve_rate_val = _safe_float(run.get("solve_rate"))
        if solve_rate_val is None:
            solve_rate_str = "N/A"
            delta_str = "N/A"
        else:
            solve_rate_str = f"{solve_rate_val:.1%}"
            if raw_solve_rate is None:
                delta_str = "N/A"
            else:
                delta = solve_rate_val - raw_solve_rate
                delta_str = f"{delta:+.1%}" if delta != 0 else "---"

        rows.append(
            {
                "config": config_name,
                "trigger_rate": trigger_rate,
                "solve_rate": solve_rate_str,
                "delta": delta_str,
            }
        )

    rows.insert(
        0,
        {
            "config": "Raw baseline",
            "trigger_rate": "--",
            "solve_rate": _rate_to_pct(raw_solve_rate),
            "delta": "---",
        },
    )
    return rows, raw_solve_rate


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def render_latex_table(rows: list[dict[str, str]]) -> str:
    latex = r"""\begin{table}[t]
\centering
\caption{Budget-stress results at $L=40$k tokens (GLM-4.7, $n=50$).}
\label{tab:budget-stress}
\begin{tabular}{lrrr}
\toprule
Configuration & Trigger Rate & Solve Rate & vs Raw \\
\midrule
"""
    for row in rows:
        config = _latex_escape(row["config"])
        trigger = _latex_escape(row["trigger_rate"])
        solve = _latex_escape(row["solve_rate"])
        delta = _latex_escape(row["delta"])
        latex += f"{config} & {trigger} & {solve} & {delta} \\\\\n"
    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output LaTeX file (default: print to stdout)",
    )
    parser.add_argument(
        "--project",
        default=os.getenv("WANDB_PROJECT", "the-complexity-trap"),
        help="WandB project name",
    )
    parser.add_argument(
        "--entity",
        default=os.getenv("WANDB_ENTITY", "ox"),
        help="WandB entity/team name",
    )
    parser.add_argument(
        "--trajectories-root",
        type=Path,
        default=Path("trajectories"),
        help="Root folder that contains trajectory run directories",
    )
    parser.add_argument(
        "--quiet-missing",
        action="store_true",
        help="Suppress warnings for missing or ambiguous trajectory directories",
    )
    args = parser.parse_args()

    df = fetch_runs(args.project, args.entity)
    rows, _ = build_budget_stress_rows(
        df=df,
        trajectories_root=args.trajectories_root,
        quiet_missing=args.quiet_missing,
    )

    if not rows:
        print("No budget-stress runs found yet. Experiments may still be running.")
        return 0

    latex = render_latex_table(rows)

    if args.output:
        args.output.write_text(latex, encoding="utf-8")
        print(f"Budget-stress table written to: {args.output}")
    else:
        print(latex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
