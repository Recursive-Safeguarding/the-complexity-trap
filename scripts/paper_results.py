#!/usr/bin/env python3
"""Generate publication-ready table and figure for the MemAgents workshop paper.

Queries WandB for all GLM-4.7 runs, classifies them as periodic vs on-demand,
computes solve rates with Wilson confidence intervals, and produces:
  - LaTeX table (Table 1)
  - matplotlib grouped bar chart (Figure 1)

Usage:
    # Show results table in terminal
    python scripts/paper_results.py

    # Generate LaTeX table
    python scripts/paper_results.py --latex

    # Write LaTeX table directly to a file (table only; no terminal header)
    python scripts/paper_results.py --latex-out background-documents/our-workshop-draft/results_table.tex

    # Generate Figure 1 as PDF
    python scripts/paper_results.py --figure paper_figure1.pdf

    # Both
    python scripts/paper_results.py --latex --figure paper_figure1.pdf
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from dashboard_shared import (
    fetch_runs, dedupe_latest_runs, PHASE3_PERIODIC, CROSS_MODEL_PERIODIC,
    THRESHOLD_SWEEP_DATA, TRIGGER_RATE_BY_THRESHOLD,
)
from compaction_trigger_stats import compute_run_stats_from_traj


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0, center - spread), min(1, center + spread)


def classify_trigger(row: pd.Series) -> str:
    """Classify a run as periodic or on-demand based on strategy + hyperparams."""
    strategy = row.get("strategy", "")
    limit_aware = _boollike(row.get("hp_limit_aware", False))

    if strategy == "raw":
        return "baseline"
    if strategy == "on_demand":
        return "on_demand"
    if limit_aware:
        return "on_demand"
    return "periodic"


def _warn(msg: str) -> None:
    print(f"WARNING: {msg}", file=sys.stderr)


def _boollike(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return False
        return bool(value)
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "y", "t")


def _floatlike(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _intlike(value: object) -> int | None:
    parsed = _floatlike(value)
    if parsed is None:
        return None
    return int(parsed)


def _pick_preferred_row(
    rows: pd.DataFrame,
    *,
    prefer_min_tokens: int | None = None,
) -> pd.Series:
    """Pick one row deterministically, preferring explicit budget if requested."""
    if rows.empty:
        raise ValueError("cannot pick from empty rows")

    work = rows.copy()

    if prefer_min_tokens is not None and "hp_limit_min_tokens" in work.columns:
        min_tokens = work["hp_limit_min_tokens"].apply(_intlike)
        preferred = work[min_tokens == prefer_min_tokens]
        if not preferred.empty:
            work = preferred

    if "created_at" in work.columns:
        work["_created_at_ts"] = pd.to_datetime(work["created_at"], errors="coerce")
    else:
        work["_created_at_ts"] = pd.NaT

    if "run_name" not in work.columns:
        work["run_name"] = ""

    # Newest timestamp first; run_name is a deterministic tie-breaker.
    ordered = work.sort_values(
        by=["_created_at_ts", "run_name"],
        ascending=[False, False],
        na_position="last",
    )
    return ordered.iloc[0]


def _resolve_run_dir(trajectories_dir: Path, run_name: str) -> tuple[Path | None, str | None]:
    """Resolve a WandB run name to a trajectory directory.

    Search order:
      1) trajectories/<run_name>
      2) trajectories/<owner>/<run_name>
    """
    if not run_name:
        return None, "run_name is empty"
    if not trajectories_dir.exists():
        return None, f"trajectories dir does not exist: {trajectories_dir}"
    if not trajectories_dir.is_dir():
        return None, f"trajectories dir is not a directory: {trajectories_dir}"
    run_path = Path(run_name)
    if run_path.is_absolute():
        return None, f"run_name must be relative, got absolute path: {run_name}"
    if ".." in run_path.parts:
        return None, f"run_name contains parent traversal: {run_name}"

    root_resolved = trajectories_dir.resolve()
    candidates: list[Path] = []
    direct = trajectories_dir / run_path
    if direct.is_dir() and (root_resolved == direct.resolve() or root_resolved in direct.resolve().parents):
        candidates.append(direct)
    for owner_dir in sorted(p for p in trajectories_dir.iterdir() if p.is_dir()):
        candidate = owner_dir / run_path
        if candidate.is_dir() and (root_resolved == candidate.resolve() or root_resolved in candidate.resolve().parents):
            candidates.append(candidate)

    deduped: list[Path] = []
    seen: set[str] = set()
    for p in candidates:
        key = str(p.resolve())
        if key not in seen:
            seen.add(key)
            deduped.append(p)

    if not deduped:
        return None, f"trajectory dir not found for run_name={run_name}"
    if len(deduped) == 1:
        return deduped[0], None

    chosen = sorted(deduped, key=lambda p: (p.stat().st_mtime, str(p)))[-1]
    choices = ", ".join(str(p) for p in deduped)
    return chosen, f"multiple trajectory dirs for run_name={run_name}; using {chosen} among [{choices}]"


def _summary_call_stats_from_traj(run_dir: Path) -> tuple[float, float]:
    """Return (avg_summary_calls, summary_trigger_rate) from local .traj files."""
    stats = compute_run_stats_from_traj(run_dir)
    if not stats:
        return 0.0, 0.0
    total = sum(s.triggers_any for s in stats)
    n_triggered = sum(1 for s in stats if s.triggers_any > 0)
    return total / len(stats), n_triggered / len(stats)


def build_results_table(
    project: str,
    entity: str | None = None,
    trajectories_dir: Path | None = None,
) -> pd.DataFrame:
    df = fetch_runs(project, entity)

    # filter to GLM-4.7 verified-mini 50-task runs (exclude full 500)
    if "instances_subset" in df.columns:
        df["instances_subset_norm"] = (
            df["instances_subset"].astype(str).str.lower().str.replace("_", "-", regex=False)
        )
    else:
        df["instances_subset_norm"] = "unknown"
    mask = (
        (df["model"] == "glm-4.7")
        & (df["instances_subset_norm"].isin(["verified-mini", "verifiedmini", "mini"]))
        & (df["n_instances"].between(40, 60))  # 50-task runs only (exclude full 500)
    )
    df = df[mask].copy()

    if "eval_complete" in df.columns:
        df = df[df["eval_complete"]].copy()

    # Dedupe *after* filtering to evaluated runs; otherwise an in-progress rerun
    # with the same display name can hide the most recent evaluated run.
    df = dedupe_latest_runs(df)

    if df.empty:
        print("No matching runs found. Experiments may still be running.")
        return pd.DataFrame()

    # Normalize summarizer naming ("same" vs explicit self model) and compute trigger
    # before we do a second, config-key-based dedupe.
    if "summarizer" in df.columns:
        def _norm_summarizer(x) -> str:
            if x is None:
                return "same"
            s = str(x).strip()
            if not s or s.lower() in ("none", "nan"):
                return "same"
            if s in ("reuse-agent-model",):
                return "same"
            return s

        df["summarizer"] = df["summarizer"].apply(_norm_summarizer)
    else:
        df["summarizer"] = "same"

    df["trigger"] = df.apply(classify_trigger, axis=1)

    def compaction_method(row):
        s = row["strategy"]
        if s == "raw":
            return "none"
        if s in ("observation_masking", "dedup_obs_masking"):
            return "masking"
        return "summary"

    df["compaction"] = df.apply(compaction_method, axis=1)

    # filter to expected hyperparams
    if {"hp_obs_n", "hp_sum_n", "hp_sum_keep_m"}.issubset(df.columns):
        def _hparams_ok(row: pd.Series) -> bool:
            strat = row.get("strategy", "")
            if strat in ("observation_masking", "dedup_obs_masking"):
                return row.get("hp_obs_n") in (10, "10")
            if strat in ("llm_summary", "on_demand", "hybrid"):
                if row.get("hp_sum_n") not in (21, "21"):
                    return False
                if row.get("hp_sum_keep_m") not in (10, "10"):
                    return False
            return True

        df = df[df.apply(_hparams_ok, axis=1)].copy()

    # Some historical runs differ only in run naming (e.g., "n=21" vs "n-21"),
    # which can lead to duplicates in the paper generator. Collapse these by a
    # "paper config key" and keep the most recent run by created_at.
    group_cols = [
        "strategy",
        "trigger",
        "summarizer",
        "hp_obs_n",
        "hp_sum_n",
        "hp_sum_keep_m",
        "hp_limit_aware",
        "hp_limit_fraction",
        "hp_limit_min_tokens",
    ]
    for c in group_cols:
        if c not in df.columns:
            df[c] = np.nan
    if "created_at" in df.columns:
        df["_created_at_ts"] = pd.to_datetime(df["created_at"], errors="coerce")
    else:
        df["_created_at_ts"] = pd.NaT
    df = (
        df.sort_values("_created_at_ts", ascending=True)
        .groupby(group_cols, dropna=False, as_index=False)
        .tail(1)
        .drop(columns=["_created_at_ts"])
    )

    valid_trajectories_dir = trajectories_dir
    if valid_trajectories_dir is not None:
        if not valid_trajectories_dir.exists():
            _warn(f"--trajectories-dir does not exist: {valid_trajectories_dir}; skipping summary-call augmentation")
            valid_trajectories_dir = None
        elif not valid_trajectories_dir.is_dir():
            _warn(f"--trajectories-dir is not a directory: {valid_trajectories_dir}; skipping summary-call augmentation")
            valid_trajectories_dir = None

    rows = []
    for _, r in df.iterrows():
        n = int(r["n_instances"])
        k = int(r["n_resolved"])
        lo, hi = wilson_ci(k, n)
        rate = k / n if n > 0 else 0.0

        avg_summary_calls = np.nan
        summary_trigger_rate = np.nan

        if valid_trajectories_dir is not None and r["compaction"] == "summary":
            run_dir, warning = _resolve_run_dir(valid_trajectories_dir, r["run_name"])
            if warning:
                _warn(warning)
            if run_dir is not None:
                avg_summary_calls, summary_trigger_rate = _summary_call_stats_from_traj(run_dir)

        rows.append({
            "run_name": r["run_name"],
            "strategy": r["strategy"],
            "summarizer": r.get("summarizer", "same"),
            "trigger": r["trigger"],
            "compaction": r["compaction"],
            "n": n,
            "k": k,
            "solve_rate": rate,
            "ci_lo": lo,
            "ci_hi": hi,
            "ci_half": (hi - lo) / 2,
            "avg_cost": r.get("avg_cost", np.nan),
            "avg_turns": r.get("avg_turns", np.nan),
            "eval_complete": r.get("eval_complete", False),
            "hp_limit_min_tokens": _intlike(r.get("hp_limit_min_tokens", 0)) or 0,
            "avg_summary_calls": avg_summary_calls,
            "summary_trigger_rate": summary_trigger_rate,
        })

    return pd.DataFrame(rows)


def format_terminal_table(results: pd.DataFrame) -> str:
    if results.empty:
        return "No results available."

    has_summary_calls = (
        "avg_summary_calls" in results.columns
        and results.loc[results["compaction"] == "summary", "avg_summary_calls"].notna().any()
    )

    lines = []
    header = f"{'Config':<35} {'Trigger':<12} {'Rate':>8} {'CI':>12} {'n':>5} {'Cost':>7}"
    if has_summary_calls:
        header += f" {'SumCalls':>8}"
    lines.append(header)
    lines.append("-" * (85 + (9 if has_summary_calls else 0)))

    for _, r in results.sort_values("solve_rate", ascending=False).iterrows():
        label = r["strategy"]
        if r["compaction"] == "summary":
            label += f" ({r['summarizer']})"

        rate_str = f"{r['solve_rate']:.1%}"
        ci_str = f"[{r['ci_lo']:.1%}, {r['ci_hi']:.1%}]"
        cost_str = f"${r['avg_cost']:.2f}" if pd.notna(r["avg_cost"]) else "N/A"
        eval_mark = "" if r["eval_complete"] else " *"

        line = (
            f"{label:<35} {r['trigger']:<12} {rate_str:>8} {ci_str:>12} "
            f"{r['n']:>5} {cost_str:>7}{eval_mark}"
        )
        if has_summary_calls:
            val = r.get("avg_summary_calls", np.nan)
            if r["compaction"] == "summary" and pd.notna(val):
                line += f" {float(val):>8.1f}"
            else:
                line += f" {'--':>8}"
        lines.append(line)

    lines.append("")
    lines.append("* = not yet evaluated (solve rate from WandB submission data)")
    return "\n".join(lines)


def format_cross_model_terminal_table(cm_df: pd.DataFrame) -> str:
    """Format cross-model comparison as a compact terminal table."""
    if cm_df.empty:
        return "No cross-model results available."

    lines = []
    lines.append("Cross-Model Comparison (periodic strategies, verified-mini, n=50)")
    header = f"{'Model':<18} {'Context':>7}   {'Raw':>12}  {'Masking':>12}  {'Summary (self)':>14}"
    lines.append(header)
    lines.append("-" * 72)

    for _, row in cm_df.iterrows():
        display = MODEL_DISPLAY_NAMES.get(row["model"], row["model"])
        context = _format_context_window_k(row.get("context_k"))

        cells = []
        rates = {}
        for col in ("raw", "masking", "summary_self"):
            rate = row.get(f"{col}_rate")
            k = row.get(f"{col}_k")
            rates[col] = rate
            if rate is None or (isinstance(rate, float) and not np.isfinite(rate)):
                cells.append("---")
            else:
                cells.append(f"{rate * 100:.1f}% ({int(k)})")

        valid = {c: r for c, r in rates.items() if r is not None and isinstance(r, float) and np.isfinite(r)}
        best_col = max(valid, key=valid.get) if valid else None
        formatted = []
        for col, cell in zip(("raw", "masking", "summary_self"), cells):
            if col == best_col and cell != "---":
                formatted.append(cell + "*")
            else:
                formatted.append(cell)

        lines.append(
            f"{display:<18} {context:>7}   {formatted[0]:>12}  {formatted[1]:>12}  {formatted[2]:>14}"
        )

    lines.append("")
    lines.append("* = best strategy for that model")
    return "\n".join(lines)


def _build_cm_from_hardcoded() -> pd.DataFrame:
    """Build cross-model DataFrame from CROSS_MODEL_PERIODIC hardcoded data."""
    rows = []
    for model in CROSS_MODEL_ORDER:
        fb = CROSS_MODEL_PERIODIC.get(model, {})
        row = {"model": model, "context_k": fb.get("context_k", 0)}
        for col in ("raw", "masking", "summary_self"):
            cell = fb.get(col)
            if cell is not None:
                k, n = cell["k"], cell["n"]
                rate = k / n if n else 0.0
                lo, hi = wilson_ci(k, n)
                row[f"{col}_k"] = k
                row[f"{col}_n"] = n
                row[f"{col}_rate"] = rate
                row[f"{col}_ci"] = (hi - lo) / 2
            else:
                row[f"{col}_k"] = None
                row[f"{col}_n"] = None
                row[f"{col}_rate"] = None
                row[f"{col}_ci"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def generate_latex_table(results: pd.DataFrame) -> str:
    if results.empty:
        return "% No results available"

    def _pct(x: float, *, signed: bool = False) -> str:
        """Format proportion as LaTeX percent."""
        val = x * 100.0
        if signed:
            return f"{val:+.1f}\\%"
        return f"{val:.1f}\\%"

    def _latex_escape(text: str) -> str:
        return text.replace("_", r"\_")

    # 5-tuple: (strategy, trigger, compaction, summarizer, min_tokens_filter)
    # min_tokens_filter=None means "don't filter by threshold"
    order = [
        ("raw", "baseline", "none", "same", None),
        ("observation_masking", "periodic", "masking", "same", None),
        ("observation_masking", "on_demand", "masking", "same", None),
        ("llm_summary", "periodic", "summary", "same", None),
        ("on_demand", "on_demand", "summary", "same", 0),       # R4: L=170k (min_tokens=0 → fraction-driven)
        ("on_demand", "on_demand", "summary", "same", 40000),   # B1: L=40k
        ("llm_summary", "periodic", "summary", "minimax-m2.1", None),
        ("on_demand", "on_demand", "summary", "minimax-m2.1", None),
    ]

    labels = {
        ("raw", "baseline", "same", None): "Raw (no compaction)",
        ("observation_masking", "periodic", "same", None): "Periodic masking",
        ("observation_masking", "on_demand", "same", None): "On-demand masking",
        ("llm_summary", "periodic", "same", None): "Periodic summary (self)",
        ("llm_summary", "periodic", "minimax-m2.1", None): "Periodic summary (MiniMax)",
        ("on_demand", "on_demand", "same", 0): "On-demand summary (self, L=170k)",
        ("on_demand", "on_demand", "same", 40000): "On-demand summary (self, L=40k)",
        ("on_demand", "on_demand", "minimax-m2.1", None): "On-demand summary (MiniMax)",
    }

    raw_rows = results[results["strategy"] == "raw"]
    raw_rate = _pick_preferred_row(raw_rows)["solve_rate"] if len(raw_rows) > 0 else 0.64
    use_phase3_fallback = (
        "run_name" in results.columns
        and len(results) > 0
        and results["run_name"].astype(str).str.startswith("glm-4.7__").all()
    )
    phase3_on_demand_fallback = {
        ("observation_masking", "on_demand", "same", None): (29, 50),
        ("on_demand", "on_demand", "same", 0): (31, 50),
        ("on_demand", "on_demand", "same", 40000): (23, 50),
        ("on_demand", "on_demand", "minimax-m2.1", None): (28, 50),
    }

    has_summary_calls = (
        "avg_summary_calls" in results.columns
        and results.loc[results["compaction"] == "summary", "avg_summary_calls"].notna().any()
    )

    if has_summary_calls:
        col_spec = "lccccc"
        header_row = (
            r"\textbf{Configuration} $\downarrow$ & \textbf{Trigger} & \textbf{Solved} "
            r"& \textbf{Rate} (\%) & $\Delta$ vs \textbf{Raw} & \textbf{Summary Calls} \\"
        )
    else:
        col_spec = "lcccc"
        header_row = (
            r"\textbf{Configuration} $\downarrow$ & \textbf{Trigger} & \textbf{Solved} "
            r"& \textbf{Rate} (\%) & $\Delta$ vs \textbf{Raw} \\"
        )

    caption = (
        r"Solve rates on SWE-bench Verified-Mini (50 instances; \texttt{verified-mini}) with GLM-4.7. "
        r"On-demand compaction triggers at a token threshold $L$; we test $L$=170k (85\% of context) and $L$=40k."
    )
    if has_summary_calls:
        caption += (
            r" Summary Calls are computed from local trajectory \texttt{summaries} and apply only to summary-based methods."
        )

    has_summary_calls = (
        "avg_summary_calls" in results.columns
        and results.loc[results["compaction"] == "summary", "avg_summary_calls"].notna().any()
    )

    if has_summary_calls:
        col_spec = "lccccc"
        header_row = (
            r"\textbf{Configuration} $\downarrow$ & \textbf{Trigger} & \textbf{Solved} "
            r"& \textbf{Rate} (\%) & $\Delta$ vs \textbf{Raw} & \textbf{Summary Calls} \\"
        )
    else:
        col_spec = "lcccc"
        header_row = (
            r"\textbf{Configuration} $\downarrow$ & \textbf{Trigger} & \textbf{Solved} "
            r"& \textbf{Rate} (\%) & $\Delta$ vs \textbf{Raw} \\"
        )

    caption = (
        r"Solve rates on SWE-bench Verified-Mini (50 instances; \texttt{verified-mini}) with GLM-4.7. "
        r"On-demand compaction triggers at a token threshold $L$; we test $L$=170k (85\% of context) and $L$=40k."
    )
    if has_summary_calls:
        caption += (
            r" Summary Calls are computed from local trajectory \texttt{summaries} and apply only to summary-based methods."
        )

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:results}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        header_row,
        r"\midrule",
    ]

    summary_calls_suffix = " & --" if has_summary_calls else ""

    for strat, trigger, compaction, summarizer, min_tokens_filter in order:
        key = (strat, trigger, summarizer, min_tokens_filter)
        label = labels.get(key, f"{strat} ({trigger})")
        trigger_cell = _latex_escape(trigger)

        mask = (results["strategy"] == strat) & (results["trigger"] == trigger)
        if summarizer != "same":
            mask &= results["summarizer"] == summarizer
        else:
            mask &= results["summarizer"].isin(["same", "glm-4.7", ""])
        if min_tokens_filter is not None and "hp_limit_min_tokens" in results.columns:
            min_tokens = results["hp_limit_min_tokens"].apply(_intlike)
            mask &= min_tokens == min_tokens_filter

        matching = results[mask]
        if matching.empty:
            if use_phase3_fallback and key in phase3_on_demand_fallback:
                k, n = phase3_on_demand_fallback[key]
                lo, hi = wilson_ci(k, n)
                rate = k / n if n else 0.0
                ci = (hi - lo) / 2
                delta = rate - raw_rate
                delta_str = _pct(delta, signed=True)
                n_k = f"{k}/{n}"
                fallback_summary_calls = " & --" if has_summary_calls else ""
                lines.append(
                    f"  {label} & {trigger_cell} & {n_k} & {_pct(rate)} $\\pm$ {_pct(ci)} & {delta_str}{fallback_summary_calls} \\\\"
                )
                if strat == "observation_masking" and trigger == "on_demand":
                    lines.append(r"  \midrule")
                continue

            # fall back to Phase-3 hardcoded numbers
            if trigger == "periodic":
                phase3_key = None
                if strat == "raw":
                    phase3_key = "raw"
                elif strat == "observation_masking":
                    phase3_key = "masking"
                elif strat == "llm_summary" and summarizer == "same":
                    phase3_key = "summary_self"
                elif strat == "llm_summary" and summarizer == "minimax-m2.1":
                    phase3_key = "summary_minimax"
                if phase3_key and phase3_key in PHASE3_PERIODIC:
                    data = PHASE3_PERIODIC[phase3_key]
                    k, n = int(data["k"]), int(data["n"])
                    lo, hi = wilson_ci(k, n)
                    rate = k / n if n else 0.0
                    ci = (hi - lo) / 2
                    delta = rate - raw_rate
                    delta_str = _pct(delta, signed=True)
                    n_k = f"{k}/{n}"
                    if strat == "raw":
                        lines.append(
                            f"  {label} & -- & {n_k} & {_pct(rate)} $\\pm$ {_pct(ci)} & --{summary_calls_suffix} \\\\"
                        )
                        lines.append(r"  \midrule")
                    else:
                        lines.append(
                            f"  {label} & {trigger_cell} & {n_k} & {_pct(rate)} $\\pm$ {_pct(ci)} & {delta_str}{summary_calls_suffix} \\\\"
                        )
                        if strat == "observation_masking" and trigger == "on_demand":
                            lines.append(r"  \midrule")
                    continue
            pending_suffix = " & --" if has_summary_calls else ""
            lines.append(f"  {label} & {trigger_cell} & \\textit{{pending}} & -- & --{pending_suffix} \\\\")
            continue

        r = _pick_preferred_row(matching, prefer_min_tokens=min_tokens_filter)
        rate = r["solve_rate"]
        ci = r["ci_half"]
        delta = rate - raw_rate
        delta_str = _pct(delta, signed=True)
        n_k = f"{int(r['k'])}/{int(r['n'])}"

        summary_calls_cell = ""
        if has_summary_calls:
            val = r.get("avg_summary_calls", np.nan)
            if r["compaction"] == "summary" and pd.notna(val):
                summary_calls_cell = f" & {float(val):.1f}"
            else:
                summary_calls_cell = " & --"

        if strat == "raw":
            lines.append(
                f"  {label} & -- & {n_k} & {_pct(rate)} $\\pm$ {_pct(ci)} & --{summary_calls_cell} \\\\"
            )
            lines.append(r"  \midrule")
        else:
            lines.append(
                f"  {label} & {trigger_cell} & {n_k} & {_pct(rate)} $\\pm$ {_pct(ci)} & {delta_str}{summary_calls_cell} \\\\"
            )
            if strat == "observation_masking" and trigger == "on_demand":
                lines.append(r"  \midrule")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


MODEL_DISPLAY_NAMES = {
    "kimi-2.5": "Kimi-2.5",
    "glm-5": "GLM-5",
    "deepseek-chat": "DeepSeek V3.2",
    "glm-4.7": "GLM-4.7",
    "minimax-m2.1": "MiniMax M2.1",
}

# display order: descending raw solve rate, then alphabetical
CROSS_MODEL_ORDER = ["kimi-2.5", "glm-5", "deepseek-chat", "glm-4.7", "minimax-m2.1"]


def build_cross_model_table(
    project: str,
    entity: str | None = None,
) -> pd.DataFrame:
    """Build a cross-model comparison table from WandB, falling back to hardcoded data."""
    target_models = set(CROSS_MODEL_PERIODIC.keys())
    strategy_map = {
        "raw": "raw",
        "observation_masking": "masking",
        "llm_summary": "summary_self",
    }

    results: dict[tuple[str, str], dict] = {}

    try:
        df = fetch_runs(project, entity)
        if "instances_subset" in df.columns:
            df["instances_subset_norm"] = (
                df["instances_subset"].astype(str).str.lower().str.replace("_", "-", regex=False)
            )
        else:
            df["instances_subset_norm"] = "unknown"

        mask = (
            df["model"].isin(target_models)
            & df["instances_subset_norm"].isin(["verified-mini", "verifiedmini", "mini"])
            & df["n_instances"].between(40, 60)
        )
        df = df[mask].copy()
        if "eval_complete" in df.columns:
            df = df[df["eval_complete"]].copy()
        df = dedupe_latest_runs(df)

        for _, r in df.iterrows():
            model = r["model"]
            strat = r["strategy"]
            col_key = strategy_map.get(strat)
            if col_key is None:
                continue
            # for summary, only keep self-summarization
            if strat == "llm_summary":
                summ = str(r.get("summarizer", "same")).strip().lower()
                if summ not in ("same", model, "", "none", "nan", "reuse-agent-model"):
                    continue
            # skip limit-aware / on-demand variants
            trigger = classify_trigger(r)
            if trigger not in ("baseline", "periodic"):
                continue

            k = int(r["n_resolved"])
            n = int(r["n_instances"])
            results[(model, col_key)] = {"k": k, "n": n}
    except Exception:
        pass

    # fill gaps from hardcoded fallback
    rows = []
    for model in CROSS_MODEL_ORDER:
        fallback = CROSS_MODEL_PERIODIC.get(model, {})
        context_k = fallback.get("context_k", 0)
        row = {"model": model, "context_k": context_k}
        for col in ("raw", "masking", "summary_self"):
            cell = results.get((model, col))
            if cell is None:
                fb = fallback.get(col)
                if fb is not None:
                    cell = {"k": fb["k"], "n": fb["n"]}
            if cell is not None:
                k, n = cell["k"], cell["n"]
                rate = k / n if n else 0.0
                lo, hi = wilson_ci(k, n)
                row[f"{col}_k"] = k
                row[f"{col}_n"] = n
                row[f"{col}_rate"] = rate
                row[f"{col}_ci"] = (hi - lo) / 2
            else:
                row[f"{col}_k"] = None
                row[f"{col}_n"] = None
                row[f"{col}_rate"] = None
                row[f"{col}_ci"] = None
        rows.append(row)
    return pd.DataFrame(rows)


def generate_cross_model_latex(cm_df: pd.DataFrame) -> str:
    """Generate a compact LaTeX table for cross-model comparison."""
    if cm_df.empty:
        return "% No cross-model results available"

    def _fmt_cell(rate: float | None, is_best: bool) -> str:
        if rate is None or (isinstance(rate, float) and not np.isfinite(rate)):
            return "---"
        pct = f"{rate * 100:.1f}\\%"
        if is_best:
            return f"\\textbf{{{pct}}}"
        return pct

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Cross-model solve rates on \texttt{verified-mini} (periodic strategies). "
        r"Bold indicates the best strategy for each model.}",
        r"\label{tab:cross-model}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Model & Context & Raw & Masking & Summary (self) \\",
        r"\midrule",
    ]

    for _, row in cm_df.iterrows():
        display = MODEL_DISPLAY_NAMES.get(row["model"], row["model"])
        context = _format_context_window_k(row.get("context_k"))

        rates = {}
        for col in ("raw", "masking", "summary_self"):
            rates[col] = row.get(f"{col}_rate")

        valid_rates = {c: r for c, r in rates.items() if r is not None and np.isfinite(r)}
        best_col = max(valid_rates, key=valid_rates.get) if valid_rates else None

        cells = []
        for col in ("raw", "masking", "summary_self"):
            cells.append(_fmt_cell(rates[col], col == best_col))

        lines.append(f"  {display} & {context} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def build_threshold_sweep_table(
    project: str,
    entity: str | None = None,
) -> pd.DataFrame:
    """Build a threshold sweep table from WandB, falling back to THRESHOLD_SWEEP_DATA."""
    RAW_RATE = 0.64  # 32/50

    strat_group_map = {
        "observation_masking": "masking",
        "hybrid": "hybrid",
        "llm_summary": "summary",
        "on_demand": "summary",
    }

    wandb_data: dict[tuple, dict] = {}

    try:
        df = fetch_runs(project, entity)
        if "instances_subset" in df.columns:
            df["instances_subset_norm"] = (
                df["instances_subset"].astype(str).str.lower().str.replace("_", "-", regex=False)
            )
        else:
            df["instances_subset_norm"] = "unknown"

        mask = (
            (df["model"] == "glm-4.7")
            & df["instances_subset_norm"].isin(["verified-mini", "verifiedmini", "mini"])
            & df["n_instances"].between(40, 60)
        )
        df = df[mask].copy()
        if "eval_complete" in df.columns:
            df = df[df["eval_complete"]].copy()
        df = dedupe_latest_runs(df)

        for _, r in df.iterrows():
            strat = r["strategy"]
            group = strat_group_map.get(strat)
            if group is None:
                continue

            limit_aware = _boollike(r.get("hp_limit_aware", False))
            min_tokens = _intlike(r.get("hp_limit_min_tokens", 0)) or 0

            if strat == "raw":
                continue

            if strat == "on_demand" or limit_aware:
                trigger_type = "on_demand"
                if min_tokens == 0:  # fraction-driven → effective 170k
                    L = 170000
                else:
                    L = min_tokens
            else:
                trigger_type = "periodic"
                L = None

            summ = str(r.get("summarizer", "same")).strip().lower()
            if summ in ("same", "", "none", "nan", "reuse-agent-model", "glm-4.7"):
                summarizer = "glm-4.7" if group in ("summary", "hybrid") else None
            else:
                summarizer = summ

            k = int(r["n_resolved"])
            n = int(r["n_instances"])
            n_eval = int(r.get("n_evaluated", 0))

            key = (group, summarizer, L, trigger_type)
            wandb_data[key] = {"k": k, "n": n, "preds": n_eval}
    except Exception:
        pass

    all_keys = set(THRESHOLD_SWEEP_DATA.keys()) | set(wandb_data.keys())

    rows = []
    for key in sorted(all_keys, key=lambda x: (x[0], x[1] or "", x[2] or 0, x[3])):
        group, summarizer, L, trigger_type = key

        cell = wandb_data.get(key)
        if cell is None:
            fb = THRESHOLD_SWEEP_DATA.get(key)
            if fb is None:
                continue
            cell = fb.copy()

        k, n = cell["k"], cell["n"]
        preds = cell.get("preds", n)
        trigger_pct = cell.get("trigger_pct")
        if trigger_pct is None and L is not None:
            trigger_pct = TRIGGER_RATE_BY_THRESHOLD.get(L, 0)
        elif trigger_pct is None:
            trigger_pct = 100 if trigger_type == "periodic" else 0

        rate = k / n if n > 0 else 0.0
        lo, hi = wilson_ci(k, n)
        vs_raw = rate - RAW_RATE

        rows.append({
            "strategy_group": group,
            "summarizer": summarizer,
            "L": L,
            "trigger_type": trigger_type,
            "trigger_pct": trigger_pct,
            "preds": preds,
            "k": k,
            "n": n,
            "solve_rate": rate,
            "ci_lo": lo,
            "ci_hi": hi,
            "ci_half": (hi - lo) / 2,
            "vs_raw": vs_raw,
        })

    return pd.DataFrame(rows)


def _fmt_L(L: int | float | None) -> str:
    if L is None or (isinstance(L, float) and not np.isfinite(L)):
        return "---"
    L_int = int(L)
    if L_int >= 1000:
        return f"{L_int // 1000}k"
    return str(L_int)


def _format_context_window_k(context_k: object) -> str:
    """Format context-window values that are encoded in thousands of tokens."""
    val = _floatlike(context_k)
    if val is None:
        return "---"
    if val >= 1000:
        m = val / 1000.0
        m_txt = f"{m:.1f}".rstrip("0").rstrip(".")
        return f"{m_txt}M"
    k_txt = f"{val:.1f}".rstrip("0").rstrip(".")
    return f"{k_txt}K"


def generate_threshold_sweep_latex(ts_df: pd.DataFrame) -> str:
    """Generate LaTeX table for threshold sweep (Table 8)."""
    if ts_df.empty:
        return "% No threshold sweep results available"

    def _pct(x: float, *, signed: bool = False) -> str:
        val = x * 100.0
        if signed:
            return f"{val:+.1f}\\%"
        return f"{val:.1f}\\%"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Solve rate vs.\ budget threshold $L$ (GLM-4.7, on-demand triggering, $n$=50). "
        r"Periodic rows fire every $N$ turns regardless of budget.}",
        r"\label{tab:threshold-sweep}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Strategy & Summarizer & $L$ & Trigger Rate & Preds & Solve Rate (\%) & $\Delta$ vs Raw \\",
        r"\midrule",
    ]

    group_order = ["masking", "hybrid", "summary"]
    group_labels = {"masking": "Masking", "hybrid": "Hybrid", "summary": "Summary"}

    first_group = True
    for group in group_order:
        group_rows = ts_df[ts_df["strategy_group"] == group].copy()
        if group_rows.empty:
            continue

        if not first_group:
            lines.append(r"  \midrule")
        first_group = False

        # sort: on_demand by L ascending, then periodic at the end
        group_rows = group_rows.sort_values(
            by=["trigger_type", "summarizer", "L"],
            key=lambda col: col.map(lambda v: (0 if v == "on_demand" else 1) if col.name == "trigger_type"
                                    else (v or "") if col.name == "summarizer"
                                    else (v if v is not None else 999999)),
            ascending=True,
        )

        best_rate = group_rows["solve_rate"].max()

        for _, row in group_rows.iterrows():
            strat_label = f"on-demand {group_labels[group].lower()}" if row["trigger_type"] == "on_demand" else f"periodic {group_labels[group].lower()}"
            summ = row["summarizer"]
            summ_cell = "---" if summ is None else summ.replace("_", r"\_")
            L_cell = _fmt_L(row["L"])
            trigger_cell = f"$\\sim${row['trigger_pct']}\\%" if row["trigger_pct"] < 100 else "100\\%"
            preds_cell = f"{int(row['preds'])}/{int(row['n'])}"
            rate_str = _pct(row["solve_rate"])
            delta_str = _pct(row["vs_raw"], signed=True)

            if row["solve_rate"] == best_rate:
                rate_str = f"\\textbf{{{rate_str}}}"

            lines.append(
                f"  {strat_label} & {summ_cell} & {L_cell} & {trigger_cell} & {preds_cell} & {rate_str} & {delta_str} \\\\"
            )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def format_threshold_sweep_terminal(ts_df: pd.DataFrame) -> str:
    """Format threshold sweep table for terminal display."""
    if ts_df.empty:
        return "No threshold sweep results available."

    lines = []
    header = f"{'Strategy':<25} {'Summarizer':<15} {'L':>6} {'Trig%':>6} {'Preds':>7} {'Rate':>8} {'CI':>14} {'vs Raw':>8}"
    lines.append(header)
    lines.append("-" * len(header))

    group_order = ["masking", "hybrid", "summary"]
    first_group = True

    for group in group_order:
        group_rows = ts_df[ts_df["strategy_group"] == group].copy()
        if group_rows.empty:
            continue
        if not first_group:
            lines.append("")
        first_group = False

        group_rows = group_rows.sort_values(
            by=["trigger_type", "summarizer", "L"],
            key=lambda col: col.map(lambda v: (0 if v == "on_demand" else 1) if col.name == "trigger_type"
                                    else (v or "") if col.name == "summarizer"
                                    else (v if v is not None else 999999)),
            ascending=True,
        )

        for _, row in group_rows.iterrows():
            strat_label = f"od-{group}" if row["trigger_type"] == "on_demand" else f"periodic-{group}"
            summ = row["summarizer"] or "---"
            L_str = _fmt_L(row["L"])
            trig_str = f"~{row['trigger_pct']}%"
            preds_str = f"{int(row['preds'])}/{int(row['n'])}"
            rate_str = f"{row['solve_rate']:.1%}"
            ci_str = f"[{row['ci_lo']:.1%}, {row['ci_hi']:.1%}]"
            delta_str = f"{row['vs_raw']:+.1%}"

            lines.append(
                f"{strat_label:<25} {summ:<15} {L_str:>6} {trig_str:>6} {preds_str:>7} {rate_str:>8} {ci_str:>14} {delta_str:>8}"
            )

    return "\n".join(lines)


def generate_figure(results: pd.DataFrame, output_path: str) -> None:
    """Generate Figure 1: grouped bar chart comparing periodic vs on-demand."""
    import matplotlib.pyplot as plt
    import matplotlib
    import seaborn as sns

    sns.set_theme(style="whitegrid", font="Times New Roman")
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.figsize": (5.5, 3.5),
        "figure.dpi": 300,
    })

    categories = []
    use_phase3_fallback = (
        "run_name" in results.columns
        and len(results) > 0
        and results["run_name"].astype(str).str.startswith("glm-4.7__").all()
    )

    def _rate_ci_from_counts(k: int, n: int) -> tuple[float, float]:
        lo, hi = wilson_ci(k, n)
        return (k / n if n else 0.0), (hi - lo) / 2

    def _p3_rate_ci(key: str, default_rate: float, default_ci: float) -> tuple[float, float]:
        data = PHASE3_PERIODIC.get(key)
        if not data:
            return default_rate, default_ci
        k, n = int(data["k"]), int(data["n"])
        lo, hi = wilson_ci(k, n)
        return (k / n if n else default_rate), (hi - lo) / 2

    raw_rows = results[results["strategy"] == "raw"]
    raw_rate, raw_ci = (
        _pick_preferred_row(raw_rows)["solve_rate"],
        _pick_preferred_row(raw_rows)["ci_half"],
    ) if len(raw_rows) > 0 else _p3_rate_ci("raw", 0.64, 0.068)

    # Masking.
    periodic_mask = results[
        (results["strategy"] == "observation_masking") & (results["trigger"] == "periodic")
    ]
    od_mask = results[
        (results["strategy"] == "observation_masking") & (results["trigger"] == "on_demand")
    ]
    # when multiple thresholds exist, prefer the fraction-driven one (min_tokens=0 → L=170k)
    if "hp_limit_min_tokens" in od_mask.columns and len(od_mask) > 1:
        min_tokens = od_mask["hp_limit_min_tokens"].apply(_intlike)
        primary = od_mask[min_tokens == 0]
        if len(primary) > 0:
            od_mask = primary
    if len(periodic_mask) > 0:
        periodic_mask_row = _pick_preferred_row(periodic_mask)
        pm_rate, pm_ci = periodic_mask_row["solve_rate"], periodic_mask_row["ci_half"]
    else:
        pm_rate, pm_ci = _p3_rate_ci("masking", 0.62, 0.070)
    if len(od_mask) > 0:
        od_mask_row = _pick_preferred_row(od_mask, prefer_min_tokens=0)
        odm_rate = od_mask_row["solve_rate"]
        odm_ci = od_mask_row["ci_half"]
    elif use_phase3_fallback:
        odm_rate, odm_ci = _rate_ci_from_counts(29, 50)
    else:
        odm_rate, odm_ci = None, None

    if (odm_rate is not None) or use_phase3_fallback:
        categories.append(("Masking", pm_rate, pm_ci, odm_rate, odm_ci))

    # summary (self)
    periodic_sum_self = results[
        (results["strategy"] == "llm_summary") & (results["trigger"] == "periodic")
        & (results["summarizer"].isin(["same", "glm-4.7", ""]))
    ]
    od_sum_self = results[
        (results["strategy"] == "on_demand") & (results["trigger"] == "on_demand")
        & (results["summarizer"].isin(["same", "glm-4.7", ""]))
    ]
    # when multiple thresholds exist, prefer the fraction-driven one (min_tokens=0 → L=170k)
    if "hp_limit_min_tokens" in od_sum_self.columns and len(od_sum_self) > 1:
        min_tokens = od_sum_self["hp_limit_min_tokens"].apply(_intlike)
        primary = od_sum_self[min_tokens == 0]
        if len(primary) > 0:
            od_sum_self = primary

    if len(periodic_sum_self) > 0:
        periodic_sum_self_row = _pick_preferred_row(periodic_sum_self)
        ps_rate, ps_ci = periodic_sum_self_row["solve_rate"], periodic_sum_self_row["ci_half"]
    else:
        ps_rate, ps_ci = _p3_rate_ci("summary_self", 0.56, 0.133)
    if len(od_sum_self) > 0:
        od_sum_self_row = _pick_preferred_row(od_sum_self, prefer_min_tokens=0)
        ods_rate = od_sum_self_row["solve_rate"]
        ods_ci = od_sum_self_row["ci_half"]
    elif use_phase3_fallback:
        ods_rate, ods_ci = _rate_ci_from_counts(31, 50)
    else:
        ods_rate = None
        ods_ci = None

    categories.append(("Summary\n(self)", ps_rate, ps_ci, ods_rate, ods_ci))

    # summary (minimax)
    periodic_sum_mm = results[
        (results["strategy"] == "llm_summary") & (results["trigger"] == "periodic")
        & (results["summarizer"] == "minimax-m2.1")
    ]
    od_sum_mm = results[
        (results["strategy"] == "on_demand") & (results["trigger"] == "on_demand")
        & (results["summarizer"] == "minimax-m2.1")
    ]

    if len(periodic_sum_mm) > 0:
        periodic_sum_mm_row = _pick_preferred_row(periodic_sum_mm)
        pmm_rate, pmm_ci = periodic_sum_mm_row["solve_rate"], periodic_sum_mm_row["ci_half"]
    else:
        pmm_rate, pmm_ci = _p3_rate_ci("summary_minimax", 0.54, 0.133)
    if len(od_sum_mm) > 0:
        od_sum_mm_row = _pick_preferred_row(od_sum_mm)
        odmm_rate = od_sum_mm_row["solve_rate"]
        odmm_ci = od_sum_mm_row["ci_half"]
    elif use_phase3_fallback:
        odmm_rate, odmm_ci = _rate_ci_from_counts(28, 50)
    else:
        odmm_rate = None
        odmm_ci = None

    categories.append(("Summary\n(MiniMax)", pmm_rate, pmm_ci, odmm_rate, odmm_ci))

    # NOTE: We intentionally omit the "kimi" summarizer from the main paper
    # figure. It is an optional sensitivity run and often incomplete; including
    # placeholders is distracting for a 4-page workshop submission.

    # plot
    pal = sns.color_palette("muted")
    c_periodic, c_ondemand = pal[0], pal[1]

    fig, ax = plt.subplots(figsize=(6.4, 3.35))
    x = np.arange(len(categories)) * 1.18
    width = 0.28

    periodic_rates = [c[1] for c in categories]
    periodic_cis = [c[2] for c in categories]
    od_rates = [c[3] if c[3] is not None else 0 for c in categories]
    od_cis = [c[4] if c[4] is not None else 0 for c in categories]
    od_available = [c[3] is not None for c in categories]

    # raw baseline line (thin, no CI band to avoid clutter)
    ax.axhline(y=raw_rate, color="#555", linestyle="--", linewidth=0.7,
               label=f"Raw baseline ({raw_rate:.0%})", zorder=1)

    ax.bar(x - width / 2, periodic_rates, width,
           color=c_periodic, alpha=0.85, label="Periodic",
           edgecolor="white", linewidth=0.5, zorder=2)

    for i, avail in enumerate(od_available):
        rate = od_rates[i]
        ci = od_cis[i]
        if avail:
            ax.bar(x[i] + width / 2, rate, width,
                   color=c_ondemand, alpha=0.85, edgecolor="white", linewidth=0.5, zorder=2)
        else:
            ax.bar(x[i] + width / 2, 0.01, width, color="none",
                   edgecolor="#bbb", linewidth=0.8, linestyle="--", hatch="///", zorder=2)
            ax.text(x[i] + width / 2, 0.04, "?", ha="center", va="bottom",
                    fontsize=7, color="#999", weight="bold")

    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], color="#555", linestyle="--", linewidth=0.7, label=f"Raw baseline ({raw_rate:.0%})"),
        Patch(facecolor=c_periodic, alpha=0.85, edgecolor="white", label="Periodic"),
        Patch(facecolor=c_ondemand, alpha=0.85, edgecolor="white", label="On-demand"),
    ]

    labels = [c[0] for c in categories]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Solve Rate")
    ax.set_ylim(0, 0.80)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=3,
        framealpha=0.9,
        borderaxespad=0.2,
    )
    ax.set_title("Periodic vs On-Demand Context Compaction (GLM-4.7, n=50)", pad=22)
    ax.margins(x=0.06)

    plt.tight_layout(rect=[0, 0.0, 1.0, 0.98])
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.02, dpi=300)
    print(f"Figure saved to {output_path}")
    plt.close()


def generate_cross_model_figure(cm_df: pd.DataFrame, output_path: str) -> None:
    """Generate a cross-model grouped bar chart for periodic strategies."""
    import matplotlib.pyplot as plt
    import matplotlib
    import seaborn as sns

    if cm_df.empty:
        print("No cross-model data available; skipping cross-model figure.")
        return

    sns.set_theme(style="whitegrid", font="Times New Roman")
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.figsize": (6.4, 3.55),
        "figure.dpi": 300,
    })

    if "model" in cm_df.columns:
        cm_df = cm_df.copy()
        cm_df["model"] = pd.Categorical(cm_df["model"], categories=CROSS_MODEL_ORDER, ordered=True)
        cm_df = cm_df.sort_values("model")

    labels = []
    for _, row in cm_df.iterrows():
        model = MODEL_DISPLAY_NAMES.get(row["model"], row["model"])
        ctx = _format_context_window_k(row.get("context_k")).lower()
        labels.append(f"{model}\n({ctx})")

    x = np.arange(len(cm_df)) * 1.28
    width = 0.24

    fig, ax = plt.subplots(figsize=(6.8, 3.45))

    muted = sns.color_palette("muted")
    strategy_specs = [
        ("raw", "Raw", muted[2]),
        ("masking", "Masking", muted[0]),
        ("summary_self", "Summary (self)", muted[1]),
    ]
    offsets = [-width, 0.0, width]

    for offset, (prefix, legend_label, color) in zip(offsets, strategy_specs):
        rates = []
        cis = []
        available = []
        for _, row in cm_df.iterrows():
            rate = row.get(f"{prefix}_rate")
            ci = row.get(f"{prefix}_ci")
            if rate is None or (isinstance(rate, float) and not np.isfinite(rate)):
                rates.append(0.0)
                cis.append(0.0)
                available.append(False)
            else:
                rates.append(float(rate))
                cis.append(float(ci) if ci is not None else 0.0)
                available.append(True)

        for i, present in enumerate(available):
            xpos = x[i] + offset
            if present:
                ax.bar(
                    xpos,
                    rates[i],
                    width,
                    color=color,
                    alpha=0.88,
                    edgecolor="white",
                    linewidth=0.5,
                    zorder=3,
                )
            else:
                ax.bar(
                    xpos,
                    0.008,
                    width,
                    color="none",
                    edgecolor="#9e9e9e",
                    linewidth=0.75,
                    linestyle="--",
                    hatch="///",
                    zorder=2,
                )
                ax.text(
                    xpos,
                    0.022,
                    "N/A",
                    ha="center",
                    va="bottom",
                    fontsize=6.6,
                    color="#777",
                )

    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=color, alpha=0.88, edgecolor="white", label=legend_label)
        for _, legend_label, color in strategy_specs
    ]

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Solve Rate")
    ax.set_ylim(0, 0.80)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.set_title("Cross-Model Periodic Compaction (SWE-bench Verified mini, n=50)", pad=20)
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        framealpha=0.92,
        borderaxespad=0.25,
    )
    ax.grid(axis="y", linestyle="-", alpha=0.20, zorder=0)
    ax.grid(axis="x", visible=False)
    ax.margins(x=0.04)

    plt.tight_layout(rect=[0, 0.0, 1.0, 0.98])
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.02, dpi=300)
    print(f"Cross-model figure saved to {output_path}")
    plt.close()


def fetch_instance_tokens(project: str, entity: str | None = None) -> pd.DataFrame:
    """Fetch per-instance token data from WandB run_table artifacts.

    Returns a DataFrame with columns: instance_id, strategy, summarizer, trigger,
    total_input_tokens, total_tokens, n_turns, run_name.
    """
    import wandb

    df = fetch_runs(project, entity)

    # same filter as build_results_table: GLM-4.7, verified-mini, 50-instance evaluated runs
    if "instances_subset" in df.columns:
        df["instances_subset_norm"] = (
            df["instances_subset"].astype(str).str.lower().str.replace("_", "-", regex=False)
        )
    else:
        df["instances_subset_norm"] = "unknown"
    mask = (
        (df["model"] == "glm-4.7")
        & (df["instances_subset_norm"].isin(["verified-mini", "verifiedmini", "mini"]))
        & (df["n_instances"].between(40, 60))
    )
    df = df[mask].copy()
    if "eval_complete" in df.columns:
        df = df[df["eval_complete"]].copy()
    df = dedupe_latest_runs(df)

    if df.empty:
        print("No matching runs for token histogram.")
        return pd.DataFrame()

    # classify trigger/strategy per run
    df["trigger"] = df.apply(classify_trigger, axis=1)

    api = wandb.Api()
    frames = []
    for _, run_row in df.iterrows():
        run_id = run_row["run_id"]
        try:
            run = api.run(f"{entity}/{project}/{run_id}" if entity else f"{project}/{run_id}")
            art = next(
                (a for a in run.logged_artifacts() if "instances" in a.name),
                None,
            )
            if art is None:
                continue
            table = art.get("instances")
            if table is None:
                continue
            idf = pd.DataFrame(table.data, columns=table.columns)
        except Exception as exc:
            print(f"  skip {run_row['run_name']}: {exc}")
            continue

        # compute token totals
        raw_in = pd.to_numeric(idf.get("raw_input_tokens", 0), errors="coerce").fillna(0)
        cached_in = pd.to_numeric(idf.get("cached_input_tokens", 0), errors="coerce").fillna(0)
        out_tok = pd.to_numeric(idf.get("output_tokens", 0), errors="coerce").fillna(0)

        idf["total_input_tokens"] = raw_in + cached_in
        idf["total_tokens"] = idf["total_input_tokens"] + out_tok

        # estimate peak context window per instance: for SWE-agent the context
        # grows roughly linearly from a base (~5k system prompt) to the peak
        # at the last turn.  avg ≈ (base + peak)/2  =>  peak ≈ 2*avg - base.
        n_turns_col = pd.to_numeric(idf.get("n_turns", 1), errors="coerce").fillna(1).clip(lower=1)
        avg_input_per_turn = idf["total_input_tokens"] / n_turns_col
        idf["peak_context_estimate"] = (2 * avg_input_per_turn - 5000).clip(lower=0)

        idf["strategy"] = run_row["strategy"]
        idf["summarizer"] = run_row.get("summarizer", "same")
        idf["trigger"] = run_row["trigger"]
        idf["run_name"] = run_row["run_name"]

        keep = ["instance_id", "strategy", "summarizer", "trigger",
                "total_input_tokens", "total_tokens", "n_turns",
                "peak_context_estimate", "run_name"]
        available = [c for c in keep if c in idf.columns]
        frames.append(idf[available])

    if not frames:
        print("No instance-level token data found in artifacts.")
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def generate_token_histogram(token_df: pd.DataFrame, output_path: str) -> None:
    """Generate a histogram of per-instance input tokens with budget threshold lines."""
    import matplotlib.pyplot as plt
    import matplotlib
    import seaborn as sns

    sns.set_theme(style="whitegrid", font="Times New Roman")
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.figsize": (5.5, 3.5),
        "figure.dpi": 300,
    })

    token_df = token_df.copy()

    # use peak context estimate (estimated last-turn input) for budget analysis
    if "peak_context_estimate" not in token_df.columns:
        token_df["peak_context_estimate"] = token_df["total_input_tokens"]

    token_df["peak_k"] = token_df["peak_context_estimate"] / 1000.0

    # build strategy labels for hue
    def _label(row):
        s = row["strategy"]
        t = row["trigger"]
        if s == "raw":
            return "Raw"
        if s == "observation_masking":
            return f"Masking ({t})"
        if s in ("llm_summary", "on_demand"):
            return f"Summary ({t})"
        return s
    token_df["label"] = token_df.apply(_label, axis=1)

    # order: raw LAST so it draws on top (most prominent, not occluded)
    label_order = [lbl for lbl in sorted(token_df["label"].unique()) if lbl != "Raw"]
    label_order.append("Raw")

    pal = sns.color_palette("muted", n_colors=len(label_order))

    fig, ax = plt.subplots()
    sns.histplot(
        data=token_df, x="peak_k", hue="label", hue_order=label_order,
        bins=25, alpha=0.55, kde=False, palette=pal, ax=ax, element="bars",
        multiple="layer",
    )

    # threshold lines at candidate compaction budgets (context window sizes)
    raw_peak = token_df.loc[token_df["label"] == "Raw", "peak_context_estimate"]
    n_raw = len(raw_peak)
    thresholds = [
        (40, "40k"),
        (60, "60k"),
        (80, "80k"),
        (100, "100k"),
        (170, "170k (85%)"),
    ]
    colors_thresh = ["#9467bd", "#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]

    for (val_k, lbl), color in zip(thresholds, colors_thresh):
        ax.axvline(x=val_k, color=color, linestyle="--", linewidth=0.9, alpha=0.8)
        pct_above = (raw_peak > val_k * 1000).sum() / n_raw * 100 if n_raw else 0
        ax.text(
            val_k + 1, ax.get_ylim()[1] * 0.92, f"{lbl}\n{pct_above:.0f}%",
            fontsize=6.5, color=color, va="top", ha="left",
        )

    legend = ax.get_legend()
    if legend:
        legend.set_title("Strategy")

    ax.set_xlabel("Estimated Peak Context Window per Instance (thousands)")
    ax.set_ylabel("Count")
    ax.set_title("Peak Context Distribution (GLM-4.7, n=50)")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Token histogram saved to {output_path}")
    plt.close()


def generate_turns_histogram(token_df: pd.DataFrame, output_path: str) -> None:
    """Generate a histogram of per-instance turn counts with strategy threshold lines."""
    import matplotlib.pyplot as plt
    import matplotlib
    import seaborn as sns

    sns.set_theme(style="whitegrid", font="Times New Roman")
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.figsize": (5.5, 3.5),
        "figure.dpi": 300,
    })

    df = token_df.copy()
    turns_col = pd.to_numeric(df.get("n_turns", 0), errors="coerce").fillna(0)
    df["n_turns"] = turns_col

    # reuse the same labeling logic
    def _label(row):
        s = row["strategy"]
        t = row["trigger"]
        if s == "raw":
            return "Raw"
        if s == "observation_masking":
            return f"Masking ({t})"
        if s in ("llm_summary", "on_demand"):
            return f"Summary ({t})"
        return s
    df["label"] = df.apply(_label, axis=1)

    # raw last so it draws on top
    label_order = [lbl for lbl in sorted(df["label"].unique()) if lbl != "Raw"]
    label_order.append("Raw")

    pal = sns.color_palette("muted", n_colors=len(label_order))

    fig, ax = plt.subplots()
    sns.histplot(
        data=df, x="n_turns", hue="label", hue_order=label_order,
        bins=25, alpha=0.55, kde=False, palette=pal, ax=ax, element="bars",
        multiple="layer",
    )

    # threshold lines at strategy-relevant turn counts
    # stagger y-positions to avoid overlapping labels
    raw_turns = df.loc[df["label"] == "Raw", "n_turns"]
    n_raw = len(raw_turns)
    thresholds = [
        (10, "M=10\n(obs)", 0.92),
        (21, "N=21\n(sum)", 0.72),
        (43, "N=43\n(hybrid)", 0.92),
    ]
    colors_thresh = ["#2ca02c", "#d62728", "#9467bd"]

    for (val, lbl, y_frac), color in zip(thresholds, colors_thresh):
        ax.axvline(x=val, color=color, linestyle="--", linewidth=0.9, alpha=0.8)
        pct_above = (raw_turns > val).sum() / n_raw * 100 if n_raw else 0
        ax.text(
            val + 1, ax.get_ylim()[1] * y_frac, f"{lbl}\n{pct_above:.0f}%",
            fontsize=6.5, color=color, va="top", ha="left",
        )

    legend = ax.get_legend()
    if legend:
        legend.set_title("Strategy")

    ax.set_xlabel("Number of Turns per Instance")
    ax.set_ylabel("Count")
    ax.set_title("Turn Count Distribution (GLM-4.7, n=50)")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Turns histogram saved to {output_path}")

    # print summary statistics
    for lbl in label_order:
        subset = df.loc[df["label"] == lbl, "n_turns"]
        if subset.empty:
            continue
        q25, median, q75 = subset.quantile([0.25, 0.5, 0.75])
        print(f"  {lbl:30s}  median={median:.0f}  IQR=[{q25:.0f}, {q75:.0f}]  "
              f"min={subset.min():.0f}  max={subset.max():.0f}")

    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate paper results from WandB data")
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "the-complexity-trap"))
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--latex", action="store_true", help="Output LaTeX table")
    parser.add_argument(
        "--latex-out",
        type=str,
        help="Write LaTeX table to a file (table only; no terminal preamble)",
    )
    parser.add_argument("--figure", type=str, help="Output figure to path (e.g., figure1.pdf)")
    parser.add_argument("--token-histogram", type=str, help="Output token usage histogram to path (e.g., token_hist.pdf)")
    parser.add_argument("--turns-histogram", type=str, help="Output turns distribution histogram to path (e.g., turns_hist.pdf)")
    parser.add_argument(
        "--trajectories-dir",
        type=str,
        help="Path to trajectories dir for local summary-call stats",
    )
    parser.add_argument(
        "--cross-model-latex-out",
        type=str,
        help="Write cross-model LaTeX table to a file",
    )
    parser.add_argument(
        "--cross-model-figure",
        type=str,
        help="Output cross-model figure to path (e.g., cross_model_figure.pdf)",
    )
    parser.add_argument("--threshold-sweep", action="store_true", help="Show threshold sweep table in terminal")
    parser.add_argument(
        "--threshold-sweep-latex-out",
        type=str,
        help="Write threshold sweep LaTeX table to a file",
    )
    parser.add_argument("--no-fetch", action="store_true", help="Use hardcoded baselines only (no WandB)")
    parser.add_argument("--quiet", action="store_true", help="Suppress terminal table output")
    args = parser.parse_args()

    if args.no_fetch:
        if not args.quiet:
            print("Using hardcoded Phase 3 baselines (no WandB fetch)")
        rows = []
        phase3_map = [
            # (strategy, summarizer, trigger, compaction, phase3_key)
            ("raw", "same", "baseline", "none", "raw"),
            ("observation_masking", "same", "periodic", "masking", "masking"),
            ("llm_summary", "same", "periodic", "summary", "summary_self"),
            ("llm_summary", "minimax-m2.1", "periodic", "summary", "summary_minimax"),
            ("llm_summary", "kimi-2.5", "periodic", "summary", "summary_kimi"),
            ("hybrid", "minimax-m2.1", "periodic", "summary", "hybrid_minimax"),
        ]
        for strat, summ, trigger, comp, p3key in phase3_map:
            data = PHASE3_PERIODIC[p3key]
            k, n = data["k"], data["n"]
            lo, hi = wilson_ci(k, n)
            rows.append({
                "run_name": f"glm-4.7__{strat}__{summ}",
                "strategy": strat,
                "summarizer": summ,
                "trigger": trigger,
                "compaction": comp,
                "n": n, "k": k,
                "solve_rate": k / n,
                "ci_lo": lo, "ci_hi": hi, "ci_half": (hi - lo) / 2,
                "avg_cost": data.get("cost", np.nan) or np.nan,
                "avg_turns": np.nan,
                "eval_complete": True,
            })
        results = pd.DataFrame(rows)
    else:
        traj_dir = Path(args.trajectories_dir) if args.trajectories_dir else None
        results = build_results_table(args.project, args.entity, trajectories_dir=traj_dir)

    if results.empty:
        print("No results. Run with --no-fetch to use hardcoded baselines.")
        return

    if not args.quiet and not args.latex_out:
        print(format_terminal_table(results))
        print()

    # build cross-model table (used by terminal display, --latex, and --cross-model-latex-out)
    if args.no_fetch:
        cm_df = _build_cm_from_hardcoded()
    else:
        cm_df = build_cross_model_table(args.project, args.entity)

    if not cm_df.empty and not args.quiet and not args.latex_out:
        print(format_cross_model_terminal_table(cm_df))
        print()

    if args.latex or args.latex_out:
        latex = generate_latex_table(results)
        if args.latex:
            print("=" * 60)
            print("LaTeX Table 1:")
            print("=" * 60)
            print(latex)
            print()
            if not cm_df.empty:
                cm_latex = generate_cross_model_latex(cm_df)
                print("=" * 60)
                print("LaTeX Table 2 (Cross-Model):")
                print("=" * 60)
                print(cm_latex)
                print()
        if args.latex_out:
            Path(args.latex_out).write_text(latex + "\n", encoding="utf-8")
            if not args.quiet:
                print(f"Wrote LaTeX table to {args.latex_out}")

    if args.cross_model_latex_out:
        cm_latex = generate_cross_model_latex(cm_df)
        Path(args.cross_model_latex_out).write_text(cm_latex + "\n", encoding="utf-8")
        if not args.quiet:
            print(f"Wrote cross-model LaTeX table to {args.cross_model_latex_out}")

    if args.cross_model_figure:
        generate_cross_model_figure(cm_df, args.cross_model_figure)

    if args.threshold_sweep or args.threshold_sweep_latex_out:
        if args.no_fetch:
            ts_rows = []
            for key, data in sorted(
                THRESHOLD_SWEEP_DATA.items(),
                key=lambda x: (x[0][0], x[0][1] or "", x[0][2] or 0, x[0][3]),
            ):
                group, summarizer, L, trigger_type = key
                k, n = data["k"], data["n"]
                lo, hi = wilson_ci(k, n)
                rate = k / n if n else 0.0
                ts_rows.append({
                    "strategy_group": group,
                    "summarizer": summarizer,
                    "L": L,
                    "trigger_type": trigger_type,
                    "trigger_pct": data["trigger_pct"],
                    "preds": data["preds"],
                    "k": k, "n": n,
                    "solve_rate": rate,
                    "ci_lo": lo, "ci_hi": hi, "ci_half": (hi - lo) / 2,
                    "vs_raw": rate - 0.64,
                })
            ts_df = pd.DataFrame(ts_rows)
        else:
            ts_df = build_threshold_sweep_table(args.project, args.entity)

        if args.threshold_sweep and not ts_df.empty:
            print("=" * 60)
            print("Threshold Sweep (GLM-4.7, n=50):")
            print("=" * 60)
            print(format_threshold_sweep_terminal(ts_df))
            print()

        if args.threshold_sweep_latex_out and not ts_df.empty:
            ts_latex = generate_threshold_sweep_latex(ts_df)
            Path(args.threshold_sweep_latex_out).write_text(ts_latex + "\n", encoding="utf-8")
            if not args.quiet:
                print(f"Wrote threshold sweep LaTeX table to {args.threshold_sweep_latex_out}")

    if args.figure:
        generate_figure(results, args.figure)

    # instance-level histograms share the same WandB fetch
    needs_instances = args.token_histogram or args.turns_histogram
    if needs_instances:
        if args.no_fetch:
            print("--token-histogram / --turns-histogram require WandB (incompatible with --no-fetch)")
        else:
            token_df = fetch_instance_tokens(args.project, args.entity)
            if not token_df.empty:
                if args.token_histogram:
                    generate_token_histogram(token_df, args.token_histogram)
                if args.turns_histogram:
                    generate_turns_histogram(token_df, args.turns_histogram)


if __name__ == "__main__":
    main()
