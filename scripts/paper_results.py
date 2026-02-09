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
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))
from dashboard_shared import fetch_runs, dedupe_latest_runs, OUR_BASELINES

# Phase 3 periodic results (GLM-4.7, n=50, verified-mini)
# OUR_BASELINES doesn't distinguish by summarizer, so we store them here.
PHASE3_PERIODIC = {
    "raw": {"k": 32, "n": 50, "rate": 0.640, "cost": 1.00},
    "masking": {"k": 31, "n": 50, "rate": 0.620, "cost": 0.68},
    # NOTE: self-summary run had 49/50 coverage (one instance produced no prediction),
    # but for solve_rate we still divide by the full 50-instance benchmark slice.
    "summary_self": {"k": 28, "n": 50, "rate": 0.560, "cost": 1.43},
    "summary_minimax": {"k": 27, "n": 50, "rate": 0.540, "cost": 1.34},
    "summary_kimi": {"k": 7, "n": 50, "rate": 0.140, "cost": None},
    "hybrid_minimax": {"k": 28, "n": 50, "rate": 0.560, "cost": 0.42},
    }


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
    limit_aware = row.get("hp_limit_aware", False)

    if strategy == "raw":
        return "baseline"
    if strategy == "on_demand":
        return "on_demand"
    if limit_aware:
        return "on_demand"
    return "periodic"


def build_results_table(project: str, entity: str | None = None) -> pd.DataFrame:
    """Fetch runs and build the paper's results table."""
    df = fetch_runs(project, entity)
    df = dedupe_latest_runs(df)

    # only GLM-4.7, verified-mini, evaluated
    # NOTE: be strict about selecting 50-task runs; otherwise full Verified (500)
    # runs can leak into the paper table/figure.
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
    # For paper reporting, rely on evaluated results only. Pending/partial runs are
    # represented as "pending" rows in the LaTeX table.
    if "eval_complete" in df.columns:
        df = df[df["eval_complete"]].copy()

    if df.empty:
        print("No matching runs found. Experiments may still be running.")
        return pd.DataFrame()

    df["trigger"] = df.apply(classify_trigger, axis=1)

    # determine compaction method
    def compaction_method(row):
        s = row["strategy"]
        if s == "raw":
            return "none"
        if s in ("observation_masking", "dedup_obs_masking"):
            return "masking"
        return "summary"

    df["compaction"] = df.apply(compaction_method, axis=1)

    # Enforce expected paper hyperparams if present (avoid mixing variants).
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

    # compute CIs
    rows = []
    for _, r in df.iterrows():
        n = int(r["n_instances"])
        k = int(r["n_resolved"])
        lo, hi = wilson_ci(k, n)
        rate = k / n if n > 0 else 0.0
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
        })

    return pd.DataFrame(rows)


def format_terminal_table(results: pd.DataFrame) -> str:
    """Pretty-print results for terminal."""
    if results.empty:
        return "No results available."

    lines = []
    lines.append(f"{'Config':<35} {'Trigger':<12} {'Rate':>8} {'CI':>12} {'n':>5} {'Cost':>7}")
    lines.append("-" * 85)

    for _, r in results.sort_values("solve_rate", ascending=False).iterrows():
        label = r["strategy"]
        if r["compaction"] == "summary":
            label += f" ({r['summarizer']})"

        rate_str = f"{r['solve_rate']:.1%}"
        ci_str = f"[{r['ci_lo']:.1%}, {r['ci_hi']:.1%}]"
        cost_str = f"${r['avg_cost']:.2f}" if pd.notna(r["avg_cost"]) else "N/A"
        eval_mark = "" if r["eval_complete"] else " *"

        lines.append(
            f"{label:<35} {r['trigger']:<12} {rate_str:>8} {ci_str:>12} "
            f"{r['n']:>5} {cost_str:>7}{eval_mark}"
        )

    lines.append("")
    lines.append("* = not yet evaluated (solve rate from WandB submission data)")
    return "\n".join(lines)


def generate_latex_table(results: pd.DataFrame) -> str:
    """Generate LaTeX table for the paper."""
    if results.empty:
        return "% No results available"

    def _pct(x: float, *, signed: bool = False) -> str:
        """Format a proportion as LaTeX percent (escaping '%')."""
        val = x * 100.0
        if signed:
            return f"{val:+.1f}\\%"
        return f"{val:.1f}\\%"

    def _latex_escape(text: str) -> str:
        # Minimal escaping for our table cells.
        return text.replace("_", r"\_")

    # group: raw, periodic masking, on-demand masking, periodic summary (self),
    # on-demand summary (self), on-demand summary (minimax), on-demand summary (kimi)
    order = [
        ("raw", "baseline", "none", "same"),
        ("observation_masking", "periodic", "masking", "same"),
        ("observation_masking", "on_demand", "masking", "same"),
        ("on_demand", "on_demand", "summary", "same"),
        ("on_demand", "on_demand", "summary", "minimax-m2.1"),
        ("on_demand", "on_demand", "summary", "kimi-k2"),
        ("llm_summary", "periodic", "summary", "same"),
        ("llm_summary", "periodic", "summary", "minimax-m2.1"),
    ]

    labels = {
        ("raw", "baseline", "same"): "Raw (no compaction)",
        ("observation_masking", "periodic", "same"): "Periodic masking",
        ("observation_masking", "on_demand", "same"): "On-demand masking",
        ("on_demand", "on_demand", "same"): "On-demand summary (self)",
        ("on_demand", "on_demand", "minimax-m2.1"): "On-demand summary (minimax)",
        ("on_demand", "on_demand", "kimi-k2"): "On-demand summary (kimi)",
        ("llm_summary", "periodic", "same"): "Periodic summary (self)",
        ("llm_summary", "periodic", "minimax-m2.1"): "Periodic summary (minimax)",
    }

    # get raw baseline rate for delta computation
    raw_rows = results[results["strategy"] == "raw"]
    raw_rate = raw_rows["solve_rate"].iloc[0] if len(raw_rows) > 0 else 0.64

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Solve rates on SWE-bench Verified-Mini (50 instances; \texttt{verified-mini}) with GLM-4.7. "
        r"On-demand compaction triggers only at 85\% context utilization.}",
        r"\label{tab:results}",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Configuration & Trigger & Solved & Rate (\%) & $\Delta$ vs Raw \\",
        r"\midrule",
    ]

    for strat, trigger, compaction, summarizer in order:
        key = (strat, trigger, summarizer)
        label = labels.get(key, f"{strat} ({trigger})")
        trigger_cell = _latex_escape(trigger)

        # find matching row
        mask = (results["strategy"] == strat) & (results["trigger"] == trigger)
        if summarizer != "same":
            mask &= results["summarizer"] == summarizer
        else:
            mask &= results["summarizer"].isin(["same", "glm-4.7", ""])

        matching = results[mask]
        if matching.empty:
            # For periodic baselines, prefer the hardcoded Phase-3 numbers so LaTeX
            # output stays complete even if WandB fetch is flaky.
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
                            f"  {label} & -- & {n_k} & {_pct(rate)} $\\pm$ {_pct(ci)} & -- \\\\"
                        )
                    else:
                        lines.append(
                            f"  {label} & {trigger_cell} & {n_k} & {_pct(rate)} $\\pm$ {_pct(ci)} & {delta_str} \\\\"
                        )
                    continue
            lines.append(f"  {label} & {trigger_cell} & \\textit{{pending}} & -- & -- \\\\")
            continue

        r = matching.iloc[0]
        rate = r["solve_rate"]
        ci = r["ci_half"]
        delta = rate - raw_rate
        delta_str = _pct(delta, signed=True)
        n_k = f"{int(r['k'])}/{int(r['n'])}"

        if strat == "raw":
            lines.append(
                f"  {label} & -- & {n_k} & {_pct(rate)} $\\pm$ {_pct(ci)} & -- \\\\"
            )
        else:
            lines.append(
                f"  {label} & {trigger_cell} & {n_k} & {_pct(rate)} $\\pm$ {_pct(ci)} & {delta_str} \\\\"
            )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def generate_figure(results: pd.DataFrame, output_path: str) -> None:
    """Generate Figure 1: grouped bar chart comparing periodic vs on-demand."""
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.figsize": (5.5, 3.5),
        "figure.dpi": 300,
    })

    # build data for grouped bars: periodic vs on-demand for each compaction type
    categories = []  # (label, periodic_rate, periodic_ci, od_rate, od_ci)

    def _p3_rate_ci(key: str, default_rate: float, default_ci: float) -> tuple[float, float]:
        data = PHASE3_PERIODIC.get(key)
        if not data:
            return default_rate, default_ci
        k, n = int(data["k"]), int(data["n"])
        lo, hi = wilson_ci(k, n)
        return (k / n if n else default_rate), (hi - lo) / 2

    raw_rows = results[results["strategy"] == "raw"]
    raw_rate, raw_ci = (
        raw_rows["solve_rate"].iloc[0],
        raw_rows["ci_half"].iloc[0],
    ) if len(raw_rows) > 0 else _p3_rate_ci("raw", 0.64, 0.068)

    # masking
    periodic_mask = results[
        (results["strategy"] == "observation_masking") & (results["trigger"] == "periodic")
    ]
    od_mask = results[
        (results["strategy"] == "observation_masking") & (results["trigger"] == "on_demand")
    ]

    pm_rate, pm_ci = (
        periodic_mask["solve_rate"].iloc[0],
        periodic_mask["ci_half"].iloc[0],
    ) if len(periodic_mask) > 0 else _p3_rate_ci("masking", 0.62, 0.070)
    odm_rate = od_mask["solve_rate"].iloc[0] if len(od_mask) > 0 else None
    odm_ci = od_mask["ci_half"].iloc[0] if len(od_mask) > 0 else None

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

    ps_rate, ps_ci = (
        periodic_sum_self["solve_rate"].iloc[0],
        periodic_sum_self["ci_half"].iloc[0],
    ) if len(periodic_sum_self) > 0 else _p3_rate_ci("summary_self", 0.56, 0.133)
    ods_rate = od_sum_self["solve_rate"].iloc[0] if len(od_sum_self) > 0 else None
    ods_ci = od_sum_self["ci_half"].iloc[0] if len(od_sum_self) > 0 else None

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

    pmm_rate, pmm_ci = (
        periodic_sum_mm["solve_rate"].iloc[0],
        periodic_sum_mm["ci_half"].iloc[0],
    ) if len(periodic_sum_mm) > 0 else _p3_rate_ci("summary_minimax", 0.54, 0.133)
    odmm_rate = od_sum_mm["solve_rate"].iloc[0] if len(od_sum_mm) > 0 else None
    odmm_ci = od_sum_mm["ci_half"].iloc[0] if len(od_sum_mm) > 0 else None

    categories.append(("Summary\n(minimax)", pmm_rate, pmm_ci, odmm_rate, odmm_ci))

    # summary (kimi)
    periodic_sum_k = results[
        (results["strategy"] == "llm_summary") & (results["trigger"] == "periodic")
        & (results["summarizer"] == "kimi-k2")
    ]
    od_sum_k = results[
        (results["strategy"] == "on_demand") & (results["trigger"] == "on_demand")
        & (results["summarizer"] == "kimi-k2")
    ]

    pk_rate, pk_ci = (
        periodic_sum_k["solve_rate"].iloc[0],
        periodic_sum_k["ci_half"].iloc[0],
    ) if len(periodic_sum_k) > 0 else _p3_rate_ci("summary_kimi", 0.14, 0.060)
    odk_rate = od_sum_k["solve_rate"].iloc[0] if len(od_sum_k) > 0 else None
    odk_ci = od_sum_k["ci_half"].iloc[0] if len(od_sum_k) > 0 else None

    categories.append(("Summary\n(kimi)", pk_rate, pk_ci, odk_rate, odk_ci))

    # plot
    fig, ax = plt.subplots()
    x = np.arange(len(categories))
    width = 0.30

    periodic_rates = [c[1] for c in categories]
    periodic_cis = [c[2] for c in categories]
    od_rates = [c[3] if c[3] is not None else 0 for c in categories]
    od_cis = [c[4] if c[4] is not None else 0 for c in categories]
    od_available = [c[3] is not None for c in categories]

    # raw baseline line (thin, no CI band to avoid clutter)
    ax.axhline(y=raw_rate, color="#555", linestyle="--", linewidth=0.7,
               label=f"Raw baseline ({raw_rate:.0%})", zorder=1)

    bars_p = ax.bar(x - width / 2, periodic_rates, width, yerr=periodic_cis,
                    capsize=3, color="#4C72B0", alpha=0.85, label="Periodic",
                    edgecolor="white", linewidth=0.5, zorder=2)

    # on-demand bars: solid color when available, hatched placeholder when pending
    for i, avail in enumerate(od_available):
        rate = od_rates[i]
        ci = od_cis[i]
        if avail:
            ax.bar(x[i] + width / 2, rate, width, yerr=ci, capsize=3,
                   color="#DD8452", alpha=0.85, edgecolor="white", linewidth=0.5, zorder=2)
        else:
            ax.bar(x[i] + width / 2, 0.01, width, color="none",
                   edgecolor="#bbb", linewidth=0.8, linestyle="--", hatch="///", zorder=2)
            ax.text(x[i] + width / 2, 0.04, "?", ha="center", va="bottom",
                    fontsize=7, color="#999", weight="bold")

    # manual legend entry for on-demand
    from matplotlib.patches import Patch
    legend_elements = [
        plt.Line2D([0], [0], color="#555", linestyle="--", linewidth=0.7, label=f"Raw baseline ({raw_rate:.0%})"),
        Patch(facecolor="#4C72B0", alpha=0.85, edgecolor="white", label="Periodic"),
        Patch(facecolor="#DD8452", alpha=0.85, edgecolor="white", label="On-demand"),
    ]

    labels = [c[0] for c in categories]
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Solve Rate")
    ax.set_ylim(0, 0.80)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)
    ax.set_title("Periodic vs On-Demand Context Compaction (GLM-4.7, n=50)")

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    print(f"Figure saved to {output_path}")
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
            ("llm_summary", "kimi-k2", "periodic", "summary", "summary_kimi"),
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
        results = build_results_table(args.project, args.entity)

    if results.empty:
        print("No results. Run with --no-fetch to use hardcoded baselines.")
        return

    if not args.quiet and not args.latex_out:
        print(format_terminal_table(results))
        print()

    if args.latex or args.latex_out:
        latex = generate_latex_table(results)
        if args.latex:
            print("=" * 60)
            print("LaTeX Table 1:")
            print("=" * 60)
            print(latex)
            print()
        if args.latex_out:
            Path(args.latex_out).write_text(latex + "\n", encoding="utf-8")
            if not args.quiet:
                print(f"Wrote LaTeX table to {args.latex_out}")

    if args.figure:
        generate_figure(results, args.figure)


if __name__ == "__main__":
    main()
