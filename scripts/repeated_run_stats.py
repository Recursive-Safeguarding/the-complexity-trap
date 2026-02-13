#!/usr/bin/env python3
"""Repeated-run aggregation for WandB experiment configs.

This script answers a different question than per-run Wilson intervals:
- `paper_results.py`: uncertainty within one run (instance-level Bernoulli model).
- `repeated_run_stats.py`: uncertainty across independent reruns of the same config.

It groups runs by a strict config key, then computes run-level mean solve rate,
sample std, and a 95% t-interval across repeats.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tabulate import tabulate

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent))

from dashboard_shared import dedupe_latest_runs, fetch_runs, get_project_config


T_CRITICAL_95: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}

GROUP_COLUMNS = [
    "model",
    "strategy",
    "summarizer_norm",
    "instances_subset_norm",
    "hp_obs_n",
    "hp_sum_n",
    "hp_sum_keep_m",
    "hp_limit_aware",
    "hp_limit_fraction",
    "hp_limit_min_tokens",
]


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a single binomial proportion."""
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - spread), min(1.0, center + spread)


def t_critical_95(df: int) -> float:
    """Return 95% two-sided t critical value for degrees of freedom."""
    if df <= 0:
        return 0.0
    if df in T_CRITICAL_95:
        return T_CRITICAL_95[df]
    return 1.96


def normalize_subset(value: Any) -> str:
    if value is None:
        return "unknown"
    return str(value).strip().lower().replace("_", "-")


def normalize_summarizer(value: Any) -> str:
    if value is None:
        return "same"
    s = str(value).strip()
    if not s:
        return "same"
    if s.lower() in ("none", "nan", "same", "reuse-agent-model"):
        return "same"
    return s


def _to_bool(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "t")
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return False
        return bool(value)
    return bool(value)


def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    required_defaults: dict[str, Any] = {
        "run_id": "",
        "run_name": "",
        "created_at": "",
        "model": "unknown",
        "strategy": "unknown",
        "summarizer": "same",
        "instances_subset": "verified",
        "eval_complete": False,
        "hp_obs_n": 0,
        "hp_sum_n": 0,
        "hp_sum_keep_m": 0,
        "hp_limit_aware": False,
        "hp_limit_fraction": np.nan,
        "hp_limit_min_tokens": 0,
        "n_instances": 0,
        "n_resolved": 0,
    }

    for col, default in required_defaults.items():
        if col not in work.columns:
            work[col] = default

    int_cols = ["hp_obs_n", "hp_sum_n", "hp_sum_keep_m", "hp_limit_min_tokens", "n_instances", "n_resolved"]
    for col in int_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0).astype(int)

    work["hp_limit_fraction"] = pd.to_numeric(work["hp_limit_fraction"], errors="coerce")
    work["hp_limit_aware"] = work["hp_limit_aware"].apply(_to_bool)
    work["eval_complete"] = work["eval_complete"].apply(_to_bool)
    work["instances_subset_norm"] = work["instances_subset"].apply(normalize_subset)
    work["summarizer_norm"] = work["summarizer"].apply(normalize_summarizer)
    return work


def filter_runs(
    df: pd.DataFrame,
    *,
    model_filter: str | None,
    strategy_filter: str | None,
    instances_subset: str,
    eval_only: bool,
    n_instances_min: int,
    n_instances_max: int,
) -> pd.DataFrame:
    """Apply selection filters and return deduped candidate runs."""
    work = _ensure_columns(df)

    if model_filter:
        work = work[work["model"].str.contains(model_filter, case=False, na=False)]
    if strategy_filter:
        work = work[work["strategy"].str.contains(strategy_filter, case=False, na=False)]

    subset_norm = normalize_subset(instances_subset)
    work = work[work["instances_subset_norm"] == subset_norm]

    if eval_only:
        work = work[work["eval_complete"]]

    work = work[work["n_instances"].between(n_instances_min, n_instances_max)]
    work = work[work["n_instances"] > 0]

    # Remove duplicate API snapshots for the same run name while keeping the latest.
    work = dedupe_latest_runs(work)
    return work.reset_index(drop=True)


def _format_hp_fraction(value: float) -> str:
    if pd.isna(value):
        return "na"
    return f"{float(value):g}"


def _hp_signature(
    *,
    hp_obs_n: int,
    hp_sum_n: int,
    hp_sum_keep_m: int,
    hp_limit_aware: bool,
    hp_limit_fraction: float,
    hp_limit_min_tokens: int,
) -> str:
    return (
        f"obs_n={hp_obs_n};sum_n={hp_sum_n};keep_m={hp_sum_keep_m};"
        f"limit_aware={hp_limit_aware};limit_fraction={_format_hp_fraction(hp_limit_fraction)};"
        f"limit_min_tokens={hp_limit_min_tokens}"
    )


def compute_repeated_run_stats(runs: pd.DataFrame, *, min_repeats: int = 2) -> pd.DataFrame:
    """Compute run-level uncertainty across repeated runs for each strict config."""
    if min_repeats < 1:
        raise ValueError("min_repeats must be >= 1")
    if runs.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    grouped = runs.groupby(GROUP_COLUMNS, dropna=False, sort=True)
    for keys, group in grouped:
        n_runs = int(len(group))
        if n_runs < min_repeats:
            continue

        group = group.sort_values("created_at")
        rates = (group["n_resolved"] / group["n_instances"]).astype(float)
        mean_rate = float(rates.mean())
        std_rate = float(rates.std(ddof=1)) if n_runs > 1 else 0.0
        sem_rate = std_rate / math.sqrt(n_runs) if n_runs > 1 else 0.0
        t_crit = t_critical_95(n_runs - 1)
        ci_half = t_crit * sem_rate
        t_ci_low = max(0.0, mean_rate - ci_half)
        t_ci_high = min(1.0, mean_rate + ci_half)

        total_resolved = int(group["n_resolved"].sum())
        total_instances = int(group["n_instances"].sum())
        pooled_rate = total_resolved / total_instances if total_instances > 0 else 0.0
        pooled_ci_low, pooled_ci_high = wilson_ci(total_resolved, total_instances)

        row = dict(zip(GROUP_COLUMNS, keys, strict=True))
        row.update(
            {
                "summarizer": row.pop("summarizer_norm"),
                "instances_subset": row.pop("instances_subset_norm"),
                "hp_signature": _hp_signature(
                    hp_obs_n=int(row["hp_obs_n"]),
                    hp_sum_n=int(row["hp_sum_n"]),
                    hp_sum_keep_m=int(row["hp_sum_keep_m"]),
                    hp_limit_aware=bool(row["hp_limit_aware"]),
                    hp_limit_fraction=float(row["hp_limit_fraction"])
                    if not pd.isna(row["hp_limit_fraction"])
                    else np.nan,
                    hp_limit_min_tokens=int(row["hp_limit_min_tokens"]),
                ),
                "n_runs": n_runs,
                "mean_rate": mean_rate,
                "std_rate": std_rate,
                "sem_rate": sem_rate,
                "t_critical": t_crit,
                "t_ci_low": t_ci_low,
                "t_ci_high": t_ci_high,
                "t_ci_half": ci_half,
                "pooled_rate": pooled_rate,
                "pooled_ci_low": pooled_ci_low,
                "pooled_ci_high": pooled_ci_high,
                "total_resolved": total_resolved,
                "total_instances": total_instances,
                "run_names": group["run_name"].astype(str).tolist(),
                "run_ids": group["run_id"].astype(str).tolist(),
                "run_rates": [float(x) for x in rates.tolist()],
            }
        )
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["model", "strategy", "summarizer", "instances_subset", "hp_signature"], ascending=True
    ).reset_index(drop=True)


def _format_pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _format_ci(low: float, high: float) -> str:
    return f"[{_format_pct(low)}, {_format_pct(high)}]"


def _build_display_dataframe(stats: pd.DataFrame, *, show_runs: bool) -> pd.DataFrame:
    columns: dict[str, Any] = {
        "model": stats["model"],
        "strategy": stats["strategy"],
        "summarizer": stats["summarizer"],
        "subset": stats["instances_subset"],
        "hp_signature": stats["hp_signature"],
        "n_runs": stats["n_runs"],
        "mean_rate": stats["mean_rate"].map(_format_pct),
        "t_ci_95": [_format_ci(lo, hi) for lo, hi in zip(stats["t_ci_low"], stats["t_ci_high"], strict=True)],
        "std_pp": (stats["std_rate"] * 100).map(lambda x: f"{x:.2f}"),
        "pooled_rate": stats["pooled_rate"].map(_format_pct),
        "pooled_wilson_95": [
            _format_ci(lo, hi) for lo, hi in zip(stats["pooled_ci_low"], stats["pooled_ci_high"], strict=True)
        ],
    }
    table = pd.DataFrame(columns)

    if show_runs:
        table["run_names"] = stats["run_names"].map(lambda xs: ", ".join(xs))
        table["run_rates"] = stats["run_rates"].map(lambda xs: ", ".join(f"{x * 100:.1f}%" for x in xs))
    return table


def _to_builtin(value: Any) -> Any:
    if isinstance(value, list):
        return [_to_builtin(v) for v in value]
    if isinstance(value, tuple):
        return [_to_builtin(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if pd.isna(value):
        return None
    return value


def render_results(
    stats: pd.DataFrame,
    *,
    output_format: str,
    show_runs: bool,
    file: TextIO,
) -> None:
    if stats.empty:
        file.write("No repeated configs matched filters (or none meet --min-repeats).\n")
        return

    if output_format in ("table", "markdown"):
        display_df = _build_display_dataframe(stats, show_runs=show_runs)
        tablefmt = "simple" if output_format == "table" else "github"
        file.write(tabulate(display_df, headers="keys", tablefmt=tablefmt, showindex=False))
        file.write("\n")
        return

    machine_df = stats.copy()
    if not show_runs:
        machine_df = machine_df.drop(columns=["run_names", "run_ids", "run_rates"], errors="ignore")

    if output_format == "json":
        records = [{k: _to_builtin(v) for k, v in rec.items()} for rec in machine_df.to_dict(orient="records")]
        json.dump(records, file, indent=2)
        file.write("\n")
        return

    if output_format == "csv":
        machine_df.to_csv(file, index=False)
        return

    raise ValueError(f"Unsupported format: {output_format}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate repeated runs and estimate run-level uncertainty.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/repeated_run_stats.py
  python scripts/repeated_run_stats.py --model glm-4.7 --strategy on_demand
  python scripts/repeated_run_stats.py --format markdown --output /tmp/repeats.md
  python scripts/repeated_run_stats.py --show-runs --min-repeats 2
        """,
    )
    parser.add_argument("--project", "-p", help="WandB project (default from env)")
    parser.add_argument("--entity", "-e", help="WandB entity (default from env)")
    parser.add_argument("--model", "-m", help="Filter model (substring, case-insensitive)")
    parser.add_argument("--strategy", "-s", help="Filter strategy (substring, case-insensitive)")
    parser.add_argument(
        "--instances-subset",
        default="verified-mini",
        help="Subset filter (default: verified-mini; normalization handles _ vs -)",
    )
    parser.add_argument(
        "--eval-only",
        dest="eval_only",
        action="store_true",
        default=True,
        help="Only include eval_complete runs (default: true).",
    )
    parser.add_argument(
        "--include-unevaluated",
        dest="eval_only",
        action="store_false",
        help="Include runs without completed evaluation.",
    )
    parser.add_argument("--n-instances-min", type=int, default=40, help="Minimum n_instances (default: 40)")
    parser.add_argument("--n-instances-max", type=int, default=60, help="Maximum n_instances (default: 60)")
    parser.add_argument("--min-repeats", type=int, default=2, help="Minimum reruns per config (default: 2)")
    parser.add_argument(
        "--format",
        "-f",
        default="table",
        choices=["table", "markdown", "json", "csv"],
        help="Output format",
    )
    parser.add_argument("--output", help="Write output to this file path instead of stdout")
    parser.add_argument("--show-runs", action="store_true", help="Include run names/IDs and per-run rates")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    project = args.project
    entity = args.entity

    if not project:
        env_project, env_entity = get_project_config()
        project = env_project
        if not entity:
            entity = env_entity

    if not project:
        print("ERROR: No project specified. Set WANDB_PROJECT or pass --project.", file=sys.stderr)
        return 1

    try:
        df = fetch_runs(project, entity)
    except Exception as exc:
        print(f"ERROR: Failed to fetch runs: {exc}", file=sys.stderr)
        return 1

    filtered = filter_runs(
        df,
        model_filter=args.model,
        strategy_filter=args.strategy,
        instances_subset=args.instances_subset,
        eval_only=args.eval_only,
        n_instances_min=args.n_instances_min,
        n_instances_max=args.n_instances_max,
    )

    stats = compute_repeated_run_stats(filtered, min_repeats=args.min_repeats)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            render_results(stats, output_format=args.format, show_runs=args.show_runs, file=handle)
    else:
        render_results(stats, output_format=args.format, show_runs=args.show_runs, file=sys.stdout)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
