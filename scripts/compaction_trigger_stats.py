#!/usr/bin/env python3
"""Compute compaction trigger frequency from trajectory data.

Two data sources, selected via --source:

  traj  -- Parse `<instance>.traj` JSON files (summaries list).
           Counts summary compactions for periodic, on-demand, and hybrid runs.
           Note: masking-only compactions are not recorded in .traj files.
  log   -- Parse `*.debug.log` for "triggering compaction" lines.
           Only works for limit-aware (on-demand) runs.
  auto  -- (default) Analyze both logs and .traj files together.

Usage:
  python scripts/compaction_trigger_stats.py trajectories/<user>/<run_dir>
  python scripts/compaction_trigger_stats.py --source traj trajectories/<user>/<run_dir>
  python scripts/compaction_trigger_stats.py --json trajectories/<user>/<run_dir>
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class InstanceStats:
    instance: str
    triggers_any: int
    triggers_by_label: dict[str, int]
    turns: int = 0
    exit_status: str = ""
    summary_cost: float = 0.0
    summary_tokens: int = 0


def _label_from_line(line: str) -> str:
    if "LastNObservations:" in line:
        return "masking"
    if "SummarizeEveryNTurns:" in line:
        return "summary"
    return "unknown"


def compute_run_stats_from_log(run_dir: Path) -> list[InstanceStats]:
    stats: list[InstanceStats] = []

    for inst_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        debug_logs = sorted(inst_dir.glob("*.debug.log"))
        if not debug_logs:
            continue

        preferred = inst_dir / f"{inst_dir.name}.debug.log"
        log_path = preferred if preferred.exists() else debug_logs[0]
        try:
            text = log_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        triggers_by_label: dict[str, int] = defaultdict(int)
        triggers_any = 0
        for line in text.splitlines():
            if "triggering compaction" not in line:
                continue
            triggers_any += 1
            triggers_by_label[_label_from_line(line)] += 1

        stats.append(
            InstanceStats(
                instance=inst_dir.name,
                triggers_any=triggers_any,
                triggers_by_label=dict(triggers_by_label),
            )
        )

    return stats


def compute_run_stats_from_traj(run_dir: Path) -> list[InstanceStats]:
    stats: list[InstanceStats] = []

    for inst_dir in sorted(p for p in run_dir.iterdir() if p.is_dir()):
        traj_files = sorted(inst_dir.glob("*.traj"))
        if not traj_files:
            continue

        preferred = inst_dir / f"{inst_dir.name}.traj"
        traj_path = preferred if preferred.exists() else traj_files[0]

        try:
            data = json.loads(traj_path.read_text(encoding="utf-8", errors="replace"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue

        raw_summaries = data.get("summaries")
        if raw_summaries is None:
            summaries = []
        elif isinstance(raw_summaries, list):
            summaries = raw_summaries
        else:
            continue
        history = data.get("history") or []
        raw_info = data.get("info")
        info = raw_info if isinstance(raw_info, dict) else {}

        n_compactions = len(summaries)
        turns = 0
        if isinstance(history, list):
            turns = sum(1 for item in history if isinstance(item, dict) and item.get("message_type") == "action")
            if turns == 0:
                turns = len(history) // 2
        exit_status = info.get("exit_status", "")

        total_cost = 0.0
        total_tokens = 0
        for s in summaries:
            if not isinstance(s, dict):
                continue
            st = s.get("statistics") or {}
            if not isinstance(st, dict):
                continue
            total_cost += _safe_float(st.get("cost", 0.0), default=0.0)
            tok = st.get("tokens") or {}
            if isinstance(tok, dict):
                total_tokens += _safe_int(tok.get("raw_input", 0), default=0)
                total_tokens += _safe_int(tok.get("output", 0), default=0)

        triggers_by_label: dict[str, int] = {}
        if n_compactions > 0:
            triggers_by_label["summary"] = n_compactions

        stats.append(
            InstanceStats(
                instance=inst_dir.name,
                triggers_any=n_compactions,
                triggers_by_label=triggers_by_label,
                turns=turns,
                exit_status=exit_status,
                summary_cost=total_cost,
                summary_tokens=total_tokens,
            )
        )

    return stats


MIN_ACTIVE_TURNS = 10


def _has_traj_files(run_dir: Path) -> bool:
    for inst_dir in run_dir.iterdir():
        if inst_dir.is_dir() and next(inst_dir.glob("*.traj"), None):
            return True
    return False


def compute_run_stats(run_dir: Path, source: str = "auto") -> tuple[list[InstanceStats], list[InstanceStats]]:
    """Return (log_stats, traj_stats) for the requested source mode."""
    if source not in ("log", "traj", "auto"):
        raise ValueError(f"Invalid source: {source}")
    if source == "log":
        return compute_run_stats_from_log(run_dir), []
    if source == "traj":
        return [], compute_run_stats_from_traj(run_dir)
    return compute_run_stats_from_log(run_dir), compute_run_stats_from_traj(run_dir)


def _rate(numer: int, denom: int) -> float | None:
    if denom <= 0:
        return None
    return numer / denom


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(parsed):
        return default
    return parsed


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        try:
            return int(float(value))
        except (TypeError, ValueError, OverflowError):
            return default


def summarize_run(run_dir: Path, source: str = "auto") -> dict[str, Any]:
    """Return machine-readable run-level trigger and summary-call aggregates.

    Notes:
    - `log` section reports limit-aware trigger events from debug logs.
    - `traj` section reports summary calls from `.traj` `summaries`.
    """
    log_stats, traj_stats = compute_run_stats(run_dir, source=source)

    log_n = len(log_stats)
    log_triggered_any = sum(1 for s in log_stats if s.triggers_any > 0)
    log_total_triggers = sum(s.triggers_any for s in log_stats)
    log_label_instances: Counter[str] = Counter()
    log_label_triggers: Counter[str] = Counter()
    for s in log_stats:
        for label, n in s.triggers_by_label.items():
            if n > 0:
                log_label_instances[label] += 1
                log_label_triggers[label] += n

    traj_n = len(traj_stats)
    traj_with_summaries = sum(1 for s in traj_stats if s.triggers_any > 0)
    traj_total_summary_calls = sum(s.triggers_any for s in traj_stats)
    traj_total_summary_cost = sum(s.summary_cost for s in traj_stats)
    traj_total_summary_tokens = sum(s.summary_tokens for s in traj_stats)
    active_traj = [s for s in traj_stats if s.turns > MIN_ACTIVE_TURNS]
    active_n = len(active_traj)
    active_total_summary_calls = sum(s.triggers_any for s in active_traj)
    active_avg_summary_calls = (
        active_total_summary_calls / active_n if active_n > 0 else None
    )

    return {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "source": source,
        "log": {
            "n_instances": log_n,
            "n_triggered_any": log_triggered_any,
            "trigger_rate": _rate(log_triggered_any, log_n),
            "n_never_triggered": log_n - log_triggered_any,
            "total_triggers": log_total_triggers,
            "instances_by_label": dict(log_label_instances),
            "triggers_by_label": dict(log_label_triggers),
        },
        "traj": {
            "n_instances": traj_n,
            "n_with_summaries": traj_with_summaries,
            "summary_rate": _rate(traj_with_summaries, traj_n),
            "total_summary_calls": traj_total_summary_calls,
            "avg_summary_calls_per_traj_instance": (
                traj_total_summary_calls / traj_n if traj_n > 0 else None
            ),
            "total_summary_cost": traj_total_summary_cost,
            "total_summary_tokens": traj_total_summary_tokens,
            "n_active_instances": active_n,
            "active_total_summary_calls": active_total_summary_calls,
            "active_avg_summary_calls_per_instance": active_avg_summary_calls,
        },
    }


def _print_log_report(stats: list[InstanceStats], run_dir: Path) -> None:
    n = len(stats)
    n_triggered = sum(1 for s in stats if s.triggers_any > 0)
    n_never = n - n_triggered

    print(f"Run: {run_dir.name}")
    print(f"Source: debug logs")
    print(f"Instances with logs: {n}")
    print(f"Triggered any: {n_triggered} ({n_triggered / n:.1%})")
    print(f"Never triggered: {n_never} ({n_never / n:.1%})")
    print()

    label_instances = Counter()
    label_triggers = Counter()
    trigger_hist = Counter()

    for s in stats:
        trigger_hist[s.triggers_any] += 1
        for label, k in s.triggers_by_label.items():
            if k > 0:
                label_instances[label] += 1
                label_triggers[label] += k

    print("By label:")
    for label in sorted(set(label_instances) | set(label_triggers)):
        inst = label_instances[label]
        trig = label_triggers[label]
        print(f"  {label}: instances={inst} ({inst / n:.1%}) total_triggers={trig}")
    print()

    print("Trigger count distribution:")
    for k in sorted(trigger_hist):
        v = trigger_hist[k]
        print(f"  {k}: {v} instance{'s' if v != 1 else ''}")


def _print_traj_report(stats: list[InstanceStats], run_dir: Path) -> None:
    n = len(stats)

    print(f"Run: {run_dir.name}")
    print(f"Source: .traj files")
    print()

    print(f"{'Instance':<45} {'Turns':>5} {'Compactions':>11} {'Sum Cost':>10}  {'Exit'}")
    print("-" * 100)
    for s in sorted(stats, key=lambda x: x.instance):
        cost_str = f"${s.summary_cost:.3f}" if s.summary_cost > 0 else "-"
        print(f"{s.instance:<45} {s.turns:>5} {s.triggers_any:>11} {cost_str:>10}  {s.exit_status}")
    print()

    active = [s for s in stats if s.turns > MIN_ACTIVE_TURNS]
    n_active = len(active)
    print(f"Active instances (>{MIN_ACTIVE_TURNS} turns): {n_active}/{n}")
    print()

    trigger_hist: Counter[int] = Counter()
    for s in active:
        trigger_hist[s.triggers_any] += 1

    print("Compaction distribution (active instances):")
    for k in sorted(trigger_hist):
        v = trigger_hist[k]
        print(f"  {k}: {v} instance{'s' if v != 1 else ''}")
    print()

    total_compactions = sum(s.triggers_any for s in active)
    total_cost = sum(s.summary_cost for s in active)
    avg = total_compactions / n_active if n_active else 0

    print(f"Total compactions (active): {total_compactions}")
    print(f"Avg compactions/active instance: {avg:.1f}")
    if total_cost > 0:
        print(f"Total summary cost: ${total_cost:.3f}")


def _print_auto_report(log_stats: list[InstanceStats], traj_stats: list[InstanceStats], run_dir: Path) -> None:
    print(f"Run: {run_dir.name}")
    print("Source: auto (logs + .traj)")
    print()

    if log_stats:
        n_logs = len(log_stats)
        n_triggered = sum(1 for s in log_stats if s.triggers_any > 0)
        print("Limit-aware trigger stats (from debug logs):")
        print(f"  Instances with logs: {n_logs}")
        print(f"  Triggered any: {n_triggered} ({n_triggered / n_logs:.1%})")

        label_instances = Counter()
        label_triggers = Counter()
        for s in log_stats:
            for label, k in s.triggers_by_label.items():
                if k > 0:
                    label_instances[label] += 1
                    label_triggers[label] += k
        for label in sorted(set(label_instances) | set(label_triggers)):
            inst = label_instances[label]
            trig = label_triggers[label]
            print(f"  {label}: instances={inst} ({inst / n_logs:.1%}) total_triggers={trig}")
    else:
        print("Limit-aware trigger stats (from debug logs): unavailable")
    print()

    if traj_stats:
        n_traj = len(traj_stats)
        n_with_summary = sum(1 for s in traj_stats if s.triggers_any > 0)
        total_compactions = sum(s.triggers_any for s in traj_stats)
        total_cost = sum(s.summary_cost for s in traj_stats)
        print("Summary-call stats (from .traj summaries):")
        print(f"  Instances with traj: {n_traj}")
        print(f"  Instances with summaries: {n_with_summary} ({n_with_summary / n_traj:.1%})")
        print(f"  Total summary calls: {total_compactions}")
        print(f"  Avg summary calls per traj instance: {total_compactions / n_traj:.2f}")
        if total_cost > 0:
            print(f"  Total summary cost: ${total_cost:.3f}")
    else:
        print("Summary-call stats (from .traj summaries): unavailable")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute compaction trigger stats from trajectory data"
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Path to a run directory containing instance subfolders",
    )
    parser.add_argument(
        "--source",
        choices=["traj", "log", "auto"],
        default="auto",
        help="Data source: traj (.traj JSON), log (debug logs), auto (default)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output machine-readable JSON summary",
    )
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory is not a directory: {run_dir}")

    log_stats, traj_stats = compute_run_stats(run_dir, source=args.source)

    if args.source == "traj":
        if not traj_stats:
            print("No .traj files found.")
            return 1
        if args.json:
            print(json.dumps(summarize_run(run_dir, source="traj"), indent=2, sort_keys=True))
        else:
            _print_traj_report(traj_stats, run_dir)
    elif args.source == "log":
        if not log_stats:
            print("No debug logs found.")
            return 1
        if args.json:
            print(json.dumps(summarize_run(run_dir, source="log"), indent=2, sort_keys=True))
        else:
            _print_log_report(log_stats, run_dir)
    else:
        if not log_stats and not traj_stats:
            print("No debug logs or .traj files found.")
            return 1
        if args.json:
            print(json.dumps(summarize_run(run_dir, source="auto"), indent=2, sort_keys=True))
        else:
            _print_auto_report(log_stats, traj_stats, run_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
