#!/usr/bin/env python3
"""Compute limit-aware compaction trigger frequency from trajectory logs.

Why this exists:
Reviewers will (reasonably) ask "did on-demand compaction actually trigger?"
This script answers that from the per-instance `*.debug.log` files produced by
SWE-agent runs.

It reports:
- how many instances have any "triggering compaction" events
- breakdown by history processor label (LastNObservations vs SummarizeEveryNTurns)
- basic distribution of trigger counts

Usage:
  python scripts/compaction_trigger_stats.py trajectories/<user>/<run_dir>

Notes:
- Works on both local and VPS run directories.
- Assumes standard SWE-agent log filenames: `<instance>/*.debug.log`.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


def _label_from_line(line: str) -> str:
    if "LastNObservations:" in line:
        return "masking"
    if "SummarizeEveryNTurns:" in line:
        return "summary"
    return "unknown"


@dataclass(frozen=True)
class InstanceStats:
    instance: str
    triggers_any: int
    triggers_by_label: dict[str, int]


def compute_run_stats(run_dir: Path) -> list[InstanceStats]:
    stats: list[InstanceStats] = []

    for inst_dir in sorted([p for p in run_dir.iterdir() if p.is_dir()]):
        debug_logs = sorted(inst_dir.glob("*.debug.log"))
        if not debug_logs:
            continue

        # Prefer the canonical `<instance>.debug.log` if present; else take the first.
        preferred = inst_dir / f"{inst_dir.name}.debug.log"
        log_path = preferred if preferred.exists() else debug_logs[0]

        text = log_path.read_text(encoding="utf-8", errors="replace")

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute compaction trigger frequency from trajectory logs")
    parser.add_argument("run_dir", type=Path, help="Path to a run directory containing instance subfolders")
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")

    stats = compute_run_stats(run_dir)
    if not stats:
        print("No instance debug logs found; nothing to report.")
        return 1

    n = len(stats)
    n_triggered = sum(1 for s in stats if s.triggers_any > 0)
    n_never = n - n_triggered

    print(f"run_dir: {run_dir}")
    print(f"instances_with_logs: {n}")
    print(f"triggered_any: {n_triggered} ({n_triggered / n:.1%})")
    print(f"never_triggered: {n_never} ({n_never / n:.1%})")
    print()

    # label breakdown
    label_instance_counts: Counter[str] = Counter()
    label_trigger_counts: Counter[str] = Counter()
    trigger_count_hist: Counter[int] = Counter()

    for s in stats:
        trigger_count_hist[s.triggers_any] += 1
        for label, k in s.triggers_by_label.items():
            if k > 0:
                label_instance_counts[label] += 1
                label_trigger_counts[label] += k

    print("by_label:")
    for label in sorted(set(label_instance_counts.keys()) | set(label_trigger_counts.keys())):
        inst = label_instance_counts[label]
        trig = label_trigger_counts[label]
        print(f"  {label}: instances={inst} ({inst / n:.1%}) total_triggers={trig}")
    print()

    print("trigger_count_histogram:")
    for k in sorted(trigger_count_hist.keys()):
        v = trigger_count_hist[k]
        print(f"  {k}: {v}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
