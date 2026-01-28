#!/usr/bin/env python3
"""Update WandB runs with evaluation results retroactively.

Finds runs that have evaluation results (*.eval.json files) and updates
the corresponding WandB run with solve_rate, eval_pass_rate, and eval_coverage.

Usage:
    python scripts/update_wandb_with_evals.py --project the-complexity-trap --entity ox
    python scripts/update_wandb_with_evals.py --dry-run  # Preview changes
"""

import argparse
import json
import math
import os
import re
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import wandb


def _coerce_int(value, default: int = 0) -> int:
    """Best-effort int parsing that tolerates None/NaN/strings."""
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return default
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return default
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default


def _coerce_float(value, default: float = 0.0) -> float:
    """Best-effort float parsing that tolerates None/NaN/strings."""
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return default
        return float(value)
    if isinstance(value, str):
        try:
            parsed = float(value)
            return parsed if math.isfinite(parsed) else default
        except ValueError:
            return default
    return default


def find_eval_results(project_dir: Path) -> dict[str, Path]:
    """Map run_id -> eval results path. Checks *__eval.json in root, then logs/run_evaluation/."""
    results = {}

    for eval_file in project_dir.glob("*__eval.json"):
        filename = eval_file.stem
        if filename.endswith("__eval"):
            filename = filename[:-6]

        timestamp_pattern = r"__\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}"
        match = re.search(timestamp_pattern, filename)
        if match:
            run_id_with_ts = filename[:match.end()]
            run_id = re.sub(timestamp_pattern, "", run_id_with_ts)
        else:
            parts = filename.split(".")
            run_id = parts[0] if parts else filename

        results[run_id] = eval_file

    # fallback: logs/run_evaluation/
    logs_dir = project_dir / "logs" / "run_evaluation"
    if logs_dir.exists():
        for eval_dir in logs_dir.iterdir():
            if not eval_dir.is_dir():
                continue

            report_files = list(eval_dir.rglob("*.json"))
            if not report_files:
                continue

            latest = max(report_files, key=lambda p: p.stat().st_mtime)
            run_id = eval_dir.name.replace("__eval", "")
            run_id = re.sub(r"__\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", "", run_id)

            if run_id not in results:
                results[run_id] = latest

    return results


def parse_eval_results(results_path: Path) -> dict:
    """Parse eval JSON. Handles aggregated {"resolved_instances": N} or per-instance {id: {"resolved": bool}}."""
    with open(results_path) as f:
        data = json.load(f)

    # handle list-wrapped eval results (e.g., [{instance_id: ..., resolved: true}, ...])
    if isinstance(data, list):
        n_resolved = sum(
            1 for item in data
            if isinstance(item, dict) and item.get("resolved")
        )
        return {"n_resolved": n_resolved, "n_evaluated": len(data), "n_instances": 0}

    if not isinstance(data, dict):
        return {"n_resolved": 0, "n_evaluated": 0, "n_instances": 0}

    if "resolved_instances" in data:
        return {
            "n_resolved": data.get("resolved_instances", 0),
            "n_evaluated": data.get("completed_instances", data.get("submitted_instances", 0)),
            # Do not treat submitted/evaluated instances as the total slice size.
            "n_instances": 0,
        }

    # Unwrap common container keys if present.
    for key in ("predictions", "preds", "instances", "data"):
        wrapped = data.get(key)
        if isinstance(wrapped, dict):
            data = wrapped
            break

    # Count only per-instance entries; ignore top-level metadata keys.
    instance_ids = [
        k
        for k, v in data.items()
        if isinstance(v, dict) and ("resolved" in v or "result" in v or "status" in v)
    ]

    if "resolved" in data or "resolved_ids" in data:
        resolved = data.get("resolved", data.get("resolved_ids", []))
        if isinstance(resolved, list):
            # If we have explicit per-instance entries, intersect with them.
            # Otherwise, fall back to trusting the resolved list length.
            if instance_ids:
                n_resolved = len([rid for rid in resolved if rid in instance_ids])
            else:
                n_resolved = len(resolved)
        elif isinstance(resolved, bool):
            # Single-instance or legacy format; fall back to per-instance flags.
            n_resolved = len([k for k in instance_ids if data.get(k, {}).get("resolved")])
        else:
            n_resolved = 1 if resolved else 0
    else:
        n_resolved = len([k for k in instance_ids if data.get(k, {}).get("resolved")])

    n_evaluated = len(instance_ids)

    return {
        "n_resolved": n_resolved,
        "n_evaluated": n_evaluated,
        # Prefer backfilling from the WandB run summary when possible.
        "n_instances": 0,
    }


def update_wandb_run(run, metrics: dict, dry_run: bool = False):
    n_resolved = _coerce_int(metrics.get("n_resolved", 0), default=0)
    n_evaluated = _coerce_int(metrics.get("n_evaluated", 0), default=0)
    n_instances = _coerce_int(metrics.get("n_instances", 0), default=0)
    if not n_instances:
        # Prefer explicit totals, fall back to legacy "instances" key, then to n_evaluated.
        n_instances_raw = run.summary.get("n_instances") or run.summary.get("instances", n_evaluated)
        n_instances = _coerce_int(n_instances_raw, default=n_evaluated)

    eval_pass_rate = n_resolved / n_evaluated if n_evaluated else 0
    eval_coverage = n_evaluated / n_instances if n_instances else 0
    solve_rate = n_resolved / n_instances if n_instances else 0

    updates = {
        "n_resolved": n_resolved,
        "n_evaluated": n_evaluated,
        "eval_pass_rate": eval_pass_rate,
        "eval_coverage": eval_coverage,
        "solve_rate": solve_rate,
        "eval_complete": True,
    }

    if dry_run:
        print(f"  Would update: {updates}")
    else:
        run.summary.update(updates)
        run.update()
        print(f"  Updated: solve_rate={solve_rate:.1%}, pass_rate={eval_pass_rate:.1%}, coverage={eval_coverage:.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="Backfill WandB runs with retroactive evaluation results."
    )
    parser.add_argument(
        "--project",
        default=os.getenv("WANDB_PROJECT", "the-complexity-trap"),
        help="WandB project name (use 'the-complexity-trap').",
    )
    parser.add_argument(
        "--entity",
        default=os.getenv("WANDB_ENTITY", "ox"),
        help="WandB entity/team.",
    )
    parser.add_argument(
        "--project-dir",
        default=".",
        help="Project directory to search for eval results.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview updates without writing to WandB."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing solve_rate values.",
    )
    args = parser.parse_args()

    project_dir = Path(args.project_dir)
    if not project_dir.exists():
        print(f"Project directory not found: {project_dir}")
        return

    eval_results = find_eval_results(project_dir)
    print(f"Found {len(eval_results)} evaluation results")

    if not eval_results:
        return

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}")

    wandb_runs_by_name = {}
    for run in runs:
        if run.name not in wandb_runs_by_name:
            wandb_runs_by_name[run.name] = []
        wandb_runs_by_name[run.name].append(run)
    print(f"Found {len(runs)} WandB runs ({len(wandb_runs_by_name)} unique names)")

    updated = 0
    for run_id, results_path in eval_results.items():
        wandb_run = None
        if run_id in wandb_runs_by_name:
            wandb_run = max(wandb_runs_by_name[run_id], key=lambda r: r.created_at)

        if not wandb_run:
            for name, run_list in wandb_runs_by_name.items():
                if name.startswith(run_id) or run_id.startswith(name):
                    wandb_run = max(run_list, key=lambda r: r.created_at)
                    break

        if not wandb_run:
            print(f"⚠ No WandB run found for: {run_id}")
            continue

        existing_rate = _coerce_float(wandb_run.summary.get("solve_rate", 0.0), default=0.0)
        if existing_rate > 0 and not args.force and not args.dry_run:
            print(f"✓ Already evaluated: {run_id} (solve_rate={existing_rate:.1%})")
            continue

        print(f"→ Processing: {run_id}")
        try:
            metrics = parse_eval_results(results_path)
            print(f"  Results: {metrics['n_resolved']}/{metrics['n_evaluated']} resolved")
            update_wandb_run(wandb_run, metrics, dry_run=args.dry_run)
            updated += 1
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print(f"\n{'Would update' if args.dry_run else 'Updated'} {updated} runs")


if __name__ == "__main__":
    main()
