#!/usr/bin/env python3
"""
One-shot backfill for runs with preds.json but no results.json.

Unlike watch_eval.py (deleted), this script:
- Runs once and exits (no polling daemon)
- No tranche evaluation (handled by WandBHook.batch_eval_interval)
- No state management (stateless)

Usage:
    python scripts/evaluate_missing.py --subset verified-mini
    python scripts/evaluate_missing.py --subset verified --project ox/the-complexity-trap
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import wandb

# add parent to path once for sweagent imports (keeps script standalone)
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_run_name_from_dir(run_dir: Path) -> str:
    """Extract run name from directory (remove timestamp suffix)."""
    name = run_dir.name
    match = re.match(r"(.+)__\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$", name)
    return match.group(1) if match else name


def extract_dataset_tag(run_dir_name: str) -> str | None:
    """Extract dataset tag from run directory name."""
    # sweep-prefixed dirs use different naming convention
    if run_dir_name.startswith("sweep_"):
        return None
    parts = run_dir_name.rsplit("__", 1)
    if len(parts) < 2:
        return None
    base = parts[0]
    if "__" not in base:
        return None
    return base.split("__")[-1].lower()


def matches_subset(dataset_tag: str | None, subset: str) -> bool:
    """Check if dataset_tag matches the expected subset."""
    if not dataset_tag:
        return False
    # check verified-mini first to avoid matching "verified" prefix
    if subset == "verified-mini":
        return dataset_tag in ("mini", "verified-mini", "verifiedmini") or dataset_tag.startswith("mini")
    if subset == "verified":
        # exclude mini variants
        if dataset_tag in ("mini", "verified-mini", "verifiedmini") or dataset_tag.startswith("mini"):
            return False
        return dataset_tag in ("v", "verified") or dataset_tag.startswith("v") or dataset_tag.startswith("verified")
    if subset == "lite":
        return dataset_tag == "lite" or dataset_tag.startswith("lite")
    return dataset_tag == subset.lower() or dataset_tag.startswith(subset.lower())


def find_unevaluated_runs(base_dir: Path, subset: str) -> list[Path]:
    """Find runs with preds.json but no results.json."""
    runs = []
    if not base_dir.exists():
        return runs

    for user_dir in base_dir.iterdir():
        if not user_dir.is_dir():
            continue
        for run_dir in user_dir.iterdir():
            if not run_dir.is_dir():
                continue

            dataset_tag = extract_dataset_tag(run_dir.name)
            if not matches_subset(dataset_tag, subset):
                continue

            preds = run_dir / "preds.json"
            alt_preds = run_dir / "all.preds.json"
            results = run_dir / "results.json"

            if (preds.exists() or alt_preds.exists()) and not results.exists():
                runs.append(run_dir)

    return sorted(runs, key=lambda p: p.stat().st_mtime)


def count_instances(run_dir: Path) -> int:
    """Count completed instances in a run directory."""
    count = 0
    for item in run_dir.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            if list(item.glob("*.traj")):
                count += 1
    return count


def evaluate_run(
    run_dir: Path,
    subset: str,
    workers: int,
    timeout: int,
) -> dict | None:
    """Run Docker evaluation and return results."""
    from sweagent.utils.sbcli import run_docker_evaluation, parse_eval_results

    preds = run_dir / "preds.json"
    if not preds.exists():
        preds = run_dir / "all.preds.json"
    if not preds.exists():
        return None

    print(f"   🐳 Running Docker evaluation...")

    success, error, results_path = run_docker_evaluation(
        preds_path=preds,
        subset=subset,
        run_id=run_dir.name,
        output_dir=run_dir,
        max_workers=workers,
        per_instance_timeout=timeout,
        log_file=run_dir / "evaluation.log",
    )

    if not success:
        print(f"   ❌ {error}")
        return None

    return parse_eval_results(results_path)


def evaluate_run_sbcli(
    run_dir: Path,
    subset: str,
    workers: int = 1,
    timeout: int = 900,
) -> dict | None:
    """Run sb-cli cloud evaluation, falling back to Docker on failure."""
    from sweagent.utils.sbcli import run_sbcli_evaluation, parse_eval_results

    preds = run_dir / "preds.json"
    if not preds.exists():
        preds = run_dir / "all.preds.json"
    if not preds.exists():
        return None

    success, error, results_path = run_sbcli_evaluation(
        preds_path=preds,
        subset=subset,
        run_id=run_dir.name,
        output_dir=run_dir,
    )

    if not success:
        print(f"   ⚠️  sb-cli {error}")
        print("   🐳 Falling back to Docker...")
        return evaluate_run(run_dir, subset, workers, timeout)

    # parse results using shared parser
    result = parse_eval_results(results_path)
    if result is None:
        print("   ⚠️  Failed to parse sb-cli results")
        print("   🐳 Falling back to Docker...")
        return evaluate_run(run_dir, subset, workers, timeout)

    return result


def update_wandb_run(
    project: str,
    run_name: str,
    n_resolved: int,
    n_evaluated: int,
    n_instances: int,
) -> bool:
    """Update WandB run with evaluation metrics."""
    try:
        api = wandb.Api()
        runs = api.runs(project, filters={"display_name": run_name})
        runs_list = list(runs)
        if not runs_list:
            print(f"   ⚠️  WandB run not found: {run_name}")
            return False

        wb_run = runs_list[0]
        solve_rate = n_resolved / n_instances if n_instances else 0
        eval_pass_rate = n_resolved / n_evaluated if n_evaluated else 0
        eval_coverage = n_evaluated / n_instances if n_instances else 0

        wb_run.summary.update({
            "n_resolved": n_resolved,
            "n_evaluated": n_evaluated,
            "n_instances": n_instances,
            "eval_pass_rate": eval_pass_rate,
            "eval_coverage": eval_coverage,
            "solve_rate": solve_rate,
            "eval_complete": True,
        })
        wb_run.update()

        print(f"   📊 WandB updated: {n_resolved}/{n_evaluated} ({eval_pass_rate:.1%})")
        print(f"      solve_rate: {solve_rate:.1%}, coverage: {eval_coverage:.1%}")
        return True

    except Exception as e:
        print(f"   ❌ Error updating WandB: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Backfill evaluation for runs missing results.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--subset",
        default="verified-mini",
        help="Dataset subset (verified, verified-mini, lite)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Eval workers (keep at 1 for VPS)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Per-instance timeout in seconds",
    )
    parser.add_argument(
        "--project",
        default="ox/the-complexity-trap",
        help="WandB project (entity/project)",
    )
    parser.add_argument(
        "--backend",
        choices=["docker", "sbcli"],
        default="docker",
        help="Evaluation backend: docker (local) or sbcli (cloud)",
    )
    args = parser.parse_args()

    if args.workers < 1:
        print("Error: --workers must be at least 1")
        return 1

    print(f"🔍 Searching for unevaluated runs in trajectories/")
    print(f"   Subset filter: {args.subset}")
    print(f"   Backend: {args.backend}")
    print()

    runs = find_unevaluated_runs(Path("trajectories"), args.subset)
    if not runs:
        print("✅ No unevaluated runs found")
        return 0

    print(f"📋 Found {len(runs)} runs to evaluate")
    print()

    success = 0
    failed = 0

    for run_dir in runs:
        run_name = get_run_name_from_dir(run_dir)
        n_instances = count_instances(run_dir)
        print(f"🚀 {run_name}")
        print(f"   📁 {n_instances} instances")

        if args.backend == "sbcli":
            result = evaluate_run_sbcli(run_dir, args.subset, args.workers, args.timeout)
        else:
            result = evaluate_run(run_dir, args.subset, args.workers, args.timeout)
        if result:
            update_wandb_run(
                args.project,
                run_name,
                result["n_resolved"],
                result["n_evaluated"],
                n_instances,
            )
            success += 1
            print(f"   ✅ Complete")
        else:
            failed += 1
            print(f"   ❌ Failed")
        print()

    print(f"📊 Done: {success} success, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
