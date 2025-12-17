#!/usr/bin/env python3
"""Parse trajectory files to extract metrics for WandB logging.

Usage:
    python scripts/parse_trajectory.py path/to/trajectory
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def extract_metrics_from_trajectory(traj_path: Path) -> dict[str, Any]:
    with open(traj_path) as f:
        data = json.load(f)

    info = data.get("info", {})
    trajectory = data.get("trajectory", [])
    model_stats = info.get("model_stats", {})
    agent_stats = info.get("agent_model_stats", model_stats)
    summary_stats = info.get("summary_model_stats") or {}
    agent_tokens = agent_stats.get("tokens", {})
    summary_tokens = summary_stats.get("tokens", {}) if summary_stats else {}
    agent_cost = agent_stats.get("instance_cost", 0) or 0
    summary_cost = summary_stats.get("instance_cost", 0) if summary_stats else 0

    return {
        "instance_id": traj_path.stem,
        "exit_status": info.get("exit_status", "unknown"),
        "submitted": info.get("exit_status") == "submitted",
        "n_turns": len(trajectory),
        "total_cost": agent_cost + summary_cost,
        "agent_cost": agent_cost,
        "summary_cost": summary_cost,
        "agent_api_calls": agent_stats.get("api_calls", 0) or 0,
        "summary_api_calls": summary_stats.get("api_calls", 0) if summary_stats else 0,
        "raw_input_tokens": agent_tokens.get("raw_input", 0) or 0,
        "cached_input_tokens": agent_tokens.get("cached_input", 0) or 0,
        "output_tokens": agent_tokens.get("output", 0) or 0,
        "summary_input_tokens": summary_tokens.get("raw_input", 0) or 0,
        "summary_output_tokens": summary_tokens.get("output", 0) or 0,
    }


def extract_metrics_from_directory(traj_dir: Path) -> dict[str, Any]:
    traj_files = list(traj_dir.rglob("*.traj"))

    if not traj_files:
        return {"error": "No trajectory files found", "n_instances": 0}

    all_metrics = []
    for traj_file in traj_files:
        try:
            metrics = extract_metrics_from_trajectory(traj_file)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"Error parsing {traj_file}: {e}")

    if not all_metrics:
        return {"error": "Failed to parse any trajectories", "n_instances": 0}

    n_instances = len(all_metrics)
    n_submitted = sum(1 for m in all_metrics if m.get("submitted"))

    total_cost = sum(m.get("total_cost", 0) for m in all_metrics)
    total_agent_cost = sum(m.get("agent_cost", 0) for m in all_metrics)
    total_summary_cost = sum(m.get("summary_cost", 0) for m in all_metrics)

    total_turns = sum(m.get("n_turns", 0) for m in all_metrics)
    total_api_calls = sum(m.get("agent_api_calls", 0) for m in all_metrics)
    total_raw_input = sum(m.get("raw_input_tokens", 0) for m in all_metrics)
    total_cached_input = sum(m.get("cached_input_tokens", 0) for m in all_metrics)
    total_output = sum(m.get("output_tokens", 0) for m in all_metrics)

    return {
        "n_instances": n_instances,
        "n_submitted": n_submitted,
        "submission_rate": n_submitted / n_instances if n_instances > 0 else 0,
        "total_cost": total_cost,
        "avg_cost": total_cost / n_instances if n_instances > 0 else 0,
        "total_agent_cost": total_agent_cost,
        "total_summary_cost": total_summary_cost,
        "summary_cost_fraction": total_summary_cost / total_cost if total_cost > 0 else 0,
        "total_turns": total_turns,
        "avg_turns": total_turns / n_instances if n_instances > 0 else 0,
        "total_api_calls": total_api_calls,
        "avg_api_calls": total_api_calls / n_instances if n_instances > 0 else 0,
        "total_raw_input_tokens": total_raw_input,
        "total_cached_input_tokens": total_cached_input,
        "total_output_tokens": total_output,
        "cache_hit_rate": total_cached_input / (total_raw_input + total_cached_input) if (total_raw_input + total_cached_input) > 0 else 0,
        "instances": all_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Parse trajectory files for metrics")
    parser.add_argument("trajectory_dir", type=Path, help="Path to trajectory directory")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    if not args.trajectory_dir.exists():
        print(f"Error: {args.trajectory_dir} does not exist")
        return 1

    metrics = extract_metrics_from_directory(args.trajectory_dir)

    if args.json:
        output = {k: v for k, v in metrics.items() if k != "instances"}
        print(json.dumps(output, indent=2))
    else:
        print(f"Trajectory Directory: {args.trajectory_dir}")
        print("=" * 60)
        print(f"Instances: {metrics['n_instances']}")
        print(f"Submitted: {metrics['n_submitted']} ({metrics['submission_rate']:.1%})")
        print()
        print(f"Total Cost: ${metrics['total_cost']:.4f}")
        print(f"  Agent Cost: ${metrics['total_agent_cost']:.4f}")
        print(f"  Summary Cost: ${metrics['total_summary_cost']:.4f} ({metrics['summary_cost_fraction']:.1%})")
        print()
        print(f"Avg Cost/Instance: ${metrics['avg_cost']:.4f}")
        print(f"Avg Turns/Instance: {metrics['avg_turns']:.1f}")
        print(f"Avg API Calls/Instance: {metrics['avg_api_calls']:.1f}")
        print()
        print(f"Total Tokens: {metrics['total_raw_input_tokens'] + metrics['total_cached_input_tokens'] + metrics['total_output_tokens']:,}")
        print(f"  Raw Input: {metrics['total_raw_input_tokens']:,}")
        print(f"  Cached Input: {metrics['total_cached_input_tokens']:,} ({metrics['cache_hit_rate']:.1%} cache hit)")
        print(f"  Output: {metrics['total_output_tokens']:,}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
