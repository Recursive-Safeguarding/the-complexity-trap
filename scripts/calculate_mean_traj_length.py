#!/usr/bin/env python3

import argparse
import json
import yaml
import statistics
from pathlib import Path
from typing import Dict, List, Set, Tuple


def load_exit_statuses(exit_statuses_path: Path) -> Dict[str, List[str]]:
    """Load exit statuses from YAML file."""
    with open(exit_statuses_path, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('instances_by_exit_status', {})


def count_assistant_messages(traj_path: Path) -> int:
    """Count assistant messages in a trajectory file."""
    try:
        with open(traj_path, 'r') as f:
            traj_data = json.load(f)

        assistant_count = 0
        history = traj_data.get('history', [])
        for message in history:
            if message.get('role') == 'assistant':
                assistant_count += 1

        return assistant_count
    except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
        print(f"Error reading {traj_path}: {e}")
        return 0


def get_submitted_instances(exit_statuses: Dict[str, List[str]]) -> Set[str]:
    """Get set of submitted instance names."""
    return set(exit_statuses.get('submitted', []))


def calculate_statistics(lengths: List[int]) -> Dict[str, float]:
    """Calculate statistics for a list of lengths."""
    if not lengths:
        return {'count': 0, 'mean': 0, 'std': 0, 'min': 0, 'max': 0}

    return {
        'count': len(lengths),
        'mean': statistics.mean(lengths),
        'std': statistics.stdev(lengths) if len(lengths) > 1 else 0,
        'min': min(lengths),
        'max': max(lengths)
    }


def print_exit_status_stats(exit_statuses: Dict[str, List[str]], total_instances: int):
    """Print exit status statistics in percentages."""
    print("Exit status distribution:")
    for status, instances in exit_statuses.items():
        count = len(instances)
        percentage = (count / total_instances) * 100
        print(f"  {status}: {count} ({percentage:.1f}%)")
    print()


def calculate_averages(trajectories_dir: Path, exit_statuses_path: Path) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Calculate statistics for all and submitted instances."""

    exit_statuses = load_exit_statuses(exit_statuses_path)
    submitted_instances = get_submitted_instances(exit_statuses)

    all_lengths = []
    submitted_lengths = []

    for instance_dir in trajectories_dir.iterdir():
        if not instance_dir.is_dir():
            continue

        instance_name = instance_dir.name
        traj_file = instance_dir / f"{instance_name}.traj"

        if not traj_file.exists():
            continue

        length = count_assistant_messages(traj_file)
        all_lengths.append(length)

        if instance_name in submitted_instances:
            submitted_lengths.append(length)

    all_stats = calculate_statistics(all_lengths)
    submitted_stats = calculate_statistics(submitted_lengths)

    return all_stats, submitted_stats, exit_statuses


def main():
    parser = argparse.ArgumentParser(
        description='Calculate mean trajectory length by counting assistant messages',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s trajectories/slinko/2025-08-10_11-02-04__swesmith_infer__hosted_vllm_QwenMasked-v2-999__t-0.00__p-1.00__c-0.00___swe_bench_verified_test
  %(prog)s --concise trajectories/slinko/2025-08-10_11-02-04__swesmith_infer__hosted_vllm_QwenMasked-v2-999__t-0.00__p-1.00__c-0.00___swe_bench_verified_test
        """
    )
    parser.add_argument('trajectories_dir', type=Path, help='Path to trajectories directory')
    parser.add_argument('--exit-statuses', type=Path, default=None,
                        help='Path to run_batch_exit_statuses.yaml (default: trajectories_dir/run_batch_exit_statuses.yaml)')
    parser.add_argument('--concise', action='store_true', help='Show only mean values')

    args = parser.parse_args()

    if not args.trajectories_dir.exists():
        print(f"Error: Trajectories directory {args.trajectories_dir} does not exist")
        return 1

    if args.exit_statuses is None:
        args.exit_statuses = args.trajectories_dir / 'run_batch_exit_statuses.yaml'

    if not args.exit_statuses.exists():
        print(f"Error: Exit statuses file {args.exit_statuses} does not exist")
        return 1

    try:
        all_stats, submitted_stats, exit_statuses = calculate_averages(args.trajectories_dir, args.exit_statuses)

        if args.concise:
            print(f"All: {all_stats['mean']:.2f} (n={all_stats['count']})")
            print(f"Submitted: {submitted_stats['mean']:.2f} (n={submitted_stats['count']})")
        else:
            print(f"All instances:")
            print(f"  Count: {all_stats['count']}")
            print(f"  Mean: {all_stats['mean']:.2f}")
            print(f"  Std: {all_stats['std']:.2f}")
            print(f"  Min: {all_stats['min']}")
            print(f"  Max: {all_stats['max']}")
            print()
            print(f"Submitted instances:")
            print(f"  Count: {submitted_stats['count']}")
            print(f"  Mean: {submitted_stats['mean']:.2f}")
            print(f"  Std: {submitted_stats['std']:.2f}")
            print(f"  Min: {submitted_stats['min']}")
            print(f"  Max: {submitted_stats['max']}")
            print()
            print_exit_status_stats(exit_statuses, all_stats['count'])

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
