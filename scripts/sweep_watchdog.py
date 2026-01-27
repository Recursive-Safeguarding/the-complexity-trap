#!/usr/bin/env python3
"""Watchdog script for monitoring SWE-agent sweep progress.

Detects stalled sweeps (no progress for N minutes) and optionally kills/restarts them.
Process killing uses project-specific patterns to avoid affecting other projects.

IMPORTANT: Container killing (--kill-containers) is DISABLED by default because
find_orphaned_containers() filters only by 'swebench/sweb.eval' image, which would
kill ALL swebench containers from ANY project on the VPS. Only enable this flag if
you're certain no other projects are running swebench evaluations.

Usage:
    # Monitor mode (continuous) - kills project processes only
    python scripts/sweep_watchdog.py --monitor \
      --sweep-id ox/the-complexity-trap/SWEEP_ID \
      --timeout 30 \
      --restart

    # Monitor mode with container killing (DANGEROUS on multi-project VPS)
    python scripts/sweep_watchdog.py --monitor \
      --sweep-id ox/the-complexity-trap/SWEEP_ID \
      --kill-containers

    # One-shot check
    python scripts/sweep_watchdog.py --check \
      --trajectory-dir trajectories/root/RUN_NAME

    # Kill stuck processes only (no containers)
    python scripts/sweep_watchdog.py --kill-stuck

    # Kill stuck processes AND containers
    python scripts/sweep_watchdog.py --kill-stuck --kill-containers

    # Dry run (show what would be killed)
    python scripts/sweep_watchdog.py --kill-stuck --dry-run
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import NamedTuple

# project root for process matching
PROJECT_ROOT = Path(__file__).parent.parent
PROJECT_NAME = PROJECT_ROOT.name  # "the-complexity-trap"

# default stall timeout in minutes
DEFAULT_TIMEOUT_MINUTES = 30

# polling interval for monitor mode (seconds)
POLL_INTERVAL_SECONDS = 60


class RunProgress(NamedTuple):
    """Progress info for a single run."""
    run_dir: Path
    instance_count: int
    mtime: float
    mtime_str: str


def load_yaml_simple(path: Path) -> dict:
    """Simple YAML parser (no PyYAML dependency)."""
    content = path.read_text()
    result = {"instances_by_exit_status": {}, "total_cost": 0}

    current_status = None
    for line in content.split("\n"):
        if line.startswith("    ") and line.strip().endswith(":") and not line.startswith("        "):
            current_status = line.strip().rstrip(":")
            result["instances_by_exit_status"][current_status] = []
        elif line.startswith("    - ") and current_status:
            instance_id = line.strip().lstrip("- ")
            result["instances_by_exit_status"][current_status].append(instance_id)
        elif line.startswith("total_cost:"):
            try:
                result["total_cost"] = float(line.split(":")[1].strip())
            except (ValueError, IndexError):
                pass

    return result


def find_active_runs(base_dir: Path) -> list[Path]:
    """Find run dirs with run_batch_exit_statuses.yaml, sorted by mtime descending."""
    if not base_dir.exists():
        return []

    runs = []
    for user_dir in base_dir.iterdir():
        if not user_dir.is_dir():
            continue
        for run_dir in user_dir.iterdir():
            if not run_dir.is_dir():
                continue
            status_file = run_dir / "run_batch_exit_statuses.yaml"
            if status_file.exists():
                runs.append(run_dir)

    for run_dir in base_dir.iterdir():
        if run_dir.is_dir():
            status_file = run_dir / "run_batch_exit_statuses.yaml"
            if status_file.exists() and run_dir not in runs:
                runs.append(run_dir)

    runs.sort(key=lambda p: (p / "run_batch_exit_statuses.yaml").stat().st_mtime, reverse=True)
    return runs


def get_progress(run_dir: Path) -> RunProgress | None:
    """Get progress info, or None if unavailable."""
    status_file = run_dir / "run_batch_exit_statuses.yaml"
    if not status_file.exists():
        return None

    try:
        stat = status_file.stat()
        mtime = stat.st_mtime
        mtime_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

        data = load_yaml_simple(status_file)
        instance_count = sum(
            len(instances)
            for instances in data.get("instances_by_exit_status", {}).values()
        )

        return RunProgress(
            run_dir=run_dir,
            instance_count=instance_count,
            mtime=mtime,
            mtime_str=mtime_str,
        )
    except Exception as e:
        print(f"Warning: Failed to read {status_file}: {e}")
        return None


def is_stalled(run_dir: Path, timeout_minutes: int) -> tuple[bool, str]:
    """Check if run is stalled. Returns (is_stalled, message)."""
    progress = get_progress(run_dir)
    if progress is None:
        return False, "No status file found"

    now = time.time()
    age_minutes = (now - progress.mtime) / 60

    if age_minutes > timeout_minutes:
        return True, (
            f"STALLED: No progress for {age_minutes:.1f} minutes "
            f"(threshold: {timeout_minutes} min). "
            f"Last update: {progress.mtime_str}, instances: {progress.instance_count}"
        )

    return False, (
        f"OK: Last update {age_minutes:.1f} minutes ago "
        f"(threshold: {timeout_minutes} min). "
        f"Instances: {progress.instance_count}"
    )


def find_project_processes(dry_run: bool = False) -> list[dict]:
    """Find this project's Python processes by name pattern."""
    processes = []
    patterns = [
        f"{PROJECT_NAME}.*run_sweep",
        f"{PROJECT_NAME}.*wandb.*agent",
        f"{PROJECT_NAME}.*sweagent",
    ]

    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        for line in result.stdout.split("\n"):
            if "python" not in line.lower():
                continue

            for pattern in patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            pid = int(parts[1])
                            processes.append({
                                "pid": pid,
                                "command": " ".join(parts[10:]) if len(parts) > 10 else line,
                                "pattern": pattern,
                                "full_line": line,
                            })
                        except ValueError:
                            pass
                    break
    except Exception as e:
        print(f"Warning: Failed to list processes: {e}")

    return processes


def find_orphaned_containers() -> list[dict]:
    """Find swebench/sweb.eval containers. WARNING: Matches ALL projects on VPS."""
    containers = []

    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "ancestor=swebench/sweb.eval", "--format", "{{.ID}}\t{{.Image}}\t{{.Status}}"],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode != 0:
            # docker might not be available
            return containers

        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                containers.append({
                    "container_id": parts[0],
                    "image": parts[1] if len(parts) > 1 else "unknown",
                    "status": parts[2] if len(parts) > 2 else "unknown",
                })
    except FileNotFoundError:
        # docker not installed
        pass
    except Exception as e:
        print(f"Warning: Failed to list Docker containers: {e}")

    return containers


def kill_project_processes(dry_run: bool = False) -> int:
    """Kill this project's Python processes. Never uses blanket killall/pkill."""
    processes = find_project_processes(dry_run=dry_run)

    if not processes:
        print("No project-specific processes found.")
        return 0

    print(f"Found {len(processes)} project-specific process(es):")
    for proc in processes:
        print(f"  PID {proc['pid']}: {proc['command'][:80]}...")

    if dry_run:
        print("\n[DRY RUN] Would kill the above processes.")
        return 0

    killed = 0
    for proc in processes:
        try:
            os.kill(proc["pid"], 15)  # SIGTERM
            print(f"  Killed PID {proc['pid']}")
            killed += 1
        except ProcessLookupError:
            print(f"  PID {proc['pid']} already terminated")
        except PermissionError:
            print(f"  Permission denied to kill PID {proc['pid']}")
        except Exception as e:
            print(f"  Failed to kill PID {proc['pid']}: {e}")

    return killed


def kill_orphaned_containers(dry_run: bool = False) -> int:
    """Kill orphaned swebench containers."""
    containers = find_orphaned_containers()

    if not containers:
        print("No orphaned swebench containers found.")
        return 0

    print(f"Found {len(containers)} swebench container(s):")
    for container in containers:
        print(f"  {container['container_id']}: {container['image']} ({container['status']})")

    if dry_run:
        print("\n[DRY RUN] Would kill the above containers.")
        return 0

    killed = 0
    for container in containers:
        try:
            result = subprocess.run(
                ["docker", "kill", container["container_id"]],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0:
                print(f"  Killed container {container['container_id']}")
                killed += 1
            else:
                print(f"  Failed to kill container {container['container_id']}: {result.stderr}")
        except Exception as e:
            print(f"  Failed to kill container {container['container_id']}: {e}")

    return killed


def restart_sweep_agent(sweep_id: str, tmux_session: str = "sweep", dry_run: bool = False) -> bool:
    """Restart wandb agent in tmux."""
    if not sweep_id:
        print("No sweep ID provided, cannot restart.")
        return False

    activate_cmd = f"cd {PROJECT_ROOT} && source .venv/bin/activate && set -a && source .env && set +a"
    agent_cmd = f"wandb agent {sweep_id}"
    full_cmd = f"{activate_cmd} && {agent_cmd}"

    print(f"Restarting sweep agent in tmux session '{tmux_session}'...")
    print(f"  Sweep ID: {sweep_id}")
    print(f"  Command: {agent_cmd}")

    if dry_run:
        print("\n[DRY RUN] Would restart the sweep agent.")
        return True

    try:
        # check if tmux session exists
        result = subprocess.run(
            ["tmux", "has-session", "-t", tmux_session],
            capture_output=True,
            timeout=5,
        )
        session_exists = result.returncode == 0

        if session_exists:
            # kill existing session
            subprocess.run(
                ["tmux", "kill-session", "-t", tmux_session],
                capture_output=True,
                timeout=10,
            )
            print(f"  Killed existing tmux session '{tmux_session}'")

        # start new session with the command
        result = subprocess.run(
            ["tmux", "new-session", "-d", "-s", tmux_session, f"bash -c '{full_cmd}'"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            print(f"  Started new tmux session '{tmux_session}'")
            return True
        else:
            print(f"  Failed to start tmux session: {result.stderr}")
            return False
    except FileNotFoundError:
        print("  Error: tmux not installed")
        return False
    except Exception as e:
        print(f"  Error restarting sweep agent: {e}")
        return False


def monitor_loop(
    sweep_id: str | None,
    timeout_minutes: int,
    restart: bool,
    tmux_session: str,
    trajectories_dir: Path,
    dry_run: bool = False,
    kill_containers: bool = False,
) -> None:
    """Continuous monitoring loop.

    Checks for stalled runs every POLL_INTERVAL_SECONDS and takes action.
    """
    print(f"Starting watchdog monitor...")
    print(f"  Sweep ID: {sweep_id or 'not specified'}")
    print(f"  Stall timeout: {timeout_minutes} minutes")
    print(f"  Auto-restart: {restart}")
    print(f"  Tmux session: {tmux_session}")
    print(f"  Trajectories dir: {trajectories_dir}")
    print(f"  Poll interval: {POLL_INTERVAL_SECONDS} seconds")
    print()

    last_restart_time = 0
    min_restart_interval = 300  # don't restart more than once per 5 minutes

    while True:
        now = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # find most recent active run
        runs = find_active_runs(trajectories_dir)

        if not runs:
            print(f"[{timestamp}] No active runs found in {trajectories_dir}")
            time.sleep(POLL_INTERVAL_SECONDS)
            continue

        # check the most recent run
        latest_run = runs[0]
        is_stuck, message = is_stalled(latest_run, timeout_minutes)

        print(f"[{timestamp}] {latest_run.name}: {message}")

        if is_stuck:
            # check if we recently restarted
            if now - last_restart_time < min_restart_interval:
                wait_time = min_restart_interval - (now - last_restart_time)
                print(f"  Skipping action: restarted {int(now - last_restart_time)}s ago (min interval: {min_restart_interval}s)")
                print(f"  Will check again after cooldown ({int(wait_time)}s)")
            else:
                print(f"\n{'='*60}")
                print(f"TAKING ACTION: Run appears stalled")
                print(f"{'='*60}\n")

                # kill stuck processes
                procs_killed = kill_project_processes(dry_run=dry_run)

                # only kill containers if explicitly requested (dangerous on multi-project VPS)
                containers_killed = 0
                if kill_containers:
                    containers_killed = kill_orphaned_containers(dry_run=dry_run)

                print(f"\nKilled {procs_killed} process(es), {containers_killed} container(s)")

                # restart if requested
                if restart and sweep_id:
                    time.sleep(5)  # brief pause before restart
                    success = restart_sweep_agent(sweep_id, tmux_session, dry_run=dry_run)
                    if success:
                        last_restart_time = now
                        print(f"Sweep agent restarted at {datetime.now().strftime('%H:%M:%S')}")

                print(f"\n{'='*60}\n")

        time.sleep(POLL_INTERVAL_SECONDS)


def check_single_run(trajectory_dir: Path, timeout_minutes: int) -> int:
    """Check a single run directory and return exit code (0=ok, 1=stalled)."""
    is_stuck, message = is_stalled(trajectory_dir, timeout_minutes)
    print(message)
    return 1 if is_stuck else 0


def main():
    parser = argparse.ArgumentParser(
        description="Watchdog for SWE-agent sweeps - detects and handles stalls",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--monitor",
        action="store_true",
        help="Continuous monitoring mode",
    )
    mode_group.add_argument(
        "--check",
        action="store_true",
        help="One-shot check of a trajectory directory",
    )
    mode_group.add_argument(
        "--kill-stuck",
        action="store_true",
        help="Kill project-specific stuck processes",
    )
    mode_group.add_argument(
        "--list-processes",
        action="store_true",
        help="List project-specific processes (no action)",
    )

    # common options
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_MINUTES,
        help=f"Stall timeout in minutes (default: {DEFAULT_TIMEOUT_MINUTES})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it",
    )

    # monitor mode options
    parser.add_argument(
        "--sweep-id",
        help="WandB sweep ID for restart (e.g., ox/the-complexity-trap/abc123)",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Auto-restart sweep agent after killing stuck processes",
    )
    parser.add_argument(
        "--tmux-session",
        default="sweep",
        help="Tmux session name for restart (default: sweep)",
    )
    parser.add_argument(
        "--trajectories-dir",
        type=Path,
        default=PROJECT_ROOT / "trajectories",
        help="Base directory for trajectories",
    )
    parser.add_argument(
        "--kill-containers",
        action="store_true",
        help="Also kill orphaned swebench containers (DANGEROUS on multi-project VPS)",
    )

    # check mode options
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        help="Specific trajectory directory to check",
    )

    args = parser.parse_args()

    if args.monitor:
        if args.restart and not args.sweep_id:
            print("Warning: --restart requires --sweep-id. Restart will be skipped.")

        try:
            monitor_loop(
                sweep_id=args.sweep_id,
                timeout_minutes=args.timeout,
                restart=args.restart,
                tmux_session=args.tmux_session,
                trajectories_dir=args.trajectories_dir,
                dry_run=args.dry_run,
                kill_containers=args.kill_containers,
            )
        except KeyboardInterrupt:
            print("\nWatchdog stopped.")
            return 0

    elif args.check:
        if not args.trajectory_dir:
            parser.error("--check requires --trajectory-dir")

        return check_single_run(args.trajectory_dir, args.timeout)

    elif args.kill_stuck:
        procs_killed = kill_project_processes(dry_run=args.dry_run)

        # only kill containers if explicitly requested (consistent with --monitor mode)
        containers_killed = 0
        if args.kill_containers:
            containers_killed = kill_orphaned_containers(dry_run=args.dry_run)

        if not args.dry_run:
            print(f"\nTotal: killed {procs_killed} process(es), {containers_killed} container(s)")

        return 0

    elif args.list_processes:
        processes = find_project_processes(dry_run=True)
        containers = find_orphaned_containers()

        print(f"\n{PROJECT_NAME} processes:")
        if processes:
            for proc in processes:
                print(f"  PID {proc['pid']}: {proc['command'][:80]}...")
        else:
            print("  (none)")

        print(f"\nSwebench containers:")
        if containers:
            for container in containers:
                print(f"  {container['container_id']}: {container['image']} ({container['status']})")
        else:
            print("  (none)")

        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
