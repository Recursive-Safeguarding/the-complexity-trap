"""SWE-bench evaluation via sb-cli (cloud) or Docker (local).

Used by wandb_hook.py and evaluate_missing.py.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import TypedDict


# Dataset mappings for Docker evaluation (HuggingFace paths)
DATASET_MAP_DOCKER = {
    "verified": "SWE-bench/SWE-bench_Verified",
    "verified-mini": "MariusHobbhahn/swe-bench-verified-mini",
    "lite": "SWE-bench/SWE-bench_Lite",
}

# Dataset mappings for sb-cli cloud evaluation
# NOTE: verified-mini intentionally omitted to prevent accidental full-benchmark runs.
# sb-cli doesn't have a mini subset, so mapping to verified would run 500 instances
# instead of 50, causing unexpected costs. Use Docker evaluation for verified-mini.
DATASET_MAP_SBCLI = {
    "verified": "swe-bench_verified",
    "lite": "swe-bench_lite",
}

# Timeout configuration (seconds)
SBCLI_TIMEOUT_SUBMIT = 1800   # 30 min for cloud submission
SBCLI_TIMEOUT_STATUS = 60     # 1 min for status check
SBCLI_TIMEOUT_REPORT = 300    # 5 min for report fetch
SBCLI_POLL_INTERVAL = 30      # 30s between status checks
SBCLI_POLL_MAX_ATTEMPTS = 60  # 60 attempts = 30 min max wait

# Docker evaluation defaults
DOCKER_EVAL_MAX_WORKERS = 4       # parallel workers
DOCKER_EVAL_TIMEOUT = 900         # per-instance timeout (15 min)
DOCKER_EVAL_MIN_PROCESS_TIMEOUT = 7200  # minimum process timeout (2 hours)


class EvalResult(TypedDict):
    """Result from sb-cli or Docker evaluation."""
    n_resolved: int
    n_evaluated: int


def get_dataset_name(subset: str, backend: str = "docker") -> str | None:
    """Map subset to dataset name for the given backend."""
    if backend == "sbcli":
        return DATASET_MAP_SBCLI.get(subset)
    return DATASET_MAP_DOCKER.get(subset, f"SWE-bench/SWE-bench_{subset.title()}")


def check_sbcli_available() -> tuple[bool, str]:
    """Check if sb-cli is installed and SWEBENCH_API_KEY is set."""
    if not os.environ.get("SWEBENCH_API_KEY"):
        return False, "SWEBENCH_API_KEY not set"
    try:
        result = subprocess.run(
            ["sb-cli", "--help"], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return False, f"sb-cli --help failed: {result.stderr.strip()}"
        return True, ""
    except FileNotFoundError:
        return False, "sb-cli not installed"
    except subprocess.TimeoutExpired:
        return False, "sb-cli --help timed out"
    except Exception as e:
        return False, str(e)


def submit_to_sbcli(
    preds_path: Path,
    dataset: str,
    run_id: str,
) -> tuple[bool, str]:
    """Submit predictions to sb-cli cloud."""
    cmd = [
        "sb-cli", "submit", dataset, "test",
        "--predictions_path", str(preds_path),
        "--run_id", run_id,
        "--gen_report",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=SBCLI_TIMEOUT_SUBMIT
        )
        if result.returncode != 0:
            return False, f"exit {result.returncode}: {result.stderr}"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "submission timed out"
    except Exception as e:
        return False, str(e)


def poll_sbcli_status(run_id: str) -> tuple[bool, str]:
    """Poll sb-cli status until complete or failed."""
    # terminal failure states to detect (covers various API conventions)
    failure_states = ("failed", "canceled", "cancelled", "error", "not found", "aborted", "rejected", "timeout", "timed out")

    for attempt in range(SBCLI_POLL_MAX_ATTEMPTS):
        # check immediately on first attempt, then sleep before retries
        if attempt > 0:
            time.sleep(SBCLI_POLL_INTERVAL)
        try:
            result = subprocess.run(
                ["sb-cli", "status", run_id],
                capture_output=True, text=True, timeout=SBCLI_TIMEOUT_STATUS
            )
            if result.returncode != 0:
                return False, f"status check failed (exit {result.returncode}): {result.stderr.strip()}"
            stdout = result.stdout.lower()
            if "completed" in stdout:
                return True, ""
            if any(state in stdout for state in failure_states):
                # extract just the state for cleaner error message
                matched_state = next((s for s in failure_states if s in stdout), "unknown")
                return False, f"evaluation {matched_state}"
        except subprocess.TimeoutExpired:
            return False, "status check timed out"
        except Exception as e:
            return False, f"status check error: {e}"
    return False, f"polling timed out after {SBCLI_POLL_MAX_ATTEMPTS * SBCLI_POLL_INTERVAL}s"


def fetch_sbcli_report(run_id: str, output_path: Path) -> tuple[bool, str]:
    """Download report from sb-cli to output_path."""
    try:
        result = subprocess.run(
            ["sb-cli", "report", run_id, "--output", str(output_path)],
            capture_output=True, text=True, timeout=SBCLI_TIMEOUT_REPORT
        )
        if result.returncode != 0:
            return False, f"report fetch failed (exit {result.returncode}): {result.stderr}"
        if not output_path.exists():
            return False, "report file not created"
        # validate JSON
        try:
            json.loads(output_path.read_text())
        except json.JSONDecodeError:
            return False, "invalid JSON in report"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "report fetch timed out"
    except Exception as e:
        return False, str(e)


def run_sbcli_evaluation(
    preds_path: Path,
    subset: str,
    run_id: str,
    output_dir: Path,
) -> tuple[bool, str, Path | None]:
    """Submit to sb-cli, poll for completion, fetch results."""
    # check availability
    available, err = check_sbcli_available()
    if not available:
        return False, err, None

    # get dataset name
    dataset = get_dataset_name(subset, "sbcli")
    if not dataset:
        return False, f"unknown subset '{subset}' for sb-cli", None

    # ensure output_dir exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # submit
    print(f"   ☁️  Submitting to sb-cli ({dataset})...")
    ok, err = submit_to_sbcli(preds_path, dataset, run_id)
    if not ok:
        return False, f"submit failed: {err}", None

    # poll
    print("   ⏳ Waiting for sb-cli evaluation...")
    ok, err = poll_sbcli_status(run_id)
    if not ok:
        return False, f"polling failed: {err}", None

    # fetch report
    results_path = output_dir / "results.json"
    ok, err = fetch_sbcli_report(run_id, results_path)
    if not ok:
        return False, f"report failed: {err}", None

    return True, "", results_path


def run_docker_evaluation(
    preds_path: Path,
    subset: str,
    run_id: str,
    output_dir: Path | None = None,
    max_workers: int = DOCKER_EVAL_MAX_WORKERS,
    per_instance_timeout: int = DOCKER_EVAL_TIMEOUT,
    log_file: Path | None = None,
    n_instances: int | None = None,
) -> tuple[bool, str, Path | None]:
    """Run swebench evaluation via Docker containers."""
    import shutil
    import sys

    # get dataset name
    dataset = get_dataset_name(subset, "docker")
    if not dataset:
        return False, f"unknown subset '{subset}' for docker", None

    # build command
    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "-p", str(preds_path),
        "-d", dataset,
        "-id", run_id,
        "--max_workers", str(max_workers),
    ]
    # per_instance_timeout > 0: pass to swebench; <= 0: omit (use swebench default)
    if per_instance_timeout > 0:
        cmd.extend(["--timeout", str(per_instance_timeout)])

    # calculate dynamic process timeout
    if n_instances is None:
        try:
            data = json.loads(preds_path.read_text())
            if isinstance(data, list):
                n_instances = len(data)
            elif isinstance(data, dict):
                # Prefer common wrapper keys to avoid severe under-counting.
                for key in ("predictions", "preds", "instances", "data"):
                    wrapped = data.get(key)
                    if isinstance(wrapped, (list, dict)):
                        n_instances = len(wrapped)
                        break
                else:
                    n_instances = len(data)
            else:
                n_instances = 50
        except (json.JSONDecodeError, FileNotFoundError, IsADirectoryError):
            n_instances = 50  # fallback estimate

    # guard against division by zero
    if max_workers < 1:
        max_workers = 1

    # when per_instance_timeout <= 0, use generous default for dynamic calc
    effective_timeout = per_instance_timeout if per_instance_timeout > 0 else DOCKER_EVAL_TIMEOUT
    dynamic_timeout = max(
        DOCKER_EVAL_MIN_PROCESS_TIMEOUT,
        (effective_timeout * n_instances) // max_workers + 1800
    )

    try:
        if log_file:
            with open(log_file, "w") as f:
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    timeout=dynamic_timeout,
                )
        else:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=dynamic_timeout,
            )

        if result.returncode != 0:
            stderr = result.stderr if hasattr(result, 'stderr') and result.stderr else ""
            return False, f"exit {result.returncode}: {stderr[:500]}", None

        reports: list[Path] = []

        # 1) swebench harness typically writes under logs/run_evaluation/<run_id>/
        logs_dir = Path("logs/run_evaluation") / run_id
        if logs_dir.exists():
            reports.extend(logs_dir.glob(f"{run_id}*.json"))
            reports.extend(logs_dir.glob("report.json"))

        # 2) some harness versions write to CWD as <run_dir>.<run_id>.json
        # where <run_dir> is typically preds_path.parent.name.
        candidate_stems = {preds_path.stem, preds_path.parent.name}
        for stem in candidate_stems:
            cwd_report = Path(f"{stem}.{run_id}.json")
            if cwd_report.exists():
                reports.append(cwd_report)
                continue
            reports.extend(Path(".").glob(f"{stem}.*{run_id}*.json"))

        if not reports:
            return False, "no evaluation report found after run_evaluation", None

        # Prefer explicit aggregate reports over per-instance JSON artifacts.
        preferred_reports = [p for p in reports if p.name == "report.json"]
        if not preferred_reports:
            preferred_reports = [p for p in reports if "report" in p.stem.lower()]
        report_pool = preferred_reports or reports

        # Heuristically prefer aggregate reports that contain evaluation totals.
        aggregate_keys = {
            "resolved_ids",
            "resolved",
            "submitted_ids",
            "submitted",
            "applied",
            "unresolved",
        }
        aggregate_reports: list[Path] = []
        for candidate in report_pool:
            try:
                payload = json.loads(candidate.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and aggregate_keys.intersection(payload):
                aggregate_reports.append(candidate)
        if aggregate_reports:
            report_pool = aggregate_reports

        latest_report = max(report_pool, key=lambda p: p.stat().st_mtime)

        # copy to output_dir if specified
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            results_path = output_dir / "results.json"
            shutil.copy(latest_report, results_path)
            return True, "", results_path
        else:
            return True, "", latest_report

    except subprocess.TimeoutExpired:
        return False, f"timed out after {dynamic_timeout}s", None
    except Exception as e:
        return False, str(e), None


def parse_eval_results(results_path: Path) -> EvalResult | None:
    """Parse results.json (handles both sb-cli and Docker formats)."""
    try:
        data = json.loads(results_path.read_text())
    except (json.JSONDecodeError, FileNotFoundError, IsADirectoryError):
        return None

    # handle both resolved_ids and resolved keys (can be list or int)
    resolved = data.get("resolved_ids", data.get("resolved", []))
    if isinstance(resolved, list):
        n_resolved = len(resolved)
    elif isinstance(resolved, int):
        n_resolved = resolved
    else:
        n_resolved = 0

    # handle both submitted_ids and submitted keys (can be list or int)
    submitted = data.get("submitted_ids", data.get("submitted"))
    if isinstance(submitted, list):
        n_evaluated = len(submitted)
    elif isinstance(submitted, int):
        n_evaluated = submitted
    elif "applied" in data:
        applied = data["applied"]
        n_evaluated = len(applied) if isinstance(applied, list) else applied if isinstance(applied, int) else 0
    else:
        # fallback: resolved + unresolved (handle both as list or int)
        unresolved = data.get("unresolved", [])
        n_unresolved = len(unresolved) if isinstance(unresolved, list) else unresolved if isinstance(unresolved, int) else 0
        n_evaluated = n_resolved + n_unresolved

    return {"n_resolved": n_resolved, "n_evaluated": n_evaluated}
