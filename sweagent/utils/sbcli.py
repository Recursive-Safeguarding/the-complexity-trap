"""SWE-bench evaluation via sb-cli (cloud) or Docker (local).

Used by wandb_hook.py and evaluate_missing.py.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import TypedDict


# Dataset mappings for Docker evaluation (HuggingFace paths)
DATASET_MAP_DOCKER = {
    "verified": "SWE-bench/SWE-bench_Verified",
    "verified-mini": "MariusHobbhahn/swe-bench-verified-mini",
    "lite": "SWE-bench/SWE-bench_Lite",
}

# Dataset mappings for sb-cli cloud evaluation
# NOTE: sb-cli does not have a "verified-mini" subset. However, sb-cli evaluates only
# the instance IDs present in the submitted predictions file (and can be further
# constrained via --instance_ids). We map "verified-mini" -> swe-bench_verified and
# add a hard guardrail to prevent accidental 500-instance submissions.
DATASET_MAP_SBCLI = {
    "verified": "swe-bench_verified",
    "verified-mini": "swe-bench_verified",
    "lite": "swe-bench_lite",
}

# Timeout configuration (seconds)
SBCLI_TIMEOUT_SUBMIT = 7200   # 2h wall-time cap for sb-cli submit+wait+report

# Guardrails (to prevent accidental full-benchmark evaluation)
MAX_MINI_INSTANCES = 60

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

def _extract_instance_ids_from_preds(preds_path: Path) -> list[str]:
    """Extract instance IDs from a predictions file.

    Supports:
    - JSON list: [{"instance_id": "...", ...}, ...]
    - JSON dict keyed by instance_id: {"id": {...}, ...}
    - JSON dict wrappers: {"predictions": [...]} / {"preds": [...]} / {"data": [...]}
    - JSONL: one JSON object per line (must include instance_id)
    """
    if not preds_path.exists():
        return []

    def _dedupe_preserve_order(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for x in items:
            if x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    try:
        raw_text = preds_path.read_text()
    except OSError:
        return []

    # JSONL support (rare in this repo, but cheap to handle)
    if preds_path.suffix == ".jsonl":
        ids: list[str] = []
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                iid = obj.get("instance_id")
                if isinstance(iid, str):
                    ids.append(iid)
        return _dedupe_preserve_order(ids)

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return []

    # Unwrap common wrappers
    if isinstance(data, dict):
        for key in ("predictions", "preds", "instances", "data"):
            wrapped = data.get(key)
            if isinstance(wrapped, (list, dict)):
                data = wrapped
                break

    ids: list[str] = []
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            iid = item.get("instance_id")
            if isinstance(iid, str):
                ids.append(iid)
    elif isinstance(data, dict):
        # dict keyed by instance_id
        for k in data.keys():
            if isinstance(k, str):
                ids.append(k)

    return _dedupe_preserve_order(ids)

def submit_to_sbcli(
    preds_path: Path,
    dataset: str,
    run_id: str,
    output_dir: Path,
    *,
    instance_ids: list[str] | None = None,
) -> tuple[bool, str]:
    """Submit predictions to sb-cli cloud (waits for evaluation and writes a report).

    sb-cli's `submit` command can (optionally) wait for evaluation and generate a report.
    We use it as a single-shot "submit+wait+report" operation.
    """
    cmd: list[str] = [
        "sb-cli", "submit", dataset, "test",
        "--predictions_path", str(preds_path),
        "--run_id", run_id,
        "--output_dir", str(output_dir),
        "--overwrite", "1",
        "--verify_submission", "1",
        "--wait_for_evaluation", "1",
        "--gen_report", "1",
    ]
    if instance_ids:
        cmd.extend(["--instance_ids", ",".join(instance_ids)])
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

def _load_sbcli_report_payload(sbcli_out_dir: Path, *, dataset: str, run_id: str) -> tuple[dict | None, dict | None]:
    """Load sb-cli report and optional response payload from output directory."""
    # New sb-cli writes:
    #   <subset>__test__<run_id>.json
    #   <subset>__test__<run_id>.response.json (optional)
    base = f"{dataset}__test__{run_id}"
    report_path = sbcli_out_dir / f"{base}.json"
    response_path = sbcli_out_dir / f"{base}.response.json"

    report = None
    response = None
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text())
        except Exception:
            report = None
    if response_path.exists():
        try:
            response = json.loads(response_path.read_text())
        except Exception:
            response = None

    # Fallback: try to find the newest report-like JSON in the directory.
    if report is None and sbcli_out_dir.exists():
        candidates = sorted(sbcli_out_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for cand in candidates:
            if cand.name.endswith(".response.json"):
                continue
            try:
                payload = json.loads(cand.read_text())
            except Exception:
                continue
            if isinstance(payload, dict) and ("resolved_instances" in payload or "submitted_instances" in payload):
                report = payload
                break

    return report, response

def run_sbcli_evaluation(
    preds_path: Path,
    subset: str,
    run_id: str,
    output_dir: Path,
) -> tuple[bool, str, Path | None]:
    """Submit to sb-cli and write results.json under output_dir.

    For sb-cli we always write our own `results.json` (even if sb-cli writes its own
    report files) so downstream tooling can rely on stable keys.
    """
    # check availability
    available, err = check_sbcli_available()
    if not available:
        return False, err, None

    # get dataset name
    dataset = get_dataset_name(subset, "sbcli")
    if not dataset:
        return False, f"unknown subset '{subset}' for sb-cli", None

    instance_ids = _extract_instance_ids_from_preds(preds_path)
    if subset == "verified-mini" and not instance_ids:
        return (
            False,
            "guardrail: refusing sb-cli evaluation for subset=verified-mini with 0 extracted instance IDs. "
            "This likely means preds.json is malformed or missing instance_id fields; refusing to risk a large submission.",
            None,
        )
    if subset == "verified-mini" and len(instance_ids) > MAX_MINI_INSTANCES:
        return (
            False,
            f"guardrail: refusing sb-cli evaluation for subset=verified-mini with {len(instance_ids)} predictions "
            f"(cap={MAX_MINI_INSTANCES}). This likely indicates a non-mini run; refusing to risk a large submission.",
            None,
        )

    # ensure output_dir exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # submit+wait+report
    sbcli_out_dir = output_dir / "sb-cli-reports"
    sbcli_out_dir.mkdir(parents=True, exist_ok=True)
    print(f"   ☁️  Submitting to sb-cli ({dataset})...")
    ok, err = submit_to_sbcli(
        preds_path,
        dataset,
        run_id,
        sbcli_out_dir,
        instance_ids=instance_ids if subset == "verified-mini" else None,
    )
    if not ok:
        return False, f"submit failed: {err}", None

    report, response = _load_sbcli_report_payload(sbcli_out_dir, dataset=dataset, run_id=run_id)
    if not isinstance(report, dict):
        return False, "sb-cli did not produce a readable report JSON", None

    # Build a stable results.json payload for downstream tools.
    # Note: sb-cli report provides counts; response may include per-instance lists.
    resolved_ids: list[str] = []
    submitted_ids: list[str] = list(instance_ids)
    # Prefer IDs from the sb-cli report itself when available.
    report_resolved_ids = report.get("resolved_ids")
    if isinstance(report_resolved_ids, list) and all(isinstance(x, str) for x in report_resolved_ids):
        resolved_ids = report_resolved_ids
    report_submitted_ids = report.get("submitted_ids")
    if isinstance(report_submitted_ids, list) and all(isinstance(x, str) for x in report_submitted_ids):
        submitted_ids = report_submitted_ids
    if isinstance(response, dict):
        # Heuristic extraction for per-instance IDs if the API returns them.
        for key in ("resolved_ids", "resolved", "resolved_instances"):
            val = response.get(key)
            if isinstance(val, list) and all(isinstance(x, str) for x in val):
                resolved_ids = val
                break
        for key in ("submitted_ids", "submitted", "submitted_instances", "completed_ids", "completed"):
            val = response.get(key)
            if isinstance(val, list) and all(isinstance(x, str) for x in val):
                submitted_ids = val
                break

    results_path = output_dir / "results.json"
    results_payload = {
        "backend": "sbcli",
        "subset": subset,
        "dataset": dataset,
        "split": "test",
        "run_id": run_id,
        "submitted_ids": submitted_ids,
        "resolved_ids": resolved_ids,
        "sbcli_report": report,
    }
    # Provide stable top-level counts for downstream consumers.
    report_resolved_n = report.get("resolved_instances")
    report_submitted_n = report.get("submitted_instances")
    results_payload["resolved_instances"] = (
        report_resolved_n if isinstance(report_resolved_n, int) else len(resolved_ids)
    )
    results_payload["submitted_instances"] = (
        report_submitted_n if isinstance(report_submitted_n, int) else len(submitted_ids)
    )
    # Keep response for debugging if it exists (can contain per-instance metadata).
    if isinstance(response, dict):
        results_payload["sbcli_response"] = response

    results_path.write_text(json.dumps(results_payload, indent=2, sort_keys=True))

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

        # Persist a stable, self-describing results.json contract for downstream tools.
        #
        # Many parts of this repo expect results.json to contain at least:
        # - submitted_ids: list[str]
        # - resolved_ids:  list[str]
        # Some older harnesses also write these under "submitted"/"resolved".
        #
        # We keep the original swebench report payload under `docker_report` so callers
        # can still inspect the raw evaluator output when debugging.
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            results_path = output_dir / "results.json"
            try:
                docker_report = json.loads(latest_report.read_text())
            except Exception:
                docker_report = None

            submitted_ids: list[str] = []
            resolved_ids: list[str] = []
            if isinstance(docker_report, dict):
                # Prefer canonical list fields.
                for key in ("resolved_ids", "resolved"):
                    val = docker_report.get(key)
                    if isinstance(val, list) and all(isinstance(x, str) for x in val):
                        resolved_ids = val
                        break
                for key in ("submitted_ids", "submitted"):
                    val = docker_report.get(key)
                    if isinstance(val, list) and all(isinstance(x, str) for x in val):
                        submitted_ids = val
                        break

                # Fallback: reconstruct evaluated IDs from resolved+unresolved when available.
                if not submitted_ids:
                    unresolved = docker_report.get("unresolved")
                    if isinstance(unresolved, list) and all(isinstance(x, str) for x in unresolved):
                        submitted_ids = list(dict.fromkeys([*resolved_ids, *unresolved]))

                # Last fallback: some reports store evaluated IDs under "applied".
                if not submitted_ids:
                    applied = docker_report.get("applied")
                    if isinstance(applied, list) and all(isinstance(x, str) for x in applied):
                        submitted_ids = applied

            # Counts: prefer explicit ints if present, otherwise infer from lists.
            n_resolved = len(resolved_ids)
            n_evaluated = len(submitted_ids)
            if isinstance(docker_report, dict):
                if isinstance(docker_report.get("resolved_instances"), int):
                    n_resolved = docker_report["resolved_instances"]
                if isinstance(docker_report.get("submitted_instances"), int):
                    n_evaluated = docker_report["submitted_instances"]

            results_payload = {
                "backend": "docker",
                "subset": subset,
                "dataset": dataset,
                "split": "test",
                "run_id": run_id,
                "submitted_ids": submitted_ids,
                "resolved_ids": resolved_ids,
                "submitted_instances": n_evaluated,
                "resolved_instances": n_resolved,
                "docker_report": docker_report,
            }
            results_path.write_text(json.dumps(results_payload, indent=2, sort_keys=True))
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

    # sb-cli report format (preferred if present, since it contains authoritative counts)
    sbcli_report = data.get("sbcli_report") if isinstance(data, dict) else None
    if isinstance(sbcli_report, dict):
        res = sbcli_report.get("resolved_instances")
        sub = sbcli_report.get("submitted_instances")
        if isinstance(res, list):
            res = len(res)
        if isinstance(sub, list):
            sub = len(sub)
        if isinstance(res, int) and isinstance(sub, int):
            return {"n_resolved": res, "n_evaluated": sub}

    # sb-cli report sometimes stored at top-level (older hooks / external usage)
    if isinstance(data, dict):
        res = data.get("resolved_instances")
        sub = data.get("submitted_instances")
        if isinstance(res, list):
            res = len(res)
        if isinstance(sub, list):
            sub = len(sub)
        if isinstance(res, int) and isinstance(sub, int):
            return {"n_resolved": res, "n_evaluated": sub}

    # handle both resolved_ids and resolved keys (can be list or int)
    resolved = data.get("resolved_ids", data.get("resolved", [])) if isinstance(data, dict) else []
    if isinstance(resolved, list):
        n_resolved = len(resolved)
    elif isinstance(resolved, int):
        n_resolved = resolved
    else:
        n_resolved = 0

    # handle both submitted_ids and submitted keys (can be list or int)
    submitted = data.get("submitted_ids", data.get("submitted")) if isinstance(data, dict) else None
    if isinstance(submitted, list):
        n_evaluated = len(submitted)
    elif isinstance(submitted, int):
        n_evaluated = submitted
    elif isinstance(data, dict) and "applied" in data:
        applied = data["applied"]
        n_evaluated = len(applied) if isinstance(applied, list) else applied if isinstance(applied, int) else 0
    else:
        # fallback: resolved + unresolved (handle both as list or int)
        unresolved = data.get("unresolved", []) if isinstance(data, dict) else []
        n_unresolved = len(unresolved) if isinstance(unresolved, list) else unresolved if isinstance(unresolved, int) else 0
        n_evaluated = n_resolved + n_unresolved

    return {"n_resolved": n_resolved, "n_evaluated": n_evaluated}
