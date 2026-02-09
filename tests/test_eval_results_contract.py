from __future__ import annotations

import json

from pathlib import Path

from sweagent.utils.sbcli import parse_eval_results


def test_parse_eval_results_handles_docker_wrapper(tmp_path: Path) -> None:
    results_path = tmp_path / "results.json"
    payload = {
        "backend": "docker",
        "subset": "verified-mini",
        "dataset": "MariusHobbhahn/swe-bench-verified-mini",
        "run_id": "run",
        "split": "test",
        "submitted_ids": ["a", "b", "c"],
        "resolved_ids": ["a"],
        "submitted_instances": 3,
        "resolved_instances": 1,
        "docker_report": {"resolved_ids": ["a"], "submitted_ids": ["a", "b", "c"]},
    }
    results_path.write_text(json.dumps(payload))
    parsed = parse_eval_results(results_path)
    assert parsed == {"n_resolved": 1, "n_evaluated": 3}


def test_parse_eval_results_handles_legacy_docker_report(tmp_path: Path) -> None:
    # Older swebench reports are often written directly as results.json.
    results_path = tmp_path / "results.json"
    payload = {
        "submitted_ids": ["a", "b", "c"],
        "resolved_ids": ["a"],
        "unresolved": ["b", "c"],
    }
    results_path.write_text(json.dumps(payload))
    parsed = parse_eval_results(results_path)
    assert parsed == {"n_resolved": 1, "n_evaluated": 3}


def test_evaluate_missing_treats_invalid_results_as_unevaluated(tmp_path: Path) -> None:
    # Import locally to avoid forcing wandb imports at module import time.
    from scripts.evaluate_missing import find_unevaluated_runs

    base = tmp_path / "trajectories"
    run_dir = base / "root" / "glm-4.7__raw__mini50__2026-02-08_00-00-00"
    run_dir.mkdir(parents=True)
    (run_dir / "preds.json").write_text("[]")
    # Contract violation: results.json exists but indicates 0 evaluated instances.
    (run_dir / "results.json").write_text(json.dumps({"submitted_instances": 0, "resolved_instances": 0}))

    uneval = find_unevaluated_runs(base, subset="verified-mini")
    assert run_dir in uneval

