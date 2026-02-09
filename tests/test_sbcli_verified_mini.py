from __future__ import annotations

import json

import pytest

from sweagent.utils import sbcli


def test_extract_instance_ids_from_preds_list(tmp_path):
    preds_path = tmp_path / "preds.json"
    preds_path.write_text(
        json.dumps(
            [
                {"instance_id": "a", "model_patch": "", "model_name_or_path": "m"},
                {"instance_id": "b", "model_patch": "", "model_name_or_path": "m"},
                {"instance_id": "a", "model_patch": "", "model_name_or_path": "m"},  # dup
            ]
        )
    )
    assert sbcli._extract_instance_ids_from_preds(preds_path) == ["a", "b"]


def test_extract_instance_ids_from_preds_dict(tmp_path):
    preds_path = tmp_path / "preds.json"
    preds_path.write_text(
        json.dumps(
            {
                "x": {"model_patch": "", "model_name_or_path": "m"},
                "y": {"model_patch": "", "model_name_or_path": "m"},
            }
        )
    )
    assert sbcli._extract_instance_ids_from_preds(preds_path) == ["x", "y"]


def test_guardrail_verified_mini_blocks_large_submission(tmp_path, monkeypatch):
    # Avoid requiring sb-cli binary or API key in unit tests.
    monkeypatch.setattr(sbcli, "check_sbcli_available", lambda: (True, ""))

    preds_path = tmp_path / "preds.json"
    preds = [
        {"instance_id": f"id{i}", "model_patch": "", "model_name_or_path": "m"}
        for i in range(sbcli.MAX_MINI_INSTANCES + 1)
    ]
    preds_path.write_text(json.dumps(preds))

    ok, err, results_path = sbcli.run_sbcli_evaluation(
        preds_path=preds_path,
        subset="verified-mini",
        run_id="run",
        output_dir=tmp_path,
    )
    assert not ok
    assert isinstance(err, str) and err.startswith("guardrail:")
    assert results_path is None


def test_guardrail_verified_mini_blocks_empty_instance_ids(tmp_path, monkeypatch):
    # Avoid requiring sb-cli binary or API key in unit tests.
    monkeypatch.setattr(sbcli, "check_sbcli_available", lambda: (True, ""))

    # Ensure we never attempt to submit when the guardrail should trip.
    def _no_submit(*args, **kwargs):
        raise AssertionError("submit_to_sbcli should not be called when guardrail triggers")

    monkeypatch.setattr(sbcli, "submit_to_sbcli", _no_submit)

    preds_path = tmp_path / "preds.json"
    preds_path.write_text(json.dumps([{"model_patch": "", "model_name_or_path": "m"}]))

    ok, err, results_path = sbcli.run_sbcli_evaluation(
        preds_path=preds_path,
        subset="verified-mini",
        run_id="run",
        output_dir=tmp_path,
    )
    assert not ok
    assert isinstance(err, str) and err.startswith("guardrail:")
    assert results_path is None


def test_submit_to_sbcli_includes_instance_ids(tmp_path, monkeypatch):
    called = {}

    def fake_run(cmd, capture_output, text, timeout):
        called["cmd"] = cmd

        class Result:
            returncode = 0
            stdout = ""
            stderr = ""

        return Result()

    monkeypatch.setattr(sbcli.subprocess, "run", fake_run)

    preds_path = tmp_path / "preds.json"
    preds_path.write_text("[]")

    ok, err = sbcli.submit_to_sbcli(
        preds_path=preds_path,
        dataset="swe-bench_lite",
        run_id="run",
        output_dir=tmp_path,
        instance_ids=["a", "b", "c"],
    )
    assert ok
    assert err == ""
    assert "--instance_ids" in called["cmd"]
    idx = called["cmd"].index("--instance_ids")
    assert called["cmd"][idx + 1] == "a,b,c"


def test_extract_instance_ids_from_preds_jsonl_skips_bad_lines(tmp_path):
    preds_path = tmp_path / "preds.jsonl"
    preds_path.write_text('{"instance_id":"a"}\nnot-json\n{"instance_id":"b"}\n')
    assert sbcli._extract_instance_ids_from_preds(preds_path) == ["a", "b"]


def test_parse_eval_results_handles_sbcli_report_counts(tmp_path):
    results_path = tmp_path / "results.json"
    results_path.write_text(
        json.dumps(
            {
                "backend": "sbcli",
                "sbcli_report": {"resolved_instances": 3, "submitted_instances": 5},
            }
        )
    )
    parsed = sbcli.parse_eval_results(results_path)
    assert parsed == {"n_resolved": 3, "n_evaluated": 5}


def test_parse_eval_results_handles_list_counts(tmp_path):
    results_path = tmp_path / "results.json"
    results_path.write_text(
        json.dumps(
            {
                "resolved_instances": ["a", "b"],
                "submitted_instances": ["a", "b", "c"],
            }
        )
    )
    parsed = sbcli.parse_eval_results(results_path)
    assert parsed == {"n_resolved": 2, "n_evaluated": 3}
