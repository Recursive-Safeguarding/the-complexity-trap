from __future__ import annotations

import json
from pathlib import Path

from scripts.compaction_trigger_stats import (
    compute_run_stats,
    compute_run_stats_from_log,
    compute_run_stats_from_traj,
    summarize_run,
)


def _write_debug_log(inst_dir: Path, lines: list[str]) -> None:
    (inst_dir / f"{inst_dir.name}.debug.log").write_text("\n".join(lines))


def _write_traj(inst_dir: Path, payload: dict) -> None:
    (inst_dir / f"{inst_dir.name}.traj").write_text(json.dumps(payload))


def _make_inst(run_dir: Path, name: str) -> Path:
    d = run_dir / name
    d.mkdir(parents=True)
    return d


# -- compute_run_stats_from_log --


def test_log_counts_triggers(tmp_path: Path) -> None:
    run = tmp_path / "run"
    inst = _make_inst(run, "repo__task-1")
    _write_debug_log(inst, [
        "LastNObservations: triggering compaction (tokens=120000 >= threshold=100000)",
        "noise line",
        "SummarizeEveryNTurns: triggering compaction (tokens=130000 >= threshold=100000)",
    ])

    stats = compute_run_stats_from_log(run)
    assert len(stats) == 1
    assert stats[0].triggers_any == 2
    assert stats[0].triggers_by_label == {"masking": 1, "summary": 1}


def test_log_skips_dirs_without_debug_logs(tmp_path: Path) -> None:
    run = tmp_path / "run"
    inst = _make_inst(run, "repo__task-1")
    _write_traj(inst, {"summaries": [{"summary": "A"}]})

    assert compute_run_stats_from_log(run) == []


def test_log_zero_triggers(tmp_path: Path) -> None:
    run = tmp_path / "run"
    inst = _make_inst(run, "repo__task-1")
    _write_debug_log(inst, ["no compaction here"])

    stats = compute_run_stats_from_log(run)
    assert len(stats) == 1
    assert stats[0].triggers_any == 0
    assert stats[0].triggers_by_label == {}


# -- compute_run_stats_from_traj --


def test_traj_counts_summaries(tmp_path: Path) -> None:
    run = tmp_path / "run"
    inst = _make_inst(run, "repo__task-1")
    _write_traj(inst, {
        "summaries": [
            {"summary": "A", "statistics": {"cost": 0.005, "tokens": {"raw_input": 1000, "output": 50}}},
            {"summary": "B", "statistics": {"cost": 0.003, "tokens": {"raw_input": 800, "output": 30}}},
        ],
        "history": [{"role": "user"}, {"role": "assistant"}] * 40,
        "info": {"exit_status": "submitted"},
    })

    stats = compute_run_stats_from_traj(run)
    assert len(stats) == 1
    s = stats[0]
    assert s.triggers_any == 2
    assert s.turns == 40
    assert s.exit_status == "submitted"
    assert abs(s.summary_cost - 0.008) < 1e-9
    assert s.summary_tokens == 1880


def test_traj_handles_null_summaries(tmp_path: Path) -> None:
    run = tmp_path / "run"
    inst = _make_inst(run, "repo__task-1")
    _write_traj(inst, {
        "summaries": None,
        "history": [{"role": "user"}, {"role": "assistant"}] * 5,
        "info": {"exit_status": "exit_error"},
    })

    stats = compute_run_stats_from_traj(run)
    assert len(stats) == 1
    assert stats[0].triggers_any == 0
    assert stats[0].turns == 5
    assert stats[0].summary_cost == 0.0


def test_traj_handles_missing_fields(tmp_path: Path) -> None:
    run = tmp_path / "run"
    inst = _make_inst(run, "repo__task-1")
    _write_traj(inst, {"summaries": [{"summary": "A"}]})

    stats = compute_run_stats_from_traj(run)
    assert len(stats) == 1
    assert stats[0].triggers_any == 1
    assert stats[0].turns == 0
    assert stats[0].exit_status == ""
    assert stats[0].summary_cost == 0.0


def test_traj_skips_malformed_json(tmp_path: Path) -> None:
    run = tmp_path / "run"
    inst = _make_inst(run, "repo__task-1")
    (inst / f"{inst.name}.traj").write_text("{bad json")

    assert compute_run_stats_from_traj(run) == []


def test_traj_skips_dirs_without_traj(tmp_path: Path) -> None:
    run = tmp_path / "run"
    inst = _make_inst(run, "repo__task-1")
    _write_debug_log(inst, ["something"])

    assert compute_run_stats_from_traj(run) == []


# -- _has_traj_files --


def test_auto_mode_merges_log_and_traj(tmp_path: Path) -> None:
    run = tmp_path / "run"
    inst_log = _make_inst(run, "repo__task-log")
    _write_debug_log(inst_log, ["LastNObservations: triggering compaction"])

    inst_traj = _make_inst(run, "repo__task-traj")
    _write_traj(inst_traj, {"summaries": [{"summary": "A"}]})

    log_stats, traj_stats = compute_run_stats(run, source="auto")

    assert len(log_stats) == 1
    assert log_stats[0].instance == "repo__task-log"
    assert log_stats[0].triggers_any == 1
    assert len(traj_stats) == 1
    assert traj_stats[0].instance == "repo__task-traj"
    assert traj_stats[0].triggers_any == 1


def test_traj_ignores_non_dict_summary_items(tmp_path: Path) -> None:
    run = tmp_path / "run"
    inst = _make_inst(run, "repo__task-1")
    _write_traj(inst, {
        "summaries": [42, {"statistics": {"cost": 0.5, "tokens": {"raw_input": 10, "output": 2}}}],
        "history": [],
        "info": {},
    })

    stats = compute_run_stats_from_traj(run)
    assert len(stats) == 1
    assert stats[0].triggers_any == 2
    assert abs(stats[0].summary_cost - 0.5) < 1e-9
    assert stats[0].summary_tokens == 12


def test_traj_turns_prefer_action_message_type(tmp_path: Path) -> None:
    run = tmp_path / "run"
    inst = _make_inst(run, "repo__task-1")
    _write_traj(inst, {
        "summaries": [],
        "history": [
            {"role": "system", "message_type": "system_prompt"},
            {"role": "assistant", "message_type": "action"},
            {"role": "tool", "message_type": "observation"},
            {"role": "assistant", "message_type": "action"},
            {"role": "tool", "message_type": "observation"},
        ],
        "info": {},
    })

    stats = compute_run_stats_from_traj(run)
    assert len(stats) == 1
    assert stats[0].turns == 2


def test_summarize_run_auto_includes_log_and_traj_sections(tmp_path: Path) -> None:
    run = tmp_path / "run"
    inst_log = _make_inst(run, "repo__task-log")
    _write_debug_log(inst_log, [
        "LastNObservations: triggering compaction",
        "SummarizeEveryNTurns: triggering compaction",
    ])

    inst_traj = _make_inst(run, "repo__task-traj")
    _write_traj(inst_traj, {
        "summaries": [{"summary": "A"}, {"summary": "B"}],
        "history": [{"role": "assistant", "message_type": "action"}] * 12,
        "info": {"exit_status": "submitted"},
    })

    summary = summarize_run(run, source="auto")
    assert summary["run_name"] == "run"
    assert summary["log"]["n_instances"] == 1
    assert summary["log"]["n_triggered_any"] == 1
    assert summary["log"]["total_triggers"] == 2
    assert summary["log"]["instances_by_label"] == {"masking": 1, "summary": 1}
    assert summary["traj"]["n_instances"] == 1
    assert summary["traj"]["n_with_summaries"] == 1
    assert summary["traj"]["total_summary_calls"] == 2
    assert summary["traj"]["n_active_instances"] == 1
    assert summary["traj"]["active_total_summary_calls"] == 2


def test_summarize_run_rates_are_none_when_denominator_zero(tmp_path: Path) -> None:
    run = tmp_path / "run"
    inst = _make_inst(run, "repo__task-1")
    _write_debug_log(inst, ["no compaction"])

    summary = summarize_run(run, source="log")
    assert summary["log"]["trigger_rate"] == 0.0
    assert summary["traj"]["summary_rate"] is None
    assert summary["traj"]["avg_summary_calls_per_traj_instance"] is None


def test_traj_coerces_string_numeric_fields(tmp_path: Path) -> None:
    run = tmp_path / "run"
    inst = _make_inst(run, "repo__task-1")
    _write_traj(inst, {
        "summaries": [
            {
                "statistics": {
                    "cost": "0.5",
                    "tokens": {"raw_input": "10", "output": "2"},
                }
            }
        ],
        "history": [],
        "info": {},
    })

    stats = compute_run_stats_from_traj(run)
    assert len(stats) == 1
    assert abs(stats[0].summary_cost - 0.5) < 1e-9
    assert stats[0].summary_tokens == 12
