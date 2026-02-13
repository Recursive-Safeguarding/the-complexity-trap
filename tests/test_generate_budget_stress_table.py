from __future__ import annotations

from pathlib import Path

import pandas as pd

import scripts.generate_budget_stress_table as budget_table
from scripts.generate_budget_stress_table import (
    build_budget_stress_rows,
    render_latex_table,
    resolve_trajectory_run_dir,
    trigger_rate_from_summary,
)


def test_resolve_trajectory_run_dir_prefers_newest_when_multiple(tmp_path: Path) -> None:
    root = tmp_path / "trajectories"
    run_name = "glm-4.7__od_glm-4.7_la_lf-0p0001_lt-40000__mini__2026-02-10_00-06-03"

    run_a = root / "user" / run_name
    run_b = root / "root" / run_name
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)

    # Ensure deterministic newest pick.
    run_a.touch()
    run_b.touch()

    chosen, warning = resolve_trajectory_run_dir(root, run_name)
    assert chosen == run_b
    assert warning is not None
    assert "multiple trajectory directories" in warning


def test_resolve_trajectory_run_dir_missing_returns_warning(tmp_path: Path) -> None:
    root = tmp_path / "trajectories"
    root.mkdir(parents=True)

    chosen, warning = resolve_trajectory_run_dir(root, "missing-run")
    assert chosen is None
    assert warning is not None
    assert "not found" in warning


def test_resolve_trajectory_run_dir_rejects_non_directory_root(tmp_path: Path) -> None:
    root_file = tmp_path / "trajectories"
    root_file.write_text("not a dir")

    chosen, warning = resolve_trajectory_run_dir(root_file, "missing-run")
    assert chosen is None
    assert warning is not None
    assert "not a directory" in warning


def test_trigger_rate_prefers_log_for_limit_aware() -> None:
    row = pd.Series({"strategy": "observation_masking", "hp_limit_aware": True})
    summary = {
        "log": {"trigger_rate": 0.8},
        "traj": {"summary_rate": 0.2},
    }
    assert trigger_rate_from_summary(row, summary) == "80.0%"


def test_trigger_rate_uses_traj_for_periodic_summary() -> None:
    row = pd.Series({"strategy": "llm_summary", "hp_limit_aware": False})
    summary = {
        "log": {"trigger_rate": None},
        "traj": {"summary_rate": 0.44},
    }
    assert trigger_rate_from_summary(row, summary) == "44.0%"


def test_trigger_rate_returns_na_when_unavailable() -> None:
    row = pd.Series({"strategy": "observation_masking", "hp_limit_aware": False})
    summary = {
        "log": {"trigger_rate": None},
        "traj": {"summary_rate": None},
    }
    assert trigger_rate_from_summary(row, summary) == "N/A"


def test_build_budget_rows_without_raw_reports_na_delta(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run_name = "glm-4.7__od_glm-4.7_la_lf-0p0001_lt-40000__mini__2026-02-10_00-06-03"
    trajectories_root = tmp_path / "trajectories"
    (trajectories_root / "root" / run_name).mkdir(parents=True)

    df = pd.DataFrame(
        [
            {
                "run_name": run_name,
                "created_at": "2026-02-10T00:06:03Z",
                "model": "glm-4.7",
                "strategy": "on_demand",
                "summarizer": "same",
                "instances_subset": "verified-mini",
                "hp_limit_min_tokens": 40000,
                "hp_limit_aware": True,
                "eval_complete": True,
                "solve_rate": 0.58,
            }
        ]
    )

    monkeypatch.setattr(
        budget_table,
        "summarize_run",
        lambda run_dir, source="auto": {
            "log": {"trigger_rate": 0.9},
            "traj": {"summary_rate": 0.9},
        },
    )

    rows, raw_rate = build_budget_stress_rows(df, trajectories_root, quiet_missing=True)
    assert raw_rate is None
    assert rows[0]["config"] == "Raw baseline"
    assert rows[0]["solve_rate"] == "N/A"
    assert rows[1]["delta"] == "N/A"


def test_render_latex_table_escapes_special_chars() -> None:
    latex = render_latex_table(
        [
            {
                "config": "On-demand_summary(foo&bar)",
                "trigger_rate": "80.0%",
                "solve_rate": "50.0%",
                "delta": "+2.0%",
            }
        ]
    )
    assert r"On-demand\_summary(foo\&bar)" in latex
    assert r"80.0\%" in latex
    assert r"50.0\%" in latex
    assert r"+2.0\%" in latex


def test_build_budget_rows_missing_solve_rate_is_na(tmp_path: Path, monkeypatch) -> None:
    run_name = "glm-4.7__od_glm-4.7_la_lf-0p0001_lt-40000__mini__2026-02-10_00-06-03"
    raw_name = "glm-4.7__raw__mini__2026-02-09_00-00-00"
    trajectories_root = tmp_path / "trajectories"
    (trajectories_root / "root" / run_name).mkdir(parents=True)
    (trajectories_root / "root" / raw_name).mkdir(parents=True)

    df = pd.DataFrame(
        [
            {
                "run_name": raw_name,
                "created_at": "2026-02-09T00:00:00Z",
                "model": "glm-4.7",
                "strategy": "raw",
                "summarizer": "same",
                "instances_subset": "verified-mini",
                "hp_limit_min_tokens": 0,
                "hp_limit_aware": False,
                "eval_complete": True,
                "solve_rate": 0.64,
            },
            {
                "run_name": run_name,
                "created_at": "2026-02-10T00:06:03Z",
                "model": "glm-4.7",
                "strategy": "on_demand",
                "summarizer": "same",
                "instances_subset": "verified-mini",
                "hp_limit_min_tokens": 40000,
                "hp_limit_aware": True,
                "eval_complete": "True",  # bool-like string should be accepted
                "solve_rate": None,
            },
        ]
    )

    monkeypatch.setattr(
        budget_table,
        "summarize_run",
        lambda run_dir, source="auto": {
            "log": {"trigger_rate": 0.9},
            "traj": {"summary_rate": 0.9},
        },
    )

    rows, raw_rate = build_budget_stress_rows(df, trajectories_root, quiet_missing=True)
    assert raw_rate == 0.64
    assert len(rows) == 2
    assert rows[1]["solve_rate"] == "N/A"
    assert rows[1]["delta"] == "N/A"
