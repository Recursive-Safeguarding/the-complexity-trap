from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from paper_results import _resolve_run_dir, classify_trigger, generate_latex_table


def test_resolve_run_dir_rejects_non_directory_root(tmp_path: Path) -> None:
    root_file = tmp_path / "trajectories"
    root_file.write_text("not-a-dir")

    run_dir, warning = _resolve_run_dir(root_file, "run-name")
    assert run_dir is None
    assert warning is not None
    assert "not a directory" in warning


def test_resolve_run_dir_rejects_absolute_and_traversal(tmp_path: Path) -> None:
    trajectories = tmp_path / "trajectories"
    trajectories.mkdir()

    run_dir_abs, warning_abs = _resolve_run_dir(trajectories, str((tmp_path / "outside").resolve()))
    assert run_dir_abs is None
    assert warning_abs is not None
    assert "absolute path" in warning_abs

    run_dir_parent, warning_parent = _resolve_run_dir(trajectories, "../outside")
    assert run_dir_parent is None
    assert warning_parent is not None
    assert "parent traversal" in warning_parent


def test_resolve_run_dir_multiple_picks_newest(tmp_path: Path) -> None:
    trajectories = tmp_path / "trajectories"
    run_name = "glm-4.7__sum_glm-4.7_n-21_k-10__mini__2026-02-04_11-04-40"

    older = trajectories / "user" / run_name
    newer = trajectories / "root" / run_name
    older.mkdir(parents=True)
    newer.mkdir(parents=True)
    older.touch()
    newer.touch()

    run_dir, warning = _resolve_run_dir(trajectories, run_name)
    assert run_dir == newer
    assert warning is not None
    assert "multiple trajectory dirs" in warning


def test_classify_trigger_handles_non_bool_hp_limit_aware() -> None:
    periodic = pd.Series({"strategy": "observation_masking", "hp_limit_aware": "False"})
    ondemand = pd.Series({"strategy": "observation_masking", "hp_limit_aware": "TRUE"})
    nan_value = pd.Series({"strategy": "observation_masking", "hp_limit_aware": float("nan")})

    assert classify_trigger(periodic) == "periodic"
    assert classify_trigger(ondemand) == "on_demand"
    assert classify_trigger(nan_value) == "periodic"


def test_latex_table_uses_summary_calls_column_and_scope() -> None:
    rows = [
        {
            "run_name": "raw-run",
            "strategy": "raw",
            "trigger": "baseline",
            "summarizer": "same",
            "compaction": "none",
            "n": 50,
            "k": 32,
            "solve_rate": 0.64,
            "ci_lo": 0.50,
            "ci_hi": 0.76,
            "ci_half": 0.13,
            "avg_cost": 1.0,
            "eval_complete": True,
            "hp_limit_min_tokens": 0,
            "avg_summary_calls": float("nan"),
        },
        {
            "run_name": "mask-run",
            "strategy": "observation_masking",
            "trigger": "periodic",
            "summarizer": "same",
            "compaction": "masking",
            "n": 50,
            "k": 31,
            "solve_rate": 0.62,
            "ci_lo": 0.48,
            "ci_hi": 0.74,
            "ci_half": 0.13,
            "avg_cost": 0.7,
            "eval_complete": True,
            "hp_limit_min_tokens": 0,
            "avg_summary_calls": float("nan"),
        },
        {
            "run_name": "sum-run",
            "strategy": "llm_summary",
            "trigger": "periodic",
            "summarizer": "same",
            "compaction": "summary",
            "n": 50,
            "k": 28,
            "solve_rate": 0.56,
            "ci_lo": 0.42,
            "ci_hi": 0.69,
            "ci_half": 0.13,
            "avg_cost": 1.4,
            "eval_complete": True,
            "hp_limit_min_tokens": 0,
            "avg_summary_calls": 2.4,
        },
    ]
    results = pd.DataFrame(rows)

    latex = generate_latex_table(results)
    assert "Summary Calls" in latex
    assert "apply only to summary-based methods" in latex

    masking_line = next(line for line in latex.splitlines() if "Periodic masking" in line)
    assert masking_line.rstrip().endswith("& -- \\\\")

    summary_line = next(line for line in latex.splitlines() if "Periodic summary (self)" in line)
    assert "& 2.4 \\\\" in summary_line


def test_latex_table_prefers_latest_created_at_for_duplicate_rows() -> None:
    rows = [
        {
            "run_name": "raw-run",
            "strategy": "raw",
            "trigger": "baseline",
            "summarizer": "same",
            "compaction": "none",
            "n": 50,
            "k": 32,
            "solve_rate": 0.64,
            "ci_lo": 0.50,
            "ci_hi": 0.76,
            "ci_half": 0.13,
            "avg_cost": 1.0,
            "eval_complete": True,
            "hp_limit_min_tokens": 0,
            "avg_summary_calls": float("nan"),
            "created_at": "2026-02-10T00:00:00Z",
        },
        {
            "run_name": "mask-older",
            "strategy": "observation_masking",
            "trigger": "periodic",
            "summarizer": "same",
            "compaction": "masking",
            "n": 50,
            "k": 25,
            "solve_rate": 0.50,
            "ci_lo": 0.36,
            "ci_hi": 0.64,
            "ci_half": 0.14,
            "avg_cost": 0.8,
            "eval_complete": True,
            "hp_limit_min_tokens": 0,
            "avg_summary_calls": float("nan"),
            "created_at": "2026-02-01T00:00:00Z",
        },
        {
            "run_name": "mask-newer",
            "strategy": "observation_masking",
            "trigger": "periodic",
            "summarizer": "same",
            "compaction": "masking",
            "n": 50,
            "k": 31,
            "solve_rate": 0.62,
            "ci_lo": 0.48,
            "ci_hi": 0.74,
            "ci_half": 0.13,
            "avg_cost": 0.7,
            "eval_complete": True,
            "hp_limit_min_tokens": 0,
            "avg_summary_calls": float("nan"),
            "created_at": "2026-02-11T00:00:00Z",
        },
    ]
    latex = generate_latex_table(pd.DataFrame(rows))
    masking_line = next(line for line in latex.splitlines() if "Periodic masking" in line)
    assert "31/50" in masking_line
