from __future__ import annotations

import io
import json
import math

import pandas as pd
import pytest

from scripts.repeated_run_stats import compute_repeated_run_stats, filter_runs, render_results


def _row(
    run_name: str,
    created_at: str,
    *,
    n_resolved: int,
    n_instances: int = 50,
    model: str = "glm-4.7",
    strategy: str = "on_demand",
    summarizer: str | None = "same",
    instances_subset: str = "verified-mini",
    eval_complete: bool = True,
    hp_obs_n: int = 10,
    hp_sum_n: int = 21,
    hp_sum_keep_m: int = 10,
    hp_limit_aware: bool = True,
    hp_limit_fraction: float = 0.0001,
    hp_limit_min_tokens: int = 0,
) -> dict:
    return {
        "run_id": f"{run_name}-id",
        "run_name": run_name,
        "created_at": created_at,
        "model": model,
        "strategy": strategy,
        "summarizer": summarizer,
        "instances_subset": instances_subset,
        "eval_complete": eval_complete,
        "hp_obs_n": hp_obs_n,
        "hp_sum_n": hp_sum_n,
        "hp_sum_keep_m": hp_sum_keep_m,
        "hp_limit_aware": hp_limit_aware,
        "hp_limit_fraction": hp_limit_fraction,
        "hp_limit_min_tokens": hp_limit_min_tokens,
        "n_instances": n_instances,
        "n_resolved": n_resolved,
    }


def _filter(df: pd.DataFrame) -> pd.DataFrame:
    return filter_runs(
        df,
        model_filter="glm-4.7",
        strategy_filter="on_demand",
        instances_subset="verified-mini",
        eval_only=True,
        n_instances_min=40,
        n_instances_max=60,
    )


def test_strict_grouping_splits_by_summarizer_and_limit_threshold() -> None:
    rows = [
        _row("run-a1", "2026-02-11T10:00:00", n_resolved=30, hp_limit_min_tokens=0, summarizer=None),
        _row("run-a2", "2026-02-11T11:00:00", n_resolved=28, hp_limit_min_tokens=0, summarizer="same"),
        _row("run-b1", "2026-02-11T12:00:00", n_resolved=27, hp_limit_min_tokens=40000, summarizer="same"),
        _row("run-c1", "2026-02-11T13:00:00", n_resolved=29, hp_limit_min_tokens=0, summarizer="minimax-m2.1"),
    ]
    stats = compute_repeated_run_stats(_filter(pd.DataFrame(rows)), min_repeats=1)

    assert len(stats) == 3

    same_zero = stats[(stats["summarizer"] == "same") & (stats["hp_limit_min_tokens"] == 0)]
    assert len(same_zero) == 1
    assert int(same_zero.iloc[0]["n_runs"]) == 2


def test_dedupe_latest_run_name_keeps_only_latest_snapshot() -> None:
    rows = [
        _row("dup", "2026-02-11T10:00:00", n_resolved=20),
        _row("dup", "2026-02-11T11:00:00", n_resolved=30),
        _row("other", "2026-02-11T12:00:00", n_resolved=25),
    ]
    stats = compute_repeated_run_stats(_filter(pd.DataFrame(rows)), min_repeats=2)

    assert len(stats) == 1
    row = stats.iloc[0]
    assert int(row["n_runs"]) == 2
    assert row["mean_rate"] == pytest.approx(0.55)


def test_t_interval_and_pooled_stats_are_computed_correctly() -> None:
    rows = [
        _row("r1", "2026-02-11T10:00:00", n_resolved=30),  # 0.60
        _row("r2", "2026-02-11T11:00:00", n_resolved=35),  # 0.70
        _row("r3", "2026-02-11T12:00:00", n_resolved=40),  # 0.80
    ]
    stats = compute_repeated_run_stats(_filter(pd.DataFrame(rows)), min_repeats=2)
    row = stats.iloc[0]

    expected_mean = 0.7
    expected_std = 0.1
    expected_sem = expected_std / math.sqrt(3)
    expected_t = 4.303  # df = 2
    expected_half = expected_t * expected_sem

    assert int(row["n_runs"]) == 3
    assert row["mean_rate"] == pytest.approx(expected_mean)
    assert row["std_rate"] == pytest.approx(expected_std)
    assert row["sem_rate"] == pytest.approx(expected_sem)
    assert row["t_critical"] == pytest.approx(expected_t, rel=1e-3)
    assert row["t_ci_low"] == pytest.approx(max(0.0, expected_mean - expected_half))
    assert row["t_ci_high"] == pytest.approx(min(1.0, expected_mean + expected_half))
    assert row["pooled_rate"] == pytest.approx(0.7)


def test_subset_normalization_and_min_repeats_filter() -> None:
    rows = [
        _row("s1", "2026-02-11T10:00:00", n_resolved=30, instances_subset="verified-mini"),
        _row("s2", "2026-02-11T11:00:00", n_resolved=32, instances_subset="verified_mini"),
        _row("single", "2026-02-11T12:00:00", n_resolved=29, hp_limit_min_tokens=40000),
    ]
    filtered = _filter(pd.DataFrame(rows))
    stats = compute_repeated_run_stats(filtered, min_repeats=2)

    assert len(stats) == 1
    assert int(stats.iloc[0]["n_runs"]) == 2


def test_render_results_json_and_csv_smoke() -> None:
    rows = [
        _row("r1", "2026-02-11T10:00:00", n_resolved=30),
        _row("r2", "2026-02-11T11:00:00", n_resolved=35),
    ]
    stats = compute_repeated_run_stats(_filter(pd.DataFrame(rows)), min_repeats=2)

    json_buffer = io.StringIO()
    render_results(stats, output_format="json", show_runs=False, file=json_buffer)
    payload = json.loads(json_buffer.getvalue())
    assert isinstance(payload, list)
    assert payload[0]["n_runs"] == 2
    assert "run_names" not in payload[0]

    csv_buffer = io.StringIO()
    render_results(stats, output_format="csv", show_runs=False, file=csv_buffer)
    header = csv_buffer.getvalue().splitlines()[0]
    assert "mean_rate" in header
