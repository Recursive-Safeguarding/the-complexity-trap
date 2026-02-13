from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from query_metrics import aggregate_by_model_strategy, compute_strategy_comparison


def test_aggregate_avg_compactions_is_nan_when_missing_column() -> None:
    df = pd.DataFrame(
        [
            {
                "model": "glm-4.7",
                "strategy": "raw",
                "eval_complete": True,
                "n_instances": 50,
                "n_resolved": 32,
                "avg_cost": 1.0,
            }
        ]
    )

    out = aggregate_by_model_strategy(df)
    assert len(out) == 1
    assert np.isnan(out.iloc[0]["avg_compactions"])


def test_aggregate_avg_compactions_weighted_and_nan_safe() -> None:
    df = pd.DataFrame(
        [
            {
                "model": "glm-4.7",
                "strategy": "llm_summary",
                "eval_complete": True,
                "n_instances": 20,
                "n_resolved": 10,
                "avg_cost": 0.8,
                "avg_compactions": 2.0,
            },
            {
                "model": "glm-4.7",
                "strategy": "llm_summary",
                "eval_complete": True,
                "n_instances": 30,
                "n_resolved": 15,
                "avg_cost": 0.9,
                "avg_compactions": np.nan,
            },
        ]
    )

    out = aggregate_by_model_strategy(df)
    assert len(out) == 1
    # Only the finite compaction row should contribute.
    assert abs(out.iloc[0]["avg_compactions"] - 2.0) < 1e-9


def test_compare_strategies_keeps_nan_compactions() -> None:
    df = pd.DataFrame(
        [
            {
                "model": "glm-4.7",
                "strategy": "raw",
                "eval_complete": True,
                "n_instances": 50,
                "n_resolved": 30,
                "avg_cost": 1.0,
                "avg_compactions": np.nan,
            },
            {
                "model": "glm-4.7",
                "strategy": "observation_masking",
                "eval_complete": True,
                "n_instances": 50,
                "n_resolved": 29,
                "avg_cost": 0.6,
                "avg_compactions": np.nan,
            },
        ]
    )

    result = compute_strategy_comparison(df, model_filter="glm-4.7")
    assert "avg_compactions" in result.data.columns
    assert result.data["avg_compactions"].isna().all()
