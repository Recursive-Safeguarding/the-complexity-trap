#!/usr/bin/env python3
"""
Shared data and utilities for Complexity Trap dashboards.

This module is Streamlit-free and can be imported by both the web (Streamlit)
and TUI (Rich) dashboards without pulling in unnecessary dependencies.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd

# Paper baselines from arXiv:2508.21433 - Table 1 (all 5 model configurations)
# NOTE: Paper reports solve_rate (bugs actually fixed)
# solve_rate = n_resolved / n_instances (directly comparable to paper)
#
# Each entry includes: solve_rate, avg_cost, confidence intervals (ci), and relative change vs raw (delta)
PAPER_BASELINES = {
    "qwen3-32b": {
        "raw": {"solve_rate": 0.170, "avg_cost": 1.12, "rate_ci": 0.033, "cost_ci": 0.18},
        "observation_masking": {"solve_rate": 0.150, "avg_cost": 0.55, "rate_ci": 0.031, "cost_ci": 0.09,
                                "rate_delta": -0.118, "cost_delta": -0.509},
        "llm_summary": {"solve_rate": 0.160, "avg_cost": 0.50, "rate_ci": 0.033, "cost_ci": 0.07,
                        "rate_delta": -0.059, "cost_delta": -0.554},
    },
    "qwen3-32b-thinking": {
        "raw": {"solve_rate": 0.230, "avg_cost": 0.51, "rate_ci": 0.037, "cost_ci": 0.07},
        "observation_masking": {"solve_rate": 0.246, "avg_cost": 0.46, "rate_ci": 0.038, "cost_ci": 0.05,
                                "rate_delta": 0.070, "cost_delta": -0.098},
        "llm_summary": {"solve_rate": 0.248, "avg_cost": 0.51, "rate_ci": 0.039, "cost_ci": 0.06,
                        "rate_delta": 0.073, "cost_delta": 0.0},
    },
    "qwen3-coder-480b": {
        "raw": {"solve_rate": 0.534, "avg_cost": 1.29, "rate_ci": 0.043, "cost_ci": 0.26},
        "observation_masking": {"solve_rate": 0.548, "avg_cost": 0.61, "rate_ci": 0.044, "cost_ci": 0.06,
                                "rate_delta": 0.026, "cost_delta": -0.527},
        "llm_summary": {"solve_rate": 0.538, "avg_cost": 0.64, "rate_ci": 0.042, "cost_ci": 0.06,
                        "rate_delta": 0.007, "cost_delta": -0.504},
        # Hybrid from Section 5.3: N=43, M=W=10 on SWE-bench Verified-50
        "hybrid": {"solve_rate": 0.540, "avg_cost": 0.50, "rate_ci": 0.044, "cost_ci": 0.05,
                   "rate_delta": 0.011, "cost_delta": -0.612},
    },
    "gemini-2.5-flash": {
        "raw": {"solve_rate": 0.328, "avg_cost": 0.41, "rate_ci": 0.041, "cost_ci": 0.08},
        "observation_masking": {"solve_rate": 0.356, "avg_cost": 0.18, "rate_ci": 0.042, "cost_ci": 0.03,
                                "rate_delta": 0.085, "cost_delta": -0.561},
        "llm_summary": {"solve_rate": 0.360, "avg_cost": 0.24, "rate_ci": 0.041, "cost_ci": 0.04,
                        "rate_delta": 0.098, "cost_delta": -0.415},
    },
    "gemini-2.5-flash-thinking": {
        "raw": {"solve_rate": 0.404, "avg_cost": 0.56, "rate_ci": 0.043, "cost_ci": 0.10},
        "observation_masking": {"solve_rate": 0.364, "avg_cost": 0.24, "rate_ci": 0.042, "cost_ci": 0.04,
                                "rate_delta": -0.099, "cost_delta": -0.571},
        "llm_summary": {"solve_rate": 0.314, "avg_cost": 0.25, "rate_ci": 0.040, "cost_ci": 0.05,
                        "rate_delta": -0.223, "cost_delta": -0.554},
    },
}

# Our measured results (separate from paper baselines)
# GLM-4.7 on SWE-bench Verified-Mini (50 instances per strategy)
# Key finding: Context management HURTS GLM-4.7 (opposite of paper's qwen3-coder-480b results)
OUR_BASELINES = {
    "glm-4.7": {
        "raw": {"solve_rate": 0.640, "avg_cost": 1.00, "rate_ci": 0.068, "cost_ci": 0.15},
        "observation_masking": {"solve_rate": 0.620, "avg_cost": 0.68, "rate_ci": 0.069, "cost_ci": 0.10,
                                "rate_delta": -0.031, "cost_delta": -0.320},
        "llm_summary": {"solve_rate": 0.540, "avg_cost": 0.55, "rate_ci": 0.070, "cost_ci": 0.08,
                        "rate_delta": -0.156, "cost_delta": -0.450},
        "hybrid": {"solve_rate": 0.560, "avg_cost": 0.42, "rate_ci": 0.070, "cost_ci": 0.06,
                   "rate_delta": -0.125, "cost_delta": -0.580},
    },
}

# Table 2: LLM-Summary generation costs per model
SUMMARY_COSTS = {
    "qwen3-32b": {"cost": 0.0143, "pct": 2.86},
    "qwen3-32b-thinking": {"cost": 0.0033, "pct": 0.65},
    "qwen3-coder-480b": {"cost": 0.0439, "pct": 7.20},
    "gemini-2.5-flash": {"cost": 0.0161, "pct": 6.71},
    "gemini-2.5-flash-thinking": {"cost": 0.0131, "pct": 5.24},
}

# Hyperparameters from paper
STRATEGY_PARAMS = {
    "observation_masking": {"M": 10, "description": "Keep last M=10 observations"},
    "llm_summary": {"N": 21, "M": 10, "description": "Summarize every N=21 turns, keep M=10 tail"},
    "hybrid": {"N": 43, "M": 10, "W": 10, "description": "N=43 summary trigger, M=W=10 for masking+tail"},
}

# Exit status colors (matches WandB hook taxonomy)
EXIT_COLORS = {
    "exit_submitted": "#22c55e",  # green - success
    "exit_cost": "#f59e0b",  # amber - cost limit
    "exit_context": "#ef4444",  # red - context overflow
    "exit_timeout": "#3b82f6",  # blue - timeout
    "exit_format": "#a855f7",  # purple - format error
    "exit_other": "#6b7280",  # gray - other
}


def _unwrap_wandb_value(val):
    """Unwrap WandB summary dicts like {'last': value} to plain values."""
    if isinstance(val, dict):
        if "last" in val:
            return val["last"]
        if "value" in val:
            return val["value"]
    return val


def _safe_float(val, default=np.nan):
    """Convert value to float, returning default if not finite."""
    if val is None:
        return default
    if isinstance(val, float) and not np.isfinite(val):
        return default
    try:
        result = float(val)
        return result if np.isfinite(result) else default
    except (ValueError, TypeError):
        return default


def fetch_runs(project: str, entity: str | None = None, use_cache: bool = True) -> pd.DataFrame:
    """Fetch all runs from WandB as DataFrame."""
    import wandb

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project

    try:
        runs = api.runs(path)
    except wandb.errors.CommError as e:
        raise RuntimeError(f"Failed to connect to WandB: {e}") from e

    records = []
    for r in runs:
        config = r.config or {}
        summary = r.summary._json_dict if r.summary else {}

        def _get(key, default=None):
            """Get value from summary, unwrapping WandB dicts."""
            val = summary.get(key, default)
            return _unwrap_wandb_value(val)

        # wandb returns eval_complete in various formats
        _eval_val = _get("eval_complete", False)
        if isinstance(_eval_val, bool):
            eval_complete = _eval_val
        elif isinstance(_eval_val, (int, float)):
            eval_complete = bool(_eval_val)
        elif isinstance(_eval_val, str):
            eval_complete = _eval_val.lower() in ("true", "1", "yes")
        else:
            eval_complete = False

        exit_summary = {
            key: _unwrap_wandb_value(val)
            for key, val in summary.items()
            if isinstance(key, str) and key.startswith("exit/")
        }
        known_exit_keys = {
            "exit/submitted",
            "exit/exit_cost",
            "exit/exit_context",
            "exit/exit_timeout",
            "exit/exit_format",
            "exit/other",
        }

        def _num(val):
            if val is None:
                return 0
            try:
                return float(val)
            except (TypeError, ValueError):
                return 0

        exit_other = _num(exit_summary.get("exit/other", 0))
        for key, val in exit_summary.items():
            if key not in known_exit_keys:
                exit_other += _num(val)

        def _intlike(val):
            if val is None:
                return 0
            try:
                return int(val)
            except (TypeError, ValueError):
                try:
                    return int(float(val))
                except (TypeError, ValueError):
                    return 0

        n_instances = _intlike(_get("n_instances", 0))
        n_submitted = _intlike(_get("n_submitted", 0))
        n_resolved = _intlike(_get("n_resolved", 0))
        n_evaluated = _intlike(_get("n_evaluated", 0))
        solve_rate_val = _get("solve_rate")

        if solve_rate_val and n_resolved:
            try:
                inferred = int(round(n_resolved / float(solve_rate_val)))
                n_instances = max(n_instances, inferred)
            except (TypeError, ValueError, ZeroDivisionError):
                pass

        if n_instances <= 0:
            n_instances = max(n_evaluated, n_submitted, n_resolved, 0)
        else:
            n_instances = max(n_instances, n_evaluated, n_submitted, n_resolved)

        exit_total = (
            _num(exit_summary.get("exit/submitted", 0))
            + _num(exit_summary.get("exit/exit_cost", 0))
            + _num(exit_summary.get("exit/exit_context", 0))
            + _num(exit_summary.get("exit/exit_timeout", 0))
            + _num(exit_summary.get("exit/exit_format", 0))
            + exit_other
        )
        if n_instances and exit_total < n_instances:
            exit_other += (n_instances - exit_total)

        record = {
            "run_id": r.id,
            "run_name": r.name,
            "state": r.state,
            "created_at": r.created_at,

            "model": config.get("model", "unknown"),
            "strategy": config.get("strategy", "unknown"),
            "summarizer": config.get("summarizer_model", "same"),
            "instances_subset": config.get("instances_subset", "verified"),
            "eval_complete": eval_complete,

            "n_instances": n_instances,
            "n_submitted": n_submitted,
            "n_resolved": n_resolved,
            "n_evaluated": n_evaluated,
            # NaN for missing to avoid biasing aggregations
            "submission_rate": _safe_float(_get("submission_rate")),
            "resolved_rate": _safe_float(_get("resolved_rate")),
            "solve_rate": _safe_float(_get("solve_rate")),
            "eval_pass_rate": _safe_float(_get("eval_pass_rate")),
            "eval_coverage": _safe_float(_get("eval_coverage")),
            "avg_cost": _safe_float(_get("avg_cost")),
            "avg_turns": _safe_float(_get("avg_turns")),
            "cache_hit_rate": _safe_float(_get("cache_hit_rate")),
            "total_cost": _safe_float(_get("total_cost")),
            # wandb uses forward slash in exit metric names
            "exit_submitted": _num(_get("exit/submitted", 0)),
            "exit_cost": _num(_get("exit/exit_cost", 0)),
            "exit_context": _num(_get("exit/exit_context", 0)),
            "exit_timeout": _num(_get("exit/exit_timeout", 0)),
            "exit_format": _num(_get("exit/exit_format", 0)),
            "exit_other": _num(exit_other),

            "total_agent_cost": _safe_float(_get("total_agent_cost")),
            "total_summary_cost": _safe_float(_get("total_summary_cost")),
            "summary_cost_fraction": _safe_float(_get("summary_cost_fraction"), default=0.0),
        }
        records.append(record)

    return pd.DataFrame(records)


def dedupe_latest_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the newest run per run_name."""
    if df.empty or "run_name" not in df.columns:
        return df

    work = df.copy()
    if "run_id" in work.columns and work["run_name"].isna().any():
        work["run_name"] = work["run_name"].fillna(work["run_id"])

    sort_cols: list[str] = []
    if "created_at" in work.columns:
        sort_cols.append("created_at")
    if "run_id" in work.columns:
        sort_cols.append("run_id")

    if not sort_cols:
        return work.drop_duplicates("run_name", keep="last")

    work = work.sort_values(sort_cols)
    return work.drop_duplicates("run_name", keep="last")


def get_project_config() -> tuple[str | None, str | None]:
    """Get project and entity from environment variables."""
    project = os.environ.get("DASHBOARD_PROJECT") or os.environ.get("WANDB_PROJECT")
    entity = os.environ.get("DASHBOARD_ENTITY") or os.environ.get("WANDB_ENTITY")
    return project, entity
