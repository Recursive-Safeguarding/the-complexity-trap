"""Reference results from 'The Complexity Trap' paper (JetBrains Research).

Source: Lindenbauer, Slinko, Felder, Bogomolov & Zharov,
        "The Complexity Trap: Simple Observation Masking Is as Efficient as
         LLM Summarization for Agent Context Management"
Benchmark: SWE-bench Verified (500 instances)

Strategies:
- raw: No context management (baseline)
- obs_masking: Observation masking (keep last M observations)
- llm_summary: LLM-based trajectory summarization (every N turns)
- hybrid: Combined obs_masking + llm_summary (50-instance subset only)

Parameters:
- M=10: Rolling window size for observation masking / tail turns for LLM-Summary
- N=21: Summarize every N turns in LLM-Summary
- Summarizer: Gemini 1.5 Flash (for all LLM-Summary experiments)
"""

from __future__ import annotations

# Main Results (Table 1 - SWE-bench Verified, 500 instances)
# Format: solve_rate (%), cost_per_instance ($), ci = 95% confidence interval
PAPER_RESULTS: dict[str, dict] = {
    "Qwen3-32B": {
        "raw": {"solve_rate": 17.0, "cost_per_instance": 1.12, "sr_ci": 3.3, "cost_ci": 0.18},
        "obs_masking": {"solve_rate": 15.0, "cost_per_instance": 0.55, "M": 10, "sr_ci": 3.1, "cost_ci": 0.09},
        "llm_summary": {"solve_rate": 16.0, "cost_per_instance": 0.50, "N": 21, "M": 10, "sr_ci": 3.3, "cost_ci": 0.07},
    },
    "Qwen3-32B-thinking": {
        "raw": {"solve_rate": 23.0, "cost_per_instance": 0.51, "sr_ci": 3.7, "cost_ci": 0.07},
        "obs_masking": {"solve_rate": 24.6, "cost_per_instance": 0.46, "M": 10, "sr_ci": 3.8, "cost_ci": 0.05},
        "llm_summary": {"solve_rate": 24.8, "cost_per_instance": 0.51, "N": 21, "M": 10, "sr_ci": 3.9, "cost_ci": 0.06},
    },
    "Qwen3-Coder-480B": {
        "raw": {"solve_rate": 53.4, "cost_per_instance": 1.29, "sr_ci": 4.3, "cost_ci": 0.26},
        "obs_masking": {"solve_rate": 54.8, "cost_per_instance": 0.61, "M": 10, "sr_ci": 4.4, "cost_ci": 0.06},
        "llm_summary": {"solve_rate": 53.8, "cost_per_instance": 0.64, "N": 21, "M": 10, "sr_ci": 4.2, "cost_ci": 0.06},
    },
    "Gemini-2.5-Flash": {
        "raw": {"solve_rate": 32.8, "cost_per_instance": 0.41, "sr_ci": 4.1, "cost_ci": 0.08},
        "obs_masking": {"solve_rate": 35.6, "cost_per_instance": 0.18, "M": 10, "sr_ci": 4.2, "cost_ci": 0.03},
        "llm_summary": {"solve_rate": 36.0, "cost_per_instance": 0.24, "N": 21, "M": 10, "sr_ci": 4.1, "cost_ci": 0.04},
    },
    "Gemini-2.5-Flash-thinking": {
        "raw": {"solve_rate": 40.4, "cost_per_instance": 0.56, "sr_ci": 4.3, "cost_ci": 0.10},
        "obs_masking": {"solve_rate": 36.4, "cost_per_instance": 0.24, "M": 10, "sr_ci": 4.2, "cost_ci": 0.04},
        "llm_summary": {"solve_rate": 31.4, "cost_per_instance": 0.25, "N": 21, "M": 10, "sr_ci": 4.0, "cost_ci": 0.05},
    },
}

# LLM-Summary generation costs (direct API costs for generating summaries)
SUMMARY_GENERATION_COSTS: dict[str, dict] = {
    "Qwen3-32B": {"cost": 0.0143, "proportion": 2.86},
    "Qwen3-32B-thinking": {"cost": 0.0033, "proportion": 0.65},
    "Qwen3-Coder-480B": {"cost": 0.0439, "proportion": 7.20},
    "Gemini-2.5-Flash": {"cost": 0.0161, "proportion": 6.71},
    "Gemini-2.5-Flash-thinking": {"cost": 0.0131, "proportion": 5.24},
}

# Hybrid results (Qwen3-Coder-480B on 50-instance subset)
# Note: Paper only reports relative improvements, not absolute numbers
HYBRID_RESULTS = {
    "model": "Qwen3-Coder-480B",
    "subset_size": 50,
    "config": {"N": 43, "M": 10, "W": 10},  # W = obs masking window in hybrid
    "relative_improvements": {
        "vs_raw": "+2.6 pp solve rate",
        "vs_obs_masking": "-7% cost",
        "vs_llm_summary": "-11% cost",
    },
    "projected_savings_500": {
        "vs_obs_masking": 20,  # $20
        "vs_llm_summary": 35,  # $35
    },
}

# Cost reduction from obs_masking vs raw (%)
COST_REDUCTION_OBS_MASKING = {
    model: round(
        100 * (1 - results["obs_masking"]["cost_per_instance"] / results["raw"]["cost_per_instance"]), 1
    )
    for model, results in PAPER_RESULTS.items()
}

# Key findings
EXPECTED_BEHAVIOR = """
Key findings from the paper:
1. Observation Masking (M=10) reduces cost by 50-57% with equal/better solve rate (except thinking models)
2. LLM-Summary reduces cost by 41-55% with similar solve rate
3. Hybrid approach provides additional 7-11% cost reduction vs individual strategies
4. Thinking models show DEGRADED performance with context management strategies
5. Context management generally does NOT hurt performance for non-thinking models
"""


def compare_to_paper(model: str, strategy: str, solve_rate: float, cost: float) -> dict:
    """Compare experimental results to paper reference."""
    if model not in PAPER_RESULTS:
        return {"status": "unknown_model", "model": model, "known_models": list(PAPER_RESULTS.keys())}
    if strategy not in PAPER_RESULTS[model]:
        return {"status": "unknown_strategy", "strategy": strategy}

    ref = PAPER_RESULTS[model][strategy]
    sr_within_ci = abs(solve_rate - ref["solve_rate"]) <= ref.get("sr_ci", 5.0)
    cost_within_ci = abs(cost - ref["cost_per_instance"]) <= ref.get("cost_ci", 0.20)

    return {
        "model": model,
        "strategy": strategy,
        "measured": {"solve_rate": solve_rate, "cost_per_instance": cost},
        "paper": ref,
        "solve_rate_diff": round(solve_rate - ref["solve_rate"], 2),
        "cost_diff": round(cost - ref["cost_per_instance"], 2),
        "cost_ratio": round(cost / ref["cost_per_instance"], 2) if ref["cost_per_instance"] > 0 else None,
        "within_confidence_interval": {"solve_rate": sr_within_ci, "cost": cost_within_ci},
    }
