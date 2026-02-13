# CLI Query Tool for Experiment Analysis

CLI tool for querying WandB experiment data.

## Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Quick status
python scripts/query.py summary

# Best models ranked (markdown for slides)
python scripts/query.py --format markdown leaderboard

# Compare to paper baselines
python scripts/query.py --model glm-4.7 paper-comparison

# Strategy breakdown
python scripts/query.py --model glm-4.7 compare-strategies
```

## Commands

### `summary` - Quick One-Liner Status

Returns a single line with the best performer, suitable for quick checks.

```bash
python scripts/query.py summary
# Best: glm-4.7 raw @ 64.0% ($1.00) — vs paper: +0.0% | 4 strategies, 902 instances
```

### `leaderboard` - Best Models Ranked

Shows all model×strategy combinations ranked by solve rate, with paper comparison deltas.

```bash
python scripts/query.py leaderboard
python scripts/query.py --format markdown leaderboard  # For presentations
python scripts/query.py --strategy raw leaderboard     # Filter by strategy
```

**Output columns:**
- `rank` - Position by solve rate
- `model` - Model name
- `strategy` - Context management strategy
- `solve_rate` - Resolved / Total instances
- `rate_delta` - Difference vs paper baseline
- `avg_cost` - Average cost per instance
- `cost_delta` - Cost difference vs paper (negative = savings)
- `n_instances` - Total instances evaluated

### `paper-comparison` - Compare to Paper Baselines

Direct comparison of your results against the paper baselines (arXiv:2508.21433).

```bash
python scripts/query.py paper-comparison
python scripts/query.py --model glm-4.7 paper-comparison
python scripts/query.py --format markdown paper-comparison  # For slides
```

**Output columns:**
- `strategy` - Context management strategy
- `our_rate` / `paper_rate` - Solve rates
- `rate_delta` - Your improvement over paper
- `our_cost` / `paper_cost` - Costs
- `cost_delta` - Cost savings vs paper

### `compare-strategies` - Strategy Comparison

Compare all strategies for a specific model, showing deltas vs the raw baseline.

```bash
python scripts/query.py --model glm-4.7 compare-strategies
```

**Output columns:**
- `strategy` - Strategy name (raw shown first as baseline)
- `solve_rate` - Absolute solve rate
- `rate_vs_raw` - Delta compared to raw baseline
- `avg_cost` - Cost per instance
- `cost_vs_raw` - Cost savings vs raw

**Insight line:** Shows whether context management helps or hurts the model.

### `failures` - Exit Status Analysis

Breakdown of why instances fail (exit statuses).

```bash
python scripts/query.py failures
python scripts/query.py --model glm-4.7 failures
python scripts/query.py --strategy hybrid failures
```

**Status categories:**
- `Submitted` - Successfully produced a patch
- `Cost` - Hit cost limit
- `Context` - Context window overflow
- `Timeout` - Exceeded time limit
- `Format` - Invalid output format
- `Other` - Other failures

### `runs` - List Individual Runs

List all runs with optional filtering.

```bash
python scripts/query.py runs
python scripts/query.py --eval-only runs              # Only evaluated runs
python scripts/query.py --model glm-4.7 --eval-only runs
python scripts/query.py --min-instances 50 runs      # Runs with 50+ instances
```

## Repeated-Run Uncertainty

Use `scripts/repeated_run_stats.py` when you have independent reruns of the
same configuration and want uncertainty across runs (not just within one run).

```bash
# Default: verified-mini, evaluated runs only, n_instances in [40, 60], min 2 repeats
python scripts/repeated_run_stats.py

# Focus on one model/strategy
python scripts/repeated_run_stats.py --model glm-4.7 --strategy on_demand

# Include run names and export markdown
python scripts/repeated_run_stats.py --show-runs --format markdown --output /tmp/repeated_runs.md
```

What it reports per strict config group:
- `mean_rate`: mean solve rate across reruns
- `t_ci_95`: 95% t-interval across run-level solve rates
- `std_pp`: sample standard deviation (percentage points) across reruns
- `pooled_rate` and `pooled_wilson_95`: pooled instance-level reference

Strict grouping key:
- `model`, `strategy`, `summarizer`, `instances_subset`
- `hp_obs_n`, `hp_sum_n`, `hp_sum_keep_m`
- `hp_limit_aware`, `hp_limit_fraction`, `hp_limit_min_tokens`

Interpretation:
- `paper_results.py` Wilson intervals quantify uncertainty for **one run**
  (`k/n` over instances).
- `repeated_run_stats.py` t-intervals quantify uncertainty **across reruns**
  (run-to-run variation).

## Global Options

All options must come **before** the subcommand:

```bash
python scripts/query.py [OPTIONS] COMMAND
```

| Option | Short | Description |
|--------|-------|-------------|
| `--project` | `-p` | WandB project (default: `$WANDB_PROJECT` or `$DASHBOARD_PROJECT`) |
| `--entity` | `-e` | WandB entity (default: `$WANDB_ENTITY` or `$DASHBOARD_ENTITY`) |
| `--format` | `-f` | Output format: `table`, `markdown`, `json`, `csv` |
| `--model` | `-m` | Filter by model name (partial match) |
| `--strategy` | `-s` | Filter by strategy (partial match) |
| `--min-instances` | | Minimum instances per run (default: 10) |
| `--eval-only` | | Only include evaluated runs |
| `--debug` | | Print debug info (matched runs, counts) |

## Output Formats

### Table (default)

Rich terminal tables with color coding:
- Green: High solve rates (≥50%), cost savings
- Yellow: Medium solve rates (30-50%)
- Red: Low solve rates (<30%), cost increases

```bash
python scripts/query.py leaderboard
```

### Markdown

Copy-paste ready for slides, docs, or GitHub:

```bash
python scripts/query.py --format markdown leaderboard
```

### JSON

For scripts and pipelines:

```bash
python scripts/query.py --format json leaderboard | jq '.data[0]'
```

### CSV

For spreadsheets:

```bash
python scripts/query.py --format csv leaderboard > results.csv
```

## Environment Variables

Set these in your `.env` file or export them:

```bash
# Required
WANDB_API_KEY=your-wandb-api-key

# Project configuration (pick one pair)
WANDB_PROJECT=the-complexity-trap
WANDB_ENTITY=your-entity

# Or use dashboard-specific vars (higher priority)
DASHBOARD_PROJECT=the-complexity-trap
DASHBOARD_ENTITY=your-entity
```

## Paper Baselines Reference

The tool compares against baselines from "The Complexity Trap" paper (arXiv:2508.21433):

| Model | Strategy | Solve Rate | Avg Cost |
|-------|----------|------------|----------|
| qwen3-coder-480b | raw | 53.4% | $1.29 |
| qwen3-coder-480b | observation_masking | 54.8% | $0.61 |
| qwen3-coder-480b | llm_summary | 53.8% | $0.64 |
| qwen3-coder-480b | hybrid | 54.0%* | $0.50* |

Strategy names are normalized automatically:
- `obs_masking` → `observation_masking`
- `obs` → `observation_masking`
- `sum` → `llm_summary`
- `hyb` → `hybrid`

## Common Workflows

### Preparing for a Presentation

```bash
# Get the key numbers
python scripts/query.py summary

# Generate markdown tables for slides
python scripts/query.py --format markdown leaderboard > slides/leaderboard.md
python scripts/query.py --model glm-4.7 --format markdown compare-strategies >> slides/glm47.md
python scripts/query.py --format markdown paper-comparison >> slides/paper_comparison.md
```

### Investigating Failures

```bash
# What's the failure distribution?
python scripts/query.py failures

# Is one model worse?
python scripts/query.py --model glm-4.7 failures
python scripts/query.py --model kimi-2.5 failures

# Is one strategy worse?
python scripts/query.py --strategy hybrid failures
```

### Comparing Models

```bash
# Overall leaderboard
python scripts/query.py leaderboard

# Filter to specific strategy
python scripts/query.py --strategy raw leaderboard
python scripts/query.py --strategy observation_masking leaderboard
```

### Debugging Missing Data

```bash
# Enable debug mode
python scripts/query.py --debug --model typo-model leaderboard
# [DEBUG] Fetching from project: ox/the-complexity-trap
# [DEBUG] Fetched 45 runs
# [DEBUG] 0 runs match --model typo-model
```

## Architecture

```
scripts/
├── query.py              # CLI entry point (argparse)
├── query_metrics.py      # Pure analysis functions
├── query_formatters.py   # Output rendering (table/md/json/csv)
└── dashboard_shared.py   # Shared data fetching, paper baselines
```

The tool is designed with separation of concerns:
- **query.py**: CLI parsing and dispatch only
- **query_metrics.py**: Stateless data transformations (testable)
- **query_formatters.py**: Output rendering (pluggable formats)

## Troubleshooting

### "No evaluated runs found"

Check that runs have completed evaluation:
```bash
python scripts/query.py --debug runs
```

### Paper deltas showing "—"

Strategy name may not match. The tool normalizes common aliases, but if your WandB config uses a non-standard name, update `STRATEGY_ALIASES` in `query_metrics.py`.

### Slow queries

WandB API can be slow. Consider:
1. Filtering with `--model` or `--strategy` to reduce data
2. Using `--min-instances` to skip small test runs

### Missing cost data

Some runs may not have cost data (shows as "—"). This happens when:
- Cost tracking wasn't enabled
- Run was interrupted before cost summary
