# WandB visualization

Visualizations for comparing context management strategies (raw, observation masking, LLM summary, hybrid) across models.

## Executive summary

#### Pareto Scatter ⭐⭐⭐
| Setting | Value |
|---------|-------|
| X-axis | `avg_cost` (log scale optional) |
| Y-axis | `submission_rate` |
| Color | `config.strategy` |
| Shape | `config.model` |
| Size | `n_instances` (confidence indicator) |

Upper-left = Pareto-optimal.

#### Summary Table
| Metric | Grouped By |
|--------|------------|
| `submission_rate` | model, strategy |
| `avg_cost` | model, strategy |
| `avg_turns` | model, strategy |
| `cache_hit_rate` | model, strategy |

## Hyperparameter Analysis

#### Parallel Coordinates ⭐⭐⭐
Dimensions: `config.model` → `config.strategy` → `submission_rate` → `avg_cost` → `avg_turns` → `cache_hit_rate`

Highlight top 10% runs to spot patterns.

#### Parameter Importance ⭐⭐
Does `model` or `strategy` matter more for `submission_rate`?

## Token Snowball Analysis

#### Cumulative Token Growth ⭐⭐⭐
| Setting | Value |
|---------|-------|
| X-axis | `global_step` |
| Y-axis | `cumulative/tokens_in` |
| Lines | Grouped by `config.strategy` |
| Aggregation | Mean with stddev band |

Shows the "token snowball effect" - raw grows exponentially, masking flattens it.

**Expected**:
```
Tokens
  │    raw ────────────╱
  │               ╱
  │          ╱
  │     ╱
  │   masking ─────────────
  │   hybrid ──────────────
  └────────────────────────── Step
```

#### Per-Turn Token Box Plot
X: `config.strategy`, Y: `tokens_per_turn` (box with outliers)

## Failure Analysis

#### Exit Status Stacked Bar ⭐⭐⭐
| Setting | Value |
|---------|-------|
| X-axis | `config.strategy` (or `config.model`) |
| Y-axis | Count |
| Stacked by | `exit/*` categories |
| Colors | submitted=green, exit_cost=orange, exit_context=red |

Shows *why* runs fail.

#### Model×Strategy Heatmap ⭐⭐
| Setting | Value |
|---------|-------|
| Rows | `config.model` |
| Columns | `config.strategy` |
| Color | `submission_rate` (green=high) |

## Strategy Comparison

#### Submission Rate by Strategy
X: `config.strategy`, Y: `submission_rate`, grouped by model

#### Cost by Strategy
X: `config.strategy`, Y: `avg_cost`, grouped by model

#### Trajectory Length (Paper Fig 4)
X: `config.strategy`, Y: `n_turns` — tests whether LLM-Summary causes trajectory elongation

## Cost Breakdown

#### Cost Components
X: `config.strategy`, Y: stacked `total_agent_cost` / `total_summary_cost` / `total_rloop_cost`

#### Summary Cost Fraction Over Time
X: `n_instances`, Y: `summary_cost_fraction`, lines per strategy

## Repository Analysis

#### Submission Rate by Repo
X: `repo/*` (sorted), Y: submission rate

#### Repo Distribution
X: `repo/*`, Y: count

## Instance-Level

#### Instances Table
Columns: `instance_id`, `repo`, `exit_category`, `submitted`, `n_turns`, `total_cost`, `patch_lines`, `tokens_per_turn`

#### Patch Size vs Success
X: `patch_lines`, Y: `submitted` (jittered), color by strategy

## Caching

#### Cache Hit Over Time
X: `n_instances`, Y: `cache_hit_rate`, lines per model

#### Cache Hit by Strategy
X: `config.strategy`, Y: `cache_hit_rate`

## Report Layout

```
┌─────────────────────────────────────────────────────────────┐
│ EXECUTIVE SUMMARY                                           │
│ ┌─────────────────────┐ ┌─────────────────────────────────┐ │
│ │  Pareto Scatter     │ │  Summary Metrics Table          │ │
│ └─────────────────────┘ └─────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ HYPERPARAMETER ANALYSIS                                     │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │            Parallel Coordinates Plot                     │ │
│ └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ TOKEN SNOWBALL ANALYSIS                                     │
│ ┌─────────────────────────────────────────────────────────┐ │
│ │     Cumulative Token Growth by Strategy                  │ │
│ └─────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ FAILURE ANALYSIS                                            │
│ ┌─────────────────────┐ ┌─────────────────────────────────┐ │
│ │ Exit Status Stacked │ │ Model×Strategy Heatmap          │ │
│ └─────────────────────┘ └─────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ COST & TRAJECTORY                                           │
│ ┌─────────────────────┐ ┌─────────────────────────────────┐ │
│ │ Cost by Strategy    │ │ Turn Count Box Plot             │ │
│ └─────────────────────┘ └─────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│ DEEP DIVE: Instances Table                                  │
└─────────────────────────────────────────────────────────────┘
```

## Key Questions

| Chart | Question |
|-------|----------|
| Pareto Scatter | Best bang-for-buck model×strategy? |
| Parallel Coords | What patterns lead to success? |
| Token Growth | Does masking flatten context growth? |
| Exit Status | Why do runs fail? Cost? Context? Timeout? |
| Grouped Bars | Which strategy wins for each model? |
| Trajectory Box | Does LLM-Summary cause trajectory elongation? |
| Cost Breakdown | How much overhead does summarization add? |
| Repo Analysis | Which codebases are hardest? |
| Instances Table | What went wrong with specific failures? |

## Metrics (from wandb_hook.py)

### Per-Instance
- `instance_id`, `repo`, `exit_status`, `exit_category`, `submitted`
- `n_turns`, `total_cost`, `agent_cost`, `summary_cost`, `rloop_cost`
- `raw_input_tokens`, `cached_input_tokens`, `output_tokens`
- `cache_hit_rate`, `patch_lines`, `instance_duration_ms`, `tokens_per_turn`

### Aggregates
- `submission_rate`, `avg_cost`, `avg_turns`, `avg_tokens_per_turn`
- `avg_patch_lines`, `avg_duration_ms`
- `turn_std`, `turn_median`, `turn_min`, `turn_max`
- `summary_cost_fraction`, `rloop_cost_fraction`
- `exit/*` distribution, `repo/*` distribution
- `turn_distribution` (histogram)

### Step-Level
- `step/tokens_in`, `step/cost`, `step/cache_hit_rate`
- `cumulative/tokens_in`, `cumulative/cost`

## Implementation Notes

1. **Pareto Plot**: WandB Scatter Plot panel with custom axes
2. **Heatmap**: Custom Charts with Vega spec (no native support)
3. **Stacked Bars**: WandB Bar Plot with grouping
4. **Box Plots**: Bar Plot panel settings
5. **Histogram**: Logged via `wandb.Histogram()` in `on_end()`

## References

- [WandB Sweeps](https://docs.wandb.ai/guides/sweeps/visualize-sweep-results/)
- [The Complexity Trap](https://arxiv.org/html/2508.21433)
- [Context Rot Research](https://research.trychroma.com/context-rot)
