# WandB Sweeps for The Complexity Trap

Run experiments comparing context management strategies across models.

## Quick Start

```bash
uv sync --extra dev
wandb login

# Create a sweep
wandb sweep sweeps/quick_test.yaml --project complexity-trap-test
# Returns: <sweep_id>

# Run an agent (executes one configuration from the sweep)
wandb agent <your-entity>/complexity-trap-test/<sweep_id> --count 1

# Run more agents in parallel
wandb agent <your-entity>/complexity-trap-test/<sweep_id> --count 5
```

## Available Sweeps

| Sweep | Models | Strategies | Instances | Est. Time | Est. Cost |
|-------|--------|------------|-----------|-----------|-----------|
| `quick_test.yaml` | 4 (budget) | 4 | 3 | ~2-3 hrs | ~$5-15 |
| `bedrock_repro.yaml` | 1 (Bedrock) | 4 | 50 | varies | varies |

## Strategies

| Strategy | Config File | Description |
|----------|-------------|-------------|
| `raw` | `default_no_demo.yaml` | Baseline - no context management |
| `observation_masking` | `default_no_demo_N=1_M=10.yaml` | Keep last 10 observations (paper's key finding) |
| `llm_summary` | `default_no_demo_checkpoint_same_model_openhands_N=21_M=10.yaml` | LLM summarization every 21 turns |
| `hybrid` | `...N=21_M=10_masking_M=10.yaml` | Combined: summarization + masking |

## Models (Budget-Friendly)

| Preset | Provider | Notes |
|--------|----------|-------|
| `deepseek-chat` | DeepSeek | V3, very cheap |
| `gpt-4o-mini` | OpenAI | Cheap, reliable |
| `glm-4.6` | ZhipuAI | Via Z.AI endpoint |
| `kimi-k2` | Moonshot | Anthropic-compatible |

## Direct Usage (No Sweep)

```bash
# Run single experiment
python scripts/run_sweep.py \
    --model deepseek-chat \
    --strategy observation_masking \
    --instances-slice :3 \
    --wandb \
    --wandb-project complexity-trap-test

# Dry run (show command without executing)
python scripts/run_sweep.py \
    --model deepseek-chat \
    --strategy observation_masking \
    --dry-run
```

## Metrics Logged to WandB

X-axes are set via `wandb.define_metric()`:
- `step/*`, `instance/*`, `cumulative/*` → `global_step` (agent turns)
- `total_*`, `avg_*`, `exit/*` → `n_instances`

### Per-Step Metrics (`global_step` x-axis)

| Metric | Description |
|--------|-------------|
| `step` | Local step number within current instance (resets per instance) |
| `step/cost` | Cost of this turn in USD |
| `step/tokens_in` | Total input tokens (raw + cached) |
| `step/tokens_out` | Output tokens generated |
| `step/tokens_raw_input` | Non-cached input tokens (cost driver) |
| `step/tokens_cached_input` | Cached input tokens (cost savings) |
| `step/tokens_internal_reasoning` | Thinking tokens (o1, o3, DeepSeek-R1) |
| `step/inference_time_ms` | Model response latency |
| `step/execution_time_ms` | Environment command execution time |
| `step/cache_hit_rate` | Fraction of input tokens from cache |

`instance/*` variants reset per instance; `cumulative/*` accumulates across all.

Additional `instance/*` and `cumulative/*` metrics:
| Metric | Description |
|--------|-------------|
| `*/api_calls` | API call count |
| `*/inference_time_ms` | Total model response time |
| `*/execution_time_ms` | Total environment execution time |

### Aggregate Metrics (`n_instances` x-axis)

| Metric | Description |
|--------|-------------|
| `n_instances` | Number of SWE-bench instances run |
| `n_submitted` | Instances that submitted a patch |
| `submission_rate` | Fraction submitted |
| `cache_hit_rate` | Prompt caching efficiency |
| `avg_cost` | Average cost per instance |
| `avg_turns` | Average agent turns per instance |
| `avg_api_calls` | Average API calls per instance |
| `avg_tokens_per_turn` | Token efficiency metric |
| `total_turns` | Total agent turns |
| `total_api_calls` | Total agent API calls |
| `total_summary_api_calls` | Summarizer API calls |
| `total_rloop_api_calls` | Retry loop API calls |

### Cost Breakdown

| Metric | Description |
|--------|-------------|
| `total_cost` | Total USD cost (agent + summarizer + rloop) |
| `total_agent_cost` | Main agent model cost |
| `total_summary_cost` | LLM-Summary model cost (if enabled) |
| `total_rloop_cost` | Retry loop reviewer cost (if enabled) |
| `summary_cost_fraction` | Summary cost as % of total |
| `rloop_cost_fraction` | Retry loop cost as % of total |
| `summary_api_fraction` | Summary API calls as % of total API calls |
| `rloop_api_fraction` | Retry loop API calls as % of total API calls |

### Token Breakdown

| Metric | Description |
|--------|-------------|
| `total_raw_input_tokens` | Non-cached input tokens |
| `total_cached_input_tokens` | Cached input tokens |
| `total_output_tokens` | Output tokens |
| `total_internal_reasoning_tokens` | Thinking tokens (o1, o3, R1) |
| `total_summary_raw_input_tokens` | Summarizer non-cached input |
| `total_summary_cached_input_tokens` | Summarizer cached input |
| `total_summary_output_tokens` | Summarizer output tokens |

### Exit Status Distribution (`exit/*`)

Why instances terminated. For `"submitted (exit_cost)"`, the category is `exit_cost` (the actual reason); `n_submitted` still counts it as submitted.

| Metric | Description |
|--------|-------------|
| `exit/submitted` | Submitted, exited normally |
| `exit/exit_cost` | Cost limit hit |
| `exit/exit_context` | Context window exceeded |
| `exit/exit_timeout` | Execution or command timeout |
| `exit/exit_format` | Repeated format/syntax errors |
| `exit/exit_api` | API errors (rate limits, etc.) |
| `exit/exit_environment` | Docker/environment errors |
| `exit/exit_forfeit` | Agent forfeited |
| `exit/exit_command` | Exit command issued |
| `exit/exit_error` | Other runtime errors |
| `exit/other` | Uncategorized exit status |

### Per-Instance Table

A WandB Table (`instances`) with per-instance details:
- `instance_id`, `exit_status`, `exit_category`, `submitted`, `n_turns`
- `total_cost`, `agent_cost`, `summary_cost`, `rloop_cost`
- `agent_api_calls`, `summary_api_calls`, `rloop_api_calls`
- `raw_input_tokens`, `cached_input_tokens`, `output_tokens`, `internal_reasoning_tokens`
- `summary_raw_input_tokens`, `summary_cached_input_tokens`, `summary_output_tokens`
- `cache_hit_rate` (per-instance)
- `review_score` (if using ScoreRetryLoop)

## Weave (LLM Tracing)

`scripts/run_sweep.py` enables Weave tracing by default. This records LLM calls (via LiteLLM) so you can inspect token usage, latency, and prompts/responses in the Weave UI alongside the corresponding WandB run.

To disable Weave (e.g., if you don't want tracing overhead):

```bash
python scripts/run_sweep.py --model deepseek-chat --strategy raw --no-weave --wandb
```

(Note: Weave tracing requires running `sweagent` **in-process**. If you switch to subprocess execution, Weave will not trace the child process.)

```bash
python scripts/run_sweep.py --model deepseek-chat --strategy raw --wandb --execution subprocess
```

## Parse Existing Trajectories

```bash
# Parse and display metrics from a trajectory directory
python scripts/parse_trajectory.py trajectories/youdar/2025-12-10_...

# Output as JSON
python scripts/parse_trajectory.py trajectories/youdar/2025-12-10_... --json
```

## Environment Variables

API keys (in `.env` file):
- `OPENAI_API_KEY`
- `DEEPSEEK_API_KEY`
- `ZHIPUAI_API_KEY`
- `MOONSHOT_API_KEY`

### AWS Bedrock (for `bedrock_repro.yaml`)

- `AWS_DEFAULT_REGION=eu-west-2` (or `AWS_REGION`)
- Auth (pick one):
  - `AWS_BEARER_TOKEN_BEDROCK=...` (Bedrock API key / bearer token), OR
  - `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY` (+ optional `AWS_SESSION_TOKEN`), OR
  - `AWS_PROFILE=...` (if you use `~/.aws/config`)

(Note: For many Bedrock models, LiteLLM doesn't have a cost-map entry. Keep `cost_limit: 0.0` in sweeps and rely on `call_limit` instead.)
