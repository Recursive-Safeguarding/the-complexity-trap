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

### Aggregate Metrics
- `n_instances`: Number of SWE-bench instances run
- `n_submitted`: Instances where agent submitted a patch
- `submission_rate`: Fraction submitted (sweep optimization target)
- `total_cost`: Total USD cost (agent + summarizer)
- `avg_cost`: Average cost per instance
- `total_agent_cost`: Agent model cost
- `total_summary_cost`: Summarizer model cost
- `summary_cost_fraction`: Fraction of cost from summarization
- `total_turns` / `avg_turns`: Agent turns
- `total_api_calls` / `avg_api_calls`: API calls
- `total_raw_input_tokens`: Non-cached input tokens
- `total_cached_input_tokens`: Cached input tokens
- `total_output_tokens`: Output tokens
- `cache_hit_rate`: Prompt caching efficiency

### Per-Instance Table
A WandB Table (`instances`) with per-instance breakdown:
- `instance_id`, `exit_status`, `submitted`, `n_turns`
- `total_cost`, `agent_cost`, `summary_cost`
- `agent_api_calls`, `summary_api_calls`
- Token counts (raw_input, cached_input, output, summary_tokens)

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
