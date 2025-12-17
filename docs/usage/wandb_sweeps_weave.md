# WandB Sweeps & Weave

Run experiments with WandB logging and optional Weave tracing.

## Setup

```bash
uv sync --extra dev
uv run wandb login
```

## Quick Start

Single run with logging:

```bash
uv run python scripts/run_sweep.py \
  --model bedrock-qwen3-32b \
  --strategy observation_masking \
  --instances-slice :3 \
  --wandb --wandb-project complexity-trap-test
```

Dry run:

```bash
uv run python scripts/run_sweep.py --model deepseek-chat --strategy raw --dry-run
```

## Strategies

| Strategy | Config |
|----------|--------|
| `raw` | `default_no_demo.yaml` |
| `observation_masking` | `default_no_demo_N=1_M=10.yaml` |
| `llm_summary` | `default_no_demo_checkpoint_same_model_openhands_N=21_M=10.yaml` |
| `hybrid` | `...N=21_M=10_masking_M=10.yaml` |

## Running Sweeps

```bash
# Create sweep
uv run wandb sweep sweeps/quick_test.yaml

# Run agent (from repo root)
uv run wandb agent <entity>/<project>/<sweep_id>
```

Available sweeps:
- `sweeps/quick_test.yaml` - cheap models, small slice
- `sweeps/bedrock_repro.yaml` - Bedrock Qwen3, all 4 strategies

## Weave Tracing

Enabled by default when `--wandb` is set. Requires in-process execution (default).

Disable with `--no-weave` or `--execution subprocess`.

## Bedrock Notes

Set `AWS_REGION` (or `AWS_DEFAULT_REGION`) and auth env vars (`AWS_PROFILE`, `AWS_ACCESS_KEY_ID`+`AWS_SECRET_ACCESS_KEY`, or `AWS_BEARER_TOKEN_BEDROCK`).

Cost limits are disabled for Bedrock models (LiteLLM cost maps incomplete).

## Troubleshooting

- No WandB runs: `uv run wandb login`
- No Weave traces: ensure `--wandb` and `--execution inprocess`
- Paths not found in sweep: run `wandb agent` from repo root
