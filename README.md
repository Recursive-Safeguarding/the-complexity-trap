# The Complexity Trap: Multi-Model Evaluation

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/) [![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet.svg)](https://docs.astral.sh/uv/) [![WandB](https://img.shields.io/badge/WandB-experiment%20tracking-orange.svg)](https://wandb.ai/) [![LiteLLM](https://img.shields.io/badge/LiteLLM-multi--provider-green.svg)](https://docs.litellm.ai/) [![arXiv](https://img.shields.io/badge/arXiv-2508.21433-b31b1b.svg)](https://arxiv.org/abs/2508.21433)

*Fork of JetBrains Research's "The Complexity Trap" with multi-model evaluation infrastructure*

*Original README: [README_upstream.md](README_upstream.md)*

---

## What's New

This fork extends the original paper's experiments with:

- **Multi-provider LLM support** via LiteLLM: GLM-4.7, Kimi-K2, MiniMax-M2.1, DeepSeek, AWS Bedrock (Qwen3), OpenRouter, Anthropic, OpenAI
- **WandB sweep orchestration** with parallel agent support for systematic hyperparameter search
- **Query CLI** (`scripts/query.py`) for quick results analysis and paper comparison
- **Dashboards**: Streamlit web UI and Rich-based TUI for experiment monitoring
- **VPS deployment scripts** for long-running sweeps on remote servers
- **Decoupled evaluation** architecture to prevent Docker hangs from blocking sweeps
- **902 evaluated instances** across 4 context management strategies

---

## Evaluation Results

> **902 instances** evaluated on **SWE-bench Verified** with **GLM-4.7**

### Strategy Comparison

| Strategy | Solve Rate | vs Raw | Cost/Instance | Cost Savings |
|----------|------------|--------|---------------|--------------|
| **raw** | 64.0% | — | $1.00 | baseline |
| observation_masking | 62.0% | -2.0% | $0.61 | -39% |
| llm_summary | 54.2% | -9.8% | $0.50 | -50% |
| hybrid | 56.0% | -8.0% | $0.32 | -68% |

### Key Finding

**Context management hurts GLM-4.7 performance**, opposite of the paper's findings with qwen3-coder-480b. The raw baseline achieves the highest solve rate (64.0%), while observation masking drops performance by 2.0% (vs paper's +1.4% gain) and LLM-Summary drops it by 9.8% (vs paper's +0.4%). Cost savings remain substantial at 39-68%.

This suggests context management strategies may be model-dependent rather than universally beneficial.

### Paper Comparison

| Metric | Paper (qwen3-coder-480b) | Ours (GLM-4.7) |
|--------|--------------------------|----------------|
| Raw solve rate | 53.4% | **64.0%** |
| Best strategy | hybrid (+1.6%) | raw (baseline) |
| Cost reduction | 50-58% | 39-68% |
| Dataset size | 500 instances | 902 instances |

---

## Quick Start

```bash
# Clone and setup
git clone https://github.com/youkad/the-complexity-trap.git
cd the-complexity-trap
uv venv .venv --python 3.12 --seed
source .venv/bin/activate
uv sync --extra dev

# Configure API keys
cp .env.example .env
# Edit .env with your API keys (ZHIPUAI_API_KEY, MOONSHOT_API_KEY, etc.)

# Run a quick test (5 instances)
python scripts/run_sweep.py \
  --model glm-4.7 \
  --strategy raw \
  --instances-slice :5
```

---

## Query Results

```bash
# Quick status summary
python scripts/query.py summary
# Best: glm-4.7 raw @ 64.0% ($1.00) — 4 strategies, 902 instances

# Leaderboard (markdown for presentations)
python scripts/query.py --format markdown leaderboard

# Paper comparison
python scripts/query.py --model glm-4.7 paper-comparison

# Strategy breakdown
python scripts/query.py --model glm-4.7 compare-strategies

# Exit status analysis
python scripts/query.py failures
```

---

## Available Models

| Preset | Provider | Description | Context |
|--------|----------|-------------|---------|
| `glm-4.7` | Z.AI | GLM-4.7 (agentic coding) | 200K |
| `glm-4.6` | Z.AI | GLM-4.6 (355B MoE) | 200K |
| `kimi-2.5` | Moonshot | Kimi K2.5 (1T MoE, 32B active) | 262K |
| `minimax-m2.1` | MiniMax | M2.1 (enhanced multilingual) | 205K |
| `deepseek-chat` | DeepSeek | DeepSeek V3 | 128K |
| `bedrock-qwen3-coder-480b` | AWS Bedrock | Qwen3 Coder 480B | 262K |
| `bedrock-qwen3-32b` | AWS Bedrock | Qwen3 32B | 32K |
| `gpt-4o` | OpenAI | GPT-4o | 128K |
| `claude-sonnet-4.5` | Anthropic | Claude Sonnet 4.5 | 200K |

List all presets: `python scripts/run_model.py --list`

---

## Strategies

| Strategy | Config | Description |
|----------|--------|-------------|
| `raw` | `default_no_demo_raw.yaml` | No context management (baseline) |
| `observation_masking` | `default_no_demo_N=1_M=10.yaml` | Keep last M=10 observations |
| `llm_summary` | `default_no_demo_checkpoint_same_model_openhands_N=21_M=10.yaml` | Summarize every N=21 turns |
| `hybrid` | `default_no_demo_checkpoint_same_model_openhands_N=21_M=10_masking_M=10.yaml` | Both strategies combined |

---

## WandB Sweeps

```bash
# Create and run a sweep
wandb sweep sweeps/smart_search.yaml
wandb agent <SWEEP_ID>

# Or use the convenience script
./scripts/vps_sweep.sh sweeps/smart_search.yaml --start
```

Sweep configs in `sweeps/`:
- `smart_search.yaml` - Bayesian search across 6 models x 4 strategies
- `bedrock_repro.yaml` - Paper reproduction on AWS Bedrock
- `quick_test.yaml` - Sanity check (small slice)

---

## Dashboards

```bash
# Web dashboard (Streamlit)
streamlit run scripts/dashboard.py

# Terminal dashboard (Rich) - for SSH/VPS
DASHBOARD_PROJECT=the-complexity-trap python scripts/dashboard_tui.py
```

---

## Documentation

For original SWE-agent documentation and paper methodology, see:
- [README_upstream.md](README_upstream.md) - Original paper README
- [SWE-agent-README.md](SWE-agent-README.md) - SWE-agent framework docs
- [SWE-agent official docs](https://swe-agent.com/latest/)
- Marius Hobbhahn's [SWE bench verified mini split](https://huggingface.co/datasets/MariusHobbhahn/swe-bench-verified-mini)

---

## Attribution

This is a fork of **"The Complexity Trap: Simple Observation Masking Is as Efficient as LLM Summarization for Agent Context Management"** by Lindenbauer et al. (JetBrains Research).

```bibtex
@misc{lindenbauer2025complexitytrapsimpleobservation,
      title={The Complexity Trap: Simple Observation Masking Is as Efficient as LLM Summarization for Agent Context Management},
      author={Tobias Lindenbauer and Igor Slinko and Ludwig Felder and Egor Bogomolov and Yaroslav Zharov},
      year={2025},
      eprint={2508.21433},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2508.21433},
}
```

**Paper**: [arXiv:2508.21433](https://arxiv.org/abs/2508.21433)
**Dataset**: [HuggingFace](https://huggingface.co/datasets/JetBrains-Research/the-complexity-trap)

---

## License

MIT (same as upstream). See [LICENSE](LICENSE).
