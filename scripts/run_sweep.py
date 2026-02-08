#!/usr/bin/env python3
"""WandB sweep runner for the-complexity-trap experiments.

Translates WandB sweep parameters into sweagent CLI commands and logs metrics.
Optionally enables Weave tracing for LLM calls.

Usage:
    # Direct run (no sweep)
    python scripts/run_sweep.py --model deepseek-chat --strategy observation_masking --instances-slice :3

    # With WandB logging
    python scripts/run_sweep.py --model deepseek-chat --strategy observation_masking --wandb

    # Dry run (show command without executing)
    python scripts/run_sweep.py --model deepseek-chat --strategy observation_masking --dry-run
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from sweagent.utils.model_config import MODEL_PRESETS, get_model_args
from scripts.shared_utils import (
    str2bool,
    is_bedrock_model_name,
    has_bedrock_auth_env,
    add_model_cli_args,
)

STRATEGY_CONFIGS = {
    "raw": "config/default_no_demo_raw.yaml",
    "observation_masking": "config/default_no_demo_N=1_M=10.yaml",
    "dedup_obs_masking": "config/default_no_demo_dedup_obs_M=10.yaml",
    "llm_summary": "config/default_no_demo_checkpoint_same_model_openhands_N=21_M=10.yaml",
    "llm_summary_compact": "config/default_no_demo_checkpoint_compact_N=21_M=10.yaml",
    "llm_summary_obs_masking": "config/default_no_demo_checkpoint_same_model_openhands_N=21_M=10_masking_M=10.yaml",
    "llm_summary_synthesis": "config/default_no_demo_checkpoint_synthesis_N=21_M=10.yaml",
    "llm_summary_synthesis_obs_masking": "config/default_no_demo_checkpoint_synthesis_N=21_M=10_masking_M=10.yaml",
    "on_demand": "config/default_no_demo_on_demand_M=10.yaml",
    "hybrid": "config/default_no_demo_checkpoint_same_model_openhands_N=21_M=10_masking_M=10.yaml",
}

STRATEGY_ABBREV = {
    "raw": "raw",
    "observation_masking": "obs",
    "dedup_obs_masking": "dup_obs",
    "llm_summary": "sum",
    "llm_summary_compact": "sum_cpt",
    "llm_summary_obs_masking": "sum_obs",
    "llm_summary_synthesis": "syn",
    "llm_summary_synthesis_obs_masking": "syn_obs",
    "on_demand": "od",
    "hybrid": "hyb",
}

# Strategies that use a summarizer model (for tag building and config)
SUMMARIZER_STRATEGIES = {
    "llm_summary",
    "llm_summary_compact",
    "llm_summary_obs_masking",
    "llm_summary_synthesis",
    "llm_summary_synthesis_obs_masking",
    "on_demand",
    "hybrid",
}

# Strategies that use observation masking (for hyperparam handling)
MASKING_STRATEGIES = {
    "observation_masking",
    "dedup_obs_masking",
    "llm_summary_obs_masking",
    "llm_summary_synthesis_obs_masking",
    "hybrid",  # Backward compatibility
}

# Deprecated strategy aliases
STRATEGY_ALIASES = {"hybrid": "llm_summary_obs_masking"}
INSTANCE_ABBREV = {"verified-mini": "mini", "verified": "v", "lite": "lite"}

# Model provider tags for filtering in WandB
MODEL_PROVIDER = {
    "bedrock-qwen3-32b": "bedrock",
    "bedrock-qwen3-coder-480b": "bedrock",
    "bedrock-nova-pro": "bedrock",
    "bedrock-nova-lite": "bedrock",
    "bedrock-claude-haiku-4.5": "bedrock",
    "deepseek-chat": "deepseek",
    "glm-4.5-air": "zhipu",
    "glm-4.6": "zhipu",
    "glm-4.7": "zhipu",
    "kimi-k2": "moonshot",
    "minimax-m2": "minimax",
    "minimax-m2.1": "minimax",
    "gpt-4o-mini": "openai",
}


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


def build_tags(args) -> list[str]:
    """Build WandB tags for filtering."""
    tags = [
        f"model:{args.model}",
        f"provider:{MODEL_PROVIDER.get(args.model, 'other')}",
        f"strategy:{args.strategy}",
        f"subset:{args.instances_subset}",
    ]

    if args.strategy in SUMMARIZER_STRATEGIES:
        summarizer = args.model if args.summarizer_model == "same" else args.summarizer_model
        tags.append(f"summarizer:{summarizer}")
        tags.append(f"sum_provider:{MODEL_PROVIDER.get(summarizer, 'other')}")

    # Hyperparams worth filtering on
    if args.strategy in MASKING_STRATEGIES and args.hp_obs_n is not None:
        tags.append(f"obs_n:{args.hp_obs_n}")
    if args.strategy in SUMMARIZER_STRATEGIES:
        if args.hp_sum_n is not None:
            tags.append(f"sum_n:{args.hp_sum_n}")
        if args.hp_sum_keep_m is not None:
            tags.append(f"keep_m:{args.hp_sum_keep_m}")
    if args.hp_limit_aware:
        tags.append("limit-aware")
        if args.hp_limit_fraction is not None:
            tags.append(f"limit_frac:{args.hp_limit_fraction}")
        if args.hp_limit_min_tokens is not None:
            tags.append(f"limit_min_tokens:{args.hp_limit_min_tokens}")

    return tags


def build_run_name(args) -> str:
    """Build descriptive run name for WandB and output directories.

    Format: model__strategy_summarizer_hparams__dataset
    - Double underscore (__) separates major groups (model, strategy config, dataset)
    - Single underscore (_) separates items within strategy config group
    """
    parts = []

    def _fmt_float_for_name(val: float) -> str:
        return str(val).replace(".", "p")

    # Group 1: Model
    parts.append(_safe_name(args.model))

    # Group 2: Strategy config (strategy + summarizer + hparams, all joined with single _)
    strategy_parts = [STRATEGY_ABBREV.get(args.strategy, args.strategy)]

    if args.strategy in SUMMARIZER_STRATEGIES:
        sum_model = args.model if args.summarizer_model == "same" else args.summarizer_model
        strategy_parts.append(_safe_name(sum_model).replace("bedrock-", ""))

    # Add hyperparameters to strategy config
    # For masking-only strategies (no summarization)
    if args.strategy in ("observation_masking", "dedup_obs_masking") and args.hp_obs_n is not None:
        strategy_parts.append(f"n-{args.hp_obs_n}")
    # For summary-only strategies (no masking)
    elif args.strategy in ("llm_summary", "llm_summary_synthesis", "llm_summary_compact"):
        if args.hp_sum_n is not None:
            strategy_parts.append(f"n-{args.hp_sum_n}")
        if args.hp_sum_keep_m is not None:
            strategy_parts.append(f"k-{args.hp_sum_keep_m}")
    # For combined strategies (summary + masking)
    elif args.strategy in ("hybrid", "llm_summary_obs_masking", "llm_summary_synthesis_obs_masking"):
        if args.hp_obs_n is not None:
            strategy_parts.append(f"o-{args.hp_obs_n}")
        if args.hp_sum_n is not None:
            strategy_parts.append(f"s-{args.hp_sum_n}")
        if args.hp_sum_keep_m is not None:
            strategy_parts.append(f"k-{args.hp_sum_keep_m}")

    if args.hp_limit_aware:
        strategy_parts.append("la")
        if args.hp_limit_fraction is not None:
            strategy_parts.append(f"lf-{_fmt_float_for_name(args.hp_limit_fraction)}")
        if args.hp_limit_min_tokens is not None:
            strategy_parts.append(f"lt-{args.hp_limit_min_tokens}")

    parts.append("_".join(strategy_parts))

    # Group 3: Dataset
    inst_part = INSTANCE_ABBREV.get(args.instances_subset, args.instances_subset)
    if args.instances_slice:
        inst_part += args.instances_slice.replace(":", "")
    parts.append(inst_part)

    return "__".join(parts)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SWE-bench experiments with WandB sweep support"
    )

    parser.add_argument(
        "--model",
        required=True,
        help=f"Model preset key from MODEL_PRESETS: {list(MODEL_PRESETS.keys())}"
    )

    parser.add_argument(
        "--strategy",
        required=True,
        choices=list(STRATEGY_CONFIGS.keys()),
        help="Context management strategy"
    )

    parser.add_argument(
        "--summarizer-model",
        default="same",
        help="'same' to use main model, or a MODEL_PRESETS key"
    )

    parser.add_argument(
        "--instances-subset",
        default="verified",
        choices=["verified", "verified-mini", "lite"],
        help="SWE-bench subset"
    )
    parser.add_argument(
        "--instances-slice",
        default=None,
        help="Slice of instances to run (e.g., ':5' for first 5). Omit to run all."
    )
    parser.add_argument(
        "--instances-shuffle",
        type=str2bool,
        default=False,
        help="Shuffle instances before filtering/slicing (deterministic; default: false)",
    )
    parser.add_argument(
        "--instances-shuffle-seed",
        type=int,
        default=42,
        help="Seed for deterministic shuffling (used when --instances-shuffle=true; default: 42)",
    )

    parser.add_argument("--call-limit", type=int, default=250)
    parser.add_argument(
        "--cost-limit",
        type=float,
        default=0.0,
        help=(
            "Per-instance cost limit in USD (default: 0.0). "
            "Note: many Bedrock models are missing from LiteLLM's cost map, so non-zero cost limits can fail."
        ),
    )
    parser.add_argument(
        "--bypass-cost-limits",
        type=str2bool,
        default=False,
        help="Bypass cost limits. Cost still tracked.",
    )
    parser.add_argument("--num-workers", type=int, default=1)

    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb-project", default="the-complexity-trap")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", nargs="+", default=[])

    parser.add_argument(
        "--weave",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable Weave tracing (requires inprocess execution).",
    )
    parser.add_argument(
        "--weave-project",
        default=None,
        help="Weave project ref (e.g., 'entity/project'). Defaults to the WandB project.",
    )

    parser.add_argument(
        "--execution",
        choices=["inprocess", "subprocess"],
        default="inprocess",
        help=(
            "How to run sweagent. Use 'inprocess' to enable Weave tracing. "
            "Use 'subprocess' if you need isolation (Weave won't trace child process calls)."
        ),
    )

    parser.add_argument("--dry-run", action="store_true", help="Print command without executing")

    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run SWE-bench evaluation (docker or sb-cli).",
    )
    parser.add_argument(
        "--batch-eval-interval",
        type=int,
        default=0,
        help="Eval every N instances (0=disable).",
    )
    parser.add_argument(
        "--eval-backend",
        choices=["docker", "sbcli"],
        default="docker",
        help="Evaluation backend: docker (local) or sbcli (cloud, faster but has quotas).",
    )

    hp_group = parser.add_argument_group("History Processor Hyperparameters")
    hp_group.add_argument(
        "--hp-obs-n",
        type=int,
        default=None,
        help="LastNObservations: Number of observations to keep (default: from config)"
    )
    hp_group.add_argument(
        "--hp-sum-n",
        type=int,
        default=None,
        help="SummarizeEveryNTurns: Number of turns before summarization (default: 21)"
    )
    hp_group.add_argument(
        "--hp-sum-keep-m",
        type=int,
        default=None,
        help="SummarizeEveryNTurns: Recent turns to keep unsummarized (default: 10)"
    )
    hp_group.add_argument(
        "--hp-sum-static-checkpoint",
        type=str2bool,
        default=None,
        help="SummarizeEveryNTurns: Replace vs append summaries (default: true)"
    )
    hp_group.add_argument(
        "--hp-sum-extract-actions",
        type=str2bool,
        default=None,
        help="SummarizeEveryNTurns: Include action details in summary (default: false)"
    )
    hp_group.add_argument(
        "--hp-sum-max-action-length",
        type=int,
        default=None,
        help="SummarizeEveryNTurns: Max action length to keep (-1 = no limit, default: -1)"
    )
    hp_group.add_argument(
        "--hp-sum-max-reasoning-length",
        type=int,
        default=None,
        help="SummarizeEveryNTurns: Max reasoning length to keep (-1 = no limit, default: -1)"
    )
    hp_group.add_argument(
        "--hp-sum-omit-turns",
        type=str2bool,
        default=None,
        help="SummarizeEveryNTurns: Skip LLM summary, just mark omitted (default: false)"
    )
    hp_group.add_argument(
        "--hp-limit-aware",
        type=str2bool,
        default=None,
        help="Only compact near the context limit (default: false)"
    )
    hp_group.add_argument(
        "--hp-limit-fraction",
        type=float,
        default=None,
        help="Trigger at this fraction of the context window (default: 0.9)"
    )
    hp_group.add_argument(
        "--hp-limit-min-tokens",
        type=int,
        default=None,
        help="Trigger at this token count (overrides fraction; default: 0)"
    )

    return parser.parse_args()


def get_relevant_hparams(strategy: str, args) -> dict:
    hparams = {}
    if strategy in MASKING_STRATEGIES:
        if args.hp_obs_n is not None:
            hparams["hp_obs_n"] = args.hp_obs_n
        if args.hp_limit_aware is not None:
            hparams["hp_limit_aware"] = args.hp_limit_aware
        if args.hp_limit_fraction is not None:
            hparams["hp_limit_fraction"] = args.hp_limit_fraction
        if args.hp_limit_min_tokens is not None:
            hparams["hp_limit_min_tokens"] = args.hp_limit_min_tokens
    if strategy in SUMMARIZER_STRATEGIES:
        if args.hp_sum_n is not None:
            hparams["hp_sum_n"] = args.hp_sum_n
        if args.hp_sum_keep_m is not None:
            hparams["hp_sum_keep_m"] = args.hp_sum_keep_m
        if args.hp_sum_static_checkpoint is not None:
            hparams["hp_sum_static_checkpoint"] = args.hp_sum_static_checkpoint
        if args.hp_sum_extract_actions is not None:
            hparams["hp_sum_extract_actions"] = args.hp_sum_extract_actions
        if args.hp_sum_max_action_length is not None:
            hparams["hp_sum_max_action_length"] = args.hp_sum_max_action_length
        if args.hp_sum_max_reasoning_length is not None:
            hparams["hp_sum_max_reasoning_length"] = args.hp_sum_max_reasoning_length
        if args.hp_sum_omit_turns is not None:
            hparams["hp_sum_omit_turns"] = args.hp_sum_omit_turns
        if args.hp_limit_aware is not None:
            hparams["hp_limit_aware"] = args.hp_limit_aware
        if args.hp_limit_fraction is not None:
            hparams["hp_limit_fraction"] = args.hp_limit_fraction
        if args.hp_limit_min_tokens is not None:
            hparams["hp_limit_min_tokens"] = args.hp_limit_min_tokens

    return hparams


def generate_custom_config(base_config_path: str, args, output_path: Path, model_args: dict | None = None, summarizer_model_args: dict | None = None) -> Path:
    import yaml
    with open(base_config_path) as f:
        config = yaml.safe_load(f)

    # Init agent.model
    if "agent" not in config:
        config["agent"] = {}
    if "model" not in config["agent"]:
        config["agent"]["model"] = {}

    modified = False

    # Set extra_headers
    if model_args and model_args.get("extra_headers"):
        if "completion_kwargs" not in config["agent"]["model"]:
            config["agent"]["model"]["completion_kwargs"] = {}
        config["agent"]["model"]["completion_kwargs"]["extra_headers"] = model_args["extra_headers"]
        modified = True

    # Add extra_headers to summary_model config if present
    if summarizer_model_args and summarizer_model_args.get("extra_headers"):
        if "summary_model" not in config["agent"]:
            config["agent"]["summary_model"] = {}
        if "completion_kwargs" not in config["agent"]["summary_model"]:
            config["agent"]["summary_model"]["completion_kwargs"] = {}
        config["agent"]["summary_model"]["completion_kwargs"]["extra_headers"] = summarizer_model_args["extra_headers"]
        modified = True

    if "history_processors" in config.get("agent", {}):
        limit_aware_explicit_false = args.hp_limit_aware is False
        for processor in config["agent"]["history_processors"]:
            proc_type = processor.get("type")

            if proc_type == "last_n_observations":
                if args.hp_obs_n is not None:
                    processor["n"] = args.hp_obs_n
                    modified = True
                if args.hp_limit_aware is not None:
                    processor["enable_limit_aware_trigger"] = args.hp_limit_aware
                    modified = True
                if args.hp_limit_fraction is not None and not limit_aware_explicit_false:
                    processor["limit_trigger_fraction"] = args.hp_limit_fraction
                    modified = True
                if args.hp_limit_min_tokens is not None and not limit_aware_explicit_false:
                    processor["limit_trigger_min_tokens"] = args.hp_limit_min_tokens
                    modified = True

            elif proc_type == "summarize_every_n_turns":
                if args.hp_sum_n is not None:
                    processor["n"] = args.hp_sum_n
                    modified = True
                if args.hp_sum_keep_m is not None:
                    processor["keep_last_m_turns"] = args.hp_sum_keep_m
                    modified = True
                if args.hp_sum_static_checkpoint is not None:
                    processor["enable_static_checkpointing"] = args.hp_sum_static_checkpoint
                    modified = True
                if args.hp_sum_extract_actions is not None:
                    processor["extract_action_from_turns"] = args.hp_sum_extract_actions
                    modified = True
                if args.hp_sum_max_action_length is not None:
                    processor["max_kept_action_length"] = args.hp_sum_max_action_length
                    modified = True
                if args.hp_sum_max_reasoning_length is not None:
                    processor["max_kept_reasoning_length"] = args.hp_sum_max_reasoning_length
                    modified = True
                if args.hp_sum_omit_turns is not None:
                    processor["omit_turns"] = args.hp_sum_omit_turns
                    modified = True
                if args.hp_limit_aware is not None:
                    processor["enable_limit_aware_trigger"] = args.hp_limit_aware
                    modified = True
                if args.hp_limit_fraction is not None and not limit_aware_explicit_false:
                    processor["limit_trigger_fraction"] = args.hp_limit_fraction
                    modified = True
                if args.hp_limit_min_tokens is not None and not limit_aware_explicit_false:
                    processor["limit_trigger_min_tokens"] = args.hp_limit_min_tokens
                    modified = True

    if not modified:
        return Path(base_config_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    return output_path


def build_sweagent_command(args) -> tuple[list[str], Path]:
    model_args = get_model_args(args.model)
    # Separate cost limits
    agent_cost_limit = args.cost_limit
    if is_bedrock_model_name(model_args["name"]) and agent_cost_limit > 0:
        print("Note: Disabling cost limit for Bedrock agent model (LiteLLM cost map is often missing entries).")
        agent_cost_limit = 0.0

    # Get summarizer model args early so we can write extra_headers to config
    summarizer_model_args = None
    summarizer_cost_limit = args.cost_limit  # Independent from agent_cost_limit
    # "same" = reuse agent model config; explicit names need their own config
    if args.strategy in SUMMARIZER_STRATEGIES and args.summarizer_model != "same":
        summarizer_model_args = get_model_args(args.summarizer_model)
        if is_bedrock_model_name(summarizer_model_args["name"]) and summarizer_cost_limit > 0:
            print("Note: Disabling cost limit for Bedrock summarizer (LiteLLM cost map is often missing entries).")
            summarizer_cost_limit = 0.0

    base_config_file = STRATEGY_CONFIGS[args.strategy]
    ts = time.strftime("%Y%m%d_%H%M%S")
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "user"
    custom_config_path = Path("config/.generated") / user / f"config_{ts}_{_safe_name(args.model)}_{args.strategy}.yaml"
    config_file = str(generate_custom_config(base_config_file, args, custom_config_path, model_args, summarizer_model_args))
    cmd = [
        "sweagent", "run-batch",
        "--config", config_file,
        "--agent.model.name", model_args["name"],
        "--agent.model.per_instance_call_limit", str(args.call_limit),
        "--agent.model.per_instance_cost_limit", str(agent_cost_limit),
        "--agent.model.total_cost_limit", "0",
        "--instances.type", "swe_bench",
        "--instances.subset", args.instances_subset,
        "--instances.split", "test",
    ]
    if args.instances_slice:
        cmd.extend(["--instances.slice", args.instances_slice])
    cmd.extend([
        "--instances.shuffle", str(args.instances_shuffle),
        "--instances.shuffle_seed", str(args.instances_shuffle_seed),
        "--num_workers", str(args.num_workers),
    ])
    add_model_cli_args(cmd, model_args, "--agent.model", include_name=False, include_context_window=True)
    if args.bypass_cost_limits:
        cmd.extend(["--agent.model.bypass_cost_limits", "True"])

    if summarizer_model_args:
        cmd.extend(["--agent.summary_model.name", summarizer_model_args["name"]])
        add_model_cli_args(cmd, summarizer_model_args, "--agent.summary_model", include_name=False, include_context_window=True)
        cmd.extend(["--agent.summary_model.per_instance_cost_limit", str(summarizer_cost_limit)])
        cmd.extend(["--agent.summary_model.total_cost_limit", "0"])
        if args.bypass_cost_limits:
            cmd.extend(["--agent.summary_model.bypass_cost_limits", "True"])

    return cmd, Path("trajectories")


def find_latest_trajectory_dir(base_dir: Path, model_name: str) -> Path | None:
    import glob
    pattern = str(base_dir / "**" / f"*{model_name.replace('/', '_')}*")
    matches = glob.glob(pattern, recursive=True)

    if not matches:
        user_dirs = list(base_dir.iterdir()) if base_dir.exists() else []
        for user_dir in user_dirs:
            if user_dir.is_dir():
                run_dirs = sorted(user_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
                for run_dir in run_dirs:
                    if run_dir.is_dir() and any(run_dir.glob("*.traj")):
                        return run_dir
        return None

    return Path(max(matches, key=os.path.getmtime))


def _check_docker_available() -> tuple[bool, str]:
    """Check Docker availability."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            if "Cannot connect to the Docker daemon" in result.stderr:
                return False, "Docker daemon not running. Start with: sudo systemctl start docker"
            if "permission denied" in result.stderr.lower():
                return False, "Docker permission denied. Add user to docker group or use sudo."
            return False, f"Docker not working: {result.stderr[:200]}"
        return True, ""
    except FileNotFoundError:
        return False, "Docker not installed. Install from: https://docs.docker.com/get-docker/"
    except subprocess.TimeoutExpired:
        return False, "Docker command timed out - daemon may be unresponsive"
    except Exception as e:
        return False, f"Docker check failed: {e}"


def _check_sbcli_available() -> tuple[bool, str]:
    """Check sb-cli availability."""
    try:
        from sweagent.utils.sbcli import check_sbcli_available

        return check_sbcli_available()
    except Exception as e:
        return False, f"sb-cli check failed: {e}"


def _require_docker_available() -> None:
    """Raise if Docker unavailable."""
    ok, msg = _check_docker_available()
    if not ok:
        raise RuntimeError(f"Docker preflight check failed: {msg}")


class EvaluationError(Exception):
    """SWE-bench evaluation failed."""
    pass


def _make_retry_logger(max_attempts: int):
    """Create retry logger."""
    def _log_retry_attempt(retry_state) -> None:
        attempt = retry_state.attempt_number
        outcome = retry_state.outcome

        if outcome.failed:
            exc = outcome.exception()
            print(f"⚠️ Evaluation failed (attempt {attempt}/{max_attempts})")
            print(f"   Error: {exc}")
            if attempt < max_attempts:
                # Calculate next delay (same formula as wait_exponential)
                next_delay = min(30 * (2 ** (attempt - 1)), 300)
                print(f"   Retrying in {int(next_delay)}s...")
    return _log_retry_attempt


def _eval_with_retries(
    eval_fn,
    max_attempts: int = 5,
    preflight_fn=None,
) -> Path:
    """Run evaluation with retries. Raises EvaluationError on failure."""
    from tenacity import (
        retry,
        stop_after_attempt,
        wait_exponential,
        retry_if_exception_type,
        before_sleep,
    )

    retry_logger = _make_retry_logger(max_attempts)

    @retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=30, min=0, max=300),
        retry=retry_if_exception_type(EvaluationError),  # Only retry EvaluationError, not all exceptions
        before_sleep=retry_logger,
        reraise=True,  # Re-raises the original exception after retries exhausted
    )
    def _attempt_eval():
        # Optional preflight check before each attempt
        if preflight_fn is not None:
            ok, msg = preflight_fn()
            if not ok:
                raise EvaluationError(f"Evaluation preflight failed: {msg}")

        result = eval_fn()
        if result is None:
            raise EvaluationError("Evaluation returned no results (check logs above)")
        return result

    try:
        return _attempt_eval()
    except EvaluationError as e:
        # All retries exhausted (or first attempt failed) - CIRCUIT BREAKER
        # With reraise=True, tenacity re-raises the original EvaluationError
        raise EvaluationError(
            f"🛑 CIRCUIT BREAKER: Evaluation failed after {max_attempts} attempts. "
            f"Stopping sweep to prevent wasting money on broken infrastructure. "
            f"Last error: {e}"
        ) from e
    except Exception as e:
        # Non-retryable error (e.g., FileNotFoundError for missing swebench)
        # Don't retry these - fail immediately
        raise EvaluationError(
            f"🛑 CIRCUIT BREAKER: Evaluation failed with non-retryable error. "
            f"Error: {e}"
        ) from e


def run_evaluation(
    output_dir: Path,
    subset: str = "verified",
    max_workers: int = 4,
    eval_backend: str = "docker",
) -> Path | None:
    """Run SWE-bench evaluation with the selected backend."""
    from sweagent.utils.sbcli import parse_eval_results

    preds_path = output_dir / "preds.json"
    if not preds_path.exists():
        # Check if all.preds.json exists (alternative naming)
        alt_preds = output_dir / "all.preds.json"
        if alt_preds.exists():
            preds_path = alt_preds
        else:
            print(f"WARNING: No predictions file found in {output_dir}")
            print(f"  Looked for: {preds_path}, {alt_preds}")
            return None

    run_id = output_dir.name

    if eval_backend == "sbcli":
        from sweagent.utils.sbcli import run_sbcli_evaluation

        print("Running SWE-bench evaluation (sb-cli)...")

        success, error, results_path = run_sbcli_evaluation(
            preds_path=preds_path,
            subset=subset,
            run_id=run_id,
            output_dir=output_dir,
        )

        if not success or results_path is None:
            print(f"WARNING: sb-cli evaluation failed: {error}")
            return None

        result = parse_eval_results(results_path)
        if result:
            n_resolved = result["n_resolved"]
            n_evaluated = result["n_evaluated"]
            rate = n_resolved / n_evaluated * 100 if n_evaluated else 0
            print(f"  Resolved: {n_resolved}/{n_evaluated} ({rate:.1f}%)")

        return results_path

    from sweagent.utils.sbcli import get_dataset_name

    docker_ok, docker_error = _check_docker_available()
    if not docker_ok:
        print(f"ERROR: {docker_error}")
        print("SWE-bench evaluation requires Docker to run test containers.")
        return None

    dataset = get_dataset_name(subset, "docker")

    cmd = [
        sys.executable, "-m", "swebench.harness.run_evaluation",
        "-p", str(preds_path),
        "-d", dataset,
        "-id", run_id,
        "--max_workers", str(max_workers),
    ]

    print(f"Running SWE-bench evaluation (local)...")
    print(f"  Dataset: {dataset}")
    print(f"  Predictions: {preds_path}")
    print(f"  Max workers: {max_workers}")
    print(f"  Command: {' '.join(cmd)}")

    # Create a log file for evaluation output
    eval_log_path = output_dir / "evaluation.log"

    try:
        # Stream output to both console and log file for visibility
        with open(eval_log_path, "w") as log_file:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # line buffered
            )

            # Stream output line by line from a background thread so timeouts still fire
            output_lines = []
            timed_out = False

            def _stream_output() -> None:
                for line in iter(process.stdout.readline, ""):
                    if line == "":
                        break
                    output_lines.append(line)
                    log_file.write(line)
                    log_file.flush()
                    # Print progress indicators (Docker pulls, test results)
                    if any(k in line.lower() for k in ["pulling", "resolved", "passed", "failed", "error", "instance"]):
                        print(f"  {line.rstrip()}")

            reader_thread = threading.Thread(target=_stream_output, daemon=True)
            reader_thread.start()

            try:
                process.wait(timeout=7200)  # 2 hour timeout
            except subprocess.TimeoutExpired:
                timed_out = True
                print("WARNING: Local evaluation timed out after 2 hours; terminating process...")
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print("WARNING: Local evaluation did not terminate gracefully; killing process...")
                    process.kill()
                    process.wait()
            finally:
                if process.stdout:
                    try:
                        process.stdout.close()
                    except Exception:
                        pass
                reader_thread.join(timeout=5)

            if timed_out:
                print(f"  Full log saved to: {eval_log_path}")
                if output_lines:
                    print("  Last 20 lines of output:")
                    for line in output_lines[-20:]:
                        print(f"    {line.rstrip()}")
                return None

            if process.returncode != 0:
                print(f"WARNING: Local evaluation failed with code {process.returncode}")
                print(f"  Full log saved to: {eval_log_path}")
                # Print last 20 lines of output for debugging
                print("  Last 20 lines of output:")
                for line in output_lines[-20:]:
                    print(f"    {line.rstrip()}")
                return None

        # Find the report file in logs/run_evaluation/<run_id>/
        logs_dir = Path("logs/run_evaluation") / run_id
        report_pattern = f"{run_id}.*.json"
        reports = list(logs_dir.glob(report_pattern)) if logs_dir.exists() else []

        if not reports:
            # Also check for report.json directly
            direct_report = logs_dir / "report.json"
            if direct_report.exists():
                reports = [direct_report]

        results_path = output_dir / "results.json"
        if reports:
            # Use the most recent report
            latest_report = max(reports, key=lambda p: p.stat().st_mtime)
            shutil.copy(str(latest_report), str(results_path))
            print(f"Evaluation results saved to: {results_path}")

            # Parse and print summary
            result = parse_eval_results(results_path)
            if result:
                n_resolved = result["n_resolved"]
                n_evaluated = result["n_evaluated"]
                rate = n_resolved / n_evaluated * 100 if n_evaluated else 0
                print(f"  Resolved: {n_resolved}/{n_evaluated} ({rate:.1f}%)")

            return results_path
        else:
            print(f"WARNING: No report files found in {logs_dir}")
            print(f"  Looked for: {report_pattern}")
            print(f"  Full evaluation log: {eval_log_path}")
            # Print last 30 lines of output for debugging
            if output_lines:
                print("  Last 30 lines of evaluation output:")
                for line in output_lines[-30:]:
                    print(f"    {line.rstrip()}")
            return None

    except subprocess.TimeoutExpired:
        print("WARNING: Local evaluation timed out after 2 hours")
        return None
    except FileNotFoundError:
        print("WARNING: swebench not found. Install with: pip install swebench")
        return None
    except Exception as e:
        print(f"WARNING: Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def _run_sweagent(cmd: list[str], execution: str, wandb_hook=None) -> int:
    """Run sweagent either in-process (with hooks) or as a subprocess."""
    if execution == "subprocess":
        import subprocess
        result = subprocess.run(cmd, capture_output=False)
        return result.returncode

    sweagent_args = cmd[2:] if cmd[:2] == ["sweagent", "run-batch"] else cmd[1:]
    try:
        from sweagent.run.common import BasicCLI, ConfigHelper
        from sweagent.run.run_batch import RunBatch, RunBatchConfig

        help_text = ConfigHelper().get_help(RunBatchConfig)
        config = BasicCLI(RunBatchConfig, help_text=help_text).get_config(sweagent_args)
        rb = RunBatch.from_config(config)
        if wandb_hook:
            rb.add_hook(wandb_hook)
        rb.main()
        return 0
    except SystemExit as e:
        code = e.code
        return 0 if code is None else (code if isinstance(code, int) else 1)
    except Exception as e:
        print(f"sweagent crashed: {type(e).__name__}: {e}")
        return 1


def main():
    args = parse_args()

    # Handle deprecated strategy aliases
    if args.strategy in STRATEGY_ALIASES:
        canonical = STRATEGY_ALIASES[args.strategy]
        print(f"Warning: Strategy '{args.strategy}' is deprecated, use '{canonical}' instead.")
        # Note: We don't auto-replace to maintain backward compat in WandB tags/names

    model_preset = MODEL_PRESETS.get(args.model)
    if model_preset and is_bedrock_model_name(model_preset.name):
        if not (os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")):
            print("Warning: AWS_DEFAULT_REGION/AWS_REGION not set in environment (required for Bedrock)")
            print("Set one of them (e.g. AWS_DEFAULT_REGION=eu-west-2) or configure ~/.aws/config")
        if not has_bedrock_auth_env():
            print("Warning: No Bedrock auth env vars detected (AWS_BEARER_TOKEN_BEDROCK / AWS_ACCESS_KEY_ID+AWS_SECRET_ACCESS_KEY / AWS_PROFILE).")
            print("If you rely on ~/.aws/credentials, SSO, or instance roles, you can ignore this.")

    # "same" = reuse agent model config; explicit names need their own config
    if args.strategy in SUMMARIZER_STRATEGIES and args.summarizer_model != "same":
        sum_preset = MODEL_PRESETS.get(args.summarizer_model)
        if sum_preset and is_bedrock_model_name(sum_preset.name):
            if not (os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")):
                print("Warning: AWS_DEFAULT_REGION/AWS_REGION not set in environment (required for Bedrock summarizer)")
                print("Set one of them (e.g. AWS_DEFAULT_REGION=eu-west-2) or configure ~/.aws/config")
            if not has_bedrock_auth_env():
                print("Warning: No Bedrock auth env vars detected for summarizer.")
                print("If you rely on ~/.aws/credentials, SSO, or instance roles, you can ignore this.")

    cmd, trajectories_base = build_sweagent_command(args)
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "user"
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = build_run_name(args)
    output_dir = Path("trajectories") / user / f"{run_name}__{ts}"

    # Warn about incompatible flag combinations
    if args.wandb and args.execution == "subprocess":
        print("WARNING: WandB logging requires --execution inprocess. WandB disabled for this run.")
        args.wandb = False
    if args.evaluate and not args.wandb:
        print("Note: --evaluate without --wandb; results will be printed but not logged to WandB.")

    if args.dry_run:
        cmd.extend(["--output_dir", str(output_dir)])
        print("Command:")
        print("  " + " \\\n    ".join(cmd))
        print()
        print(f"[DRY RUN] Output dir: {output_dir}")
        print("[DRY RUN] Command not executed")
        return 0

    wandb_hook = None
    if args.wandb and args.execution == "inprocess":
        try:
            from sweagent.run.hooks.wandb_hook import WandBHook

            config = {
                "model": args.model,
                "strategy": args.strategy,
                "summarizer_model": args.summarizer_model,
                "instances_subset": args.instances_subset,
                "instances_slice": args.instances_slice,
                "call_limit": args.call_limit,
                "cost_limit": args.cost_limit,
                "num_workers": args.num_workers,
                "instances_shuffle": args.instances_shuffle,
                "instances_shuffle_seed": args.instances_shuffle_seed,
                "batch_eval_interval": args.batch_eval_interval,
                **get_relevant_hparams(args.strategy, args),
            }
            wandb_hook = WandBHook(
                project=args.wandb_project,
                entity=args.wandb_entity,
                group=args.wandb_group,
                tags=args.wandb_tags + build_tags(args),
                config=config,
                name=run_name,
                defer_finish=args.evaluate,  # Delay wandb.finish() if we'll run evaluation
                batch_eval_interval=args.batch_eval_interval,
                dataset_subset=args.instances_subset,
                model_name=args.model,
                output_dir=output_dir,
                run_final_eval=args.evaluate,
                eval_backend=args.eval_backend,
            )
        except ImportError:
            print("WARNING: wandb not installed, skipping WandB logging")
            args.wandb = False

    if args.weave and args.execution == "inprocess":
        try:
            import weave
            weave_project = args.weave_project or args.wandb_project
            if args.wandb_entity:
                weave_project = f"{args.wandb_entity}/{weave_project}"
            weave.init(weave_project)
            print(f"Weave initialized: {weave_project}")
        except ImportError:
            print("WARNING: weave not installed; skipping Weave tracing.")
        except Exception as e:
            print(f"WARNING: Weave init failed: {e}")

    cmd.extend(["--output_dir", str(output_dir)])

    print(f"Output dir: {output_dir}")
    print("Command:")
    print("  " + " \\\n    ".join(cmd))
    print()

    # Check evaluation backend availability
    if args.evaluate:
        if args.eval_backend == "sbcli":
            print("Preflight check: verifying sb-cli is available for evaluation...")
            ok, msg = _check_sbcli_available()
            if ok:
                print("  ✅ sb-cli is available")
            else:
                print(f"  ❌ sb-cli not available: {msg}")
                print()
                print("🛑 ABORTING: Cannot run with --evaluate when sb-cli is unavailable.")
                print("   Install sb-cli and set SWEBENCH_API_KEY, or use --eval-backend docker.")
                if wandb_hook:
                    wandb_hook.finalize()
                return 1
        else:
            print("Preflight check: verifying Docker is available for evaluation...")
            try:
                _require_docker_available()
                print("  ✅ Docker is available")
            except RuntimeError as e:
                print(f"  ❌ {e}")
                print()
                print("🛑 ABORTING: Cannot run with --evaluate when Docker is unavailable.")
                print("   Fix Docker first, or use --eval-backend sbcli to avoid Docker.")
                if wandb_hook:
                    wandb_hook.finalize()
                return 1

    print("Executing sweagent...")
    rc = _run_sweagent(cmd, execution=args.execution, wandb_hook=wandb_hook)

    if rc != 0:
        print(f"sweagent exited with code {rc}")

    if args.evaluate:
        print()
        try:
            results_path = _eval_with_retries(
                lambda: run_evaluation(
                    output_dir,
                    args.instances_subset,
                    eval_backend=args.eval_backend,
                ),
                max_attempts=5,
                preflight_fn=_check_sbcli_available if args.eval_backend == "sbcli" else _check_docker_available,
            )
            if wandb_hook:
                wandb_hook.update_with_evaluation_results(results_path)
            else:
                print("Note: Evaluation completed but --wandb not enabled, solve_rate not logged")
        except EvaluationError as e:
            print()
            print(f"❌ {e}")
            print()
            if wandb_hook:
                try:
                    import wandb
                    wandb.run.summary["eval_status"] = "failed"
                    wandb.run.summary["eval_error"] = str(e)
                except Exception:
                    pass
                wandb_hook.finalize()
            return 1

    if wandb_hook:
        wandb_hook.finalize()

    return rc


if __name__ == "__main__":
    sys.exit(main())
