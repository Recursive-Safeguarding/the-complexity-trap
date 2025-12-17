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
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from sweagent.utils.model_config import MODEL_PRESETS, get_model_args

STRATEGY_CONFIGS = {
    "raw": "config/default_no_demo_raw.yaml",
    "observation_masking": "config/default_no_demo_N=1_M=10.yaml",
    "llm_summary": "config/default_no_demo_checkpoint_same_model_openhands_N=21_M=10.yaml",
    "hybrid": "config/default_no_demo_checkpoint_same_model_openhands_N=21_M=10_masking_M=10.yaml",
}

def _is_bedrock_model_name(model_name: str) -> bool:
    return model_name.startswith("bedrock/")


def _has_bedrock_auth_env() -> bool:
    return bool(
        os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        or (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"))
        or os.environ.get("AWS_PROFILE")
    )


def _str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "t", "yes", "y"}


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
        default=":10",
        help="Slice of instances to run (e.g., ':5' for first 5)"
    )
    parser.add_argument(
        "--instances-shuffle",
        type=_str2bool,
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
    parser.add_argument("--num-workers", type=int, default=1)

    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb-project", default="complexity-trap")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", nargs="+", default=[])

    parser.add_argument(
        "--weave",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Enable Weave tracing (default: enabled). "
            "Weave requires running sweagent in-process to trace LLM calls."
        ),
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

    return parser.parse_args()


def _add_model_args(cmd: list[str], model_args: dict, prefix: str):
    for key in ("api_base", "api_key", "max_input_tokens", "max_output_tokens"):
        if model_args.get(key):
            cmd.extend([f"{prefix}.{key}", str(model_args[key])])


def build_sweagent_command(args) -> tuple[list[str], Path]:
    model_args = get_model_args(args.model)
    if _is_bedrock_model_name(model_args["name"]) and args.cost_limit > 0:
        print("Note: Disabling cost limit for Bedrock models (LiteLLM cost map is often missing entries).")
        args.cost_limit = 0.0

    config_file = STRATEGY_CONFIGS[args.strategy]
    cmd = [
        "sweagent", "run-batch",
        "--config", config_file,
        "--agent.model.name", model_args["name"],
        "--agent.model.per_instance_call_limit", str(args.call_limit),
        "--agent.model.per_instance_cost_limit", str(args.cost_limit),
        "--agent.model.total_cost_limit", "0",
        "--instances.type", "swe_bench",
        "--instances.subset", args.instances_subset,
        "--instances.split", "test",
        "--instances.slice", args.instances_slice,
        "--instances.shuffle", str(args.instances_shuffle),
        "--instances.shuffle_seed", str(args.instances_shuffle_seed),
        "--num_workers", str(args.num_workers),
    ]
    _add_model_args(cmd, model_args, "--agent.model")

    if args.strategy in ("llm_summary", "hybrid") and args.summarizer_model != "same":
        sum_args = get_model_args(args.summarizer_model)
        if _is_bedrock_model_name(sum_args["name"]) and args.cost_limit > 0:
            print("Note: Disabling cost limit for Bedrock summarizer (LiteLLM cost map is often missing entries).")
            args.cost_limit = 0.0
        cmd.extend(["--agent.summary_model.name", sum_args["name"]])
        _add_model_args(cmd, sum_args, "--agent.summary_model")
        cmd.extend(["--agent.summary_model.per_instance_cost_limit", str(args.cost_limit)])
        cmd.extend(["--agent.summary_model.total_cost_limit", "0"])

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

def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace(" ", "_")


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

    model_preset = MODEL_PRESETS.get(args.model)
    if model_preset and _is_bedrock_model_name(model_preset.name):
        if not (os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")):
            print("Warning: AWS_DEFAULT_REGION/AWS_REGION not set in environment (required for Bedrock)")
            print("Set one of them (e.g. AWS_DEFAULT_REGION=eu-west-2) or configure ~/.aws/config")
        if not _has_bedrock_auth_env():
            print("Warning: No Bedrock auth env vars detected (AWS_BEARER_TOKEN_BEDROCK / AWS_ACCESS_KEY_ID+AWS_SECRET_ACCESS_KEY / AWS_PROFILE).")
            print("If you rely on ~/.aws/credentials, SSO, or instance roles, you can ignore this.")

    if args.strategy in ("llm_summary", "hybrid") and args.summarizer_model not in (None, "same"):
        sum_preset = MODEL_PRESETS.get(args.summarizer_model)
        if sum_preset and _is_bedrock_model_name(sum_preset.name):
            if not (os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")):
                print("Warning: AWS_DEFAULT_REGION/AWS_REGION not set in environment (required for Bedrock summarizer)")
                print("Set one of them (e.g. AWS_DEFAULT_REGION=eu-west-2) or configure ~/.aws/config")
            if not _has_bedrock_auth_env():
                print("Warning: No Bedrock auth env vars detected for summarizer.")
                print("If you rely on ~/.aws/credentials, SSO, or instance roles, you can ignore this.")

    cmd, trajectories_base = build_sweagent_command(args)
    # Choose a deterministic output directory per run so metrics parsing is reliable.
    # (Avoids accidentally parsing a previous run when running multiple sweep points.)
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "user"
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path("trajectories") / user / f"sweep_{ts}__{_safe_name(args.model)}__{args.strategy}"

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
            }
            # Build run name: model__strategy_summarizer__slice
            strategy_part = args.strategy
            if args.strategy in ("llm_summary", "hybrid") and args.summarizer_model != "same":
                strategy_part = f"{args.strategy}_{args.summarizer_model}"
            parts = [args.model, strategy_part]
            if args.instances_slice:
                parts.append(args.instances_slice.replace(":", ""))
            run_name = "__".join(parts)
            wandb_hook = WandBHook(
                project=args.wandb_project,
                entity=args.wandb_entity,
                group=args.wandb_group,
                tags=args.wandb_tags + [args.model, args.strategy],
                config=config,
                name=run_name,
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

    print("Executing sweagent...")
    rc = _run_sweagent(cmd, execution=args.execution, wandb_hook=wandb_hook)

    if rc != 0:
        print(f"sweagent exited with code {rc}")

    return rc


if __name__ == "__main__":
    sys.exit(main())
