#!/usr/bin/env python3
"""Run sweagent with model presets.

Examples:
    python scripts/run_model.py --list
    python scripts/run_model.py kimi-k2 --config config/default_no_demo_N=1_M=10.yaml
    python scripts/run_model.py bedrock-qwen3-32b --instances-slice :5 --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from dotenv import load_dotenv

from sweagent.utils.model_config import (
    MODEL_PRESETS,
    get_model_args,
    print_models,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run SWE-agent with different model presets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "model",
        nargs="?",
        help="Model preset key (e.g., kimi-k2, glm-4.6, minimax-m2)",
    )
    parser.add_argument(
        "--list",
        "-l",
        action="store_true",
        help="List all available model presets",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config/default_no_demo_N=1_M=10.yaml",
        help="Config file to use (default: default_no_demo_N=1_M=10.yaml)",
    )
    parser.add_argument(
        "--summarizer-model",
        help="Model preset for summarizer (for LLM-Summary configs)",
    )
    parser.add_argument(
        "--instances-type",
        default="swe_bench",
        help="Instance type (default: swe_bench)",
    )
    parser.add_argument(
        "--instances-subset",
        default="verified",
        choices=["verified", "lite", "full"],
        help="SWE-bench subset (default: verified)",
    )
    parser.add_argument(
        "--instances-split",
        default="test",
        help="Dataset split (default: test)",
    )
    parser.add_argument(
        "--instances-slice",
        help="Slice of instances (e.g., :10 for first 10)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--call-limit",
        type=int,
        default=250,
        help="Per-instance call limit (default: 250)",
    )
    parser.add_argument(
        "--cost-limit",
        type=float,
        default=0.0,
        help="Per-instance cost limit in USD (default: 0.0)",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for trajectories",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print command without executing",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    parser.add_argument(
        "extra_args",
        nargs="*",
        help="Additional arguments passed to sweagent",
    )

    return parser.parse_args()


def _is_bedrock_model_name(model_name: str) -> bool:
    return model_name.startswith("bedrock/")


def _has_bedrock_auth_env() -> bool:
    return bool(
        os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        or (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"))
        or os.environ.get("AWS_PROFILE")
    )


def _add_model_args(cmd: list[str], model_args: dict, prefix: str):
    cmd.extend([f"{prefix}.name", model_args["name"]])
    for key in ("api_base", "api_key", "max_input_tokens", "max_output_tokens"):
        if model_args.get(key):
            cmd.extend([f"{prefix}.{key}", str(model_args[key])])


def build_command(args) -> list[str]:
    cmd = ["sweagent", "run-batch", "--config", args.config]

    _add_model_args(cmd, get_model_args(args.model), "--agent.model")
    cmd.extend(["--agent.model.per_instance_call_limit", str(args.call_limit)])
    cmd.extend(["--agent.model.per_instance_cost_limit", str(args.cost_limit)])

    if args.summarizer_model:
        _add_model_args(cmd, get_model_args(args.summarizer_model), "--agent.summary_model")
        cmd.extend(["--agent.summary_model.per_instance_cost_limit", str(args.cost_limit)])
        cmd.extend(["--agent.summary_model.total_cost_limit", "0"])

    cmd.extend(["--instances.type", args.instances_type])
    cmd.extend(["--instances.subset", args.instances_subset])
    cmd.extend(["--instances.split", args.instances_split])

    if args.instances_slice:
        cmd.extend(["--instances.slice", args.instances_slice])

    cmd.extend(["--num_workers", str(args.num_workers)])

    if args.output_dir:
        cmd.extend(["--output_dir", args.output_dir])

    if args.extra_args:
        cmd.extend(args.extra_args)

    return cmd


def _run_sweagent_inprocess(cmd: list[str]) -> int:
    sweagent_args = cmd[1:] if cmd and cmd[0] == "sweagent" else cmd
    try:
        from sweagent.run.run import main as sweagent_main

        sweagent_main(sweagent_args)
        return 0
    except SystemExit as e:
        code = e.code
        if code is None:
            return 0
        if isinstance(code, int):
            return code
        return 1
    except Exception as e:
        print(f"sweagent crashed in-process: {type(e).__name__}: {e}")
        return 1


def main():
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    args = parse_args()

    if args.list:
        print_models()
        return 0

    if not args.model:
        print("Error: Model required. Use --list to see available models.")
        return 1

    if args.model not in MODEL_PRESETS:
        matches = [k for k in MODEL_PRESETS if args.model.lower() in k.lower()]
        if matches:
            print(f"Did you mean: {', '.join(matches)}?")
        else:
            print(f"Unknown model: {args.model}")
            print("Use --list to see available models.")
        return 1

    preset = MODEL_PRESETS[args.model]
    if _is_bedrock_model_name(preset.name):
        if not (os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")):
            print("Warning: AWS_DEFAULT_REGION/AWS_REGION not set in environment (required for Bedrock)")
            print("Set one of them (e.g. AWS_DEFAULT_REGION=eu-west-2) or configure ~/.aws/config")
        if not _has_bedrock_auth_env():
            print("Warning: No Bedrock auth env vars detected (AWS_BEARER_TOKEN_BEDROCK / AWS_ACCESS_KEY_ID+AWS_SECRET_ACCESS_KEY / AWS_PROFILE).")
            print("If you rely on ~/.aws/credentials, SSO, or instance roles, you can ignore this.")
        if args.cost_limit > 0:
            print("Note: Disabling cost limit for Bedrock models (LiteLLM cost map is often missing entries).")
            args.cost_limit = 0.0

    if args.summarizer_model:
        sum_preset = MODEL_PRESETS.get(args.summarizer_model)
        if sum_preset and _is_bedrock_model_name(sum_preset.name):
            if not (os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION")):
                print("Warning: AWS_DEFAULT_REGION/AWS_REGION not set in environment (required for Bedrock summarizer)")
            if not _has_bedrock_auth_env():
                print("Warning: No Bedrock auth env vars detected for summarizer.")
                print("If you rely on ~/.aws/credentials, SSO, or instance roles, you can ignore this.")
            if args.cost_limit > 0:
                print("Note: Disabling cost limit for Bedrock summarizer (LiteLLM cost map is often missing entries).")
                args.cost_limit = 0.0

    cmd = build_command(args)

    if args.verbose or args.dry_run:
        print("Command:")
        print("  " + " \\\n    ".join(cmd))
        print()

    if args.dry_run:
        return 0

    if preset.api_key_var and not _is_bedrock_model_name(preset.name) and not os.environ.get(preset.api_key_var):
        print(f"Warning: {preset.api_key_var} not set in environment")
        print("Make sure to set it in .env or export it")

    try:
        return _run_sweagent_inprocess(cmd)
    except KeyboardInterrupt:
        print("\nInterrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
