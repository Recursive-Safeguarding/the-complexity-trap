"""Shared utilities for run_model.py and run_sweep.py.

Eliminates DRY violations between the two scripts.
"""
from __future__ import annotations

import json
import os


def str2bool(value: str | bool) -> bool:
    """Parse bool from string (for argparse type= parameter)."""
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "t", "yes", "y"}


def is_bedrock_model_name(model_name: str) -> bool:
    """Check if model name indicates a Bedrock model."""
    return model_name.startswith("bedrock/")


def has_bedrock_auth_env() -> bool:
    """Check if Bedrock auth environment variables are configured."""
    return bool(
        os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        or (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"))
        or os.environ.get("AWS_PROFILE")
    )


def add_model_cli_args(
    cmd: list[str],
    model_args: dict,
    prefix: str,
    *,
    include_name: bool = True,
    include_context_window: bool = False,
) -> None:
    """Add model arguments to CLI command list.

    Args:
        cmd: Command list to extend
        model_args: Model arguments dict from get_model_args()
        prefix: CLI prefix (e.g., "--agent.model")
        include_name: Whether to include the model name (default: True)
        include_context_window: Whether to include context_window (default: False)
    """
    if include_name and model_args.get("name"):
        cmd.extend([f"{prefix}.name", model_args["name"]])

    # standard keys
    keys = ["api_base", "api_key", "max_input_tokens", "max_output_tokens"]
    if include_context_window:
        keys.append("context_window")

    for key in keys:
        if model_args.get(key):
            cmd.extend([f"{prefix}.{key}", str(model_args[key])])

    # bypass_cost_limits as explicit "True" string
    if model_args.get("bypass_cost_limits"):
        cmd.extend([f"{prefix}.bypass_cost_limits", "True"])

    # extra_headers via completion_kwargs
    completion_kwargs = dict(model_args.get("completion_kwargs") or {})
    if model_args.get("extra_headers"):
        completion_kwargs.setdefault("extra_headers", model_args["extra_headers"])
    if completion_kwargs:
        cmd.extend([f"{prefix}.completion_kwargs", json.dumps(completion_kwargs)])
