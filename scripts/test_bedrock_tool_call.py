#!/usr/bin/env python3
"""Smoke-test Bedrock tool calling before running SWE-bench jobs."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

import litellm

litellm.suppress_debug_info = True

from sweagent.utils.model_config import get_model_args


def _has_region() -> bool:
    return bool(os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION"))


def _has_any_obvious_auth() -> bool:
    return bool(
        os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        or (os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"))
        or os.environ.get("AWS_PROFILE")
    )


def _require_boto3() -> bool:
    try:
        import boto3  # noqa: F401
        import botocore  # noqa: F401
    except Exception:
        return False
    return True


def _print_env_hint() -> None:
    if not _has_region():
        print("SKIP: AWS_DEFAULT_REGION/AWS_REGION not set. Example: AWS_DEFAULT_REGION=eu-west-2")
    if not _has_any_obvious_auth():
        print("Note: No Bedrock auth env vars found (may still work via ~/.aws or instance roles).")
    if not _require_boto3():
        print("ERROR: boto3/botocore not installed (required by LiteLLM Bedrock).")
        print("Run: uv sync --extra dev")


def test_tool_call(preset_key: str) -> int:
    model_args = get_model_args(preset_key)
    model_name = model_args["name"]

    print(f"\nTesting tool calling on: {preset_key} ({model_name})")

    tools = [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Echo the provided text.",
                "parameters": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                },
            },
        }
    ]

    messages = [
        {
            "role": "user",
            "content": "Call the tool `echo` with text 'hello'. Do not answer normally.",
        }
    ]

    # LiteLLM boto3 workaround: pass dummy creds when using bearer-token auth
    extra_bedrock_kwargs: dict[str, str] = {}
    if os.environ.get("AWS_BEARER_TOKEN_BEDROCK") and not (
        os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")
    ):
        extra_bedrock_kwargs["aws_access_key_id"] = "DUMMY"
        extra_bedrock_kwargs["aws_secret_access_key"] = "DUMMY"
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        if region:
            extra_bedrock_kwargs["aws_region_name"] = region

    try:
        response = litellm.completion(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.0,
            max_tokens=200,
            **extra_bedrock_kwargs,
        )
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        return 1

    choice = response.choices[0]
    msg = choice.message

    tool_calls = getattr(msg, "tool_calls", None)
    content = (msg.content or "").strip()

    print(f"content: {content!r}")
    print(f"tool_calls: {tool_calls!r}")

    if tool_calls:
        print("OK: tool_calls returned")
        return 0

    print("WARNING: No tool_calls returned.")
    return 2


def test_basic_completion(preset_key: str) -> int:
    model_args = get_model_args(preset_key)
    model_name = model_args["name"]

    print(f"\nTesting basic completion on: {preset_key} ({model_name})")

    messages = [{"role": "user", "content": "Say 'OK' and nothing else."}]

    # LiteLLM boto3 workaround
    extra_bedrock_kwargs: dict[str, str] = {}
    if os.environ.get("AWS_BEARER_TOKEN_BEDROCK") and not (
        os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY")
    ):
        extra_bedrock_kwargs["aws_access_key_id"] = "DUMMY"
        extra_bedrock_kwargs["aws_secret_access_key"] = "DUMMY"
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        if region:
            extra_bedrock_kwargs["aws_region_name"] = region

    try:
        response = litellm.completion(
            model=model_name,
            messages=messages,
            temperature=0.0,
            max_tokens=20,
            **extra_bedrock_kwargs,
        )
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
        return 1

    msg = response.choices[0].message
    content = (msg.content or "").strip()
    print(f"content: {content!r}")
    return 0 if content else 1


def main() -> int:
    _print_env_hint()
    if not _require_boto3():
        return 1
    if not _has_region():
        return 0
    if not _has_any_obvious_auth():
        print("SKIP: No Bedrock auth configured. Set AWS_BEARER_TOKEN_BEDROCK or AWS_PROFILE.")
        return 0

    rc_agent = test_tool_call("bedrock-qwen3-32b")
    rc_summarizer = test_basic_completion("bedrock-nova-pro")

    return 0 if rc_agent == 0 and rc_summarizer == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
