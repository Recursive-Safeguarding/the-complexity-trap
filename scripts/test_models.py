#!/usr/bin/env python3
"""Quick test to verify all model presets can make API calls."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import litellm

litellm.suppress_debug_info = True

from sweagent.utils.model_config import MODEL_PRESETS, get_model_args

TEST_MODELS = [
    "gpt-4o-mini",          # OpenAI
    "deepseek-chat",        # DeepSeek V3
    "deepseek-speciale",    # DeepSeek V3.2-Speciale (needs enable_thinking)
    "claude-haiku-4.5",     # Anthropic Claude 4.5 (fastest/cheapest)
    "glm-4.6",              # GLM/ZhipuAI
    "minimax-m2",           # MiniMax
    "kimi-k2",              # Kimi/Moonshot
    "bedrock-qwen3-32b",    # AWS Bedrock (main paper-repro model)
    "bedrock-nova-pro",     # AWS Bedrock (summarizer substitute if Anthropic not enabled)
    # "gemini-2.5-flash",   # Vertex AI - SKIP: requires valid GCP credentials
    "openrouter/anthropic/claude-sonnet-4",  # OpenRouter (paid, less rate-limited)
]

def test_model(preset_key: str) -> tuple[bool, str]:
    preset = MODEL_PRESETS.get(preset_key)
    if not preset:
        return False, f"Unknown preset: {preset_key}"

    is_bedrock = preset.name.startswith("bedrock/")
    model_name = get_model_args(preset_key)["name"] if is_bedrock else preset.name
    api_key = os.environ.get(preset.api_key_var) if preset.api_key_var else None
    if not api_key and preset.api_key_var and not is_bedrock:
        return False, f"Missing env var: {preset.api_key_var}"
    if is_bedrock:
        has_region = bool(os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION"))
        has_api_key = bool(os.environ.get("AWS_BEARER_TOKEN_BEDROCK"))
        has_access_keys = bool(os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"))
        has_profile = bool(os.environ.get("AWS_PROFILE"))
        if not has_region:
            return True, "SKIP (set AWS_DEFAULT_REGION or AWS_REGION)"
        if not (has_api_key or has_access_keys or has_profile):
            return True, "SKIP (no Bedrock auth env vars set)"

    try:
        kwargs = {
            "model": model_name,
            "messages": [{"role": "user", "content": "Say 'OK' and nothing else."}],
            "max_tokens": 10,
        }

        if preset.api_base:
            kwargs["api_base"] = preset.api_base
        if api_key:
            kwargs["api_key"] = api_key

        # LiteLLM boto3 workaround: pass dummy creds when using bearer-token auth
        if is_bedrock and os.environ.get("AWS_BEARER_TOKEN_BEDROCK") and not has_access_keys:
            kwargs["aws_access_key_id"] = "DUMMY"
            kwargs["aws_secret_access_key"] = "DUMMY"
            region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
            if region:
                kwargs["aws_region_name"] = region

        if "speciale" in preset_key.lower() or "reasoner" in preset.name.lower():
            kwargs["max_tokens"] = 500  # reasoner needs more tokens

        response = litellm.completion(**kwargs)
        content = response.choices[0].message.content.strip()
        return True, content[:50]  # Truncate response

    except Exception as e:
        return False, str(e)[:100]

def main():
    print("Testing model presets...")
    print("=" * 70)

    results = []
    for preset_key in TEST_MODELS:
        print(f"\n  Testing {preset_key}...", end=" ", flush=True)
        success, msg = test_model(preset_key)
        status = "✓" if success else "✗"
        print(f"{status}")
        if success:
            print(f"    Response: {msg}")
        else:
            print(f"    Error: {msg}")
        results.append((preset_key, success, msg))

    print("\n" + "=" * 70)
    print("Summary:")
    passed = sum(1 for _, s, _ in results if s)
    print(f"  {passed}/{len(results)} models working")

    if passed < len(results):
        print("\nFailed models:")
        for name, success, msg in results:
            if not success:
                print(f"  - {name}: {msg}")
        return 1
    return 0

if __name__ == "__main__":
    sys.exit(main())
