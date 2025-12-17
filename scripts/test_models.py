#!/usr/bin/env python3
"""Test script to verify all model presets can make API calls."""

import argparse
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
    "gpt-4o-mini",          # OpenAI completion API
    "gpt-5.1-codex-mini",   # OpenAI response API
    "deepseek-chat",        # DeepSeek V3
    "deepseek-reasoner",    # DeepSeek R1
    "glm-4.6",              # GLM/ZhipuAI
    "minimax-m2",           # MiniMax
    "kimi-k2",              # Kimi/Moonshot
    "bedrock-qwen3-32b",    # AWS Bedrock (paper-repro model)
    "bedrock-nova-pro",     # Amazon model on AWS Bedrock
    "openrouter/qwen/qwen3-coder:free",  # Qwen 3 480B (may hit rate limits)
    # "gemini-2.5-flash",   # Vertex AI - SKIP: requires valid GCP credentials
]

def test_model(preset_key: str, verbose: bool = False) -> tuple[bool, str]:
    preset = MODEL_PRESETS.get(preset_key)
    if not preset:
        return False, f"Unknown preset: {preset_key}"

    if verbose:
        print(f"\n{'='*70}")
        print(f"Testing: {preset_key}")
        print(f"{'='*70}")

    is_bedrock = preset.name.startswith("bedrock/")
    model_name = get_model_args(preset_key)["name"] if is_bedrock else preset.name

    if verbose:
        print(f"Model name: {model_name}")
        print(f"Description: {preset.description}")
        print(f"API base: {preset.api_base or 'default'}")
        print(f"API key var: {preset.api_key_var or 'N/A'}")
        if preset.max_input_tokens:
            print(f"Max input tokens: {preset.max_input_tokens:,}")
        if preset.max_output_tokens:
            print(f"Max output tokens: {preset.max_output_tokens:,}")
        print(f"Function calling: {'✓' if preset.supports_function_calling else '✗'}")

    api_key = os.environ.get(preset.api_key_var) if preset.api_key_var else None
    if not api_key and preset.api_key_var and not is_bedrock:
        msg = f"Missing env var: {preset.api_key_var}"
        if verbose:
            print(f"\n❌ FAILED: {msg}")
        return False, msg

    if is_bedrock:
        has_region = bool(os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION"))
        has_api_key = bool(os.environ.get("AWS_BEARER_TOKEN_BEDROCK"))
        has_access_keys = bool(os.environ.get("AWS_ACCESS_KEY_ID") and os.environ.get("AWS_SECRET_ACCESS_KEY"))
        has_profile = bool(os.environ.get("AWS_PROFILE"))
        if not has_region:
            msg = "SKIP (set AWS_DEFAULT_REGION or AWS_REGION)"
            if verbose:
                print(f"\n⊘ SKIPPED: {msg}")
            return True, msg
        if not (has_api_key or has_access_keys or has_profile):
            msg = "SKIP (no Bedrock auth env vars set)"
            if verbose:
                print(f"\n⊘ SKIPPED: {msg}")
            return True, msg

    if verbose and api_key:
        print(f"API key: {api_key[:10]}...{api_key[-4:] if len(api_key) > 14 else ''}")
        print(f"\nSending test request...")

    try:
        # Check if this is a Responses API model (gpt-5 series)
        is_response_api = "gpt-5" in model_name.lower()

        if is_response_api:
            # Responses API uses different endpoint and parameters
            # + max_output_tokens must be >= 16 for Responses API
            kwargs = {
                "model": model_name,
                "input": "Say 'OK' and nothing else.",
                "max_output_tokens": 16,
            }

            if preset.api_base:
                kwargs["api_base"] = preset.api_base
            if api_key:
                kwargs["api_key"] = api_key

            response = litellm.responses(**kwargs)
            output = response.output
            if isinstance(output, list):
                content = " ".join(str(item.text if hasattr(item, 'text') else item) for item in output).strip()
            else:
                content = str(output).strip()
        else:
            # completion API
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

        if verbose:
            print(f"\n✅ SUCCESS!")
            print(f"Response: {content[:100]}")

        return True, content[:50]

    except Exception as e:
        error_msg = str(e)[:100]
        if verbose:
            print(f"\n❌ FAILED!")
            print(f"Error: {error_msg}")
        return False, error_msg

def main():
    parser = argparse.ArgumentParser(
        description="Test model preset configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all models
  python scripts/test_models.py

  # Test specific models with verbose output
  python scripts/test_models.py kimi-k2 glm-4.6 minimax-m2 --verbose

  # List available models
  python scripts/test_models.py --list
        """
    )
    parser.add_argument(
        "models",
        nargs="*",
        help="Specific model keys to test (default: test all)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed configuration for each model"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available model presets and exit"
    )

    args = parser.parse_args()

    if args.list:
        print("Available model presets:")
        print("=" * 70)
        for key, preset in sorted(MODEL_PRESETS.items()):
            fc = "+" if preset.supports_function_calling else "-"
            print(f"  {key:45} [{fc}FC] {preset.description}")
        return 0

    test_models = args.models if args.models else TEST_MODELS

    if not args.verbose:
        print("Testing model presets...")
        print("=" * 70)
        print(f"\nTesting {len(test_models)} models: {', '.join(test_models)}\n")

    results = []
    for preset_key in test_models:
        if not args.verbose:
            print(f"  Testing {preset_key}...", end=" ", flush=True)

        success, msg = test_model(preset_key, verbose=args.verbose)
        status = "✓" if success else "✗"

        if not args.verbose:
            print(f"{status}")
            if success and not msg.startswith("SKIP"):
                print(f"    Response: {msg}")
            elif not success:
                print(f"    Error: {msg}")
            elif msg.startswith("SKIP"):
                print(f"    {msg}")

        results.append((preset_key, success, msg))

    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, s, m in results if s and not m.startswith("SKIP"))
    skipped = sum(1 for _, s, m in results if s and m.startswith("SKIP"))
    failed = sum(1 for _, s, _ in results if not s)

    print(f"\n✅ Passed: {passed}/{len(results)}")
    for name, success, msg in results:
        if success and not msg.startswith("SKIP"):
            print(f"   - {name}")

    if skipped > 0:
        print(f"\n⊘ Skipped: {skipped}/{len(results)}")
        for name, success, msg in results:
            if success and msg.startswith("SKIP"):
                print(f"   - {name}: {msg}")

    if failed > 0:
        print(f"\n❌ Failed: {failed}/{len(results)}")
        for name, success, msg in results:
            if not success:
                print(f"   - {name}: {msg}")

    print("\n" + "=" * 70)

    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
