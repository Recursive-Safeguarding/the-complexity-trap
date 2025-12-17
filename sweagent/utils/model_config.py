"""Model presets for sweagent CLI."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


@dataclass
class ModelPreset:
    name: str
    api_base: str | None = None
    api_key_var: str | None = None
    description: str = ""
    supports_function_calling: bool = True
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None

    def get_cli_args(self) -> dict[str, Any]:
        args: dict[str, Any] = {"name": self.name}
        if self.api_base:
            args["api_base"] = self.api_base
        if self.api_key_var:
            args["api_key"] = f"${self.api_key_var}"
        if self.max_input_tokens:
            args["max_input_tokens"] = self.max_input_tokens
        if self.max_output_tokens:
            args["max_output_tokens"] = self.max_output_tokens
        return args


MODEL_PRESETS: dict[str, ModelPreset] = {
    # OpenAI
    "gpt-4o": ModelPreset("gpt-4o", api_key_var="OPENAI_API_KEY", description="GPT-4o"),
    "gpt-4o-mini": ModelPreset("gpt-4o-mini", api_key_var="OPENAI_API_KEY", description="GPT-4o Mini"),
    "gpt-5.2": ModelPreset(
        name="gpt-5.2",
        api_key_var="OPENAI_API_KEY",
        description="OpenAI GPT-5.2 (frontier)",
        max_input_tokens=400000,
        max_output_tokens=128000,
    ),
    "gpt-5.2-pro": ModelPreset(
        name="gpt-5.2-pro",
        api_key_var="OPENAI_API_KEY",
        description="OpenAI GPT-5.2 Pro",
        max_input_tokens=400000,
        max_output_tokens=128000,
    ),
    "gpt-5.2-chat-latest": ModelPreset(
        name="gpt-5.2-chat-latest",
        api_key_var="OPENAI_API_KEY",
        description="OpenAI GPT-5.2 Chat (rolling alias)",
        max_input_tokens=128000,
        max_output_tokens=16384,
    ),
    "gpt-5.1": ModelPreset(
        name="gpt-5.1",
        api_key_var="OPENAI_API_KEY",
        description="OpenAI GPT-5.1",
        max_input_tokens=272000,
        max_output_tokens=128000,
    ),
    "gpt-5.1-codex": ModelPreset(
        name="gpt-5.1-codex",
        api_key_var="OPENAI_API_KEY",
        description="OpenAI GPT-5.1 Codex (agentic coding)",
        max_input_tokens=272000,
        max_output_tokens=128000,
    ),
    "gpt-5.1-codex-mini": ModelPreset(
        name="gpt-5.1-codex-mini",
        api_key_var="OPENAI_API_KEY",
        description="OpenAI GPT-5.1 Codex Mini (agentic coding, faster/cheaper)",
        max_input_tokens=272000,
        max_output_tokens=128000,
    ),
    "gpt-5.1-codex-max": ModelPreset(
        name="gpt-5.1-codex-max",
        api_key_var="OPENAI_API_KEY",
        description="OpenAI GPT-5.1 Codex Max (agentic coding, higher quality)",
        max_input_tokens=400000,
        max_output_tokens=128000,
    ),
    "gpt-5": ModelPreset(
        name="gpt-5",
        api_key_var="OPENAI_API_KEY",
        description="OpenAI GPT-5",
        max_input_tokens=272000,
        max_output_tokens=128000,
    ),
    "gpt-5-chat-latest": ModelPreset(
        name="gpt-5-chat-latest",
        api_key_var="OPENAI_API_KEY",
        description="OpenAI GPT-5 Chat (rolling alias)",
        supports_function_calling=False,
        max_input_tokens=128000,
        max_output_tokens=16384,
    ),
    "gpt-5-mini": ModelPreset(
        name="gpt-5-mini",
        api_key_var="OPENAI_API_KEY",
        description="OpenAI GPT-5 Mini (fast, cheaper)",
        max_input_tokens=272000,
        max_output_tokens=128000,
    ),
    "gpt-5-nano": ModelPreset(
        name="gpt-5-nano",
        api_key_var="OPENAI_API_KEY",
        description="OpenAI GPT-5 Nano (fastest, most cost-efficient)",
        max_input_tokens=272000,
        max_output_tokens=128000,
    ),

    # DeepSeek
    "deepseek-chat": ModelPreset("deepseek/deepseek-chat", api_base="https://api.deepseek.com/v1", api_key_var="DEEPSEEK_API_KEY", description="DeepSeek V3", max_input_tokens=128000, max_output_tokens=8192),
    "deepseek-reasoner": ModelPreset("deepseek/deepseek-reasoner", api_base="https://api.deepseek.com/v1", api_key_var="DEEPSEEK_API_KEY", description="DeepSeek R1", supports_function_calling=False, max_input_tokens=128000, max_output_tokens=8192),

    # Anthropic
    "claude-sonnet-4.5": ModelPreset("claude-sonnet-4-5-20250929", api_key_var="ANTHROPIC_API_KEY", description="Claude Sonnet 4.5"),
    "claude-haiku-4.5": ModelPreset("claude-haiku-4-5-20251001", api_key_var="ANTHROPIC_API_KEY", description="Claude Haiku 4.5"),
    "claude-opus-4.5": ModelPreset("claude-opus-4-5-20251101", api_key_var="ANTHROPIC_API_KEY", description="Claude Opus 4.5"),

    # GLM (Z.AI endpoint)
    "glm-4.6": ModelPreset("anthropic/glm-4.6", api_base="https://api.z.ai/api/anthropic", api_key_var="ZHIPUAI_API_KEY", description="GLM-4.6 (355B MoE, 32B active)", max_input_tokens=200000, max_output_tokens=131072),
    "glm-4.5-air": ModelPreset("anthropic/glm-4.5-air", api_base="https://api.z.ai/api/anthropic", api_key_var="ZHIPUAI_API_KEY", description="GLM-4.5 Air", max_input_tokens=128000, max_output_tokens=16384),

    # MiniMax
    "minimax-m2": ModelPreset("openai/MiniMax-M2", api_base="https://api.minimax.io/v1", api_key_var="MINIMAX_API_KEY", description="MiniMax M2 (230B MoE, 10B active)", max_input_tokens=204800, max_output_tokens=196608),

    # Kimi / Moonshot
    "kimi-k2": ModelPreset("anthropic/kimi-for-coding", api_base="https://api.kimi.com/coding/", api_key_var="MOONSHOT_API_KEY", description="Kimi K2 (1T MoE, 32B active)", max_input_tokens=262144, max_output_tokens=32768),
    "kimi-k2-free": ModelPreset("openrouter/moonshotai/kimi-k2:free", api_key_var="OPENROUTER_API_KEY", description="Kimi K2 (OpenRouter free)"),

    # Vertex AI
    "gemini-2.5-flash": ModelPreset("vertex_ai/gemini-2.5-flash", description="Gemini 2.5 Flash"),
    "gemini-3-pro-preview": ModelPreset("vertex_ai/gemini-3-pro-preview", description="Gemini 3 Pro Preview"),

    # OpenRouter
    "openrouter/qwen/qwen3-coder:free": ModelPreset("openrouter/qwen/qwen3-coder:free", api_key_var="OPENROUTER_API_KEY", description="Qwen3 Coder (free)"),
    "openrouter/deepseek/deepseek-chat-v3-0324:free": ModelPreset("openrouter/deepseek/deepseek-chat-v3-0324:free", api_key_var="OPENROUTER_API_KEY", description="DeepSeek V3 (free)"),
    "openrouter/google/gemini-2.5-flash-preview-05-20": ModelPreset("openrouter/google/gemini-2.5-flash-preview-05-20", api_key_var="OPENROUTER_API_KEY", description="Gemini 2.5 Flash"),
    "openrouter/anthropic/claude-sonnet-4": ModelPreset("openrouter/anthropic/claude-sonnet-4", api_key_var="OPENROUTER_API_KEY", description="Claude Sonnet 4"),
    "openrouter/meta-llama/llama-4-maverick:free": ModelPreset("openrouter/meta-llama/llama-4-maverick:free", api_key_var="OPENROUTER_API_KEY", description="Llama 4 Maverick (free)"),

    # AWS Bedrock (Converse API)
    # Auth: AWS_BEARER_TOKEN_BEDROCK, or standard AWS creds (env/profile/SSO)
    # Region: AWS_REGION or AWS_DEFAULT_REGION
    "bedrock-qwen3-32b": ModelPreset("bedrock/converse/qwen.qwen3-32b-v1:0", description="Qwen3 32B", max_input_tokens=128000, max_output_tokens=128000),
    "bedrock-qwen3-coder-480b": ModelPreset("bedrock/converse/qwen.qwen3-coder-480b-a35b-v1:0", description="Qwen3 Coder 480B", max_input_tokens=262144, max_output_tokens=262144),

    "bedrock-claude-haiku-4.5": ModelPreset("bedrock/converse/anthropic.claude-haiku-4-5-20251001-v1:0", description="Claude Haiku 4.5", max_input_tokens=200000, max_output_tokens=64000),
    "bedrock-claude-haiku-4.5-us": ModelPreset("bedrock/converse/us.anthropic.claude-haiku-4-5-20251001-v1:0", description="Claude Haiku 4.5 (US)", max_input_tokens=200000, max_output_tokens=64000),
    "bedrock-claude-haiku-4.5-eu": ModelPreset("bedrock/converse/eu.anthropic.claude-haiku-4-5-20251001-v1:0", description="Claude Haiku 4.5 (EU)", max_input_tokens=200000, max_output_tokens=64000),
    "bedrock-flash-equivalent": ModelPreset("bedrock/converse/anthropic.claude-haiku-4-5-20251001-v1:0", description="Claude Haiku 4.5 (Flash substitute)", max_input_tokens=200000, max_output_tokens=8192),

    "bedrock-nova-pro": ModelPreset("bedrock/converse/amazon.nova-pro-v1:0", description="Nova Pro", supports_function_calling=False, max_input_tokens=300000, max_output_tokens=5000),
    "bedrock-nova-pro-us": ModelPreset("bedrock/converse/us.amazon.nova-pro-v1:0", description="Nova Pro (US)", supports_function_calling=False, max_input_tokens=300000, max_output_tokens=5000),
    "bedrock-nova-pro-eu": ModelPreset("bedrock/converse/eu.amazon.nova-pro-v1:0", description="Nova Pro (EU)", supports_function_calling=False, max_input_tokens=300000, max_output_tokens=5000),

    "bedrock-nova-lite": ModelPreset("bedrock/converse/amazon.nova-lite-v1:0", description="Nova Lite", supports_function_calling=False, max_input_tokens=300000, max_output_tokens=5000),
    "bedrock-nova-lite-us": ModelPreset("bedrock/converse/us.amazon.nova-lite-v1:0", description="Nova Lite (US)", supports_function_calling=False, max_input_tokens=300000, max_output_tokens=5000),
    "bedrock-nova-lite-eu": ModelPreset("bedrock/converse/eu.amazon.nova-lite-v1:0", description="Nova Lite (EU)", supports_function_calling=False, max_input_tokens=300000, max_output_tokens=5000),

    "bedrock-nova-micro": ModelPreset("bedrock/converse/amazon.nova-micro-v1:0", description="Nova Micro", supports_function_calling=False, max_input_tokens=128000, max_output_tokens=5000),
    "bedrock-nova-micro-us": ModelPreset("bedrock/converse/us.amazon.nova-micro-v1:0", description="Nova Micro (US)", supports_function_calling=False, max_input_tokens=128000, max_output_tokens=5000),
    "bedrock-nova-micro-eu": ModelPreset("bedrock/converse/eu.amazon.nova-micro-v1:0", description="Nova Micro (EU)", supports_function_calling=False, max_input_tokens=128000, max_output_tokens=5000),
}


# Coding plan models (monthly subscription, not per-token)
CUSTOM_MODEL_PRICING: dict[str, dict] = {
    "anthropic/kimi-for-coding": {
        "input_cost_per_token": 0,
        "output_cost_per_token": 0,
        "max_tokens": 262144,
        "litellm_provider": "anthropic",
        "mode": "chat",
    },
    "anthropic/glm-4.6": {
        "input_cost_per_token": 0,
        "output_cost_per_token": 0,
        "max_tokens": 200000,
        "litellm_provider": "anthropic",
        "mode": "chat",
    },
    "glm-4.6": {
        "input_cost_per_token": 0,
        "output_cost_per_token": 0,
        "max_tokens": 200000,
        "litellm_provider": "anthropic",
        "mode": "chat",
    },
    "openai/MiniMax-M2": {
        "input_cost_per_token": 0,
        "output_cost_per_token": 0,
        "max_tokens": 204800,
        "litellm_provider": "openai",
        "mode": "chat",
    },
    "MiniMax-M2": {
        "input_cost_per_token": 0,
        "output_cost_per_token": 0,
        "max_tokens": 204800,
        "litellm_provider": "openai",
        "mode": "chat",
    },
}


def get_model_args(model_key: str) -> dict[str, Any]:
    """Return CLI args for a model preset."""
    preset = MODEL_PRESETS.get(model_key)
    if not preset:
        raise ValueError(f"Unknown model: {model_key}. Available: {', '.join(sorted(MODEL_PRESETS))}")

    args = preset.get_cli_args()

    # Auto-select regional inference profile for Bedrock Anthropic models
    if model_key in ("bedrock-claude-haiku-4.5", "bedrock-flash-equivalent"):
        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or ""
        if region.startswith("eu-"):
            args["name"] = MODEL_PRESETS["bedrock-claude-haiku-4.5-eu"].name
        elif region.startswith("us-"):
            args["name"] = MODEL_PRESETS["bedrock-claude-haiku-4.5-us"].name

    return args


def print_models():
    """Print available presets."""
    for key, p in MODEL_PRESETS.items():
        fc = "+" if p.supports_function_calling else "-"
        print(f"  {key:45} [{fc}FC] {p.description}")


if __name__ == "__main__":
    print_models()
