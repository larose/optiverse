import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Callable
from openai import OpenAI
from .evaluator import Evaluator
from .strategies import Strategy


@dataclass
class Problem:
    description: str
    initial_solution: str
    evaluator: Evaluator


@dataclass
class LLMConfig:
    model: str
    client: OpenAI


@dataclass
class OptimizerConfig:
    llm: LLMConfig
    max_iterations: int
    problem: Problem
    search_strategy: Strategy
    directory: Path


def _create_openai_client(api_key: str) -> OpenAI:
    """Create OpenAI client for OpenAI provider."""
    return OpenAI(api_key=api_key)


def _create_google_client(api_key: str) -> OpenAI:
    """Create OpenAI-compatible client for Google Gemini provider."""
    return OpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )


def _create_nvidia_client(api_key: str) -> OpenAI:
    """Create OpenAI-compatible client for NVIDIA provider."""
    return OpenAI(
        api_key=api_key,
        base_url="https://integrate.api.nvidia.com/v1",
    )


def create_llm_config_from_env() -> LLMConfig:
    """
    Create LLMConfig from environment variables.

    Required environment variables:
    - LLM_API_KEY: API key for the LLM provider
    - LLM_MODEL: Model name to use
    - LLM_PROVIDER: Provider name (openai, google, nvidia)

    Returns:
        LLMConfig: Configured LLM configuration

    Raises:
        ValueError: If any required environment variable is missing or provider is unsupported
    """
    # Check required environment variables
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise ValueError("LLM_API_KEY environment variable is required")

    model = os.getenv("LLM_MODEL")
    if not model:
        raise ValueError("LLM_MODEL environment variable is required")

    provider = os.getenv("LLM_PROVIDER")
    if not provider:
        raise ValueError("LLM_PROVIDER environment variable is required")

    # Provider configurations
    provider_configs: Dict[str, Callable[[str], OpenAI]] = {
        "openai": _create_openai_client,
        "google": _create_google_client,
        "nvidia": _create_nvidia_client,
    }

    # Check if provider is supported
    if provider.lower() not in provider_configs:
        supported_providers = ", ".join(provider_configs.keys())
        raise ValueError(
            f"Unsupported provider '{provider}'. Supported providers: {supported_providers}"
        )

    # Create client for the specified provider
    client_factory = provider_configs[provider.lower()]
    client = client_factory(api_key)

    return LLMConfig(model=model, client=client)
