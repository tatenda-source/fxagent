"""Multi-provider LLM client abstraction.

Supports OpenAI, Anthropic, and Ollama (local). Uses a two-tier strategy:
- "deep_think" LLM for judges/managers (more capable, slower)
- "quick_think" LLM for analysts/debaters (faster, cheaper)

Inspired by TradingAgents' provider-agnostic approach.
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Abstract base for LLM providers."""

    @abstractmethod
    def generate(self, prompt: str, system: str = "", temperature: float = 0.3,
                 max_tokens: int = 2000) -> str:
        pass


class OpenAIClient(LLMClient):
    """OpenAI API client (GPT-4o, GPT-4o-mini, etc.)."""

    def __init__(self, model: str = "gpt-4o-mini", api_key: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai")
        self.model = model
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def generate(self, prompt: str, system: str = "", temperature: float = 0.3,
                 max_tokens: int = 2000) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


class AnthropicClient(LLMClient):
    """Anthropic API client (Claude)."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: Optional[str] = None):
        try:
            import anthropic
        except ImportError:
            raise ImportError("pip install anthropic")
        self.model = model
        self.client = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    def generate(self, prompt: str, system: str = "", temperature: float = 0.3,
                 max_tokens: int = 2000) -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = self.client.messages.create(**kwargs)
        return response.content[0].text


class OllamaClient(LLMClient):
    """Ollama local LLM client."""

    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def generate(self, prompt: str, system: str = "", temperature: float = 0.3,
                 max_tokens: int = 2000) -> str:
        import requests

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if system:
            payload["system"] = system

        response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["response"]


def create_llm_client(provider: str = "openai", model: Optional[str] = None,
                      api_key: Optional[str] = None) -> LLMClient:
    """Factory function to create an LLM client.

    Args:
        provider: "openai", "anthropic", or "ollama"
        model: Model name override. If None, uses provider default.
        api_key: API key override. If None, reads from environment.
    """
    providers = {
        "openai": lambda: OpenAIClient(model=model or "gpt-4o-mini", api_key=api_key),
        "anthropic": lambda: AnthropicClient(model=model or "claude-sonnet-4-20250514", api_key=api_key),
        "ollama": lambda: OllamaClient(model=model or "llama3.1:8b"),
    }

    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Choose from: {list(providers.keys())}")

    return providers[provider]()
