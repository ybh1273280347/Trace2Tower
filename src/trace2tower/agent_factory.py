from __future__ import annotations

from typing import Any

from .agents import BaseAgent, LLMActionAgent, RandomAgent
from .env import require_env
from .llm import OpenAICompatibleChatClient


def build_agent(name: str, config: dict[str, Any] | None = None) -> BaseAgent:
    if name not in {"smoke_random", "random"}:
        if name == "llm_action":
            settings = config or {}
            client = OpenAICompatibleChatClient(
                model=_llm_model(settings),
                api_key=settings.get("api_key", ""),
                base_url=settings.get("base_url", ""),
                temperature=float(settings.get("temperature", 0.0)),
                max_tokens=int(settings.get("max_tokens", 128)),
                timeout=int(settings.get("timeout", 60)),
                max_retries=int(settings.get("max_retries", 3)),
            )
            return LLMActionAgent(
                client=client,
                max_skill_chars=int(settings.get("max_skill_chars", 2400)),
            )
        raise ValueError(f"Unsupported agent: {name}")

    settings = config or {}
    seed = settings.get("seed")
    return RandomAgent(seed=int(seed) if seed is not None else None)


def _llm_model(settings: dict[str, Any]) -> str:
    return require_env("LLM_MODEL")
