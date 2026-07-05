from __future__ import annotations

from typing import Any

from trace2tower.agents import BaseAgent, LLMActionAgent
from trace2tower.env import require_env
from trace2tower.llm import OpenAICompatibleChatClient


def build_agent(name: str, config: dict[str, Any] | None = None) -> BaseAgent:
    # 按名称构造 agent；目前唯一实现是调用 LLM 的动作选择 agent。
    if name == "llm_action":
        settings = config or {}
        client = OpenAICompatibleChatClient(
            model=_llm_model(),
            temperature=float(settings.get("temperature", 0.0)),
            max_tokens=int(settings.get("max_tokens", 128)),
            timeout=int(settings.get("timeout", 60)),
            max_retries=int(settings.get("max_retries", 3)),
            retry_delay=float(settings.get("retry_delay", 1.0)),
        )
        return LLMActionAgent(
            client=client,
            max_skill_chars=int(settings.get("max_skill_chars", 2400)),
        )
    raise ValueError(f"Unsupported agent: {name}")


def _llm_model() -> str:
    # 从环境变量读取实验使用的 LLM 模型名。
    return require_env("LLM_MODEL")
