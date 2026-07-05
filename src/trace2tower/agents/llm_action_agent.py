from __future__ import annotations

import json
import re
from typing import Any

from trace2tower.llm import OpenAICompatibleChatClient

from .base import BaseAgent


class LLMActionAgent(BaseAgent):
    # 基于 LLM 的动作选择 agent：把任务目标、观测、候选动作和检索到的技能拼成 prompt。
    def __init__(self, client: OpenAICompatibleChatClient, max_skill_chars: int = 2400) -> None:
        self.client = client
        self.max_skill_chars = max_skill_chars
        self.last_metadata: dict[str, Any] = {}

    def act(self, observation: str, info: dict[str, Any]) -> str:
        # 获取当前可用动作；若环境未提供则使用 noop 占位，避免后续索引错误。
        actions = info.get("admissible_actions") or ["noop"]
        messages = [
            {
                "role": "system",
                "content": self._system_prompt(),
            },
            {
                "role": "user",
                "content": self._user_prompt(
                    goal=str(info.get("goal", "")),
                    observation=observation,
                    actions=actions,
                    skills=info.get("retrieved_skills", []),
                ),
            },
        ]
        result = self.client.chat(messages)
        action = self._parse_action(result.content, actions)
        # 若 LLM 输出无法匹配任何合法动作，则回退到第一个可用动作并记录。
        fallback = action is None
        if fallback:
            action = actions[0]

        self.last_metadata = {
            "llm_response": result.content,
            "llm_action_fallback": fallback,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
        }
        return action

    def _system_prompt(self) -> str:
        return (
            "You are an action-selection agent for a text environment. "
            "Choose exactly one action from the provided admissible actions. "
            "Use the skill library only as guidance. "
            "Return strict JSON only: {\"action\": \"<one admissible action>\"}."
        )

    def _user_prompt(
        self,
        *,
        goal: str,
        observation: str,
        actions: list[str],
        skills: list[dict[str, Any]],
    ) -> str:
        return "\n\n".join(
            [
                f"Goal:\n{goal}",
                f"Observation:\n{observation[:4000]}",
                f"Relevant skills:\n{self._format_skills(skills)}",
                "Admissible actions:\n" + "\n".join(f"- {action}" for action in actions),
            ]
        )

    def _format_skills(self, skills: list[dict[str, Any]]) -> str:
        # 将检索到的技能拼接成 prompt 文本，并按 max_skill_chars 截断，控制上下文长度。
        if not skills:
            return "None."

        blocks = []
        remaining = self.max_skill_chars
        for skill in skills:
            content = "\n".join(
                [
                    f"Skill: {skill.get('name', '')}",
                    f"Granularity: {skill.get('granularity', '')}",
                    str(skill.get("content") or skill.get("embedding_text") or ""),
                ]
            )
            if len(content) > remaining:
                content = content[:remaining]
            blocks.append(content)
            remaining -= len(content)
            if remaining <= 0:
                break
        return "\n\n".join(blocks)

    def _parse_action(self, text: str, actions: list[str]) -> str | None:
        # 优先解析严格 JSON；失败后尝试在 LLM 输出中查找任意合法动作字符串。
        parsed = self._parse_json_action(text)
        if parsed in actions:
            return parsed

        for action in actions:
            if action in text:
                return action
        return None

    def _parse_json_action(self, text: str) -> str:
        # 去除常见 markdown 代码块包裹，再尝试按 JSON 解析 action 字段。
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            return ""
        return str(data.get("action", ""))
