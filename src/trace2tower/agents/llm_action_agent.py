from __future__ import annotations

import json
import re
from typing import Any

from trace2tower.llm import OpenAICompatibleChatClient

from .base import BaseAgent


class LLMActionAgent(BaseAgent):
    def __init__(self, client: OpenAICompatibleChatClient, max_skill_chars: int = 2400) -> None:
        self.client = client
        self.max_skill_chars = max_skill_chars
        self.last_metadata: dict[str, Any] = {}

    def act(self, observation: str, info: dict[str, Any]) -> str:
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
        parsed = self._parse_json_action(text)
        if parsed in actions:
            return parsed

        for action in actions:
            if action in text:
                return action
        return None

    def _parse_json_action(self, text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            return ""
        return str(data.get("action", ""))
