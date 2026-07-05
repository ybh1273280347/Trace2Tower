from __future__ import annotations

import time
from dataclasses import dataclass

import requests

from .env import load_repo_dotenv, require_env


@dataclass
class ChatResult:
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAICompatibleChatClient:
    def __init__(
        self,
        model: str,
        api_key: str = "",
        base_url: str = "",
        temperature: float = 0.0,
        max_tokens: int = 128,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        load_repo_dotenv()
        self.model = model
        self.api_key = api_key or require_env("LLM_API_KEY")
        self.base_url = (base_url or require_env("LLM_BASE_URL")).rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        if not self.api_key:
            raise RuntimeError("LLM agent requires LLM_API_KEY or agent.api_key.")

    def chat(self, messages: list[dict[str, str]]) -> ChatResult:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    time.sleep(self.retry_delay * (2 ** attempt))
                    continue
                response.raise_for_status()
                data = response.json()
                usage = data.get("usage", {})
                return ChatResult(
                    content=data["choices"][0]["message"]["content"],
                    prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
                    completion_tokens=int(usage.get("completion_tokens", 0) or 0),
                    total_tokens=int(usage.get("total_tokens", 0) or 0),
                )
            except requests.RequestException:
                if attempt >= self.max_retries:
                    raise
                time.sleep(self.retry_delay * (2 ** attempt))

        raise RuntimeError("LLM request failed without returning a response.")
