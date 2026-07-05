from __future__ import annotations

import time
from dataclasses import dataclass

import requests

from .env import load_repo_dotenv, require_env


@dataclass
class ChatResult:
    # 统一封装 LLM 返回内容以及 token 消耗，便于后续成本统计。
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class OpenAICompatibleChatClient:
    # 兼容 OpenAI 风格接口的聊天客户端；支持指数退避重试。
    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 128,
        timeout: int = 60,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        load_repo_dotenv()
        self.model = model
        self.api_key = require_env("LLM_API_KEY")
        self.base_url = require_env("LLM_BASE_URL").rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def chat(self, messages: list[dict[str, str]]) -> ChatResult:
        # 构造请求头与 payload，只使用 chat completions 接口。
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

        # 指数退避：对限流和服务端错误最多重试 max_retries 次。
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

        # 正常情况不会到达这里，保留防御性兜底。
        raise RuntimeError("LLM request failed without returning a response.")
