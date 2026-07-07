from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import requests

from .env import load_repo_dotenv, require_env


@dataclass
class EmbeddingResult:
    # 保存 embedding 矩阵和 token 统计，便于实验表记录成本。
    embeddings: np.ndarray
    prompt_tokens: int = 0
    total_tokens: int = 0


class OpenAICompatibleEmbeddingClient:
    # 兼容 OpenAI /v1/embeddings 的真实 embedding 客户端；配置只读 LLM_* 环境变量。
    def __init__(
        self,
        *,
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        batch_size: int = 8,
        batch_delay: float = 0.0,
    ) -> None:
        load_repo_dotenv()
        self.model = require_env("LLM_EMBEDDING_MODEL")
        self.api_key = require_env("LLM_API_KEY")
        self.base_url = require_env("LLM_BASE_URL").rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.batch_size = batch_size
        self.batch_delay = batch_delay
        if self.batch_size <= 0:
            raise ValueError("embedding_batch_size must be positive.")
        if self.max_retries < 0:
            raise ValueError("embedding_max_retries must be non-negative.")

    def embed(self, texts: list[str]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult(embeddings=np.zeros((0, 0), dtype=np.float32))

        batches = [
            texts[start : start + self.batch_size]
            for start in range(0, len(texts), self.batch_size)
        ]
        results = []
        prompt_tokens = 0
        total_tokens = 0
        for index, batch in enumerate(batches):
            result = self._embed_batch(batch)
            results.append(result.embeddings)
            prompt_tokens += result.prompt_tokens
            total_tokens += result.total_tokens
            if self.batch_delay > 0 and index < len(batches) - 1:
                time.sleep(self.batch_delay)
        return EmbeddingResult(
            embeddings=np.vstack(results).astype(np.float32),
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
        )

    def _embed_batch(self, texts: list[str]) -> EmbeddingResult:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "input": texts,
        }

        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    f"{self.base_url}/embeddings",
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                if response.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    time.sleep(_retry_delay(response, default=self.retry_delay * (2 ** attempt)))
                    continue
                response.raise_for_status()
                return _embedding_result(response.json(), expected_count=len(texts))
            except requests.RequestException:
                if attempt >= self.max_retries:
                    raise
                time.sleep(self.retry_delay * (2 ** attempt))

        raise RuntimeError("Embedding request failed without returning a response.")


def _embedding_result(data: dict, *, expected_count: int) -> EmbeddingResult:
    rows = data.get("data", [])
    if len(rows) != expected_count:
        raise RuntimeError(
            f"Embedding response returned {len(rows)} vectors, expected {expected_count}."
        )

    ordered = sorted(rows, key=lambda item: int(item.get("index", 0)))
    vectors = []
    for item in ordered:
        vector = item.get("embedding")
        if not isinstance(vector, list) or not vector:
            raise RuntimeError("Embedding response contains an empty or invalid embedding vector.")
        vectors.append(vector)

    matrix = np.asarray(vectors, dtype=np.float32)
    usage = data.get("usage", {})
    return EmbeddingResult(
        embeddings=matrix,
        prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
        total_tokens=int(usage.get("total_tokens", 0) or 0),
    )


def _retry_delay(response: requests.Response, *, default: float) -> float:
    retry_after = response.headers.get("Retry-After")
    if not retry_after:
        return default
    try:
        return max(default, float(retry_after))
    except ValueError:
        return default
