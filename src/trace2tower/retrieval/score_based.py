from __future__ import annotations

from typing import Any

import numpy as np

from trace2tower.embedding import EmbeddingResult, OpenAICompatibleEmbeddingClient
from trace2tower.text import cosine_text_similarity

from .base import BaseRetriever


class NoSkillRetriever(BaseRetriever):
    # 空检索器：用于 no-skill baseline，保证接口一致。
    def __init__(self) -> None:
        self.last_metadata: dict[str, Any] = {}

    def retrieve(self, model: dict, task_state: dict) -> list[dict]:
        self.last_metadata = {
            "retrieval_strategy": "none",
            "candidate_skill_count": 0,
            "returned_skill_count": 0,
        }
        return []


class ScoreBasedRetriever(BaseRetriever):
    def __init__(
        self,
        strategy: str,
        k: int = 3,
        *,
        use_embedding_similarity: bool = False,
        embedding_config: dict[str, Any] | None = None,
        level_top_k: dict[str, int] | None = None,
    ) -> None:
        self.strategy = strategy
        self.k = k
        self.level_top_k = {
            str(level): int(count)
            for level, count in (level_top_k or {}).items()
            if int(count) > 0
        }
        self.use_embedding_similarity = use_embedding_similarity
        self.embedding_client = (
            _build_embedding_client(embedding_config or {})
            if use_embedding_similarity
            else None
        )
        self._skill_cache_key: tuple[tuple[str, str], ...] | None = None
        self._skill_cache_embeddings: np.ndarray | None = None
        self._last_embedding_usage: dict[str, int] = {}
        self.last_metadata: dict[str, Any] = {}

    def retrieve(self, model: dict, task_state: dict) -> list[dict]:
        # 为每个技能计算当前 task_state 下的分数，返回前 k 个。
        self._last_embedding_usage = {}
        embedding_similarities = self._embedding_similarities(model, task_state)
        scored = []
        for skill in model.get("skills", []):
            item = dict(skill)
            metadata = dict(item.get("metadata", {}))
            # 在线检索只负责给当前 task_state 排序；离线指标在 scripts/evaluate_selectors.py 里批量算。
            score = score_skill(
                self.strategy,
                item,
                task_state,
                embedding_similarity=embedding_similarities.get(str(item.get("skill_id", ""))),
            )
            metadata["retrieval_score"] = score
            metadata["retrieval_strategy"] = self.strategy
            if self.use_embedding_similarity:
                metadata["retrieval_similarity_type"] = "embedding"
                metadata["retrieval_embedding_model"] = self.embedding_client.model
            item["metadata"] = metadata
            scored.append(item)

        scored.sort(
            key=lambda skill: skill.get("metadata", {}).get("retrieval_score", 0.0),
            reverse=True,
        )
        selected = _select_ranked_skills(scored, k=self.k, level_top_k=self.level_top_k)
        self.last_metadata = {
            "retrieval_strategy": self.strategy,
            "candidate_skill_count": len(scored),
            "returned_skill_count": len(selected),
            "level_top_k": self.level_top_k,
            "returned_by_level": _returned_by_level(selected),
        }
        if self.use_embedding_similarity:
            self.last_metadata.update(
                {
                    "retrieval_similarity_type": "embedding",
                    "retrieval_embedding_model": self.embedding_client.model,
                    **self._last_embedding_usage,
                }
            )
        return selected

    def _embedding_similarities(
        self,
        model: dict[str, Any],
        task_state: dict[str, Any],
    ) -> dict[str, float]:
        if not self.use_embedding_similarity:
            return {}
        if self.embedding_client is None:
            raise RuntimeError("Embedding retriever was requested without an embedding client.")

        skills = model.get("skills", [])
        if not skills:
            return {}

        skill_embeddings, skill_usage = self._cached_skill_embeddings(skills)
        task_result = self.embedding_client.embed([_task_state_text(task_state)])
        self._last_embedding_usage = {
            "skill_embedding_prompt_tokens": skill_usage.prompt_tokens,
            "skill_embedding_total_tokens": skill_usage.total_tokens,
            "task_embedding_prompt_tokens": task_result.prompt_tokens,
            "task_embedding_total_tokens": task_result.total_tokens,
            "embedding_prompt_tokens": skill_usage.prompt_tokens + task_result.prompt_tokens,
            "embedding_total_tokens": skill_usage.total_tokens + task_result.total_tokens,
        }
        similarities = _cosine_scores(skill_embeddings, task_result.embeddings[0])
        return {
            str(skill.get("skill_id", "")): float(score)
            for skill, score in zip(skills, similarities)
        }

    def _cached_skill_embeddings(
        self,
        skills: list[dict[str, Any]],
    ) -> tuple[np.ndarray, EmbeddingResult]:
        if self.embedding_client is None:
            raise RuntimeError("Embedding retriever was requested without an embedding client.")

        cache_key = tuple(
            (
                str(skill.get("skill_id", "")),
                _skill_text(skill),
            )
            for skill in skills
        )
        if self._skill_cache_key == cache_key and self._skill_cache_embeddings is not None:
            return self._skill_cache_embeddings, _zero_embedding_usage()

        result = self.embedding_client.embed([text for _, text in cache_key])
        self._skill_cache_key = cache_key
        self._skill_cache_embeddings = result.embeddings
        return self._skill_cache_embeddings, result


def score_skill(
    strategy: str,
    skill: dict[str, Any],
    task_state: dict[str, Any],
    *,
    embedding_similarity: float | None = None,
) -> float:
    # 根据策略名称把技能元数据或任务相似度映射为一个标量分数。
    metadata = skill.get("metadata", {})
    if strategy in {"frequency", "topk"}:
        return float(metadata.get("support", len(skill.get("members", []))) or 0.0)
    if strategy in {"success_rate", "historical_success_rate"}:
        return float(metadata.get("success_rate", 0.0) or 0.0)
    if strategy == "recent_reward_lift":
        return float(metadata.get("recent_reward_lift", 0.0) or 0.0)
    if strategy == "similarity":
        return _similarity_score(skill, task_state)
    if strategy == "embedding_similarity":
        return _required_embedding_similarity(embedding_similarity)
    if strategy == "pue_no_cost":
        return _pue_score(skill, task_state, use_cost=False)
    if strategy == "pue_no_recent":
        return _pue_score(skill, task_state, use_recent=False)
    if strategy == "pue_no_similarity":
        return _pue_score(skill, task_state, use_similarity=False)
    if strategy in {"pue", "pue_full"}:
        return _pue_score(skill, task_state)
    if strategy in {"embedding_pue", "embedding_pue_full"}:
        return _pue_score(
            skill,
            task_state,
            embedding_similarity=_required_embedding_similarity(embedding_similarity),
        )
    raise ValueError(f"Unsupported retrieval strategy: {strategy}")


def _pue_score(
    skill: dict[str, Any],
    task_state: dict[str, Any],
    *,
    use_cost: bool = True,
    use_recent: bool = True,
    use_similarity: bool = True,
    embedding_similarity: float | None = None,
) -> float:
    # PUE (Predictive Utility Estimation) 风格的启发式分数，可解释且易于消融。
    metadata = skill.get("metadata", {})
    support = float(metadata.get("support", 0.0) or 0.0)
    cost = float(metadata.get("token_cost", 0.0) or 0.0)
    cost_penalty = cost / (cost + 100.0) if use_cost else 0.0
    recent_success_rate = (
        float(metadata.get("recent_success_rate", metadata.get("success_rate", 0.0)) or 0.0)
        if use_recent
        else 0.0
    )
    recent_reward_lift = (
        float(metadata.get("recent_reward_lift", 0.0) or 0.0)
        if use_recent
        else 0.0
    )
    similarity = 0.0
    if use_similarity:
        similarity = (
            embedding_similarity
            if embedding_similarity is not None
            else _similarity_score(skill, task_state)
        )
    # 可解释 PUE proxy：近期反馈、历史收益、覆盖度、相似度加分，失败率和上下文成本扣分。
    score = (
        1.2 * recent_success_rate
        + 0.9 * float(metadata.get("avg_reward", 0.0) or 0.0)
        + 0.6 * float(metadata.get("coverage", 0.0) or 0.0)
        + 0.4 * recent_reward_lift
        + 0.3 * min(1.0, support / 20.0)
        + 0.5 * similarity
        + 0.7 * float(metadata.get("deployment_utility", 0.0) or 0.0)
        - 0.8 * float(metadata.get("failure_rate", 0.0) or 0.0)
        - 0.5 * cost_penalty
    )
    return score * float(metadata.get("deployment_weight", 1.0) or 1.0)


def _similarity_score(skill: dict[str, Any], task_state: dict[str, Any]) -> float:
    # 基于词袋余弦相似度，衡量技能内容与当前任务状态的相关性。
    return cosine_text_similarity(_skill_text(skill), _task_state_text(task_state))


def _skill_text(skill: dict[str, Any]) -> str:
    # 把技能的多字段文本拼成单一字符串用于相似度计算。
    return "\n".join(
        [
            str(skill.get("name", "")),
            str(skill.get("granularity", "")),
            str(skill.get("content", "")),
            str(skill.get("embedding_text", "")),
        ]
    )


def _task_state_text(task_state: dict[str, Any]) -> str:
    # 把目标、环境名和最近几个片段的 label/text 拼成查询文本。
    pieces = [
        str(task_state.get("goal", "")),
        str(task_state.get("env", "")),
    ]
    for segment in task_state.get("segments", [])[:5]:
        pieces.append(str(segment.get("label", "")))
        pieces.append(str(segment.get("text", "")))
    pieces.append(str(task_state.get("observation", "")))
    pieces.extend(str(action) for action in task_state.get("recent_actions", [])[-5:])
    return "\n".join(pieces)


def _build_embedding_client(config: dict[str, Any]) -> OpenAICompatibleEmbeddingClient:
    return OpenAICompatibleEmbeddingClient(
        timeout=int(config.get("embedding_timeout", 120)),
        max_retries=int(config.get("embedding_max_retries", 3)),
        retry_delay=float(config.get("embedding_retry_delay", 1.0)),
        batch_size=int(config.get("embedding_batch_size", 8)),
        batch_delay=float(config.get("embedding_batch_delay", 0.0)),
    )


def _cosine_scores(matrix: np.ndarray, query: np.ndarray) -> np.ndarray:
    matrix_norm = np.linalg.norm(matrix, axis=1)
    query_norm = float(np.linalg.norm(query))
    denominator = np.maximum(matrix_norm * query_norm, 1e-12)
    return (matrix @ query) / denominator


def _required_embedding_similarity(value: float | None) -> float:
    if value is None:
        raise RuntimeError("Embedding retrieval strategy requires real embedding similarity.")
    return value


def _zero_embedding_usage() -> EmbeddingResult:
    return EmbeddingResult(embeddings=np.zeros((0, 0), dtype=np.float32))


def _select_ranked_skills(
    scored: list[dict[str, Any]],
    *,
    k: int,
    level_top_k: dict[str, int],
) -> list[dict[str, Any]]:
    if not level_top_k:
        return scored[:k]

    selected = []
    selected_ids = set()
    for level, count in level_top_k.items():
        for skill in scored:
            if len([item for item in selected if _skill_level(item) == level]) >= count:
                break
            skill_id = str(skill.get("skill_id", ""))
            if skill_id in selected_ids or _skill_level(skill) != level:
                continue
            selected.append(skill)
            selected_ids.add(skill_id)
    return selected


def _returned_by_level(skills: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for skill in skills:
        level = _skill_level(skill)
        counts[level] = counts.get(level, 0) + 1
    return counts


def _skill_level(skill: dict[str, Any]) -> str:
    metadata = skill.get("metadata", {})
    return str(metadata.get("tower_level") or skill.get("granularity", ""))
