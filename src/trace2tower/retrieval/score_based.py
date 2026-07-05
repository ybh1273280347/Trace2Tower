from __future__ import annotations

from typing import Any

from trace2tower.text import cosine_text_similarity

from .base import BaseRetriever


class NoSkillRetriever(BaseRetriever):
    def retrieve(self, model: dict, task_state: dict) -> list[dict]:
        return []


class ScoreBasedRetriever(BaseRetriever):
    def __init__(self, strategy: str, k: int = 3) -> None:
        self.strategy = strategy
        self.k = k

    def retrieve(self, model: dict, task_state: dict) -> list[dict]:
        scored = []
        for skill in model.get("skills", []):
            item = dict(skill)
            metadata = dict(item.get("metadata", {}))
            # 在线检索只负责给当前 task_state 排序；离线指标在 scripts/evaluate_selectors.py 里批量算。
            score = score_skill(self.strategy, item, task_state)
            metadata["retrieval_score"] = score
            metadata["retrieval_strategy"] = self.strategy
            item["metadata"] = metadata
            scored.append(item)

        scored.sort(
            key=lambda skill: skill.get("metadata", {}).get("retrieval_score", 0.0),
            reverse=True,
        )
        return scored[: self.k]


def score_skill(strategy: str, skill: dict[str, Any], task_state: dict[str, Any]) -> float:
    metadata = skill.get("metadata", {})
    if strategy in {"frequency", "topk"}:
        return float(metadata.get("support", len(skill.get("members", []))) or 0.0)
    if strategy in {"success_rate", "historical_success_rate"}:
        return float(metadata.get("success_rate", 0.0) or 0.0)
    if strategy == "recent_reward_lift":
        return float(metadata.get("recent_reward_lift", 0.0) or 0.0)
    if strategy == "similarity":
        return _similarity_score(skill, task_state)
    if strategy == "pue_no_cost":
        return _pue_score(skill, task_state, use_cost=False)
    if strategy == "pue_no_recent":
        return _pue_score(skill, task_state, use_recent=False)
    if strategy == "pue_no_similarity":
        return _pue_score(skill, task_state, use_similarity=False)
    if strategy in {"pue", "pue_full"}:
        return _pue_score(skill, task_state)
    raise ValueError(f"Unsupported retrieval strategy: {strategy}")


def _pue_score(
    skill: dict[str, Any],
    task_state: dict[str, Any],
    *,
    use_cost: bool = True,
    use_recent: bool = True,
    use_similarity: bool = True,
) -> float:
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
    similarity = _similarity_score(skill, task_state) if use_similarity else 0.0
    # 可解释 PUE proxy：近期反馈、历史收益、覆盖度、相似度加分，失败率和上下文成本扣分。
    return (
        1.2 * recent_success_rate
        + 0.9 * float(metadata.get("avg_reward", 0.0) or 0.0)
        + 0.6 * float(metadata.get("coverage", 0.0) or 0.0)
        + 0.4 * recent_reward_lift
        + 0.3 * min(1.0, support / 20.0)
        + 0.5 * similarity
        - 0.8 * float(metadata.get("failure_rate", 0.0) or 0.0)
        - 0.5 * cost_penalty
    )


def _similarity_score(skill: dict[str, Any], task_state: dict[str, Any]) -> float:
    return cosine_text_similarity(_skill_text(skill), _task_state_text(task_state))


def _skill_text(skill: dict[str, Any]) -> str:
    return "\n".join(
        [
            str(skill.get("name", "")),
            str(skill.get("granularity", "")),
            str(skill.get("content", "")),
            str(skill.get("embedding_text", "")),
        ]
    )


def _task_state_text(task_state: dict[str, Any]) -> str:
    pieces = [
        str(task_state.get("goal", "")),
        str(task_state.get("env", "")),
    ]
    for segment in task_state.get("segments", [])[:5]:
        pieces.append(str(segment.get("label", "")))
        pieces.append(str(segment.get("text", "")))
    return "\n".join(pieces)
