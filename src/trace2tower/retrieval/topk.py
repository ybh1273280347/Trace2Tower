from __future__ import annotations

from .base import BaseRetriever


class TopKRetriever(BaseRetriever):
    def __init__(self, k: int = 3) -> None:
        self.k = k

    def retrieve(self, model: dict, task_state: dict) -> list[dict]:
        # 最小检索器：按支持度取前 k 个技能卡，后续再替换成相似度或图检索。
        # 若 metadata 里没有 support，则用成员数量兜底。
        skills = sorted(
            model.get("skills", []),
            key=lambda skill: skill.get("metadata", {}).get("support", len(skill.get("members", []))),
            reverse=True,
        )
        return skills[: self.k]
