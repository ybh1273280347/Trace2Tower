from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseAgent(ABC):
    # 所有 agent / baseline 都只需要实现“观测 + 环境信息 -> 动作”这个边界。
    @abstractmethod
    def act(self, observation: str, info: dict[str, Any]) -> str:
        # info 中通常包含 goal、admissible_actions、retrieved_skills 等上下文。
        raise NotImplementedError
