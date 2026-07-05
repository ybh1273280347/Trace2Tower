from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseEnv(ABC):
    # adapter 必须把不同 benchmark 统一成文本观测、奖励、终止标记和动作信息。
    @abstractmethod
    def reset(self) -> tuple[str, dict[str, Any]]:
        # 返回初始文本观测和 info（通常包含 admissible_actions）。
        raise NotImplementedError

    @abstractmethod
    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        # 返回 (观测, 奖励, 是否结束, info)。
        raise NotImplementedError
