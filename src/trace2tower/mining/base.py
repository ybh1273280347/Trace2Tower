from __future__ import annotations

from abc import ABC, abstractmethod


class BaseMiner(ABC):
    # 输入统一是切分后的轨迹片段，输出可以是技能图、层级技能或其他中间模型。
    @abstractmethod
    def mine(self, segments: list[dict]) -> dict:
        raise NotImplementedError
