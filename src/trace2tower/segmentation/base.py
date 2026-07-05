from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseSegmenter(ABC):
    # 输入是单条轨迹 dict，输出是一组片段 dict；不要在这里混入技能挖掘逻辑。
    @abstractmethod
    def segment(self, trajectory: dict[str, Any]) -> list[dict[str, Any]]:
        # trajectory 需包含 task_id、steps、success 等字段；返回的片段供后续技能挖掘使用。
        raise NotImplementedError
