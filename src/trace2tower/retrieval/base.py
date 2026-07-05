from __future__ import annotations

from abc import ABC, abstractmethod


class BaseRetriever(ABC):
    # 未来做技能注入、相似任务检索或 graph routing 时，都从这个边界进入。
    @abstractmethod
    def retrieve(self, model: dict, task_state: dict) -> list[dict]:
        # model 为技能模型（含 skills 列表），task_state 为当前任务上下文。
        raise NotImplementedError
