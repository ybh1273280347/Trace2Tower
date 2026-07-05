from __future__ import annotations

import random
from typing import Any

from .base import BaseAgent


class RandomAgent(BaseAgent):
    # 仅用于 smoke 检查真实环境链路，不作为论文对比 baseline。
    def __init__(self, seed: int | None = None) -> None:
        self._random = random.Random(seed)

    def act(self, observation: str, info: dict[str, Any]) -> str:
        actions = info.get("admissible_actions") or ["noop"]
        return self._random.choice(actions)
