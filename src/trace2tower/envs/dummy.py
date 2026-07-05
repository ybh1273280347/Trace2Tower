from __future__ import annotations

from typing import Any, Optional

from .base import BaseEnv


class DummyEnv(BaseEnv):
    # 仅保留给极小单元测试；正式实验入口不允许回落到这个环境。
    def __init__(self, name: str, mode: str, num_products: Optional[int] = None) -> None:
        self.name = name
        self.mode = mode
        self.num_products = num_products
        self._step = 0

    def reset(self) -> tuple[str, dict[str, Any]]:
        self._step = 0
        return f"{self.name}:{self.mode}:reset", {"admissible_actions": ["noop"]}

    def step(self, action: str) -> tuple[str, float, bool, dict[str, Any]]:
        self._step += 1
        done = self._step >= 3
        reward = 1.0 if done else 0.0
        return (
            f"{self.name}:{self.mode}:obs:{self._step}:{action}",
            reward,
            done,
            {"valid_action": True, "admissible_actions": ["noop"]},
        )
