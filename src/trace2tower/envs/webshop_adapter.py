from __future__ import annotations

from typing import Optional

import sys
from pathlib import Path

from .base import BaseEnv


def build_webshop_env(
    mode: str = "text",
    num_products: int = 1000,
    webshop_root: Optional[str] = None,
) -> BaseEnv:
    # WebShop 官方仓库不是标准 pip 包；这里把项目内固定 repo 接入 Python path。
    root = Path(webshop_root or ".external/webshop")
    if not root.is_absolute():
        root = Path.cwd() / root
    if not root.exists():
        raise FileNotFoundError(f"WebShop repo not found: {root}")
    sys.path.insert(0, str(root))

    try:
        import gym
        import web_agent_site.envs  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "WebShop dependencies are not available. Use .envs/trace2tower-py38."
        ) from exc

    env = gym.make("WebAgentTextEnv-v0", observation_mode="text", num_products=num_products)
    return WebShopAdapter(env, mode=mode, num_products=num_products)


class WebShopAdapter(BaseEnv):
    # 把 WebShop Gym 环境包装成与 ALFWorld 一致的文本 reset/step 协议。
    def __init__(self, env: object, mode: str = "text", num_products: int = 1000) -> None:
        self._env = env
        self.mode = mode
        self.num_products = num_products
        self.name = "webshop"

    def reset(self) -> tuple[str, dict]:
        obs = self._env.reset()
        return str(obs), {"admissible_actions": self._available_actions()}

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        obs, reward, done, info = self._env.step(action)
        return str(obs), float(reward), bool(done), {
            "admissible_actions": self._available_actions(),
            "valid_action": True,
            "raw_info": info,
        }

    def _available_actions(self) -> list[str]:
        # WebShop 原生只给 clickables 和搜索框状态；这里统一转成 action 字符串。
        raw_actions = self._env.get_available_actions()
        actions: list[str] = []
        if raw_actions.get("has_search_bar"):
            # 有搜索框时提供一个基于任务指令的默认搜索动作。
            actions.append(f"search[{self._search_query()}]")
        actions.extend(f"click[{item}]" for item in raw_actions.get("clickables", []))
        return actions

    def _search_query(self) -> str:
        # smoke agent 没有语言理解能力，先用任务指令生成一个可执行搜索词。
        instruction = self._env.get_instruction_text()
        return (
            instruction.lower()
            .replace("[sep]", " ")
            .replace("find me", "")
            .strip()
        )
