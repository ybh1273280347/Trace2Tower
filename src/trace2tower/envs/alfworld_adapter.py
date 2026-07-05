from __future__ import annotations

from typing import Optional

import os
from pathlib import Path

import yaml

from .base import BaseEnv


def build_alfworld_env(
    mode: str = "text",
    config_path: Optional[str] = None,
    data_dir: Optional[str] = None,
) -> BaseEnv:
    # ALFWorld 官方 load_config 读取命令行参数；实验框架里改成显式读取配置文件。
    try:
        _patch_importlib_resources_for_textworld()
        from alfworld.agents.environment import get_environment
    except ImportError as exc:
        raise RuntimeError(
            "ALFWorld not installed. Use .envs/trace2tower-py38 and install alfworld first."
        ) from exc

    config = _load_alfworld_config(config_path, data_dir)
    env_type = config["env"]["type"]
    env = get_environment(env_type)(config, train_eval="train")
    return ALFWorldAdapter(env.init_env(batch_size=1), mode=mode)


def _load_alfworld_config(config_path: Optional[str], data_dir: Optional[str]) -> dict:
    # 解析 yaml 配置并设置 ALFWORLD_DATA 环境变量；支持相对路径和 env var 展开。
    project_root = Path.cwd()
    config_file = Path(config_path or "configs/alfworld/base_config.yaml")
    if not config_file.is_absolute():
        config_file = project_root / config_file
    if not config_file.exists():
        raise FileNotFoundError(f"ALFWorld config not found: {config_file}")

    alfworld_data = Path(data_dir or ".external/alfworld")
    if not alfworld_data.is_absolute():
        alfworld_data = project_root / alfworld_data
    if not alfworld_data.exists():
        raise FileNotFoundError(f"ALFWorld data dir not found: {alfworld_data}")

    os.environ["ALFWORLD_DATA"] = str(alfworld_data)
    with config_file.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    return _expand_env_vars(config)


def _expand_env_vars(value):
    # 递归展开配置中的 $VAR 形式环境变量。
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, str):
        return os.path.expandvars(value)
    return value


def _patch_importlib_resources_for_textworld() -> None:
    # TextWorld 1.6 在 Python 3.8 上调用 importlib.resources.files；
    # 该 API 在标准库里较晚才完整可用，这里用 backport 补齐。
    import importlib.resources

    if hasattr(importlib.resources, "files"):
        return
    import importlib_resources

    importlib.resources.files = importlib_resources.files


class ALFWorldAdapter(BaseEnv):
    # 把 ALFWorld/TextWorld 的 batch API 压成单 episode 的 reset/step 协议。
    def __init__(self, env: object, mode: str = "text") -> None:
        self._env = env
        self.mode = mode
        self.name = "alfworld"

    def reset(self) -> tuple[str, dict]:
        # ALFWorld 返回 batch 结构；当前平台先固定 batch_size=1。
        obs, info = self._env.reset()
        return str(obs[0]), {"admissible_actions": list(info["admissible_commands"][0])}

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        # 环境原生动作也走 batch，所以这里把单个 action 包成列表再拆回单条结果。
        obs, scores, dones, infos = self._env.step([action])
        observation = str(obs[0])
        reward = float(scores[0])
        done = bool(dones[0])
        info = infos[0]
        return observation, reward, done, {
            "admissible_actions": list(info.get("admissible_commands", [])),
            "valid_action": True,
        }
