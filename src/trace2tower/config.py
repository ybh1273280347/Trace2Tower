from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from .env import load_repo_dotenv


@dataclass
class RunConfig:
    # 保留原始 dict，避免过早把实验配置建模死；新增字段先从这里加 property。
    raw: dict[str, Any]

    @property
    def env_name(self) -> str:
        return self.raw["env"]["name"]

    @property
    def env_mode(self) -> str:
        return self.raw["env"]["mode"]

    @property
    def episodes(self) -> int:
        return int(self.raw["runtime"]["episodes"])

    @property
    def max_steps(self) -> int:
        return int(self.raw["runtime"].get("max_steps", 50))

    @property
    def output_dir(self) -> Path:
        return Path(self.raw["runtime"]["output_dir"])

    @property
    def skill_model_path(self) -> Optional[Path]:
        path = self.raw["runtime"].get("skill_model_path")
        return Path(path) if path else None

    @property
    def agent_name(self) -> str:
        return self.raw["agent"]["name"]

    @property
    def agent_config(self) -> dict[str, Any]:
        return self.raw.get("agent", {})

    @property
    def segmenter_name(self) -> str:
        return self.raw["segmenter"]["name"]

    @property
    def segmenter_config(self) -> dict[str, Any]:
        return self.raw.get("segmenter", {})

    @property
    def miner_name(self) -> str:
        return self.raw["miner"]["name"]

    @property
    def miner_config(self) -> dict[str, Any]:
        return self.raw.get("miner", {})

    @property
    def retriever_name(self) -> str:
        return self.raw["retriever"]["name"]

    @property
    def retriever_config(self) -> dict[str, Any]:
        return self.raw.get("retriever", {})

    @property
    def retriever_top_k(self) -> int:
        return int(self.raw["retriever"].get("top_k", 3))

    @property
    def num_products(self) -> Optional[int]:
        return self.raw["env"].get("num_products")

    @property
    def alfworld_config_path(self) -> Path:
        return Path(self.raw["env"].get("alfworld_config_path", "configs/alfworld/base_config.yaml"))

    @property
    def alfworld_data_dir(self) -> Path:
        return Path(self.raw["env"].get("alfworld_data_dir", ".external/alfworld"))

    @property
    def webshop_root(self) -> Path:
        return Path(self.raw["env"].get("webshop_root", ".external/webshop"))


def load_config(path: Union[str, Path]) -> RunConfig:
    # 配置文件是实验复现实验的入口，所有路径和方法选择都应尽量写进 json。
    load_repo_dotenv()
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return RunConfig(raw=raw)
