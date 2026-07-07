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

    # ---------- 环境相关配置 ----------
    @property
    def env_name(self) -> str:
        # 当前支持 alfworld / webshop。
        return self.raw["env"]["name"]

    @property
    def env_mode(self) -> str:
        return self.raw["env"]["mode"]

    # ---------- 运行相关配置 ----------
    @property
    def episodes(self) -> int:
        # 单次实验运行的 episode 数量。
        return int(self.raw["runtime"]["episodes"])

    @property
    def max_steps(self) -> int:
        # 每个 episode 最多允许执行多少步，防止无限循环。
        return int(self.raw["runtime"].get("max_steps", 50))

    @property
    def output_dir(self) -> Path:
        # 实验结果写入目录。
        return Path(self.raw["runtime"]["output_dir"])

    @property
    def skill_model_path(self) -> Optional[Path]:
        # 部署阶段使用的离线技能模型；为空表示在线边跑边挖掘。
        path = self.raw["runtime"].get("skill_model_path")
        return Path(path) if path else None

    # ---------- Agent 相关配置 ----------
    @property
    def agent_name(self) -> str:
        # 例如 llm_action。
        return self.raw["agent"]["name"]

    @property
    def agent_config(self) -> dict[str, Any]:
        return self.raw.get("agent", {})

    # ---------- 轨迹切分器配置 ----------
    @property
    def segmenter_name(self) -> str:
        return self.raw["segmenter"]["name"]

    @property
    def segmenter_config(self) -> dict[str, Any]:
        return self.raw.get("segmenter", {})

    # ---------- 技能挖掘器配置 ----------
    @property
    def miner_name(self) -> str:
        # 例如 trace2tower / skillx_official / skilllens_official / no_skill。
        return self.raw["miner"]["name"]

    @property
    def miner_config(self) -> dict[str, Any]:
        return self.raw.get("miner", {})

    # ---------- 技能检索器配置 ----------
    @property
    def retriever_name(self) -> str:
        # 例如 none / topk / frequency / pue 等。
        return self.raw["retriever"]["name"]

    @property
    def retriever_config(self) -> dict[str, Any]:
        return self.raw.get("retriever", {})

    @property
    def retriever_top_k(self) -> int:
        # 默认返回前 3 个技能。
        return int(self.raw["retriever"].get("top_k", 3))

    # ---------- 各环境专属路径配置 ----------
    @property
    def num_products(self) -> Optional[int]:
        # WebShop 加载的商品数量，影响环境规模和搜索空间。
        return self.raw["env"].get("num_products")

    @property
    def alfworld_config_path(self) -> Path:
        # ALFWorld 官方 yaml 配置路径。
        return Path(self.raw["env"].get("alfworld_config_path", "configs/alfworld/base_config.yaml"))

    @property
    def alfworld_data_dir(self) -> Path:
        # ALFWorld 预生成数据目录。
        return Path(self.raw["env"].get("alfworld_data_dir", ".external/alfworld"))

    @property
    def webshop_root(self) -> Path:
        # WebShop 官方仓库在本地的根目录。
        return Path(self.raw["env"].get("webshop_root", ".external/webshop"))


def load_config(path: Union[str, Path]) -> RunConfig:
    # 配置文件是实验复现实验的入口，所有路径和方法选择都应尽量写进 json。
    # 先加载 .env 保证 LLM_* 等环境变量可用。
    load_repo_dotenv()
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return RunConfig(raw=raw)
