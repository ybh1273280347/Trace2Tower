from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .agent_factory import build_agent
from .env_factory import build_env
from .evaluation import Evaluator
from .mining import (
    BaseMiner,
    FlatSkillSummaryMiner,
    NoSkillMiner,
    RawTrajectoryMiner,
    SkillLensOfficialMiner,
    SkillXOfficialMiner,
)
from .retrieval import BaseRetriever, NoSkillRetriever, ScoreBasedRetriever, TopKRetriever
from .segmentation import BaseSegmenter, RuleSegmenter


@dataclass
class PipelineBundle:
    # 把一次实验运行需要的组件集中保存，避免 run.py 里散落构造细节。
    env: object
    agent: object
    segmenter: BaseSegmenter
    miner: BaseMiner
    retriever: BaseRetriever
    evaluator: Evaluator


def build_pipeline_bundle(config: object) -> PipelineBundle:
    # 统一把环境、agent、切分器、矿工、检索器和评测器拼到一起。
    env = build_env(
        config.env_name,
        config.env_mode,
        config.num_products,
        alfworld_config_path=str(config.alfworld_config_path),
        alfworld_data_dir=str(config.alfworld_data_dir),
        webshop_root=str(config.webshop_root),
    )
    agent = build_agent(config.agent_name, config.agent_config)
    segmenter = build_segmenter(config.segmenter_name, config.segmenter_config)
    miner_config = dict(config.miner_config)
    miner_config.setdefault("runtime_output_dir", str(config.output_dir))
    miner = build_miner(config.miner_name, miner_config)
    retriever = build_retriever(config.retriever_name, config.retriever_config)
    evaluator = Evaluator(metrics=config.raw["evaluator"]["metrics"])
    return PipelineBundle(
        env=env,
        agent=agent,
        segmenter=segmenter,
        miner=miner,
        retriever=retriever,
        evaluator=evaluator,
    )


def build_segmenter(name: str, config: Optional[dict] = None) -> BaseSegmenter:
    if name == "rule":
        return RuleSegmenter()
    raise ValueError(f"Unsupported segmenter: {name}")


def build_miner(name: str, config: Optional[dict] = None) -> BaseMiner:
    settings = config or {}
    miners: dict[str, BaseMiner] = {
        "no_skill": NoSkillMiner(),
        "raw_trajectory": RawTrajectoryMiner(),
        "flat_skill_summary": FlatSkillSummaryMiner(),
        "skillx_official": SkillXOfficialMiner(settings),
        "skilllens_official": SkillLensOfficialMiner(settings),
    }
    try:
        return miners[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported miner: {name}") from exc


def build_retriever(name: str, config: Optional[dict] = None) -> BaseRetriever:
    settings = config or {}
    top_k = int(settings.get("top_k", 3))
    if name == "none":
        return NoSkillRetriever()
    if name == "topk":
        return TopKRetriever(k=top_k)
    if name in {
        "frequency",
        "success_rate",
        "historical_success_rate",
        "similarity",
        "recent_reward_lift",
        "pue",
        "pue_full",
        "pue_no_cost",
        "pue_no_recent",
        "pue_no_similarity",
    }:
        return ScoreBasedRetriever(strategy=name, k=top_k)
    raise ValueError(f"Unsupported retriever: {name}")
