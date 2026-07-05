from __future__ import annotations

import argparse

from trace2tower.config import load_config
from trace2tower.factories.pipeline import build_pipeline_bundle
from trace2tower.io import read_json, write_json, write_jsonl, write_jsonl_dicts
from trace2tower.runtime.execution import run_episodes


def build_parser() -> argparse.ArgumentParser:
    # CLI 只接收配置路径，实验差异都放进 json，方便复现和批量跑。
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser


def run_once(config_path: str) -> None:
    # 主流水线：构造组件 -> 采集轨迹 -> 切分 -> 离线挖掘 -> 检索演示 -> 汇总并落盘。
    config = load_config(config_path)
    bundle = build_pipeline_bundle(config)

    # 若指定了 skill_model_path，则使用离线预训练技能模型做部署阶段检索。
    deployment_model = _load_deployment_model(config.skill_model_path)
    episode_output = run_episodes(
        env=bundle.env,
        agent=bundle.agent,
        segmenter=bundle.segmenter,
        retriever=bundle.retriever,
        env_name=config.env_name,
        episodes=config.episodes,
        max_steps=config.max_steps,
        deployment_model=deployment_model,
    )
    trajectories = episode_output["trajectories"]
    raw_records = episode_output["records"]
    segment_records = episode_output["segments"]
    deployment_retrieval_records = episode_output["deployment_retrieval"]

    # 用收集到的片段离线挖掘技能模型；该模型可用于后续独立检索实验。
    model = bundle.miner.mine(segment_records)
    # 对每条轨迹，使用同一个挖掘后的模型做一次检索演示，结果写入 retrieval.jsonl。
    retrieval_records = [
        {
            "task_id": raw["task_id"],
            "retrieved_skills": bundle.retriever.retrieve(
                model,
                {
                    "goal": raw["goal"],
                    "env": raw["env"],
                    "segments": raw["segments"],
                },
            ),
        }
        for raw in raw_records
    ]
    summary = bundle.evaluator.summarize(raw_records)
    summary["components"] = {
        "agent": config.agent_name,
        "segmenter": config.segmenter_name,
        "miner": config.miner_name,
        "retriever": config.retriever_name,
    }
    summary["model_method"] = model.get("method")
    summary["skill_count"] = len(model.get("skills", []))
    if deployment_model:
        summary["deployment_model_method"] = deployment_model.get("method")
        summary["deployment_skill_count"] = len(deployment_model.get("skills", []))
    output_dir = config.output_dir
    # 输出分别服务不同分析粒度：完整轨迹、原始记录、片段、模型、检索记录和汇总指标。
    write_jsonl(output_dir / "trajectories.jsonl", trajectories)
    write_jsonl_dicts(output_dir / "records.jsonl", raw_records)
    write_jsonl_dicts(output_dir / "segments.jsonl", segment_records)
    write_json(output_dir / "model.json", model)
    write_jsonl_dicts(output_dir / "retrieval.jsonl", retrieval_records)
    write_jsonl_dicts(output_dir / "deployment_retrieval.jsonl", deployment_retrieval_records)
    write_json(output_dir / "summary.json", summary)


def _load_deployment_model(path):
    # 辅助函数：空路径表示不使用离线模型，避免在调用处写冗长条件。
    if not path:
        return None
    return read_json(path)


def main() -> None:
    args = build_parser().parse_args()
    run_once(args.config)


if __name__ == "__main__":
    main()
