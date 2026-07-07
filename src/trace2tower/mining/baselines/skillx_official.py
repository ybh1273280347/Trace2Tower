from __future__ import annotations

from pathlib import Path
from typing import Any

import requests

from trace2tower.env import require_env
from trace2tower.text import action_template

from .base import (
    OfficialBaselineError,
    OfficialBaselineMiner,
    build_skill,
    final_reward,
    group_by_trajectory,
    read_json,
    segment_action,
    segment_node,
    segments_for_sources,
    write_json,
    write_jsonl,
)


class SkillXOfficialMiner(OfficialBaselineMiner):
    # 调用 SkillX 官方仓库进行技能抽取的适配器。
    method = "skillx_official"
    repo_name = "SkillX"

    def mine(self, segments: list[dict]) -> dict:
        self.ensure_repo()
        if not segments:
            return self.model_shell(
                description=(
                    "Official SkillX extraction adapter received no segments, "
                    "so no upstream extraction was launched."
                ),
                skills=[],
                nodes=[],
                extra_metadata={"skipped_reason": "no_segments"},
            )

        self.work_dir.mkdir(parents=True, exist_ok=True)
        input_path = self.work_dir / "skillx_input.jsonl"
        output_dir = self.work_dir / "official_output"
        params_path = self.work_dir / "skillx_params.json"
        runner_path = self.work_dir / "run_skillx_official.py"

        # 把 Trace2Tower 片段格式转成 SkillX 可接受的 trajectory 格式。
        write_jsonl(input_path, _skillx_records(segments, self.config))
        # 动态生成一个调用 SkillX pipeline 的入口脚本，避免污染外部仓库。
        _write_skillx_runner(runner_path, package_name=self.repo_root.name)
        write_json(
            params_path,
            {
                "input": str(input_path),
                "output": str(output_dir),
                "model": self.llm_model,
                "benchmark": str(self.config.get("benchmark", "appworld")),
                "skill_type": str(self.config.get("skill_type", "hybrid")),
                "domain": str(self.config.get("domain", "")),
                "plan_strategy": str(self.config.get("plan_strategy", "shortest")),
                "epochs": int(self.config.get("epochs", 1)),
                "threshold": float(self.config.get("threshold", 0.0)),
                "batch_size": int(self.config.get("batch_size", 4)),
                "max_concurrent": int(self.config.get("max_concurrent", 2)),
                "filter_timing": str(self.config.get("filter_timing", "none")),
                "embedding_base_url": _embedding_base_url(),
                "embedding_model": _embedding_model(),
                "embedding_api_key": _embedding_api_key(),
                "embedding_timeout": int(self.config.get("embedding_timeout", 120)),
            },
        )
        # 预先检查 embedding 端点可用，避免官方 pipeline 跑了一半才失败。
        _validate_embedding_endpoint(self.config)

        self.run_command(
            [self.python, str(runner_path), str(params_path)],
            cwd=self.repo_root.parent,
            env=self.env_with_pythonpath(self.repo_root.parent),
            log_prefix="skillx_pipeline",
        )
        library_path = output_dir / "extraction_skill_library.json"
        if not library_path.exists():
            raise OfficialBaselineError(
                f"SkillX completed but did not write {library_path}."
            )
        library = read_json(library_path)
        skills = _skills_from_skillx(library, segments, library_path)
        return self.model_shell(
            description=(
                "Official SkillX extraction run through the upstream pipeline "
                "on Trace2Tower-converted trajectories."
            ),
            skills=skills,
            nodes=[segment_node(segment) for segment in segments],
            extra_metadata={
                "official_input_path": str(input_path),
                "official_output_path": str(library_path),
                "official_benchmark": self.config.get("benchmark", "appworld"),
                "official_skill_type": self.config.get("skill_type", "hybrid"),
                "official_embedding_base_url": _embedding_base_url(),
                "official_embedding_model": _embedding_model(),
            },
        )


def _embedding_base_url() -> str:
    # SkillX 需要的 embedding base_url 不带 /v1 后缀；这里做兼容裁剪。
    value = require_env("LLM_BASE_URL")
    stripped = value.rstrip("/")
    return stripped[:-3] if stripped.endswith("/v1") else stripped


def _embedding_api_key() -> str:
    return require_env("LLM_API_KEY")


def _embedding_model() -> str:
    return require_env("LLM_EMBEDDING_MODEL")


def _validate_embedding_endpoint(config: dict[str, Any]) -> None:
    # 对 embedding 端点做一次 preflight，确保模型可用且网络可达。
    model = _embedding_model()
    url = f"{_embedding_base_url()}/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {_embedding_api_key()}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(
            url,
            headers=headers,
            json={"model": model, "input": ["trace2tower embedding preflight"]},
            timeout=int(config.get("embedding_timeout", 120)),
        )
    except requests.RequestException as exc:
        raise OfficialBaselineError(
            f"SkillX embedding preflight failed for LLM_EMBEDDING_MODEL={model} at {url}: {exc}"
        ) from exc

    if response.status_code >= 400:
        body = response.text[:500].replace("\n", " ")
        raise OfficialBaselineError(
            f"SkillX embedding preflight failed for LLM_EMBEDDING_MODEL={model} "
            f"at {url}: HTTP {response.status_code} {body}"
        )


def _write_skillx_runner(path: Path, *, package_name: str) -> None:
    # 动态生成调用 SkillX IterativeSkillPipeline 的 asyncio 入口脚本。
    source = f"""from __future__ import annotations

import asyncio
import json
import sys

from {package_name}.clustering.embedding import EmbeddingService
from {package_name}.data.loaders import TrajectoryLoader
from {package_name}.llm.client import LLM
from {package_name}.pipeline import IterativeSkillPipeline


async def main(params_path: str) -> None:
    with open(params_path, encoding="utf-8") as handle:
        params = json.load(handle)

    trajectories = TrajectoryLoader.load(params["input"])
    llm = LLM(model=params["model"])
    pipeline = IterativeSkillPipeline(
        llm=llm,
        benchmark=params["benchmark"],
        skill_type=params["skill_type"],
        domain=params["domain"],
        plan_strategy=params["plan_strategy"],
        output_dir=params["output"],
        verbose=True,
    )

    if params["embedding_base_url"]:
        pipeline.clusterer.embedding_service = EmbeddingService(
            model=params["embedding_model"],
            base_url=params["embedding_base_url"],
            api_key=params["embedding_api_key"],
            timeout=params["embedding_timeout"],
        )

    results = await pipeline.run(
        trajectories,
        num_epochs=params["epochs"],
        filter_threshold=params["threshold"],
        batch_size=params["batch_size"],
        max_concurrent=params["max_concurrent"],
        filter_timing=params["filter_timing"],
    )
    pipeline.save_results(results)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _skillx_records(segments: list[dict], config: dict[str, Any]) -> list[dict[str, Any]]:
    # 将 Trace2Tower 的 segments 重构为 SkillX 期望的 user/assistant/tool 对话历史。
    records = []
    for trajectory_id, items in sorted(group_by_trajectory(segments).items()):
        metadata = items[0].get("metadata", {}) if items else {}
        goal = str(metadata.get("goal", ""))
        history = [
            {
                "role": "user",
                "content": goal,
            }
        ]
        for segment in items:
            template = action_template(segment_action(segment))
            history.append(
                {
                    "role": "assistant",
                    "content": segment_action(segment),
                    "tool_calls": [
                        {
                            "name": f"trace2tower.{template}",
                            "arguments": {
                                "raw_action": segment_action(segment),
                                "label": segment.get("label", "Unknown"),
                            },
                        }
                    ],
                }
            )
            history.append(
                {
                    "role": "tool",
                    "content": segment.get("text", ""),
                    "name": f"trace2tower.{template}",
                }
            )
        records.append(
            {
                "trajectory_id": trajectory_id,
                "task_id": trajectory_id,
                "user_task": goal,
                "task_history": history,
                "reward": final_reward(items[-1]) if items else 0.0,
                "metadata": {
                    "trace2tower_env": metadata.get("env", ""),
                    "segment_ids": [segment.get("segment_id") for segment in items],
                    "adapter": "skillx_official",
                    "official_benchmark": config.get("benchmark", "appworld"),
                },
            }
        )
    return records


def _skills_from_skillx(
    library: dict[str, Any],
    segments: list[dict],
    output_path: Path,
) -> list[dict[str, Any]]:
    # 解析 SkillX 输出的技能库，把 planning / functional / atomic 三类技能映射到统一格式。
    skills = []
    skill_groups = library.get("skills", {})
    for index, (task, plan) in enumerate(sorted(skill_groups.get("planning", {}).items())):
        content = "\n".join(
            [
                f"Task: {task}",
                str(plan.get("plan", "")),
            ]
        )
        members = segments_for_sources([task], segments)
        if not members:
            raise OfficialBaselineError(
                "SkillX planning skill cannot be mapped back to Trace2Tower segments: "
                f"task={task}, output={output_path}"
            )
        skills.append(
            build_skill(
                skill_id=f"skillx_official_planning_{index:03d}",
                name=f"Planning {task}",
                granularity="planning",
                segments=members,
                all_segment_count=len(segments),
                source_method="skillx_official",
                content=content,
                extra_metadata={
                    "official_output_path": str(output_path),
                    "skillx_source_type": "planning",
                },
            )
        )
    for granularity in ("functional", "atomic"):
        for index, item in enumerate(skill_groups.get(granularity, [])):
            metadata = item.get("metadata", {})
            members = segments_for_sources(metadata.get("source_tasks", []), segments)
            name = str(item.get("name") or f"{granularity.title()} Skill {index}")
            if not members:
                raise OfficialBaselineError(
                    "SkillX skill cannot be mapped back to Trace2Tower segments: "
                    f"name={name}, granularity={granularity}, "
                    f"source_tasks={metadata.get('source_tasks', [])}, output={output_path}"
                )
            content = "\n".join(
                [
                    str(item.get("document", "")),
                    str(item.get("content", "")),
                    "Tools: " + ", ".join(str(tool) for tool in item.get("tools", [])),
                ]
            ).strip()
            skills.append(
                build_skill(
                    skill_id=f"skillx_official_{granularity}_{index:03d}",
                    name=name,
                    granularity=granularity,
                    segments=members,
                    all_segment_count=len(segments),
                    source_method="skillx_official",
                    content=content,
                    extra_metadata={
                        "official_output_path": str(output_path),
                        "skillx_metadata": metadata,
                        "skillx_tools": item.get("tools", []),
                    },
                )
            )
    return skills
