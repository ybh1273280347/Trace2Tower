from __future__ import annotations

import json
import os
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Optional

from trace2tower.text import action_template, compact_counter, tokenize
from trace2tower.env import load_repo_dotenv, require_env

from .base import BaseMiner


class OfficialBaselineError(RuntimeError):
    pass


class OfficialBaselineMiner(BaseMiner):
    method = "official_baseline"
    repo_name = ""

    def __init__(self, config: Optional[dict[str, Any]] = None) -> None:
        load_repo_dotenv()
        self.config = config or {}
        self.repo_root = _resolve_path(
            self.config.get("external_root"),
            default=_repo_root() / ".external" / "baselines" / self.repo_name,
        )
        self.python = _resolve_executable(self.config.get("python", "python"))
        self.llm_model = _llm_model()
        self.timeout_sec = int(self.config.get("timeout_sec", 3600))
        self.runtime_output_dir = _resolve_path(
            self.config.get("runtime_output_dir"),
            default=_repo_root() / "experiments" / "official_baselines",
        )
        self.work_dir = _resolve_path(
            self.config.get("work_dir"),
            default=self.runtime_output_dir / self.method,
        )

    def _ensure_repo(self) -> None:
        if not self.repo_root.exists():
            raise OfficialBaselineError(
                f"{self.method} requires the upstream repository at {self.repo_root}."
            )

    def _run(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: Optional[dict[str, str]] = None,
        log_prefix: str,
    ) -> subprocess.CompletedProcess[str]:
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=merged_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout_sec,
        )
        self.work_dir.mkdir(parents=True, exist_ok=True)
        (self.work_dir / f"{log_prefix}.stdout.log").write_text(
            completed.stdout,
            encoding="utf-8",
        )
        (self.work_dir / f"{log_prefix}.stderr.log").write_text(
            completed.stderr,
            encoding="utf-8",
        )
        if completed.returncode != 0:
            stderr_tail = completed.stderr[-2000:] if completed.stderr else ""
            stdout_tail = completed.stdout[-1000:] if completed.stdout else ""
            raise OfficialBaselineError(
                f"{self.method} official command failed with exit code "
                f"{completed.returncode}. See {self.work_dir}. "
                f"stderr tail:\n{stderr_tail}\nstdout tail:\n{stdout_tail}"
            )
        return completed

    def _env_with_pythonpath(self, *paths: Path) -> dict[str, str]:
        values = [str(path) for path in paths]
        existing = os.environ.get("PYTHONPATH")
        if existing:
            values.append(existing)
        env = {"PYTHONPATH": os.pathsep.join(values)}
        env["OPENAI_API_KEY"] = require_env("LLM_API_KEY")
        env["OPENAI_BASE_URL"] = require_env("LLM_BASE_URL")
        return env

    def _env_with_pythonpath_and_llm(self, *paths: Path) -> dict[str, str]:
        env = self._env_with_pythonpath(*paths)
        return env

    def _model_shell(
        self,
        *,
        description: str,
        skills: list[dict[str, Any]],
        nodes: list[dict[str, Any]],
        edges: Optional[list[dict[str, Any]]] = None,
        extra_metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        metadata = {
            "uses_official_runtime": True,
            "official_code_reference": str(self.repo_root),
            "official_work_dir": str(self.work_dir),
            "official_llm_model": self.llm_model,
        }
        metadata.update(extra_metadata or {})
        return {
            "method": self.method,
            "description": description,
            "nodes": nodes,
            "edges": edges or [],
            "skills": skills,
            "metadata": metadata,
        }


class SkillLensOfficialMiner(OfficialBaselineMiner):
    method = "skilllens_official"
    repo_name = "SkillLens"

    def mine(self, segments: list[dict]) -> dict:
        self._ensure_repo()
        if not segments:
            return self._model_shell(
                description=(
                    "Official SkillLens extraction adapter received no "
                    "segments, so no upstream extraction was launched."
                ),
                skills=[],
                nodes=[],
                extra_metadata={"skipped_reason": "no_segments"},
            )
        self.work_dir.mkdir(parents=True, exist_ok=True)
        input_path = self.work_dir / "skilllens_input.json"
        output_dir = self.work_dir / "official_output"
        _write_json(input_path, _skilllens_input(segments, self.config))

        command = [
            self.python,
            "-m",
            "skilllens",
            "extract",
            "-i",
            str(input_path),
            "-o",
            str(output_dir),
        ]
        for key, value in self._overrides().items():
            command.extend(["--set", f"{key}={value}"])
        if self.config.get("verbose", False):
            command.append("--verbose")

        self._run(
            command,
            cwd=self.repo_root,
            env=self._env_with_pythonpath(self.repo_root),
            log_prefix="skilllens_extract",
        )
        skill_set_path = _latest_file(output_dir, "skill_set.json")
        skill_set = _read_json(skill_set_path)
        skills = [
            _skill_from_skilllens(
                index=index,
                item=item,
                segments=segments,
                output_path=skill_set_path,
            )
            for index, item in enumerate(skill_set.get("skills", []))
        ]
        return self._model_shell(
            description=(
                "Official SkillLens extraction run through the upstream "
                "`skilllens extract` CLI on Trace2Tower-converted trajectories."
            ),
            skills=skills,
            nodes=[_segment_node(segment) for segment in segments],
            extra_metadata={
                "official_input_path": str(input_path),
                "official_output_path": str(skill_set_path),
                "official_extractor_model": skill_set.get("extractor_model"),
                "official_extraction_method": skill_set.get("extraction_method"),
            },
        )

    def _overrides(self) -> dict[str, Any]:
        overrides = {
            "input.benchmark": self.config.get("benchmark", "trace2tower"),
            "output.run_name": self.config.get("run_name", "trace2tower"),
            "llm.provider": self.config.get("provider", "openai"),
            "llm.model": self.llm_model,
            "llm.api_style": self.config.get("api_style", "chat"),
            "extraction.method": self.config.get("method", "parallel"),
            "extraction.max_skills": int(self.config.get("max_skills", 8)),
            "extraction.batch_size": int(self.config.get("batch_size", 0)),
            "extraction.merge_group_size": int(self.config.get("merge_group_size", 4)),
            "extraction.max_concurrency": int(self.config.get("max_concurrency", 4)),
        }
        optional_keys = {
            "temperature": "llm.temperature",
            "max_tokens": "llm.max_tokens",
            "max_tool_rounds": "extraction.max_tool_rounds",
            "meta_skill_guidance": "extraction.meta_skill_guidance",
        }
        for config_key, override_key in optional_keys.items():
            if config_key in self.config and self.config[config_key] is not None:
                overrides[override_key] = self.config[config_key]
        return overrides


class SkillXOfficialMiner(OfficialBaselineMiner):
    method = "skillx_official"
    repo_name = "SkillX"

    def mine(self, segments: list[dict]) -> dict:
        self._ensure_repo()
        if not segments:
            return self._model_shell(
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
        _write_jsonl(input_path, _skillx_records(segments, self.config))
        _write_skillx_runner(runner_path, package_name=self.repo_root.name)
        _write_json(
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
                "embedding_base_url": _embedding_base_url(self.config),
                "embedding_model": _embedding_model(self.config),
                "embedding_api_key": _embedding_api_key(self.config),
                "embedding_timeout": int(self.config.get("embedding_timeout", 120)),
            },
        )

        self._run(
            [self.python, str(runner_path), str(params_path)],
            cwd=self.repo_root.parent,
            env=self._env_with_pythonpath_and_llm(self.repo_root.parent),
            log_prefix="skillx_pipeline",
        )
        library_path = output_dir / "extraction_skill_library.json"
        if not library_path.exists():
            raise OfficialBaselineError(
                f"SkillX completed but did not write {library_path}."
            )
        library = _read_json(library_path)
        skills = _skills_from_skillx(library, segments, library_path)
        return self._model_shell(
            description=(
                "Official SkillX extraction run through the upstream pipeline "
                "on Trace2Tower-converted trajectories."
            ),
            skills=skills,
            nodes=[_segment_node(segment) for segment in segments],
            extra_metadata={
                "official_input_path": str(input_path),
                "official_output_path": str(library_path),
                "official_benchmark": self.config.get("benchmark", "appworld"),
                "official_skill_type": self.config.get("skill_type", "hybrid"),
                "official_embedding_base_url": _embedding_base_url(self.config),
                "official_embedding_model": _embedding_model(self.config),
            },
        )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_path(value: Any, *, default: Path) -> Path:
    if value in (None, ""):
        return default
    path = Path(str(value))
    return path if path.is_absolute() else _repo_root() / path


def _resolve_executable(value: Any) -> str:
    executable = str(value or "python")
    path = Path(executable)
    if path.is_absolute():
        return str(path)
    if "/" in executable or "\\" in executable:
        return str(_repo_root() / path)
    return executable


def _llm_model() -> str:
    return require_env("LLM_MODEL")


def _embedding_base_url(config: dict[str, Any]) -> str:
    value = require_env("LLM_BASE_URL")
    stripped = value.rstrip("/")
    return stripped[:-3] if stripped.endswith("/v1") else stripped


def _embedding_api_key(config: dict[str, Any]) -> str:
    return require_env("LLM_API_KEY")


def _embedding_model(config: dict[str, Any]) -> str:
    return require_env("LLM_EMBEDDING_MODEL")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_skillx_runner(path: Path, *, package_name: str) -> None:
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


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _latest_file(root: Path, name: str) -> Path:
    candidates = sorted(root.rglob(name), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise OfficialBaselineError(f"Could not find {name} under {root}.")
    return candidates[-1]


def _skilllens_input(segments: list[dict], config: dict[str, Any]) -> dict[str, Any]:
    trajectories = []
    for trajectory_id, items in sorted(_group_by_trajectory(segments).items()):
        metadata = items[0].get("metadata", {}) if items else {}
        goal = str(metadata.get("goal", ""))
        success = _segment_success(items[-1]) if items else False
        steps = [
            {
                "role": "user",
                "content": goal,
                "metadata": {
                    "trace2tower_event": "goal",
                    "trajectory_id": trajectory_id,
                },
            }
        ]
        for segment in items:
            steps.append(
                {
                    "role": "agent",
                    "content": f"Action: {_segment_action(segment)}",
                    "observation": segment.get("text", ""),
                    "metadata": {
                        "segment_id": segment.get("segment_id"),
                        "label": segment.get("label", "Unknown"),
                        "step_index": segment.get("step_index"),
                        "reward": _step_reward(segment),
                    },
                }
            )
        trajectories.append(
            {
                "id": trajectory_id,
                "task_id": trajectory_id,
                "task_name": goal,
                "agent": config.get("agent", "trace2tower"),
                "steps": steps,
                "reward": _final_reward(items[-1]) if items else 0.0,
                "benchmark": metadata.get("env", config.get("benchmark", "trace2tower")),
                "outcome": "resolved" if success else "unresolved",
                "source_format": "trace2tower_segments",
                "metadata": {
                    "trace2tower_env": metadata.get("env", ""),
                    "trace2tower_goal": goal,
                    "segment_ids": [segment.get("segment_id") for segment in items],
                },
            }
        )
    return {
        "trajectories": trajectories,
        "source": "trace2tower",
        "metadata": {
            "segment_count": len(segments),
            "adapter": "skilllens_official",
        },
    }


def _skillx_records(segments: list[dict], config: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for trajectory_id, items in sorted(_group_by_trajectory(segments).items()):
        metadata = items[0].get("metadata", {}) if items else {}
        goal = str(metadata.get("goal", ""))
        history = [
            {
                "role": "user",
                "content": goal,
            }
        ]
        for segment in items:
            template = action_template(_segment_action(segment))
            history.append(
                {
                    "role": "assistant",
                    "content": _segment_action(segment),
                    "tool_calls": [
                        {
                            "name": f"trace2tower.{template}",
                            "arguments": {
                                "raw_action": _segment_action(segment),
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
                "reward": _final_reward(items[-1]) if items else 0.0,
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
    skills = []
    skill_groups = library.get("skills", {})
    for index, (task, plan) in enumerate(sorted(skill_groups.get("planning", {}).items())):
        content = "\n".join(
            [
                f"Task: {task}",
                str(plan.get("plan", "")),
            ]
        )
        members = _segments_for_sources([task], segments)
        skills.append(
            _build_skill(
                skill_id=f"skillx_official_planning_{index:03d}",
                name=f"Planning {task}",
                granularity="planning",
                segments=members or segments,
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
            sources = metadata.get("source_tasks", [])
            members = _segments_for_sources(sources, segments)
            name = str(item.get("name") or f"{granularity.title()} Skill {index}")
            content = "\n".join(
                [
                    str(item.get("document", "")),
                    str(item.get("content", "")),
                    "Tools: " + ", ".join(str(tool) for tool in item.get("tools", [])),
                ]
            ).strip()
            skills.append(
                _build_skill(
                    skill_id=f"skillx_official_{granularity}_{index:03d}",
                    name=name,
                    granularity=granularity,
                    segments=members or segments,
                    all_segment_count=len(segments),
                    source_method="skillx_official",
                    content=content,
                    extra_metadata={
                        "official_output_path": str(output_path),
                        "skillx_metadata": metadata,
                        "skillx_tools": item.get("tools", []),
                        "source_match": "source_tasks" if members else "fallback_all_segments",
                    },
                )
            )
    return skills


def _skill_from_skilllens(
    *,
    index: int,
    item: dict[str, Any],
    segments: list[dict],
    output_path: Path,
) -> dict[str, Any]:
    sources = item.get("source_trajectories", [])
    members = _segments_for_sources(sources, segments)
    name = str(item.get("name") or f"SkillLens Skill {index}")
    references = item.get("references", [])
    scripts = item.get("scripts", [])
    content = "\n".join(
        [
            str(item.get("description", "")),
            str(item.get("body", "")),
            _named_blocks("References", references),
            _named_blocks("Scripts", scripts),
        ]
    ).strip()
    return _build_skill(
        skill_id=f"skilllens_official_{index:03d}",
        name=name,
        granularity="skilllens_skill",
        segments=members or segments,
        all_segment_count=len(segments),
        source_method="skilllens_official",
        content=content,
        extra_metadata={
            "official_output_path": str(output_path),
            "source_trajectories": sources,
            "source_match": "source_trajectories" if members else "fallback_all_segments",
        },
    )


def _named_blocks(title: str, blocks: Any) -> str:
    if not blocks:
        return ""
    if not isinstance(blocks, list):
        return f"{title}: {blocks}"
    pieces = [f"{title}:"]
    for block in blocks:
        if isinstance(block, dict):
            name = block.get("name") or block.get("path") or block.get("title") or "item"
            value = block.get("content") or block.get("body") or block
            pieces.append(f"- {name}: {value}")
        else:
            pieces.append(f"- {block}")
    return "\n".join(pieces)


def _build_skill(
    *,
    skill_id: str,
    name: str,
    granularity: str,
    segments: list[dict[str, Any]],
    all_segment_count: int,
    source_method: str,
    content: str,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    labels = Counter(segment.get("label", "Unknown") for segment in segments)
    templates = Counter(action_template(_segment_action(segment)) for segment in segments)
    rewards = [_final_reward(segment) for segment in segments]
    step_rewards = [_step_reward(segment) for segment in segments]
    successes = [_segment_success(segment) for segment in segments]
    recent = segments[-min(10, len(segments)) :] if segments else []
    avg_reward = _mean(rewards)
    metadata = {
        "support": len(segments),
        "coverage": len(segments) / all_segment_count if all_segment_count else 0.0,
        "success_rate": _mean(successes),
        "failure_rate": 1.0 - _mean(successes),
        "avg_reward": avg_reward,
        "avg_step_reward": _mean(step_rewards),
        "recent_success_rate": _mean([_segment_success(segment) for segment in recent]),
        "recent_reward_lift": _mean([_final_reward(segment) for segment in recent]) - avg_reward,
        "token_cost": len(tokenize(content)),
        "labels": compact_counter(labels),
        "action_templates": compact_counter(templates),
        "trajectory_count": len({_trajectory_id(segment) for segment in segments}),
        "source_method": source_method,
    }
    metadata.update(extra_metadata or {})
    skill = {
        "skill_id": skill_id,
        "name": name,
        "granularity": granularity,
        "members": [segment["segment_id"] for segment in segments],
        "content": content,
        "embedding_text": "",
        "metadata": metadata,
    }
    skill["embedding_text"] = "\n".join([name, granularity, content])
    return skill


def _segments_for_sources(sources: Iterable[Any], segments: list[dict]) -> list[dict]:
    by_trajectory = _group_by_trajectory(segments)
    by_goal = defaultdict(list)
    for segment in segments:
        goal = str(segment.get("metadata", {}).get("goal", ""))
        if goal:
            by_goal[goal].append(segment)

    selected = []
    seen = set()
    for source in sources or []:
        key = str(source)
        for segment in by_trajectory.get(key, []) + by_goal.get(key, []):
            segment_id = segment.get("segment_id")
            if segment_id not in seen:
                selected.append(segment)
                seen.add(segment_id)
    return selected


def _group_by_trajectory(segments: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped = defaultdict(list)
    for segment in segments:
        grouped[_trajectory_id(segment)].append(segment)
    return dict(grouped)


def _trajectory_id(segment: dict[str, Any]) -> str:
    return str(segment.get("metadata", {}).get("trajectory_id", "unknown"))


def _segment_action(segment: dict[str, Any]) -> str:
    return str(segment.get("metadata", {}).get("action", ""))


def _segment_success(segment: dict[str, Any]) -> bool:
    metadata = segment.get("metadata", {})
    if "trajectory_success" in metadata:
        return bool(metadata["trajectory_success"])
    return _final_reward(segment) > 0


def _final_reward(segment: dict[str, Any]) -> float:
    return float(segment.get("metadata", {}).get("final_reward", 0.0) or 0.0)


def _step_reward(segment: dict[str, Any]) -> float:
    return float(segment.get("metadata", {}).get("step_reward", 0.0) or 0.0)


def _mean(values: Iterable[float | bool]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(mean(float(value) for value in values))


def _segment_node(segment: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": segment["segment_id"],
        "label": segment.get("label", "Unknown"),
        "text": segment.get("text", ""),
        "metadata": segment.get("metadata", {}),
    }


def _first_markdown_heading(content: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    return ""
