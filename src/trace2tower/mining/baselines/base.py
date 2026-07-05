from __future__ import annotations

import json
import os
import subprocess
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from trace2tower.env import load_repo_dotenv, require_env
from trace2tower.text import action_template, compact_counter, tokenize


class BaselineMiner(ABC):
    # 所有技能挖掘器的抽象基类，输入片段列表，输出统一格式的技能模型。
    method = "baseline"

    @abstractmethod
    def mine(self, segments: list[dict]) -> dict:
        raise NotImplementedError


class OfficialBaselineError(RuntimeError):
    # 官方 baseline 外部仓库缺失或命令失败时抛出，便于上层统一捕获。
    pass


class OfficialBaselineMiner(BaselineMiner):
    # 调用官方 baseline 外部仓库的适配器基类；子类只需指定 repo_name。
    method = "official_baseline"
    repo_name = ""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        load_repo_dotenv()
        self.config = config or {}
        self.repo_root = resolve_path(
            self.config.get("external_root"),
            default=repo_root() / ".external" / "baselines" / self.repo_name,
        )
        self.python = resolve_executable(self.config.get("python", "python"))
        self.llm_model = require_env("LLM_MODEL")
        self.timeout_sec = int(self.config.get("timeout_sec", 3600))
        self.runtime_output_dir = resolve_path(
            self.config.get("runtime_output_dir"),
            default=repo_root() / "experiments" / "official_baselines",
        )
        self.work_dir = resolve_path(
            self.config.get("work_dir"),
            default=self.runtime_output_dir / self.method,
        )

    def ensure_repo(self) -> None:
        # 要求外部仓库必须存在，避免在缺失时生成空模型误导实验结果。
        if not self.repo_root.exists():
            raise OfficialBaselineError(
                f"{self.method} requires the upstream repository at {self.repo_root}."
            )

    def run_command(
        self,
        command: list[str],
        *,
        cwd: Path,
        env: dict[str, str] | None = None,
        log_prefix: str,
    ) -> subprocess.CompletedProcess[str]:
        # 运行外部命令并捕获输出；失败时把 stderr/stdout  tail 写入异常方便排查。
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

    def env_with_pythonpath(self, *paths: Path) -> dict[str, str]:
        # 构造运行外部仓库所需的 PYTHONPATH 和 OpenAI 风格 API 环境变量。
        values = [str(path) for path in paths]
        existing = os.environ.get("PYTHONPATH")
        if existing:
            values.append(existing)
        return {
            "PYTHONPATH": os.pathsep.join(values),
            "OPENAI_API_KEY": require_env("LLM_API_KEY"),
            "OPENAI_BASE_URL": require_env("LLM_BASE_URL"),
        }

    def model_shell(
        self,
        *,
        description: str,
        skills: list[dict[str, Any]],
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]] | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        # 构造统一输出格式，并标记使用了官方运行时以便追溯。
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


def repo_root() -> Path:
    # 定位仓库根目录；当前文件位于 src/trace2tower/mining/baselines/base.py。
    return Path(__file__).resolve().parents[4]


def resolve_path(value: Any, *, default: Path) -> Path:
    # 把配置中的相对路径解析为仓库根目录下的绝对路径。
    if value in (None, ""):
        return default
    path = Path(str(value))
    return path if path.is_absolute() else repo_root() / path


def resolve_executable(value: Any) -> str:
    # 解析 python 可执行路径；若是裸命令名则交给 PATH 查找。
    executable = str(value or "python")
    path = Path(executable)
    if path.is_absolute():
        return str(path)
    if "/" in executable or "\\" in executable:
        return str(repo_root() / path)
    return executable


def build_skill(
    *,
    skill_id: str,
    name: str,
    granularity: str,
    segments: list[dict[str, Any]],
    all_segment_count: int,
    source_method: str,
    content: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # 基于一组片段构造统一格式的技能记录，并预计算各类统计元数据。
    labels = Counter(segment.get("label", "Unknown") for segment in segments)
    templates = Counter(action_template(segment_action(segment)) for segment in segments)
    rewards = [final_reward(segment) for segment in segments]
    step_rewards = [step_reward(segment) for segment in segments]
    successes = [segment_success(segment) for segment in segments]
    # 取最近最多 10 个片段计算近期指标，用于反映技能的时效性。
    recent = segments[-min(10, len(segments)) :] if segments else []
    avg_reward = safe_mean(rewards)
    skill_content = content or default_skill_content(name, granularity, labels, templates, avg_reward, successes)
    metadata = {
        "support": len(segments),
        "coverage": len(segments) / all_segment_count if all_segment_count else 0.0,
        "success_rate": safe_mean(successes),
        "failure_rate": 1.0 - safe_mean(successes),
        "avg_reward": avg_reward,
        "avg_step_reward": safe_mean(step_rewards),
        "recent_success_rate": safe_mean([segment_success(segment) for segment in recent]),
        "recent_reward_lift": safe_mean([final_reward(segment) for segment in recent]) - avg_reward,
        "token_cost": len(tokenize(skill_content)),
        "labels": compact_counter(labels),
        "action_templates": compact_counter(templates),
        "trajectory_count": len({trajectory_id(segment) for segment in segments}),
        "source_method": source_method,
    }
    metadata.update(extra_metadata or {})
    skill = {
        "skill_id": skill_id,
        "name": name,
        "granularity": granularity,
        "members": [segment["segment_id"] for segment in segments],
        "content": skill_content,
        "embedding_text": "",
        "metadata": metadata,
    }
    skill["embedding_text"] = skill_embedding_text(skill)
    return skill


def default_skill_content(
    name: str,
    granularity: str,
    labels: Counter[str],
    templates: Counter[str],
    avg_reward: float,
    successes: Iterable[bool],
) -> str:
    # 当外部没有提供 content 时，自动生成一段人类可读的技能描述文本。
    label_text = ", ".join(f"{label} ({count})" for label, count in labels.most_common(5)) or "none"
    template_text = ", ".join(f"{template} ({count})" for template, count in templates.most_common(5)) or "none"
    return "\n".join(
        [
            f"Skill: {name}",
            f"Granularity: {granularity}",
            f"Event patterns: {label_text}",
            f"Action templates: {template_text}",
            f"Historical success rate: {safe_mean(list(successes)):.3f}",
            f"Historical reward: {avg_reward:.3f}",
        ]
    )


def skill_embedding_text(skill: dict[str, Any]) -> str:
    # 用于向量检索或相似度计算的拼接文本。
    return "\n".join(
        [
            skill.get("name", ""),
            skill.get("granularity", ""),
            skill.get("content", ""),
        ]
    )


def segment_node(segment: dict[str, Any]) -> dict[str, Any]:
    # 把 segment 转成图节点格式，供技能模型中的 nodes 列表使用。
    return {
        "node_id": segment["segment_id"],
        "label": segment.get("label", "Unknown"),
        "text": segment.get("text", ""),
        "metadata": segment.get("metadata", {}),
    }


def sequential_edges(segments: list[dict[str, Any]], relation: str) -> list[dict[str, Any]]:
    # 在相邻 segment 之间建立顺序边，保留轨迹中的时序结构。
    return [
        {
            "source": previous["segment_id"],
            "target": current["segment_id"],
            "relation": relation,
            "weight": 1.0,
        }
        for previous, current in zip(segments, segments[1:])
    ]


def group_by_trajectory(segments: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    # 按轨迹 id 分组，很多 baseline 需要以整条轨迹为单位处理。
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for segment in segments:
        grouped[trajectory_id(segment)].append(segment)
    return dict(grouped)


def segments_for_sources(sources: Iterable[Any], segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # 根据官方 baseline 返回的 source 标识（ trajectory_id 或 goal）匹配本地片段。
    by_trajectory = group_by_trajectory(segments)
    by_goal: dict[str, list[dict[str, Any]]] = defaultdict(list)
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


def trajectory_id(segment: dict[str, Any]) -> str:
    return str(segment.get("metadata", {}).get("trajectory_id", "unknown"))


def segment_action(segment: dict[str, Any]) -> str:
    return str(segment.get("metadata", {}).get("action", ""))


def segment_success(segment: dict[str, Any]) -> bool:
    # 优先使用轨迹级成功标记；不存在时按最终奖励是否大于 0 推断。
    metadata = segment.get("metadata", {})
    if "trajectory_success" in metadata:
        return bool(metadata["trajectory_success"])
    return final_reward(segment) > 0


def final_reward(segment: dict[str, Any]) -> float:
    return float(segment.get("metadata", {}).get("final_reward", 0.0) or 0.0)


def step_reward(segment: dict[str, Any]) -> float:
    return float(segment.get("metadata", {}).get("step_reward", 0.0) or 0.0)


def safe_mean(values: Iterable[float | bool]) -> float:
    # 空序列返回 0.0，避免 statistics.mean 抛出异常。
    values = list(values)
    if not values:
        return 0.0
    return float(mean(float(value) for value in values))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def latest_file(root: Path, name: str) -> Path:
    # 官方 baseline 输出目录中可能有多份文件；按修改时间取最新一份。
    candidates = sorted(root.rglob(name), key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise OfficialBaselineError(f"Could not find {name} under {root}.")
    return candidates[-1]
