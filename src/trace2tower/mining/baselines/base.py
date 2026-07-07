from __future__ import annotations

import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from trace2tower.env import load_repo_dotenv, require_env
from trace2tower.mining.common import (
    build_skill,
    final_reward,
    group_by_trajectory,
    latest_file as common_latest_file,
    read_json,
    safe_mean,
    segment_action,
    segment_node,
    segment_success,
    segments_for_sources,
    sequential_edges,
    step_reward,
    trajectory_id,
    write_json,
    write_jsonl,
)


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


def latest_file(root: Path, name: str) -> Path:
    try:
        return common_latest_file(root, name)
    except FileNotFoundError as exc:
        raise OfficialBaselineError(str(exc)) from exc
