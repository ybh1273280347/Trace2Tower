from __future__ import annotations

import os
from pathlib import Path


def load_repo_dotenv() -> None:
    # 从仓库根目录的 .env 文件加载环境变量；不覆盖已存在的变量。
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        # 跳过空行与注释行。
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _unquote(value.strip())
        if key and key not in os.environ:
            os.environ[key] = value

    # 官方 baseline 多数只认 OPENAI_*；唯一事实源仍是 LLM_*。
    if "LLM_API_KEY" in os.environ:
        os.environ["OPENAI_API_KEY"] = os.environ["LLM_API_KEY"]
    if "LLM_BASE_URL" in os.environ:
        os.environ["OPENAI_BASE_URL"] = os.environ["LLM_BASE_URL"]


def require_env(name: str) -> str:
    # 读取指定环境变量；不存在时给出明确错误，避免在运行时才发现缺失。
    load_repo_dotenv()
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _unquote(value: str) -> str:
    # 去除 .env 值两端的一对引号，保留中间内容。
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
