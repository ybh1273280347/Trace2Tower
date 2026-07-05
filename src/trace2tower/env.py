from __future__ import annotations

import os
from pathlib import Path


def load_repo_dotenv() -> None:
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
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
    load_repo_dotenv()
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
