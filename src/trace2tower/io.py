from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any


def write_jsonl(path: Path, records: list[object]) -> None:
    # dataclass 轨迹写 jsonl，便于后续逐行流式分析或合并实验结果。
    # 自动创建父目录，避免调用方遗漏。
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            # ensure_ascii=False 保留中文观测文本，便于人工查看。
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def write_jsonl_dicts(path: Path, records: list[dict[str, Any]]) -> None:
    # dict 版本用于中间产物，比如 segments 和带模型输出的原始记录。
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            # 每条记录独立一行，方便 pandas / jq 等工具流式处理。
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, record: dict[str, Any]) -> None:
    # summary 用普通 json，方便直接查看和被脚本读取。
    path.parent.mkdir(parents=True, exist_ok=True)
    # indent=2 让人类可读；ensure_ascii=False 保留非 ASCII 字符。
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    # 读取由 write_json 写入的 summary 或外部模型文件。
    return json.loads(path.read_text(encoding="utf-8"))
