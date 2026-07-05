from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any


def write_jsonl(path: Path, records: list[object]) -> None:
    # dataclass 轨迹写 jsonl，便于后续逐行流式分析或合并实验结果。
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def write_jsonl_dicts(path: Path, records: list[dict[str, Any]]) -> None:
    # dict 版本用于中间产物，比如 segments 和带模型输出的原始记录。
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, record: dict[str, Any]) -> None:
    # summary 用普通 json，方便直接查看和被脚本读取。
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
