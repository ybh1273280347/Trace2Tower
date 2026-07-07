from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from trace2tower.text import action_template, compact_counter, tokenize


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
    # 按轨迹 id 分组，Trace2Tower 和 baseline 都需要以整条轨迹为单位处理转移。
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for segment in segments:
        grouped[trajectory_id(segment)].append(segment)
    return dict(grouped)


def segments_for_sources(sources: Iterable[Any], segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # 根据官方 baseline 返回的 source 标识（trajectory_id 或 goal）匹配本地片段。
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
        raise FileNotFoundError(f"Could not find {name} under {root}.")
    return candidates[-1]
