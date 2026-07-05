from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any, Iterable

from trace2tower.text import action_template, compact_counter, tokenize

from .base import BaseMiner


class NoSkillMiner(BaseMiner):
    def mine(self, segments: list[dict]) -> dict:
        return {
            "method": "no_skill",
            "description": "Runnable no-skill baseline. It intentionally exposes no reusable skills.",
            "nodes": [],
            "edges": [],
            "skills": [],
            "metadata": {
                "segment_count": len(segments),
            },
        }


class RawTrajectoryMiner(BaseMiner):
    def mine(self, segments: list[dict]) -> dict:
        grouped = _group_by_trajectory(segments)
        skills = []
        nodes = []
        edges = []

        for index, (trajectory_id, items) in enumerate(sorted(grouped.items())):
            skill = _build_skill(
                skill_id=f"traj_{index:03d}",
                name=f"Trajectory Memory {trajectory_id}",
                granularity="trajectory",
                segments=items,
                all_segment_count=len(segments),
                source_method="raw_trajectory",
            )
            skill["metadata"]["trajectory_id"] = trajectory_id
            skill["content"] = _trajectory_content(items)
            skill["embedding_text"] = _skill_embedding_text(skill)
            skill["metadata"]["token_cost"] = len(tokenize(skill["content"]))
            skills.append(skill)

            for segment in items:
                nodes.append(_segment_node(segment))
            edges.extend(_sequential_edges(items, relation="trajectory_order"))

        return {
            "method": "raw_trajectory",
            "description": "Runnable episodic-memory baseline: each collected trajectory becomes one retrievable memory card.",
            "nodes": nodes,
            "edges": edges,
            "skills": skills,
        }


class FlatSkillSummaryMiner(BaseMiner):
    def mine(self, segments: list[dict]) -> dict:
        grouped = defaultdict(list)
        for segment in segments:
            grouped[segment.get("label", "Unknown")].append(segment)

        skills = [
            _build_skill(
                skill_id=f"flat_{index:03d}",
                name=label,
                granularity="flat",
                segments=items,
                all_segment_count=len(segments),
                source_method="flat_skill_summary",
            )
            for index, (label, items) in enumerate(sorted(grouped.items()))
        ]
        return {
            "method": "flat_skill_summary",
            "description": "Runnable deterministic flat-skill baseline grouped by rule-segment labels.",
            "nodes": [_segment_node(segment) for segment in segments],
            "edges": [],
            "skills": skills,
        }


def _build_skill(
    *,
    skill_id: str,
    name: str,
    granularity: str,
    segments: list[dict[str, Any]],
    all_segment_count: int,
    source_method: str,
) -> dict[str, Any]:
    labels = Counter(segment.get("label", "Unknown") for segment in segments)
    templates = Counter(action_template(_segment_action(segment)) for segment in segments)
    rewards = [_final_reward(segment) for segment in segments]
    step_rewards = [_step_reward(segment) for segment in segments]
    successes = [_segment_success(segment) for segment in segments]
    recent = segments[-min(10, len(segments)) :] if segments else []
    recent_success_rate = _mean([_segment_success(segment) for segment in recent])
    avg_reward = _mean(rewards)
    recent_reward = _mean([_final_reward(segment) for segment in recent])
    content = _skill_content(name, granularity, labels, templates, avg_reward, successes)
    metadata = {
        "support": len(segments),
        "coverage": len(segments) / all_segment_count if all_segment_count else 0.0,
        "success_rate": _mean(successes),
        "failure_rate": 1.0 - _mean(successes),
        "avg_reward": avg_reward,
        "avg_step_reward": _mean(step_rewards),
        "recent_success_rate": recent_success_rate,
        "recent_reward_lift": recent_reward - avg_reward,
        "token_cost": len(tokenize(content)),
        "labels": compact_counter(labels),
        "action_templates": compact_counter(templates),
        "trajectory_count": len({_trajectory_id(segment) for segment in segments}),
        "source_method": source_method,
    }
    skill = {
        "skill_id": skill_id,
        "name": name,
        "granularity": granularity,
        "members": [segment["segment_id"] for segment in segments],
        "content": content,
        "embedding_text": "",
        "metadata": metadata,
    }
    skill["embedding_text"] = _skill_embedding_text(skill)
    return skill


def _skill_content(
    name: str,
    granularity: str,
    labels: Counter[str],
    templates: Counter[str],
    avg_reward: float,
    successes: Iterable[bool],
) -> str:
    label_text = ", ".join(f"{label} ({count})" for label, count in labels.most_common(5)) or "none"
    template_text = ", ".join(f"{template} ({count})" for template, count in templates.most_common(5)) or "none"
    success_rate = _mean(list(successes))
    return "\n".join(
        [
            f"Skill: {name}",
            f"Granularity: {granularity}",
            f"Event patterns: {label_text}",
            f"Action templates: {template_text}",
            f"Historical success rate: {success_rate:.3f}",
            f"Historical reward: {avg_reward:.3f}",
        ]
    )


def _skill_embedding_text(skill: dict[str, Any]) -> str:
    return "\n".join(
        [
            skill.get("name", ""),
            skill.get("granularity", ""),
            skill.get("content", ""),
        ]
    )


def _trajectory_content(segments: list[dict[str, Any]]) -> str:
    goal = segments[0].get("metadata", {}).get("goal", "") if segments else ""
    actions = " -> ".join(_segment_action(segment) for segment in segments[:20])
    return "\n".join(
        [
            f"Goal: {goal}",
            f"Observed action path: {actions}",
        ]
    )


def _segment_node(segment: dict[str, Any]) -> dict[str, Any]:
    return {
        "node_id": segment["segment_id"],
        "label": segment.get("label", "Unknown"),
        "text": segment.get("text", ""),
        "metadata": segment.get("metadata", {}),
    }


def _sequential_edges(segments: list[dict[str, Any]], relation: str) -> list[dict[str, Any]]:
    return [
        {
            "source": previous["segment_id"],
            "target": current["segment_id"],
            "relation": relation,
            "weight": 1.0,
        }
        for previous, current in zip(segments, segments[1:])
    ]


def _group_by_trajectory(segments: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for segment in segments:
        grouped[_trajectory_id(segment)].append(segment)
    return grouped


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
