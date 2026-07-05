from __future__ import annotations

from typing import Any

from .base import (
    BaselineMiner,
    build_skill,
    group_by_trajectory,
    segment_action,
    segment_node,
    sequential_edges,
)


class RawTrajectoryMiner(BaselineMiner):
    # 把每条轨迹整体作为一个可检索的 episodic memory，不做抽象。
    def mine(self, segments: list[dict]) -> dict:
        grouped = group_by_trajectory(segments)
        skills = []
        nodes = []
        edges = []

        for index, (trajectory_id, items) in enumerate(sorted(grouped.items())):
            skill = build_skill(
                skill_id=f"traj_{index:03d}",
                name=f"Trajectory Memory {trajectory_id}",
                granularity="trajectory",
                segments=items,
                all_segment_count=len(segments),
                source_method="raw_trajectory",
                content=_trajectory_content(items),
            )
            skill["metadata"]["trajectory_id"] = trajectory_id
            skills.append(skill)

            for segment in items:
                nodes.append(segment_node(segment))
            edges.extend(sequential_edges(items, relation="trajectory_order"))

        return {
            "method": "raw_trajectory",
            "description": "Runnable episodic-memory baseline: each collected trajectory becomes one retrievable memory card.",
            "nodes": nodes,
            "edges": edges,
            "skills": skills,
        }


def _trajectory_content(segments: list[dict[str, Any]]) -> str:
    # 生成轨迹级技能的内容摘要：目标 + 观测到的动作路径（最多 20 步）。
    goal = segments[0].get("metadata", {}).get("goal", "") if segments else ""
    actions = " -> ".join(segment_action(segment) for segment in segments[:20])
    return "\n".join(
        [
            f"Goal: {goal}",
            f"Observed action path: {actions}",
        ]
    )

