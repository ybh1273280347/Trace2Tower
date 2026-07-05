from __future__ import annotations

from collections import defaultdict

from .base import BaselineMiner, build_skill, segment_node


class FlatSkillSummaryMiner(BaselineMiner):
    # 按规则切分器生成的动作标签做扁平聚类，每个标签对应一个技能。
    def mine(self, segments: list[dict]) -> dict:
        grouped = defaultdict(list)
        for segment in segments:
            grouped[segment.get("label", "Unknown")].append(segment)

        skills = [
            build_skill(
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
            "nodes": [segment_node(segment) for segment in segments],
            "edges": [],
            "skills": skills,
        }

