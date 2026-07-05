from __future__ import annotations

from .base import BaselineMiner


class NoSkillMiner(BaselineMiner):
    # 无技能基线：用于对比验证技能库是否真正带来性能提升。
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

