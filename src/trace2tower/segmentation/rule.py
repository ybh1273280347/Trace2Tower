from __future__ import annotations

from typing import Any

from .base import BaseSegmenter


class RuleSegmenter(BaseSegmenter):
    def segment(self, trajectory: dict[str, Any]) -> list[dict[str, Any]]:
        # 先按动作模板做最小切分，后续可以替换成 LLM 切分或更细的规则切分。
        # 目前每步生成一个 segment，保留足够元数据供后续统计使用。
        return [
            {
                "segment_id": f"{trajectory['task_id']}_seg_{step['t']}",
                "label": self._label(step["action"]),
                "step_index": step["t"],
                "text": self._segment_text(trajectory, step),
                "metadata": {
                    "trajectory_id": trajectory["task_id"],
                    "env": trajectory["env"],
                    "goal": trajectory["goal"],
                    "trajectory_success": trajectory["success"],
                    "final_reward": trajectory["final_reward"],
                    "step_reward": step.get("reward", 0.0),
                    "done": step.get("done", False),
                    "action": step["action"],
                },
            }
            for step in trajectory["steps"]
        ]

    def _label(self, action: str) -> str:
        # 这里先把 WebShop / ALFWorld 常见动作映射成最粗的事件标签。
        if action.startswith("search["):
            return "QueryFormulation"
        if action.startswith("click[buy"):
            return "PurchaseDecision"
        if action.startswith("click["):
            return "CandidateInteraction"
        if action.startswith("take "):
            return "AcquireObject"
        if action.startswith("put "):
            return "PlaceObject"
        if action.startswith("go to "):
            return "NavigateToTarget"
        if action.startswith("heat ") or action.startswith("cool ") or action.startswith("clean "):
            return "TransformObject"
        return action.split("[", 1)[0]

    def _segment_text(self, trajectory: dict[str, Any], step: dict[str, Any]) -> str:
        # 把动作、观测、目标、结果拼成自然语言文本，作为技能内容和检索查询的素材。
        observation = step.get("observation", "")
        return "\n".join(
            [
                f"Env: {trajectory['env']}",
                f"Goal: {trajectory['goal']}",
                f"Action: {step['action']}",
                f"Observation: {observation[:500]}",
                f"Outcome: success={trajectory['success']}, reward={trajectory['final_reward']}",
            ]
        )
