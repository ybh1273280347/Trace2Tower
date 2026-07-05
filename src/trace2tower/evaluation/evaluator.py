from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Evaluator:
    # metrics 先保留配置字段，后续可以按名称分发到更细的指标实现。
    metrics: list[str] = field(default_factory=list)

    def summarize(self, trajectories: list[dict]) -> dict:
        # 计算一组轨迹的核心指标：成功率、步数、奖励、无效动作率、循环率、LLM 成本等。
        success_count = sum(1 for item in trajectories if item["success"])
        total = len(trajectories) or 1
        avg_steps = sum(len(item["steps"]) for item in trajectories) / total
        avg_reward = sum(item["final_reward"] for item in trajectories) / total
        invalid_actions = sum(
            1
            for item in trajectories
            for step in item["steps"]
            if self._is_invalid_step(step)
        )
        total_steps = sum(len(item["steps"]) for item in trajectories) or 1
        token_cost = sum(
            int(step.get("info", {}).get("agent", {}).get("total_tokens", 0) or 0)
            for item in trajectories
            for step in item["steps"]
        )
        llm_fallbacks = sum(
            1
            for item in trajectories
            for step in item["steps"]
            if step.get("info", {}).get("agent", {}).get("llm_action_fallback")
        )
        loop_rate = sum(self._loop_count(item["steps"]) for item in trajectories) / total_steps
        return {
            "success_rate": success_count / total,
            "episodes": len(trajectories),
            "avg_steps": avg_steps,
            "avg_reward": avg_reward,
            "invalid_action_rate": invalid_actions / total_steps,
            "loop_rate": loop_rate,
            "avg_token_cost": token_cost / total,
            "llm_action_fallback_rate": llm_fallbacks / total_steps,
        }

    def _is_invalid_step(self, step: dict) -> bool:
        # 优先使用环境显式标记的 valid_action；否则从观测文本中匹配常见无效提示。
        info = step.get("info", {})
        if "valid_action" in info:
            return not bool(info["valid_action"])

        observation = step.get("observation", "").lower()
        return any(
            marker in observation
            for marker in [
                "invalid",
                "not a valid",
                "you can't",
                "cannot",
            ]
        )

    def _loop_count(self, steps: list[dict]) -> int:
        # 统计连续两步动作相同的次数，作为循环/抖动行为的代理指标。
        return sum(
            1
            for previous, current in zip(steps, steps[1:])
            if previous.get("action") == current.get("action")
        )
