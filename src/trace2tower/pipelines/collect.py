from __future__ import annotations

from pathlib import Path

from ..agents import BaseAgent
from ..envs import BaseEnv
from ..schemas import StepRecord, TrajectoryRecord


def collect_trajectories(env: BaseEnv, agent: BaseAgent, episodes: int, output: Path) -> list[TrajectoryRecord]:
    # 旧采集入口保留给小实验；主线输出 records/segments/summary 走 run.py。
    output.mkdir(parents=True, exist_ok=True)
    trajectories: list[TrajectoryRecord] = []

    for episode in range(episodes):
        observation, info = env.reset()
        steps: list[StepRecord] = []
        done = False
        reward = 0.0
        t = 0

        while not done:
            # 这个旧函数没有 max_steps，正式跑真实 benchmark 时不要直接用它。
            t += 1
            action = agent.act(observation, info)
            observation, reward, done, info = env.step(action)
            steps.append(StepRecord(t=t, observation=observation, action=action, reward=reward, done=done, info=info))

        trajectories.append(
            TrajectoryRecord(
                task_id=f"episode_{episode:03d}",
                env=getattr(env, "name", "unknown"),
                goal=str(steps[0].observation[:200]) if steps else "",
                success=reward > 0,
                final_reward=reward,
                steps=steps,
            )
        )

    return trajectories
