from __future__ import annotations

from typing import Any

from .schemas import StepRecord, TrajectoryRecord


def run_episodes(
    *,
    env: Any,
    agent: Any,
    segmenter: Any,
    retriever: Any,
    env_name: str,
    episodes: int,
    max_steps: int,
    deployment_model: dict[str, Any] | None = None,
) -> dict[str, list]:
    trajectories: list[TrajectoryRecord] = []
    raw_records: list[dict[str, Any]] = []
    segment_records: list[dict[str, Any]] = []
    deployment_retrieval_records: list[dict[str, Any]] = []

    for episode in range(episodes):
        observation, info = env.reset()
        goal = extract_goal(env_name, observation)
        steps: list[StepRecord] = []
        recent_actions: list[str] = []
        done = False
        reward = 0.0
        t = 0
        task_id = f"{env_name}_{episode:03d}"

        while not done and t < max_steps:
            t += 1
            retrieved_skills = retrieve_deployment_skills(
                model=deployment_model,
                retriever=retriever,
                goal=goal,
                env=env_name,
                observation=observation,
                recent_actions=recent_actions,
            )
            if retrieved_skills:
                deployment_retrieval_records.append(
                    {
                        "task_id": task_id,
                        "step": t,
                        "retrieved_skills": retrieved_skills,
                    }
                )

            agent_info = dict(info)
            agent_info["goal"] = goal
            agent_info["retrieved_skills"] = retrieved_skills
            action = agent.act(observation, agent_info)
            observation, reward, done, info = env.step(action)
            step_info = dict(info)
            agent_metadata = getattr(agent, "last_metadata", {})
            if agent_metadata:
                step_info["agent"] = agent_metadata
            steps.append(
                StepRecord(
                    t=t,
                    observation=observation,
                    action=action,
                    reward=reward,
                    done=done,
                    info=step_info,
                )
            )
            recent_actions.append(action)

        trajectory = TrajectoryRecord(
            task_id=task_id,
            env=env_name,
            goal=goal,
            success=reward > 0,
            final_reward=reward,
            steps=steps,
        )
        trajectories.append(trajectory)
        raw = {
            "task_id": trajectory.task_id,
            "env": trajectory.env,
            "goal": trajectory.goal,
            "success": trajectory.success,
            "final_reward": trajectory.final_reward,
            "steps": [
                {
                    "t": step.t,
                    "observation": step.observation,
                    "action": step.action,
                    "reward": step.reward,
                    "done": step.done,
                    "info": step.info,
                }
                for step in steps
            ],
        }
        segments = segmenter.segment(raw)
        raw["segments"] = segments
        raw_records.append(raw)
        segment_records.extend(segments)

    return {
        "trajectories": trajectories,
        "records": raw_records,
        "segments": segment_records,
        "deployment_retrieval": deployment_retrieval_records,
    }


def retrieve_deployment_skills(
    *,
    model: dict[str, Any] | None,
    retriever: Any,
    goal: str,
    env: str,
    observation: str,
    recent_actions: list[str],
) -> list[dict[str, Any]]:
    if not model:
        return []
    return retriever.retrieve(
        model,
        {
            "goal": goal,
            "env": env,
            "observation": observation,
            "recent_actions": recent_actions[-5:],
            "segments": [],
        },
    )


def extract_goal(env_name: str, observation: str) -> str:
    if env_name == "webshop" and "Instruction:" in observation:
        tail = observation.split("Instruction:", 1)[1]
        parts = [
            part.strip()
            for part in tail.split("[SEP]")
            if part.strip() and part.strip().lower() not in {"instruction:"}
        ]
        return parts[0] if parts else ""
    if env_name == "alfworld":
        return observation.split("\n", 1)[0].strip()
    return observation[:200]
