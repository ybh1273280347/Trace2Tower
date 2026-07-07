from __future__ import annotations

from typing import Any

from .base import (
    OfficialBaselineError,
    OfficialBaselineMiner,
    build_skill,
    final_reward,
    group_by_trajectory,
    latest_file,
    read_json,
    segment_action,
    segment_node,
    segment_success,
    segments_for_sources,
    step_reward,
    write_json,
)


class SkillLensOfficialMiner(OfficialBaselineMiner):
    # 调用 SkillLens 官方仓库 `skilllens extract` CLI 的技能抽取适配器。
    method = "skilllens_official"
    repo_name = "SkillLens"

    def mine(self, segments: list[dict]) -> dict:
        self.ensure_repo()
        if not segments:
            return self.model_shell(
                description=(
                    "Official SkillLens extraction adapter received no "
                    "segments, so no upstream extraction was launched."
                ),
                skills=[],
                nodes=[],
                extra_metadata={"skipped_reason": "no_segments"},
            )

        self.work_dir.mkdir(parents=True, exist_ok=True)
        input_path = self.work_dir / "skilllens_input.json"
        output_dir = self.work_dir / "official_output"
        # 把 Trace2Tower 片段转成 SkillLens 的 trajectory 输入格式。
        write_json(input_path, _skilllens_input(segments, self.config))

        # 构造 `skilllens extract` 命令，并用 --set 注入配置覆盖。
        command = [
            self.python,
            "-m",
            "skilllens",
            "extract",
            "-i",
            str(input_path),
            "-o",
            str(output_dir),
        ]
        for key, value in self._overrides().items():
            command.extend(["--set", f"{key}={value}"])
        if self.config.get("verbose", False):
            command.append("--verbose")

        self.run_command(
            command,
            cwd=self.repo_root,
            env=self.env_with_pythonpath(self.repo_root),
            log_prefix="skilllens_extract",
        )
        # SkillLens 可能在输出目录产生多个 skill_set.json，取最新的一份。
        skill_set_path = latest_file(output_dir, "skill_set.json")
        skill_set = read_json(skill_set_path)
        skills = [
            _skill_from_skilllens(
                index=index,
                item=item,
                segments=segments,
                output_path=skill_set_path,
            )
            for index, item in enumerate(skill_set.get("skills", []))
        ]
        return self.model_shell(
            description=(
                "Official SkillLens extraction run through the upstream "
                "`skilllens extract` CLI on Trace2Tower-converted trajectories."
            ),
            skills=skills,
            nodes=[segment_node(segment) for segment in segments],
            extra_metadata={
                "official_input_path": str(input_path),
                "official_output_path": str(skill_set_path),
                "official_extractor_model": skill_set.get("extractor_model"),
                "official_extraction_method": skill_set.get("extraction_method"),
            },
        )

    def _overrides(self) -> dict[str, Any]:
        # 把 Trace2Tower 配置映射到 SkillLens Hydra 风格的配置键。
        overrides = {
            "input.benchmark": self.config.get("benchmark", "trace2tower"),
            "output.run_name": self.config.get("run_name", "trace2tower"),
            "llm.provider": self.config.get("provider", "openai"),
            "llm.model": self.llm_model,
            "llm.api_style": self.config.get("api_style", "chat"),
            "extraction.method": self.config.get("method", "parallel"),
            "extraction.max_skills": int(self.config.get("max_skills", 8)),
            "extraction.batch_size": int(self.config.get("batch_size", 0)),
            "extraction.merge_group_size": int(self.config.get("merge_group_size", 4)),
            "extraction.max_concurrency": int(self.config.get("max_concurrency", 4)),
        }
        optional_keys = {
            "temperature": "llm.temperature",
            "max_tokens": "llm.max_tokens",
            "max_tool_rounds": "extraction.max_tool_rounds",
            "meta_skill_guidance": "extraction.meta_skill_guidance",
        }
        for config_key, override_key in optional_keys.items():
            if config_key in self.config and self.config[config_key] is not None:
                overrides[override_key] = self.config[config_key]
        return overrides


def _skilllens_input(segments: list[dict], config: dict[str, Any]) -> dict[str, Any]:
    # 将 Trace2Tower 片段重构为 SkillLens 输入格式：trajectory 列表 + steps 对话历史。
    trajectories = []
    for trajectory_id, items in sorted(group_by_trajectory(segments).items()):
        metadata = items[0].get("metadata", {}) if items else {}
        goal = str(metadata.get("goal", ""))
        success = segment_success(items[-1]) if items else False
        steps = [
            {
                "role": "user",
                "content": goal,
                "metadata": {
                    "trace2tower_event": "goal",
                    "trajectory_id": trajectory_id,
                },
            }
        ]
        for segment in items:
            steps.append(
                {
                    "role": "agent",
                    "content": f"Action: {segment_action(segment)}",
                    "observation": segment.get("text", ""),
                    "metadata": {
                        "segment_id": segment.get("segment_id"),
                        "label": segment.get("label", "Unknown"),
                        "step_index": segment.get("step_index"),
                        "reward": step_reward(segment),
                    },
                }
            )
        trajectories.append(
            {
                "id": trajectory_id,
                "task_id": trajectory_id,
                "task_name": goal,
                "agent": config.get("agent", "trace2tower"),
                "steps": steps,
                "reward": final_reward(items[-1]) if items else 0.0,
                "benchmark": metadata.get("env", config.get("benchmark", "trace2tower")),
                "outcome": "resolved" if success else "unresolved",
                "source_format": "trace2tower_segments",
                "metadata": {
                    "trace2tower_env": metadata.get("env", ""),
                    "trace2tower_goal": goal,
                    "segment_ids": [segment.get("segment_id") for segment in items],
                },
            }
        )
    return {
        "trajectories": trajectories,
        "source": "trace2tower",
        "metadata": {
            "segment_count": len(segments),
            "adapter": "skilllens_official",
        },
    }


def _skill_from_skilllens(
    *,
    index: int,
    item: dict[str, Any],
    segments: list[dict],
    output_path,
) -> dict[str, Any]:
    # 把 SkillLens 输出的单个 skill 项映射为统一技能格式。
    sources = item.get("source_trajectories", [])
    members = segments_for_sources(sources, segments)
    name = str(item.get("name") or f"SkillLens Skill {index}")
    if not members:
        raise OfficialBaselineError(
            "SkillLens output skill cannot be mapped back to Trace2Tower segments: "
            f"name={name}, source_trajectories={sources}, output={output_path}"
        )

    content = "\n".join(
        [
            str(item.get("description", "")),
            str(item.get("body", "")),
            _named_blocks("References", item.get("references", [])),
            _named_blocks("Scripts", item.get("scripts", [])),
        ]
    ).strip()
    return build_skill(
        skill_id=f"skilllens_official_{index:03d}",
        name=name,
        granularity="skilllens_skill",
        segments=members,
        all_segment_count=len(segments),
        source_method="skilllens_official",
        content=content,
        extra_metadata={
            "official_output_path": str(output_path),
            "source_trajectories": sources,
        },
    )


def _named_blocks(title: str, blocks: Any) -> str:
    # 把 SkillLens 的 references / scripts 块格式化为可读列表。
    if not blocks:
        return ""
    if not isinstance(blocks, list):
        return f"{title}: {blocks}"
    pieces = [f"{title}:"]
    for block in blocks:
        if isinstance(block, dict):
            name = block.get("name") or block.get("path") or block.get("title") or "item"
            value = block.get("content") or block.get("body") or block
            pieces.append(f"- {name}: {value}")
        else:
            pieces.append(f"- {block}")
    return "\n".join(pieces)
