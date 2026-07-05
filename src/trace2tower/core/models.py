from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StepRecord:
    # 单步交互记录，是所有后续切分、挖掘和评测的最小单位。
    t: int                    # 当前步在 episode 内的序号（从 1 开始）。
    observation: str          # 环境返回的文本观测。
    action: str               # agent 执行的动作字符串。
    reward: float = 0.0       # 该步获得的即时奖励。
    done: bool = False        # 该步之后 episode 是否结束。
    info: dict[str, Any] = field(default_factory=dict)  # 环境或 agent 附带的元信息。


@dataclass
class TrajectoryRecord:
    # 一个 episode 的完整轨迹；正式实验时 token/latency 可用于统计 LLM 成本。
    task_id: str
    env: str
    goal: str
    success: bool
    final_reward: float
    steps: list[StepRecord] = field(default_factory=list)
    token_cost: int = 0
    latency_sec: float = 0.0


@dataclass
class SegmentRecord:
    # 切分后的片段结构，后续会成为技能诱导的节点候选。
    segment_id: str
    label: str
    step_index: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillRecord:
    # 技能层的通用记录格式，先保留 granularity 方便表达多层级技能。
    skill_id: str
    name: str
    granularity: str          # 例如 trajectory / flat / planning / atomic 等。
    members: list[str] = field(default_factory=list)    # 属于该技能的 segment_id 列表。
    metadata: dict[str, Any] = field(default_factory=dict)  # 支持度、成功率、覆盖率等统计量。
