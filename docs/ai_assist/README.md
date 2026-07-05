# Trace2Tower 扩展占位说明

这份文档只回答一件事：**新算法、新论文思路、新环境接进来时，应该插到哪一层。**

## 工程层次

```text
env -> agent -> collection -> segmentation -> mining -> retrieval -> evaluation
```

## 该改哪里

| 新想法 | 放哪一层 | 说明 |
|---|---|---|
| 新环境 | `src/trace2tower/envs/` | 写 adapter，统一 `reset/step`。 |
| 新 agent | `src/trace2tower/agents/` | 只要输出 action。 |
| 新轨迹切分 | `src/trace2tower/segmentation/` | 输出 segment 列表。 |
| 新技能抽取 | `src/trace2tower/mining/` | 输入 segments，输出 graph/skills/model。 |
| 新检索 / 路由 | `src/trace2tower/retrieval/` | 输入 model + task_state，输出可注入技能。 |
| 新评测指标 | `src/trace2tower/evaluation/` | 汇总 trajectories / records。 |

## 接入原则

1. 先加接口，再加实现。
2. 先让新思路能单独跑通，再接到主流水线。
3. 不要改掉旧实现，除非新方法已经稳定替代它。
4. 一个新想法先用最小可运行实现落位，不保留伪结果。
5. 如果新方法需要额外中间产物，就在 `run.py` 同步输出新的 `.jsonl` 或 `.json`。

## 接手必读

- `06_research_experiment_brief_for_ai.md`：研究问题、实验设计、PUE 评估和禁止事项。
- `07_metrics_story_map.md`：每个指标能说明什么，以及如何支撑 RQ 和 baseline 对比。
- `08_core_code_map.md`：核心代码位置、数据流、接算法位置和输出文件。
- `09_llm_integration_status.md`：LLM 主流程、API 配置注意事项、smoke 结果和下一步。
- `05_experiment_skeleton_handoff.md`：当前代码骨架、baseline、selector 和接算法位置。

## 新算法接入模板

```python
class NewMiner(BaseMiner):
    def mine(self, segments: list[dict]) -> dict:
        ...
```

```python
class NewRetriever(BaseRetriever):
    def retrieve(self, model: dict, task_state: dict) -> list[dict]:
        ...
```

```json
{
  "miner": {"name": "new_method"},
  "retriever": {"name": "new_retriever"}
}
```

## 第一版最小标准

- 能跑。
- 能记录轨迹。
- 能导出 segments。
- 能导出 summary。
- 能把新方法插成一层，不影响旧层。
